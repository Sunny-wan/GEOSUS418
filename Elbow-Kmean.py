import os
import time
import argparse
import rasterio
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# Performance optimization settings
os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_count() // 2))
np.set_printoptions(suppress=True)


def read_raster(input_path):
    """Read raster data"""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: File does not exist - {input_path}")
        raise FileNotFoundError(f"File not found: {input_path}")

    try:
        with rasterio.open(input_path) as src:
            data = src.read()
            profile = src.profile.copy()
        print(f"Raster data loaded: {input_path}, shape: {data.shape}")
        return data, profile
    except Exception as e:
        print(f"Failed to read raster: {str(e)}")
        raise


def preprocess_data(data, fill_value=None, sample_ratio=1.0, max_sample_size=500000):
    """Preprocess data (default sample ratio 1.0)"""
    original_shape = data.shape
    num_bands = data.shape[0]
    print(f"Preprocessing data, original shape: {original_shape}")

    # Handle fill values
    data = data.astype(np.float32, copy=False)
    if fill_value is not None:
        np.putmask(data, data == fill_value, np.nan)
        print(f"Processed fill value: {fill_value}")

    # Create valid mask and get valid indices
    valid_mask = ~np.isnan(data).any(axis=0)
    valid_indices = np.flatnonzero(valid_mask)
    total_valid = len(valid_indices)
    print(f"Valid pixels: {total_valid} ({total_valid / valid_mask.size * 100:.1f}%)")

    # Extract valid data and normalize
    flat_data = data.reshape(num_bands, -1).T[valid_indices]

    # Sampling process (use full data when sample ratio is 1)
    if sample_ratio < 1.0 and total_valid > 10000:
        theoretical_size = int(total_valid * sample_ratio)
        sample_size = max(10000, min(theoretical_size, max_sample_size))
        sample_size = min(sample_size, total_valid)

        rng = np.random.default_rng(42)
        sample_indices = rng.choice(total_valid, sample_size, replace=False, shuffle=False)
        sampled_data = flat_data[sample_indices]
        print(
            f"Sampled evaluation data: {sample_size} samples (total valid: {total_valid}, sample ratio: {sample_ratio:.1f})")
    else:
        sampled_data = flat_data
        sample_indices = np.arange(total_valid)
        print(f"Using full valid data: {len(sampled_data)} pixels (sample ratio: {sample_ratio:.1f})")

    # Calculate mean and std (based on full valid data)
    mean = np.mean(flat_data, axis=0, dtype=np.float32)
    std = np.std(flat_data, axis=0, dtype=np.float32)
    std[std < 1e-6] = 1.0

    # Normalization
    normalized_data = (flat_data - mean) / std
    normalized_sampled = (sampled_data - mean) / std

    print("Data preprocessing completed")
    return (normalized_data, normalized_sampled, sample_indices,
            valid_indices, original_shape, mean, std, flat_data, total_valid)


def perform_clustering(data, n_clusters):
    """Perform clustering"""
    print(f"Performing {n_clusters} clusters...")
    try:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init='auto',
            max_iter=100,
            batch_size=1024 * 16,
            verbose=0
        )
        clusters = kmeans.fit_predict(data)
        return clusters, kmeans.cluster_centers_, kmeans.inertia_  # inertia is SSE
    except Exception as e:
        print(f"Clustering failed: {str(e)}")
        raise


def save_cluster_result(clusters, valid_indices, original_shape, profile, output_path):
    """Save clustering result"""
    try:
        result = np.zeros(original_shape[1] * original_shape[2], dtype=np.uint8)
        result[valid_indices] = clusters + 1

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress='deflate',
            predictor=2
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(result.reshape(original_shape[1:]), 1)

        print(f"Result saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to save result: {str(e)}")
        raise


def save_cluster_preview(clusters, valid_indices, original_shape, output_path, n_clusters):
    """Save clustering preview image"""
    try:
        result = np.zeros(original_shape[1] * original_shape[2], dtype=np.uint8)
        result[valid_indices] = clusters + 1
        result_2d = result.reshape(original_shape[1:])

        if max(result_2d.shape) > 2000:
            scale = 2000 / max(result_2d.shape)
            result_2d = result_2d[::int(1 / scale), ::int(1 / scale)]

        plt.figure(figsize=(8, 6), dpi=100)
        plt.imshow(result_2d, cmap='viridis')
        plt.colorbar(label='Cluster class')
        plt.title(f'K={n_clusters} Clustering Result (Elbow Method)')
        plt.axis('off')

        preview_path = output_path.parent / f"preview_{n_clusters}.png"
        plt.savefig(preview_path, bbox_inches='tight')
        plt.close()
        print(f"Preview saved: {preview_path}")

    except Exception as e:
        print(f"Failed to save preview: {str(e)}")


def calculate_evaluation_metrics(normalized_data, clusters, n_clusters, inertia):
    """Calculate evaluation metrics (SSE only)"""
    # Mean inter-cluster distance (optional, doesn't affect elbow method)
    unique_labels = np.unique(clusters)
    centroids = np.array([np.mean(normalized_data[clusters == label], axis=0)
                          for label in unique_labels])
    mean_inter_distance = 0.0
    if len(centroids) > 1:
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distances.append(np.linalg.norm(centroids[i] - centroids[j]))
        mean_inter_distance = np.mean(distances) if distances else 0

    return {
        'mean_inter_distance': mean_inter_distance,
        'inertia': inertia  # Use SSE from K-means directly
    }


def find_elbow_point(k_list, inertia_list):
    if len(k_list) < 3:
        return k_list[-1]

    # Calculate first-order difference (SSE change)
    first_diffs = np.diff(inertia_list)
    # Calculate second-order difference (change rate of SSE change)
    second_diffs = np.diff(first_diffs)

    # Find position of maximum second-order difference (elbow point)
    elbow_idx = np.argmax(second_diffs) + 1  # Align with original K list index
    best_k = k_list[elbow_idx + 1]  # +1 to compensate for second-order difference dimension loss

    return best_k


def save_evaluation_chart(evaluation_results, output_dir, k_range):
    """Save elbow method evaluation chart (SSE curve only) - 样式优化版"""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        k_list = [r['n_clusters'] for r in evaluation_results]
        inertia_list = [r['inertia'] for r in evaluation_results]

        # Select optimal K using elbow method
        best_k = find_elbow_point(k_list, inertia_list)
        best_idx = k_list.index(best_k)

        # Create chart (SSE curve only) - 样式优化核心代码
        plt.figure(figsize=(10, 6), dpi=600)

        # 绘制SSE曲线
        plt.plot(k_list, inertia_list, 'o-', color='darkblue', linewidth=3, markersize=8,
                 label='SSE (Sum of Squared Errors)')
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.8, linewidth=2,
                    label=f'Elbow point K: {best_k} (SSE={inertia_list[best_idx]:.2f})')

        # 1. 统一设置Arial字体 + 放大字号
        font_config = {'family': 'Arial', 'size': 14}  # 轴标签字体大小
        plt.xlabel('Number of clusters', **font_config)
        plt.ylabel('SSE (Sum of Squared Errors)', **font_config)


        # 刻度字体放大
        plt.xticks(fontproperties='Arial', size=12)
        plt.yticks(fontproperties='Arial', size=12)

        # 2. 去掉图例背景框 + 图例字体放大
        plt.legend(prop={'family': 'Arial', 'size': 14}, frameon=False)  # frameon=False 移除背景框

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        chart_path = output_dir / 'cluster_evaluation_elbow_method.jpg'
        plt.savefig(chart_path)
        plt.close()
        print(f"Elbow method evaluation chart saved to: {chart_path}")

        return best_k

    except Exception as e:
        print(f"Failed to save evaluation chart: {str(e)}")
        return None


def process_single_cluster(n_clusters, normalized_data, normalized_sampled, sample_indices,
                           valid_indices, original_shape, profile, output_dir,
                           original_flat_data, mean, std, k_range):
    """Single clustering task processing function (RMSE removed)"""
    try:
        start_time = time.time()

        # Perform clustering (using full data)
        clusters, centers, inertia = perform_clustering(normalized_data, n_clusters)
        sampled_clusters = clusters[sample_indices]

        # Save results
        output_path = output_dir / f"clusters_{n_clusters}.tif"
        save_cluster_result(clusters, valid_indices, original_shape, profile, output_path)
        save_cluster_preview(clusters, valid_indices, original_shape, output_path, n_clusters)

        # Calculate evaluation metrics (SSE only)
        metrics = calculate_evaluation_metrics(normalized_sampled, sampled_clusters, n_clusters, inertia)

        elapsed_time = time.time() - start_time
        result = {
            'n_clusters': n_clusters,
            'inertia': metrics['inertia'],  # SSE
            'mean_inter_distance': metrics['mean_inter_distance'],
            'output_path': str(output_path),
            'time': elapsed_time
        }

        print(
            f"Clustering {n_clusters} completed: SSE={metrics['inertia']:.2f}, "
            f"time elapsed={elapsed_time:.1f}s"
        )
        return result

    except Exception as e:
        print(f"Error processing K={n_clusters}: {str(e)}")
        return None


def main(args):
    try:
        start_total = time.time()

        # Initialization
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Starting K-means clustering analysis (elbow method, sample ratio={args.sample_ratio}), results saved to: {output_dir}")

        # Read and preprocess data
        print(f"Processing data: {args.input_path}")
        data, profile = read_raster(args.input_path)
        (normalized_data, normalized_sampled, sample_indices,
         valid_indices, original_shape, mean, std, original_flat_data, total_valid) = preprocess_data(
            data,
            fill_value=args.fill_value,
            sample_ratio=args.sample_ratio,
            max_sample_size=args.max_sample_size
        )

        # Set clustering range
        clusters_range = range(args.min_clusters, args.max_clusters + 1)
        print(f"Clustering range: {args.min_clusters} to {args.max_clusters}")

        # Parallel processing of clustering tasks
        cluster_func = partial(
            process_single_cluster,
            normalized_data=normalized_data,
            normalized_sampled=normalized_sampled,
            sample_indices=sample_indices,
            valid_indices=valid_indices,
            original_shape=original_shape,
            profile=profile,
            output_dir=output_dir,
            original_flat_data=original_flat_data,
            mean=mean,
            std=std,
            k_range=clusters_range
        )

        if args.parallel:
            max_workers = min(len(clusters_range), max(1, cpu_count() - 1))
            print(f"Using parallel computing, number of processes: {max_workers}")
            with Pool(processes=max_workers) as pool:
                results = list(tqdm(
                    pool.imap(cluster_func, clusters_range),
                    total=len(clusters_range),
                    desc="Clustering progress"
                ))
        else:
            results = [cluster_func(n) for n in tqdm(clusters_range, desc="Clustering progress")]

        # Filter valid results
        evaluation_results = [r for r in results if r is not None]
        if not evaluation_results:
            print("All clustering tasks failed")
            return

        # Select optimal K using elbow method
        best_k = save_evaluation_chart(evaluation_results, output_dir, clusters_range)
        if best_k is None:
            best_k = find_elbow_point([r['n_clusters'] for r in evaluation_results],
                                      [r['inertia'] for r in evaluation_results])

        best_result = next(r for r in evaluation_results if r['n_clusters'] == best_k)

        # Output optimal results (RMSE removed)
        print(f"\n===== Optimal Clustering Results (Elbow Method) =====")
        print(f"Optimal K value: {best_k} clusters (SSE inflection point)")
        print(f"SSE: {best_result['inertia']:.2f}")
        print(f"Result file: {best_result['output_path']}")

        total_time = time.time() - start_total
        print(f"\nAll clustering completed! Total time elapsed: {total_time:.1f}s")
        print(f"Total valid pixels: {total_valid}")
        print(f"Sample ratio: {args.sample_ratio}")

    except Exception as e:
        print(f"Program execution error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='KMeans Raster Clustering Tool using Elbow Method (default sample ratio 1.0)')
    parser.add_argument('--input_path', type=str,
                        default=r"H:\ArcGIS_workspace\YGCB22\do2\eledata\data\final_fused_result2.tif",
                        help='Input raster data path')
    parser.add_argument('--output_dir', type=str,
                        default=r"H:\ArcGIS_workspace\YGCB22\do2\eledata\cluster_results2",
                        help='Output results directory')
    parser.add_argument('--min_clusters', type=int, default=3)
    parser.add_argument('--max_clusters', type=int, default=10)
    parser.add_argument('--fill_value', type=float,
                        default=-3.4028234663852886e+38, help='No-data value marker')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Sample ratio (default: 1.0 for full data)')
    parser.add_argument('--max_sample_size', type=int, default=500000,
                        help='Maximum sample size (effective when sample ratio < 1)')
    parser.add_argument('--parallel', action='store_true', help='Use parallel computing')

    args = parser.parse_args()
    main(args)