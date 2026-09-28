import os
import torch
import numpy as np
import rasterio
from tqdm import tqdm
import argparse
import traceback

# Import the Erosion Fusion Model from the data model file
from data_model import (
    ErosionFusionUNet,
    collect_data_paths  # Reuse path collection logic
)


# ----------------------------
# Path configuration fully aligned with main.py (1:1 replication)
# ----------------------------
class FixedConfig:
    """Path configuration is exactly the same as main.py, with input paths fixed to your erosion data directory"""
    # 1. Input paths (the three types of erosion data directories you specified)
    rusle_dir = r"H:\ArcGIS_workspace\YGCB22\do1\rusle"
    wind_dir = r"H:\ArcGIS_workspace\YGCB22\do1\wind"
    freeze_dir = r"H:\ArcGIS_workspace\YGCB22\do1\freeze"

    # Input file name format (the format you specified)
    rusle_filename = "{}.tif"
    wind_filename = "{}.tif"
    freeze_filename = "{}.tif"

    # 2. Output paths (exactly the same as main.py)
    output_dir = r"H:\ArcGIS_workspace\YGCB22\do1\outt"  # Root output directory (same as main)
    model_dir = os.path.join(output_dir, "models")  # Model weight directory (same as main)
    loss_dir = os.path.join(output_dir, "loss_records")  # Loss record directory (same as main; not used during prediction but structure retained)
    prediction_dir = os.path.join(output_dir, "predictions")  # Prediction result directory (extended based on main root directory to maintain consistency)

    # 3. Model and patch parameters (consistent with training configuration in main.py)
    bilinear = True
    transformer_depth = 2
    transformer_heads = 4
    patch_size = 256
    overlap = 64
    zero_threshold = 0.5  # Consistent with patch filtering threshold during main training


def load_model(model_path, device):
    """Load model (parameters fully aligned with training configuration in main.py)"""
    model = ErosionFusionUNet(
        n_erosion_channels=3,  # Three types of erosion inputs (RUSLE+Wind+Freeze)
        bilinear=FixedConfig.bilinear,
        transformer_depth=FixedConfig.transformer_depth,
        transformer_heads=FixedConfig.transformer_heads
    ).to(device)

    # Load the model trained and saved by main.py (default path consistent with main)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def fixed_process_single_patch(model, erosion_input_paths, x, y, block_width, block_height,
                               patch_size, overlap, device, prediction, weight):
    """Patch processing logic is fully aligned with main training to ensure data consistency"""
    erosion_data = []
    for path in erosion_input_paths:
        with rasterio.open(path) as src:
            # Window boundary check (avoid out-of-bounds, consistent with training logic)
            window_x = max(0, min(x, src.width - 1))
            window_y = max(0, min(y, src.height - 1))
            window_width = min(block_width, src.width - window_x)
            window_height = min(block_height, src.height - window_y)

            # Handle invalid windows (consistent with null value processing during training)
            if window_width <= 0 or window_height <= 0:
                print(f"Warning: Invalid patch window ({window_x},{window_y}), filling with all zeros")
                data = np.zeros((patch_size, patch_size), dtype=np.float32)
            else:
                window = rasterio.windows.Window(window_x, window_y, window_width, window_height)
                data = src.read(1, window=window).astype(np.float32)

                # Unify patch size (consistent with patch padding logic during training)
                padded_data = np.zeros((patch_size, patch_size), dtype=np.float32)
                h, w = data.shape
                padded_data[:h, :w] = data

            # Data cleaning (exactly the same as preprocessing during main training to ensure unified distribution)
            data = np.nan_to_num(padded_data, nan=0.0, posinf=1.0, neginf=0.0)
            data = np.clip(data, 0.0, 1.0)
            erosion_data.append(data)

    # Construct model input (format exactly the same as training: [1, 3, patch_size, patch_size])
    erosion_data = np.stack(erosion_data, axis=0)
    input_tensor = torch.tensor(erosion_data, device=device, dtype=torch.float32).contiguous()
    input_tensor = input_tensor.unsqueeze(0)

    # Model prediction (output erosion fusion value)
    with torch.no_grad():
        output = model(input_tensor)
        pred_block = output.squeeze().cpu().numpy()

    # Fix abnormal prediction shape (ensure patch compatibility, consistent with training logic)
    if pred_block.shape != (patch_size, patch_size):
        print(f"Warning: Abnormal prediction patch shape (expected {patch_size}x{patch_size}, actual {pred_block.shape}), auto-adjusting")
        pred_block_fixed = np.zeros((patch_size, patch_size), dtype=np.float32)
        h_pred, w_pred = pred_block.shape
        pred_block_fixed[:min(h_pred, patch_size), :min(w_pred, patch_size)] = pred_block[:min(h_pred, patch_size),
                                                                               :min(w_pred, patch_size)]
        pred_block = pred_block_fixed

    # Crop to actual edge size (consistent with patch stitching logic during training)
    final_block_height = min(block_height, prediction.shape[0] - y)
    final_block_width = min(block_width, prediction.shape[1] - x)
    pred_block = pred_block[:final_block_height, :final_block_width]

    # Edge weight smoothing (avoid patch seams, consistent with weight logic during training)
    block_weight = np.ones((final_block_height, final_block_width), dtype=np.float32)
    border = overlap // 2

    if y == 0 and final_block_height > border:
        block_weight[:border, :] *= np.linspace(0, 1, border)[:, np.newaxis]
    if (y + final_block_height) >= prediction.shape[0] and final_block_height > border:
        block_weight[-border:, :] *= np.linspace(1, 0, border)[:, np.newaxis]
    if x == 0 and final_block_width > border:
        block_weight[:, :border] *= np.linspace(0, 1, border)[np.newaxis, :]
    if (x + final_block_width) >= prediction.shape[1] and final_block_width > border:
        block_weight[:, -border:] *= np.linspace(1, 0, border)[np.newaxis, :]

    # Safely update prediction results (consistent with patch update logic during training)
    y_end = min(y + final_block_height, prediction.shape[0])
    x_end = min(x + final_block_width, prediction.shape[1])
    prediction[y:y_end, x:x_end] += pred_block[:y_end - y, :x_end - x] * block_weight[:y_end - y, :x_end - x]
    weight[y:y_end, x:x_end] += block_weight[:y_end - y, :x_end - x]

    return prediction, weight


def fixed_predict_with_patching_strategy(model, erosion_input_paths, patch_size, overlap,
                                         device, reference_tif_path, start_from_top_left=True):
    # Get metadata from RUSLE file in fixed path (replace landscape file, no landscape dependency)
    with rasterio.open(reference_tif_path) as ref_src:
        height, width = ref_src.height, ref_src.width
        profile = ref_src.profile  # Retain original geographic coordinates (consistent with image projection during training)

    prediction = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    step = max(1, patch_size - overlap)
    model.eval()

    with torch.no_grad():
        if start_from_top_left:
            # Patch from top to bottom (consistent with patch order during training)
            y = 0
            while y < height:
                current_y = min(y, height - 1)
                x = 0
                while x < width:
                    current_x = min(x, width - 1)
                    block_height = min(patch_size, height - current_y)
                    block_width = min(patch_size, width - current_x)

                    prediction, weight = fixed_process_single_patch(
                        model, erosion_input_paths, current_x, current_y, block_width, block_height,
                        patch_size, overlap, device, prediction, weight
                    )

                    next_x = current_x + step
                    if next_x + patch_size > width:
                        break
                    x = next_x

                next_y = current_y + step
                if next_y + patch_size > height:
                    break
                y = next_y
        else:
            # Patch from bottom to top (complementary to eliminate seams, consistent with supplementary strategy during training)
            y = max(0, height - patch_size)
            while y >= 0:
                x = max(0, width - patch_size)
                while x >= 0:
                    block_height = min(patch_size, height - y)
                    block_width = min(patch_size, width - x)

                    prediction, weight = fixed_process_single_patch(
                        model, erosion_input_paths, x, y, block_width, block_height,
                        patch_size, overlap, device, prediction, weight
                    )

                    next_x = x - step
                    if next_x < 0:
                        break
                    x = next_x

                next_y = y - step
                if next_y < 0:
                    break
                y = next_y

    # Handle zero weight (avoid division by zero, consistent with post-processing during training)
    result = np.divide(prediction, weight, out=np.zeros_like(prediction), where=weight != 0)
    return result, profile


def predict_year(model, year, device):
    """Predict a single year (2000-2023), paths fully aligned with main.py"""
    # Construct fixed input paths (the erosion data directory you specified)
    erosion_paths = [
        os.path.join(FixedConfig.rusle_dir, FixedConfig.rusle_filename.format(year)),
        os.path.join(FixedConfig.wind_dir, FixedConfig.wind_filename.format(year)),
        os.path.join(FixedConfig.freeze_dir, FixedConfig.freeze_filename.format(year))
    ]

    # Reference image path: fixed to use RUSLE file (no landscape dependency, matches image size during training)
    reference_path = os.path.join(FixedConfig.rusle_dir, FixedConfig.rusle_filename.format(year))

    # Check input data integrity (consistent with file verification logic during main training)
    missing_files = [p for p in erosion_paths + [reference_path] if not os.path.exists(p)]
    if missing_files:
        print(f"Warning: Missing files {[os.path.basename(p) for p in missing_files]} for year {year}, skipped")
        return False

    # Ensure output directory exists (paths unified with main.py)
    os.makedirs(FixedConfig.prediction_dir, exist_ok=True)
    # Output file name: retain naming convention from main training, clearly indicate erosion fusion results
    output_path = os.path.join(FixedConfig.prediction_dir, f"fused_erosion_{year}.tif")

    # Execute prediction (logic consistent with forward propagation during main training)
    try:
        print(f"Start predicting year {year} (data path: {FixedConfig.rusle_dir})...")
        # Dual-strategy prediction (consistent with patch strategy during training to ensure result quality)
        print("  - Patch prediction from top to bottom, left to right...")
        pred1, profile = fixed_predict_with_patching_strategy(
            model=model,
            erosion_input_paths=erosion_paths,
            patch_size=FixedConfig.patch_size,
            overlap=FixedConfig.overlap,
            device=device,
            reference_tif_path=reference_path,
            start_from_top_left=True
        )

        print("  - Patch prediction from bottom to top, right to left...")
        pred2, _ = fixed_predict_with_patching_strategy(
            model=model,
            erosion_input_paths=erosion_paths,
            patch_size=FixedConfig.patch_size,
            overlap=FixedConfig.overlap,
            device=device,
            reference_tif_path=reference_path,
            start_from_top_left=False
        )

        # Fusion results (consistent with patch fusion logic during training)
        mask1 = (pred1 != 0)
        mask2 = (pred2 != 0)
        final_fused = np.zeros_like(pred1, dtype=np.float32)
        final_fused[mask1 & ~mask2] = pred1[mask1 & ~mask2]
        final_fused[~mask1 & mask2] = pred2[~mask1 & mask2]
        both_valid = mask1 & mask2
        final_fused[both_valid] = (pred1[both_valid] + pred2[both_valid]) / 2

        # Save results (metadata format consistent with input images during training)
        profile.update(
            dtype=np.float32,
            count=1,
            nodata=None,
            compress='lzw'  # Consistent with compression used during main training to reduce file size
        )
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(final_fused, 1)

        print(f"Prediction for year {year} completed, results saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error predicting year {year}: {str(e)}")
        traceback.print_exc()
        return False


def batch_predict_2000_2023():
    """Batch predict 2000-2023 (paths fully unified with main.py)"""
    parser = argparse.ArgumentParser(description='Batch predict fusion results of three erosion types for 2000-2023 (consistent with main.py paths)')
    # Default model path is exactly the same as model save path in main.py
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(FixedConfig.model_dir, "best_erosion_fusion_unet.pth"),
                        help=f'Model weight path (default: {os.path.join(FixedConfig.model_dir, "best_erosion_fusion_unet.pth")})')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU prediction (default prioritizes GPU/MPS, consistent with device logic during main training)')
    args = parser.parse_args()

    # Device selection (consistent with device priority during main.py training)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.force_cpu else
        'mps' if torch.backends.mps.is_available() and not args.force_cpu else
        'cpu'
    )
    print(f"Using computing device: {device}")
    print(f"Project root output directory (same as main): {FixedConfig.output_dir}")
    print(f"Fixed erosion data input directory: {FixedConfig.rusle_dir}")
    print(f"Fixed prediction year range: 2000-2023")

    # Detect valid data for 2000-2023 (consistent with valid year detection logic during main training)
    print("Detecting valid erosion data under fixed paths for 2000-2023...")
    valid_erosion_paths, valid_years = collect_data_paths(
        rusle_dir=FixedConfig.rusle_dir,
        wind_dir=FixedConfig.wind_dir,
        freeze_dir=FixedConfig.freeze_dir,
        rusle_filename=FixedConfig.rusle_filename,
        wind_filename=FixedConfig.wind_filename,
        freeze_filename=FixedConfig.freeze_filename,
        years=range(2000, 2024)  # Fixed year range
    )

    if not valid_years:
        print(f"Error: No valid erosion data for 2000-2023 detected under fixed path {FixedConfig.rusle_dir}!")
        return
    print(f"Detected {len(valid_years)} valid years in total: {sorted(valid_years)}")

    # Verify model file (path consistent with model path saved by main)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist! Path: {args.model_path}")
        return

    # Load model (consistent with model initialization parameters during main training)
    print(f"Loading Erosion Fusion Model: {args.model_path}")
    model = load_model(args.model_path, device)

    # Batch prediction (progress display consistent with tqdm style during main training)
    success_count = 0
    for year in tqdm(sorted(valid_years), desc="Batch prediction progress for 2000-2023"):
        if predict_year(model, year, device):
            success_count += 1

    # Output summary (format consistent with summary during main training)
    print(f"\nBatch prediction for 2000-2023 completed!")
    print(f"Total valid years: {len(valid_years)} | Successful predictions: {success_count} | Failed predictions: {len(valid_years) - success_count}")
    print(f"Model weight directory (same as main): {FixedConfig.model_dir}")
    print(f"Prediction result directory (based on main root directory): {FixedConfig.prediction_dir}")
    print(f"Loss record directory (same as main): {FixedConfig.loss_dir}")


if __name__ == "__main__":
    batch_predict_2000_2023()