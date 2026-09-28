import os
import torch
import torch.nn as nn
import numpy as np
import rasterio
from tqdm import tqdm
from torch.utils.data import Dataset
import math


class EnhancedDoubleConv(nn.Module):
    """Enhanced double convolution block (with residual connection)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection adaptation layer
        self.residual = nn.Conv2d(in_channels, out_channels, 1,
                                  bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual  # Residual connection
        return self.relu(x)


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention module"""

    def __init__(self, dim, heads=4, reduction_ratio=2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.reduction_ratio = reduction_ratio

        assert self.head_dim * heads == dim, "Dimension must be divisible by number of heads"

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, bias=False)
        self.scale = math.sqrt(self.head_dim)

        self.reduce = nn.Conv2d(dim, dim, kernel_size=reduction_ratio,
                                stride=reduction_ratio, bias=False)
        self.restore = nn.Upsample(scale_factor=reduction_ratio, mode='bilinear', align_corners=True)

    def forward(self, x):
        batch, channels, height, width = x.shape

        x_reduced = self.reduce(x)
        r_height, r_width = x_reduced.shape[2], x_reduced.shape[3]
        seq_len = r_height * r_width

        qkv = self.qkv(x_reduced).view(batch, 3, self.heads, self.head_dim, seq_len)
        q, k, v = qkv.unbind(1)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, channels, r_height, r_width)

        out = self.restore(out)
        out = self.out(out)

        return out + x


class TransformerBlock(nn.Module):
    """Improved Transformer block"""

    def __init__(self, dim, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = EfficientSelfAttention(dim, heads)
        self.norm2 = nn.BatchNorm2d(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mlp_dim, dim, 1, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ErosionFeatureFusion(nn.Module):
    """Feature fusion module for three erosion factors"""

    def __init__(self, in_channels=3, branch_channels=32, out_channels=96):
        super().__init__()
        # Separate feature extraction branches for each erosion factor
        # Three branches, each outputting branch_channels channels, total = 3*branch_channels
        self.rusle_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.wind_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.freeze_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Attention mechanism for dynamic weight adjustment
        self.attention = nn.Sequential(
            nn.Conv2d(3 * branch_channels, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * branch_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [batch, 3, height, width] containing three erosion factors

        # Separate three erosion factors
        rusle = x[:, 0:1, :, :]  # First channel
        wind = x[:, 1:2, :, :]  # Second channel
        freeze = x[:, 2:3, :, :]  # Third channel

        # Extract individual features (each branch outputs branch_channels channels)
        rusle_feat = self.rusle_branch(rusle)  # [B, C, H, W]
        wind_feat = self.wind_branch(wind)  # [B, C, H, W]
        freeze_feat = self.freeze_branch(freeze)  # [B, C, H, W]

        # Concatenate features (total channels = 3*C)
        combined = torch.cat([rusle_feat, wind_feat, freeze_feat], dim=1)  # [B, 3C, H, W]

        # Calculate attention weights
        attn_weights = self.attention(combined)  # [B, 3, H, W]

        # Apply attention weights
        attn_rusle = attn_weights[:, 0:1, :, :] * rusle_feat  # [B, C, H, W]
        attn_wind = attn_weights[:, 1:2, :, :] * wind_feat  # [B, C, H, W]
        attn_freeze = attn_weights[:, 2:3, :, :] * freeze_feat  # [B, C, H, W]

        # Weighted fusion
        attended_combined = torch.cat([attn_rusle, attn_wind, attn_freeze], dim=1)  # [B, 3C, H, W]

        # Final fusion to get comprehensive features
        out = self.fusion(attended_combined)  # [B, out_channels, H, W]

        return out


# ----------------------------
# UNet with Feature Fusion Model
# ----------------------------
class Down(nn.Module):
    """Downsampling module"""

    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            EnhancedDoubleConv(in_channels, out_channels)
        )
        self.attention = EfficientSelfAttention(out_channels) if use_attention else None

    def forward(self, x):
        x = self.pool_conv(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class Up(nn.Module):
    """Upsampling module"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = EnhancedDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = EnhancedDoubleConv(in_channels, out_channels)

        self.attention = EfficientSelfAttention(out_channels) if use_attention else None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad to align dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        if self.attention is not None:
            x = self.attention(x)
        return x


class OutConv(nn.Module):
    """Output convolution layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class ErosionFusionUNet(nn.Module):
    """UNet model for erosion factor fusion"""

    def __init__(self, n_erosion_channels=3, bilinear=True,
                 transformer_depth=2, transformer_heads=4):
        super().__init__()
        self.n_channels = n_erosion_channels  # Input channels: 3 erosion factors
        self.bilinear = bilinear

        # Feature fusion parameters (ensure channels are divisible by 3)
        self.branch_channels = 32  # Feature channels for each erosion factor
        self.fused_channels = 3 * self.branch_channels  # Total fused channels (3×32=96)

        # Initial feature extraction and fusion
        self.initial_fusion = ErosionFeatureFusion(
            in_channels=n_erosion_channels,
            branch_channels=self.branch_channels,
            out_channels=self.fused_channels
        )

        # Encoder (adapted to patch size)
        self.inc = EnhancedDoubleConv(self.fused_channels, 128)  # Input is fused channels
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512, use_attention=True)
        factor = 2 if bilinear else 1
        self.down3 = Down(512, 1024 // factor, use_attention=True)

        # Transformer processing
        self.transformer = nn.Sequential(*[
            TransformerBlock(1024 // factor, transformer_heads)
            for _ in range(transformer_depth)
        ])

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear, use_attention=True)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.outc = OutConv(128, 1)  # Output comprehensive erosion factor (single channel)

    def forward(self, x):
        # x: [batch, 3, patch_height, patch_width]

        # First perform feature fusion (3 channels → 96 channels)
        fused = self.initial_fusion(x)  # [B, 96, H, W]

        # Feature extraction and refinement through UNet
        x1 = self.inc(fused)  # [B, 128, H, W]
        x2 = self.down1(x1)  # [B, 256, H/2, W/2]
        x3 = self.down2(x2)  # [B, 512, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, H/8, W/8] (if bilinear=True)

        x4 = self.transformer(x4)

        x = self.up1(x4, x3)  # [B, 256, H/4, W/4]
        x = self.up2(x, x2)  # [B, 128, H/2, W/2]
        x = self.up3(x, x1)  # [B, 128, H, W]
        logits = self.outc(x)  # [B, 1, H, W]

        return logits


# ----------------------------
# Patch Processing Logic (Unchanged)
# ----------------------------
def collect_data_paths(rusle_dir, wind_dir, freeze_dir,
                       rusle_filename, wind_filename, freeze_filename,
                       years=range(2000, 2024)):
    """Collect data paths for three erosion factors"""
    all_erosion = []
    valid_years = []

    for year in tqdm(years, desc="Collecting data paths"):
        rusle_path = os.path.join(rusle_dir, rusle_filename.format(year))
        wind_path = os.path.join(wind_dir, wind_filename.format(year))
        freeze_path = os.path.join(freeze_dir, freeze_filename.format(year))

        if all(os.path.exists(p) for p in [rusle_path, wind_path, freeze_path]):
            all_erosion.append([rusle_path, wind_path, freeze_path])
            valid_years.append(year)
        else:
            missing = [p for p in [rusle_path, wind_path, freeze_path] if not os.path.exists(p)]
            print(f"Year {year} missing files: {[os.path.basename(p) for p in missing]}, skipped")

    return all_erosion, valid_years


class ErosionDataset(Dataset):
    """Erosion dataset (contains only three erosion factors, used for fusion training)"""

    def __init__(self, erosion_paths_list, valid_years,
                 patch_size=256, overlap=64, zero_threshold=0.5):
        self.erosion_paths = erosion_paths_list
        self.valid_years = valid_years
        self.patch_size = patch_size
        self.overlap = overlap
        self.zero_threshold = zero_threshold

        assert len(self.erosion_paths) == len(self.valid_years), \
            "Mismatch between number of erosion data and valid years"

        # Get size information from the first erosion file
        with rasterio.open(erosion_paths_list[0][0]) as src:
            self.height, self.width = src.height, src.width

        self.patches = self._generate_and_filter_patches()
        self.total_patches = len(self.patches)

    def _generate_and_filter_patches(self):
        patches = []
        step = max(1, self.patch_size - self.overlap)

        for year_idx in range(len(self.valid_years)):
            y = 0
            while y < self.height:
                x = 0
                while x < self.width:
                    if self._is_patch_valid(year_idx, x, y):
                        patches.append({
                            'year_idx': year_idx,
                            'y_start': y,
                            'x_start': x
                        })
                    x += step
                    if x + self.patch_size > self.width:
                        break
                y += step
                if y + self.patch_size > self.height:
                    break
        return patches

    def _is_patch_valid(self, year_idx, x, y):
        try:
            block_height = min(self.patch_size, self.height - y)
            block_width = min(self.patch_size, self.width - x)

            # Check validity of all erosion factors
            for path in self.erosion_paths[year_idx]:
                with rasterio.open(path) as src:
                    window = rasterio.windows.Window(x, y, block_width, block_height)
                    erosion_data = src.read(1, window=window).astype(np.float32)
                    total_pixels = erosion_data.size
                    zero_pixels = np.sum(erosion_data == 0)
                    if zero_pixels / total_pixels > self.zero_threshold:
                        return False

            return True
        except Exception as e:
            print(f"Error checking patch validity: {str(e)}")
            return False

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        year_idx = patch_info['year_idx']
        y = patch_info['y_start']
        x = patch_info['x_start']

        block_height = min(self.patch_size, self.height - y)
        block_width = min(self.patch_size, self.width - x)

        # Read three erosion factors
        erosion_data = []
        for path in self.erosion_paths[year_idx]:
            with rasterio.open(path) as src:
                window = rasterio.windows.Window(x, y, block_width, block_height)
                data = src.read(1, window=window).astype(np.float32)
                padded_data = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                padded_data[:block_height, :block_width] = data
                erosion_data.append(padded_data)

        # Stack into 3-channel input
        erosion_data = np.stack(erosion_data, axis=0)

        # Target is the fused comprehensive erosion factor
        target = np.mean(erosion_data, axis=0, keepdims=True)

        # Data cleaning
        erosion_data = np.nan_to_num(erosion_data, nan=0.0, posinf=1.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

        # Data clipping
        erosion_data = np.clip(erosion_data, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)

        return {
            "input": erosion_data,
            "target": target,
            "year": self.valid_years[year_idx]
        }


# ----------------------------
# Patch Prediction Logic (Unchanged)
# ----------------------------
def _process_single_patch(model, erosion_input_paths, x, y, block_width, block_height,
                          patch_size, overlap, device, prediction, weight):
    """Process single patch"""
    erosion_data = []
    for path in erosion_input_paths:
        with rasterio.open(path) as src:
            window = rasterio.windows.Window(x, y, block_width, block_height)
            data = src.read(1, window=window).astype(np.float32)
            padded_data = np.zeros((patch_size, patch_size), dtype=np.float32)
            padded_data[:block_height, :block_width] = data
            data = np.nan_to_num(padded_data, nan=0.0, posinf=1.0, neginf=0.0)
            data = np.clip(data, 0.0, 1.0)
            erosion_data.append(data)

    erosion_data = np.stack(erosion_data, axis=0)
    input_tensor = torch.tensor(erosion_data, device=device, dtype=torch.float32).contiguous()
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, list):
            output = output[0]
        pred_block = output.squeeze().cpu().numpy()

    pred_block = pred_block[:block_height, :block_width]
    block_weight = np.ones((block_height, block_width), dtype=np.float32)
    border = overlap // 2

    # Edge weight processing
    if y == 0:
        block_weight[:border, :] *= np.linspace(0, 1, border)[:, np.newaxis]
    if y + block_height >= prediction.shape[0]:
        block_weight[-border:, :] *= np.linspace(1, 0, border)[:, np.newaxis]
    if x == 0:
        block_weight[:, :border] *= np.linspace(0, 1, border)[np.newaxis, :]
    if x + block_width >= prediction.shape[1]:
        block_weight[:, -border:] *= np.linspace(1, 0, border)[np.newaxis, :]

    # Update prediction results
    y_end = min(y + block_height, prediction.shape[0])
    x_end = min(x + block_width, prediction.shape[1])
    prediction[y:y_end, x:x_end] += pred_block[:y_end - y, :x_end - x] * block_weight[:y_end - y, :x_end - x]
    weight[y:y_end, x:x_end] += block_weight[:y_end - y, :x_end - x]

    return prediction, weight


def _predict_with_patching_strategy(model, erosion_input_paths, patch_size, overlap,
                                    device, reference_tif_path, start_from_top_left=True):
    """Patch prediction strategy"""
    with rasterio.open(reference_tif_path) as ref_src:
        height, width = ref_src.height, ref_src.width

    prediction = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    step = max(1, patch_size - overlap)
    model.eval()

    with torch.no_grad():
        if start_from_top_left:
            y = 0
            while y < height:
                x = 0
                while x < width:
                    block_height = min(patch_size, height - y)
                    block_width = min(patch_size, width - x)
                    prediction, weight = _process_single_patch(
                        model, erosion_input_paths, x, y, block_width, block_height,
                        patch_size, overlap, device, prediction, weight
                    )
                    next_x = x + step
                    if next_x + patch_size > width:
                        break
                    x = next_x
                next_y = y + step
                if next_y + patch_size > height:
                    break
                y = next_y
        else:
            y = max(0, height - patch_size)
            while y >= 0:
                x = max(0, width - patch_size)
                while x >= 0:
                    block_height = min(patch_size, height - y)
                    block_width = min(patch_size, width - x)
                    prediction, weight = _process_single_patch(
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

    return np.divide(prediction, weight, out=np.zeros_like(prediction), where=weight != 0)


def predict_full_image(model, erosion_input_paths, output_tif_path, patch_size=256,
                       overlap=64, device='cpu', reference_tif_path=None):
    """Predict full image and save results"""
    print("Predicting with patch strategy from top to bottom, left to right...")
    pred1 = _predict_with_patching_strategy(
        model, erosion_input_paths, patch_size, overlap,
        device, reference_tif_path, start_from_top_left=True
    )

    print("Predicting with patch strategy from bottom to top, right to left...")
    pred2 = _predict_with_patching_strategy(
        model, erosion_input_paths, patch_size, overlap,
        device, reference_tif_path, start_from_top_left=False
    )

    # Fuse prediction results from two directions
    mask1 = (pred1 != 0)
    mask2 = (pred2 != 0)
    final_pred = np.zeros_like(pred1, dtype=np.float32)
    final_pred[mask1 & ~mask2] = pred1[mask1 & ~mask2]
    final_pred[~mask1 & mask2] = pred2[~mask1 & mask2]
    both_non_zero = mask1 & mask2
    final_pred[both_non_zero] = (pred1[both_non_zero] + pred2[both_non_zero]) / 2

    # Save results
    with rasterio.open(reference_tif_path) as ref_src:
        profile = ref_src.profile
        profile.update(dtype=np.float32, count=1, nodata=None)

    with rasterio.open(output_tif_path, 'w', **profile) as dst:
        dst.write(final_pred, 1)

    return final_pred