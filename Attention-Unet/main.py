import os
import torch
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

# Import improved model (retain patch processing logic)
from data_model import (
    ErosionDataset,
    ErosionFusionUNet,  # Use erosion fusion network
    collect_data_paths
)


class Config:
    """Configuration parameters"""
    rusle_dir = r"H:\ArcGIS_workspace\YGCB22\do1\rusle"
    wind_dir = r"H:\ArcGIS_workspace\YGCB22\do1\wind"
    freeze_dir = r"H:\ArcGIS_workspace\YGCB22\do1\freeze"

    rusle_filename = "{}.tif"
    wind_filename = "{}.tif"
    freeze_filename = "{}.tif"

    output_dir = r"H:\ArcGIS_workspace\YGCB22\do1\outt"
    model_dir = os.path.join(output_dir, "models")
    loss_dir = os.path.join(output_dir, "loss_records")

    epochs = 100
    batch_size = 8
    lr = 1e-4
    weight_decay = 1e-5
    patch_size = 256  # Patch size remains unchanged
    overlap = 64
    zero_threshold = 0.5
    val_split = 0.2
    bilinear = True

    # Transformer parameters
    transformer_depth = 2  # Number of Transformer blocks
    transformer_heads = 4  # Number of attention heads


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for erosion factor fusion model (retain patch processing)')
    parser.add_argument('--years', nargs='+', type=int, default=range(2000, 2024))
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    parser.add_argument('--lr', type=float, default=Config.lr)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--loss_file', type=str)
    return parser.parse_args()


def init_directories():
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.loss_dir, exist_ok=True)


def main():
    args = parse_args()
    init_directories()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}")

    # Data collection and loading
    print("Collecting training data paths...")
    all_erosion, valid_years = collect_data_paths(
        rusle_dir=Config.rusle_dir,
        wind_dir=Config.wind_dir,
        freeze_dir=Config.freeze_dir,
        rusle_filename=Config.rusle_filename,
        wind_filename=Config.wind_filename,
        freeze_filename=Config.freeze_filename,
        years=args.years
    )

    if not valid_years:
        print("Error: No valid data years found, cannot proceed with training!")
        return

    # Dataset and data loaders
    dataset = ErosionDataset(
        erosion_paths_list=all_erosion,
        valid_years=valid_years,
        patch_size=Config.patch_size,
        overlap=Config.overlap,
        zero_threshold=Config.zero_threshold
    )

    val_size = int(len(dataset) * Config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Model initialization
    model = ErosionFusionUNet(
        n_erosion_channels=3,
        bilinear=Config.bilinear,
        transformer_depth=Config.transformer_depth,
        transformer_heads=Config.transformer_heads
    ).to(device)

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=Config.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    loss_records = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'timestamp': []
    }

    start_time = datetime.now()
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Training]"):
            inputs = torch.tensor(batch["input"], device=device, dtype=torch.float32).contiguous()
            targets = torch.tensor(batch["target"], device=device, dtype=torch.float32).contiguous()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print(f"Warning: Training loss is NaN, skipped current batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Validation]"):
                inputs = torch.tensor(batch["input"], device=device, dtype=torch.float32).contiguous()
                targets = torch.tensor(batch["target"], device=device, dtype=torch.float32).contiguous()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        loss_records['epoch'].append(epoch)
        loss_records['train_loss'].append(avg_train_loss)
        loss_records['val_loss'].append(avg_val_loss)
        loss_records['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        print(f"Epoch {epoch}/{args.epochs} - "
              f"Training loss: {avg_train_loss:.6f}, "
              f"Validation loss: {avg_val_loss:.6f}")

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(Config.model_dir, "best_erosion_fusion_unet.pth"))
            print(f"Best model updated (validation loss: {best_val_loss:.6f})")

    end_time = datetime.now()
    print(f"Training completed, time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {str(end_time - start_time)}")

    # Save loss records
    loss_filename = args.loss_file if args.loss_file else \
        f"erosion_fusion_unet_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    if not loss_filename.endswith('.xlsx'):
        loss_filename += '.xlsx'
    loss_filepath = os.path.join(Config.loss_dir, loss_filename)
    pd.DataFrame(loss_records).to_excel(loss_filepath, index=False)
    print(f"Loss records saved to: {loss_filepath}")


if __name__ == "__main__":
    main()