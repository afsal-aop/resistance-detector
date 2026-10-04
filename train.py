"""
Training script for MobileNetV3 / MobileNetV2 Resistor Band Object Detector & Classifier.
Trains on custom resistor color band dataset and saves model weights.
"""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dataset import ResistorBandDataset, collate_fn, CLASS_NAMES
from model import get_resistor_detector

def evaluate_model(model, dataloader, device):
    """Calculate validation loss."""
    model.train()  # In torchvision, Faster R-CNN computes losses in train() mode
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1
    return total_loss / max(1, num_batches)

def train(
    data_dir: str = 'resistor_images',
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 0.0005,
    output_dir: str = 'weights',
    freeze_backbone: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Resistor Band Detector Training on {device} ===")
    
    # Datasets
    train_dataset = ResistorBandDataset(data_dir=data_dir, split='train', augment=True)
    val_dataset = ResistorBandDataset(data_dir=data_dir, split='valid', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    # Model
    model = get_resistor_detector(num_classes=len(CLASS_NAMES) + 1, pretrained=True)
    
    if freeze_backbone:
        for p in model.backbone.body.parameters():
            p.requires_grad = False
        print("[INFO] Backbone feature layers frozen for fast transfer learning.")
        
    model.to(device)
    
    # Optimizer & LR Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_weights_path = os.path.join(output_dir, 'resistor_detector.pth')
    
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
            optimizer.step()
            
            running_loss += losses.item()
            batch_count += 1
            
            if (i + 1) % 25 == 0 or (i + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Batch [{i+1}/{len(train_loader)}] Loss: {losses.item():.4f}")
                
        lr_scheduler.step()
        
        train_loss = running_loss / max(1, batch_count)
        val_loss = evaluate_model(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        
        print(f"--> Epoch {epoch:02d}/{epochs:02d} Complete | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        if val_loss < best_val_loss or epoch == 1:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'class_names': CLASS_NAMES
            }, best_weights_path)
            print(f"    [CHECKPOINT] Saved best model to {best_weights_path} (Val Loss: {val_loss:.4f})")
            
    print(f"\n[DONE] Training completed in {time.time() - start_total_time:.1f}s! Best Val Loss: {best_val_loss:.4f}")
    return best_weights_path

if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    train(epochs=epochs, batch_size=4)
