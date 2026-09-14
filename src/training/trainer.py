import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np


class P3Trainer:
    """
    Configurable, reproducible training and fine-tuning engine for P3 on TopAneu-26.
    Designed for memory-constrained environments (e.g., RTX 5050 8GB VRAM) using
    mixed precision (AMP FP16) and gradient accumulation.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: dict,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.experiment_id = config.get("experiment_id", "P4-P3-001")
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 1)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8)
        self.lr = float(config.get("lr", 1e-4))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.pos_weight = float(config.get("pos_weight", 1.25))
        self.use_amp = config.get("use_amp", True) and (self.device.type == "cuda")
        self.output_dir = os.path.join(config.get("output_dir", "experiments/results"), self.experiment_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # Loss function
        pw_tensor = torch.tensor([self.pos_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        # Optimizer: Tune heads, deep stages, or full network as specified
        freeze_encoder = config.get("freeze_encoder", False)
        freeze_encoder_stages = config.get("freeze_encoder_stages", None)

        if freeze_encoder:
            print("[TRAINER] Freezing all conv encoder blocks. Only training cross-attention pooling & classification heads.")
            for p in self.model.conv_encoder_blocks.parameters():
                p.requires_grad = False
        elif freeze_encoder_stages is not None and freeze_encoder_stages > 0:
            print(f"[TRAINER] Freezing early encoder stages 0..{freeze_encoder_stages-1}. Fine-tuning deep stages {freeze_encoder_stages}+ & cross-attention heads.")
            for i in range(freeze_encoder_stages):
                for p in self.model.conv_encoder_blocks[i].parameters():
                    p.requires_grad = False

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[TRAINER] Trainable parameters: {n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.2f}%)")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.05
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # DataLoaders (pin_memory=False to avoid Windows host locked-memory page thrashing)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        self.history = []
        self.best_val_auc = -1.0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        all_preds = []
        all_targets = []
        t0 = time.time()

        for step, batch in enumerate(self.train_loader, 1):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["presence"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                # Run classification forward (encoder + cross-attention classification heads)
                cls_preds = self.model(images, only_forward_cls=True)
                presence_logits = cls_preds[0] # (B, 1)
                loss = self.criterion(presence_logits, targets)
                # Scale loss for gradient accumulation
                loss_scaled = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss_scaled).backward()

            if step % self.gradient_accumulation_steps == 0 or step == num_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            probs = torch.sigmoid(presence_logits).detach().cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())

            if step % 25 == 0 or step == num_batches:
                step_elapsed = time.time() - t0
                print(f"  [Epoch {epoch:02d}/{self.epochs:02d}] Step {step:>3d}/{num_batches} ({step_elapsed:.1f}s) | Step Loss: {loss.item():.4f}", flush=True)

        self.scheduler.step()
        train_time = time.time() - t0
        avg_loss = total_loss / num_batches
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        bin_preds = (all_preds >= 0.5).astype(int)
        train_acc = accuracy_score(all_targets, bin_preds)
        try:
            train_auc = roc_auc_score(all_targets, all_preds)
        except Exception:
            train_auc = 0.5

        return {
            "loss": avg_loss,
            "accuracy": train_acc,
            "auc": train_auc,
            "time": train_time
        }

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        t0 = time.time()

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["presence"].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    cls_preds = self.model(images, only_forward_cls=True)
                    presence_logits = cls_preds[0]
                    loss = self.criterion(presence_logits, targets)

                total_loss += loss.item()
                probs = torch.sigmoid(presence_logits).cpu().numpy().flatten()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy().flatten())

        eval_time = time.time() - t0
        avg_loss = total_loss / max(len(dataloader), 1)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        bin_preds = (all_preds >= 0.5).astype(int)
        acc = accuracy_score(all_targets, bin_preds)
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except Exception:
            auc = 0.5

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "auc": auc,
            "time": eval_time
        }

    def train(self):
        print("=" * 70)
        print(f"STARTING TRAINING RUN: {self.experiment_id}")
        print(f"Device: {self.device} | AMP: {self.use_amp} | Batch Size: {self.batch_size} | Grad Accum: {self.gradient_accumulation_steps}")
        print(f"Epochs: {self.epochs} | LR: {self.lr} | Pos Weight: {self.pos_weight}")
        print("=" * 70)

        # Baseline evaluation before training
        print("\nEvaluating initial checkpoint performance on validation split...")
        init_val = self.evaluate(self.val_loader)
        print(f"Initial Validation: Loss={init_val['loss']:.4f}, Acc={init_val['accuracy']*100:.2f}%, AUC={init_val['auc']:.4f}")

        for epoch in range(1, self.epochs + 1):
            t_epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            epoch_duration = time.time() - t_epoch_start

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:02d}/{self.epochs:02d} Summary [{epoch_duration:.1f}s]:", flush=True)
            print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']*100:.2f}%, AUC={train_metrics['auc']:.4f}", flush=True)
            print(f"  Val  : Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']*100:.2f}%, AUC={val_metrics['auc']:.4f} (LR={current_lr:.2e})", flush=True)

            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_auc": train_metrics["auc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "duration_seconds": epoch_duration
            }
            self.history.append(epoch_record)

            # Save best checkpoint by validation AUC (and loss)
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                best_path = os.path.join(self.output_dir, "best_checkpoint.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                    "val_metrics": val_metrics
                }, best_path)
                print(f"  >>> New Best Checkpoint Saved: Val AUC = {self.best_val_auc:.4f} -> {best_path}")

            # Save last checkpoint
            last_path = os.path.join(self.output_dir, "last_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "val_metrics": val_metrics
            }, last_path)

            # Persist history
            history_path = os.path.join(self.output_dir, "training_history.json")
            with open(history_path, "w") as jf:
                json.dump(self.history, jf, indent=2)

        print("\n" + "=" * 70)
        print(f"TRAINING COMPLETE: {self.experiment_id}")
        print(f"Best Val AUC: {self.best_val_auc:.4f} | Artifacts in: {self.output_dir}")
        print("=" * 70)
        return self.history
