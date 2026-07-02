"""
Training loop for the BERT sentiment classifier.
Supports mixed precision, gradient clipping, learning rate scheduling,
and checkpoint saving based on best validation accuracy.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.model.classifier import SentimentClassifier
from src.model.tokenizer import SentimentTokenizer
from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    MAX_SEQ_LENGTH,
    NUM_LABELS,
    LABELS,
)
from src.utils.logger import setup_logger

logger = setup_logger("model.trainer")


class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenized text + labels."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer: SentimentTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer.encode_text(self.texts[idx])
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class Trainer:
    """Training manager for the BERT sentiment classifier."""

    def __init__(
        self,
        model: SentimentClassifier,
        tokenizer: SentimentTokenizer,
        output_dir: str | Path = "models/sentiment_classifier",
        device: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.device = self._get_device(device)
        self.model.to(self.device)

        self.training_history: list[dict] = []
        self.best_val_accuracy = 0.0

        logger.info(f"Trainer initialized on device: {self.device}")

    def _get_device(self, device: str | None) -> torch.device:
        """Determine the best available device (GPU or CPU)."""
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        logger.info("No GPU found — using CPU (training will be slower)")
        return torch.device("cpu")

    def _create_dataloader(
        self, texts: list[str], labels: list[int], batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader from texts and labels."""
        dataset = SentimentDataset(texts, labels, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        warmup_steps: int = WARMUP_STEPS,
        weight_decay: float = WEIGHT_DECAY,
        max_grad_norm: float = MAX_GRAD_NORM,
        use_fp16: bool = True,
    ) -> dict:
        """Execute the full training loop.

        Args:
            train_texts: Training text samples.
            train_labels: Training labels.
            val_texts: Validation texts (optional).
            val_labels: Validation labels (optional).
            epochs: Number of training epochs.
            batch_size: Batch size.
            learning_rate: Peak learning rate for AdamW.
            warmup_steps: Number of warmup steps for scheduler.
            weight_decay: AdamW weight decay.
            max_grad_norm: Maximum gradient norm for clipping.
            use_fp16: Whether to use mixed precision training.

        Returns:
            Dictionary with training history and best accuracy.
        """
        logger.info(f"Starting training for {epochs} epochs (batch_size={batch_size})")
        start_time = time.time()

        # Prepare data loaders
        train_loader = self._create_dataloader(train_texts, train_labels, batch_size)
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_loader = self._create_dataloader(val_texts, val_labels, batch_size, shuffle=False)

        # Optimizer & scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and self.device.type == "cuda")

        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, scheduler, scaler, max_grad_norm
            )
            epoch_time = time.time() - epoch_start

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "epoch_time_s": epoch_time,
            }

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader)
                epoch_record["val_loss"] = val_loss
                epoch_record["val_accuracy"] = val_acc

                # Save best model
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.save_checkpoint(f"best_model_epoch_{epoch}")
                    logger.info(
                        f"Epoch {epoch}: New best validation accuracy: {val_acc:.4f}"
                    )
            else:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}")

            self.training_history.append(epoch_record)

            # Build the optional validation portion of the log line
            if val_loader is not None:
                val_part = (
                    f"Val Loss: {epoch_record.get('val_loss', 0):.4f} │ "
                    f"Val Acc: {epoch_record.get('val_accuracy', 0):.4f} │ "
                )
            else:
                val_part = ""

            logger.info(
                f"Epoch {epoch}/{epochs} │ "
                f"Loss: {train_loss:.4f} │ Acc: {train_acc:.4f} │ "
                f"{val_part}"
                f"Time: {epoch_time:.1f}s"
            )

        total_time = time.time() - start_time
        logger.info(
            f"Training complete! Best validation accuracy: {self.best_val_accuracy:.4f} "
            f"in {total_time:.1f}s"
        )

        return {
            "history": self.training_history,
            "best_val_accuracy": self.best_val_accuracy,
            "total_time_s": total_time,
        }

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: AdamW,
        scheduler,
        scaler: torch.cuda.amp.GradScaler,
        max_grad_norm: float,
    ) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(dataloader, desc="Training", leave=False)
        for batch in progress:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_loss += loss.item()
            preds = self.model.predict_class(outputs["logits"])
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Run evaluation on a dataset.

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]

            total_loss += loss.item()
            preds = self.model.predict_class(outputs["logits"])
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def save_checkpoint(self, name: str) -> None:
        """Save a model checkpoint.

        Args:
            name: Checkpoint name (subdirectory).
        """
        save_path = self.output_dir / name
        self.model.save(save_path)
        self.tokenizer.save(save_path)

    def save_best_model(self) -> None:
        """Save the best model to the main output directory."""
        self.model.save(self.output_dir)
        self.tokenizer.save(self.output_dir)
        logger.info(f"Best model saved to {self.output_dir}")
