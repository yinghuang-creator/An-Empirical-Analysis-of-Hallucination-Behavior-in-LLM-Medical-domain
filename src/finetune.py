"""
finetune.py
Supervised fine-tuning (SFT) of BioGPT on MedQA training split.

What this does:
  - Formats each MedQA sample as a prompt → answer string
  - Fine-tunes BioGPT with next-token prediction loss (standard SFT)
  - Saves the best checkpoint based on validation loss
  - Logs train/val loss per epoch so you can plot learning curves

Compute: runs on a single A100 (BU SCC) in ~1-2 hrs for 200 samples,
         ~4-6 hrs for the full MedQA train split (~10k samples).
         Start with n=500 to validate the pipeline, then scale up.

Install: pip install transformers datasets torch accelerate
"""

from __future__ import annotations
import json, os
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from load_dataset import load_medqa, stratified_sample, QASample

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class FinetuneConfig:
    model_name: str        = "microsoft/biogpt"
    output_dir: str        = "checkpoints/biogpt-medqa"

    # data
    train_samples: int     = -1      # -1 = full train split
    val_samples: int       = 200
    seed: int              = 42

    # training
    epochs: int            = 3
    batch_size: int        = 4
    grad_accum_steps: int  = 4      # effective batch = 4×4 = 16
    lr: float              = 2e-5
    warmup_ratio: float    = 0.1
    max_length: int        = 512
    weight_decay: float    = 0.01

    # compute
    fp16: bool             = True   # set False if not on GPU
    device: str            = "cuda" if torch.cuda.is_available() else "cpu"


# ── Prompt formatter ──────────────────────────────────────────────────────────
def format_training_example(sample: QASample) -> str:
    """
    Formats a MedQA sample as a single string for SFT.
    The model learns to predict the answer given the question + choices.

    Format:
        Question: ...
        A. ...  B. ...  C. ...  D. ...
        Answer: B
    """
    choices_str = "  ".join(
        f"{k}. {v}" for k, v in (sample.choices or {}).items()
    )
    return (
        f"Question: {sample.question}\n"
        f"{choices_str}\n"
        f"Answer: {sample.gold_answer}"
    )


# ── Dataset ───────────────────────────────────────────────────────────────────
class MedQADataset(Dataset):
    """
    Tokenises each example as a full prompt+answer string.
    Labels are identical to input_ids (standard causal LM SFT).
    Padding tokens in labels are set to -100 so they don't
    contribute to the loss.
    """
    def __init__(
        self,
        samples: list[QASample],
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.texts      = [format_training_example(s) for s in samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        # mask padding in labels so it doesn't inflate the loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Trainer ───────────────────────────────────────────────────────────────────
class SFTTrainer:
    def __init__(self, cfg: FinetuneConfig):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        print(f"Loading {cfg.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        # BioGPT has no pad token by default — use EOS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        self.model.to(cfg.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
        self.history: list[dict] = []

    def _make_loader(self, samples, shuffle: bool) -> DataLoader:
        ds = MedQADataset(samples, self.tokenizer, self.cfg.max_length)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
        )

    def _run_epoch(self, loader, optimizer, scheduler, train: bool) -> float:
        self.model.train(train)
        total_loss, steps = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=self.cfg.fp16):
                out  = self.model(**batch)
                loss = out.loss / self.cfg.grad_accum_steps

            if train:
                self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.grad_accum_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * self.cfg.grad_accum_steps
            steps += 1

        return total_loss / steps

    def train(
        self,
        train_samples: list[QASample],
        val_samples: list[QASample],
    ):
        train_loader = self._make_loader(train_samples, shuffle=True)
        val_loader   = self._make_loader(val_samples,   shuffle=False)

        total_steps  = (
            len(train_loader) * self.cfg.epochs // self.cfg.grad_accum_steps
        )
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_loss = float("inf")

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._run_epoch(
                train_loader, optimizer, scheduler, train=True
            )
            with torch.no_grad():
                val_loss = self._run_epoch(
                    val_loader, optimizer, scheduler, train=False
                )

            entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
            }
            self.history.append(entry)
            print(
                f"Epoch {epoch}/{self.cfg.epochs} — "
                f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}"
            )

            # save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(self.out / "best")
                self.tokenizer.save_pretrained(self.out / "best")
                print(f"  ✓ New best checkpoint saved (val_loss={val_loss:.4f})")

        # always save final epoch too
        self.model.save_pretrained(self.out / "final")
        self.tokenizer.save_pretrained(self.out / "final")

        # save training log
        with open(self.out / "training_log.json", "w") as f:
            json.dump({
                "config": vars(self.cfg),
                "history": self.history,
            }, f, indent=2)
        print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        return self.out / "best"


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = FinetuneConfig()

    # load data
    all_train = load_medqa(split="train")
    all_val   = load_medqa(split="validation")

    train_samples = (
        all_train if cfg.train_samples == -1
        else stratified_sample(all_train, cfg.train_samples, cfg.seed)
    )
    val_samples = stratified_sample(all_val, cfg.val_samples, cfg.seed)

    print(f"Train: {len(train_samples)} samples | Val: {len(val_samples)} samples")

    trainer = SFTTrainer(cfg)
    best_ckpt = trainer.train(train_samples, val_samples)
    print(f"Best checkpoint at: {best_ckpt}")
    print("To run inference with fine-tuned model:")
    print(f'  runner = ModelRunner("{best_ckpt}")')