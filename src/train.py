import time

import torch
from sacrebleu.metrics import BLEU
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    D_FF,
    D_MODEL,
    DEVICE,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LABEL_SMOOTHING,
    MAX_LEN,
    N_HEAD,
    N_LAYER,
    PAD_ID,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    TRAIN_FILE,
    VAL_FILE,
    WARMUP_STEPS,
    DEFAULT_BEAM_SIZE,
)
from src.dataset import TranslationDataset, collate_fn
from src.model import Transformer
from src.utils import (
    make_src_pad_mask,
    make_tgt_mask,
    generate_decoded,
    ensure_checkpoint_dir,
    save_checkpoint,
)
from src.vocab import Vocab, decode_tgt, load_src_vocab, load_tgt_vocab

class LabelSmoothingLoss(nn.Module):
    """Label-smoothed cross-entropy loss"""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.pad_id = pad_id

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ``log_probs``: (B, T, V), ``target``: (B, T)
        smooth_loss = -log_probs.mean(dim=-1)  # uniform mass term
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        mask = (target != self.pad_id).float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

def noam_lr(step: int, d_model: int, warmup_steps: int, factor: float = 1) -> float:
    """Noam learning-rate scheduler."""
    step = max(step, 1)
    result = factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return result

def compute_loss(
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_out: torch.Tensor,
        model: Transformer,
        criterion: LabelSmoothingLoss,
        device: str,
) -> tuple[torch.Tensor, torch.types.Number]:
    """Compute loss and tokens count."""
    src = src.to(device)
    tgt_in = tgt_in.to(device)
    tgt_out = tgt_out.to(device)
    src_mask = make_src_pad_mask(src, PAD_ID)
    tgt_mask = make_tgt_mask(tgt_in, PAD_ID)

    log_probs = model(src, tgt_in, src_mask, tgt_mask)
    # Reshape for the loss: (B*T, V) and (B*T,)
    loss = criterion(
        log_probs.reshape(-1, log_probs.size(-1)),
        tgt_out.reshape(-1),
    )

    n_tokens = (tgt_out != PAD_ID).sum().item()

    return loss, n_tokens

def compute_validation_loss(
    model: Transformer,
    loader: DataLoader,
    criterion: LabelSmoothingLoss,
    device: str,
) -> float:
    """Compute average loss for validation"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        progress = tqdm(loader, desc=f"Validation Loss")
        for src, tgt_in, tgt_out in progress:
            loss, n_tokens = compute_loss(
                src=src,
                tgt_in=tgt_in,
                tgt_out=tgt_out,
                model=model,
                criterion=criterion,
                device=device,
            )

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def compute_validation_bleu(
    model: Transformer,
    dataset: TranslationDataset,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: str,
    n_samples: int = 50,
    beam_size: int = DEFAULT_BEAM_SIZE,
    max_len: int = MAX_LEN
) -> float:
    """Compute average corpus BLEU score for validation."""
    model.eval()
    n = min(n_samples, len(dataset))
    bleu = BLEU(tokenize="zh")
    sys: list[str]  = []
    refs: list[str] = []

    for i in  tqdm(range(n), desc=f"Validation BLEU"):
        src_ids, tgt_ids = dataset[i]
        pred_ids = generate_decoded(
            model=model,
            src=src_ids,
            beam_size=beam_size,
            max_len=max_len
        )
        sys.append(decode_tgt(pred_ids, tgt_vocab))
        refs.append(decode_tgt(tgt_ids, tgt_vocab))

    try:
        score = bleu.corpus_score(sys, [refs]).score
    except ZeroDivisionError:
        score = 0.0
    return score

def train() -> None:
    """Train on the train and validation set and save the best model."""
    ensure_checkpoint_dir()
    print(f"[train] device: {DEVICE}")

    # Datasets & DataLoaders
    src_vocab = load_src_vocab()
    tgt_vocab = load_tgt_vocab()
    train_ds = TranslationDataset(TRAIN_FILE, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(VAL_FILE, src_vocab, tgt_vocab)
    print(
        f"[train] train={len(train_ds)} val={len(val_ds)}",
        f"src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model parameters: {n_params:,}")

    # Label Smoothing Loss
    criterion = LabelSmoothingLoss(
        vocab_size=TGT_VOCAB_SIZE,
        smoothing=LABEL_SMOOTHING,
        pad_id=PAD_ID,
    )

    # Adam Optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # Noam lr Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_lr(step + 1, D_MODEL, WARMUP_STEPS),
    )

    # Training loop
    best_val_loss = float("inf")
    best_bleu = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        start = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")

        for src, tgt_in, tgt_out in progress:
            loss, n_tokens = compute_loss(
                src=src,
                tgt_in=tgt_in,
                tgt_out=tgt_out,
                model=model,
                criterion=criterion,
                device=DEVICE,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping
            # mitigates Transformer training instability on small batches.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            running_loss += loss.item() * n_tokens
            running_tokens += n_tokens
            progress.set_postfix(
                loss=f"{running_loss / max(running_tokens, 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        train_loss = running_loss / max(running_tokens, 1)

        # validation
        val_loss = compute_validation_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )
        val_bleu = compute_validation_bleu(
            model=model,
            dataset=val_ds,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=DEVICE,
            n_samples=50
        )

        elapsed = time.time() - start
        print(
            f"[epoch {epoch:02d}]",
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            f"val_bleu={val_bleu:.2f} lr={scheduler.get_last_lr()[0]:.2e}",
            f"({elapsed:.1f}s)"
        )

        # Checkpoint best model by validation loss
        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_bleu = val_bleu
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                config=None,
                epoch=best_epoch,
                val_loss=val_loss,
                val_bleu=val_bleu
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"[epoch {epoch:02d}] early stop – no improvement for {EARLY_STOP_PATIENCE} epochs"
                )
                break

    print(
        f"[train] done. best val_loss={best_val_loss:.4f}  best val_bleu={best_bleu:.2f} best epoch={best_epoch} checkpoint={CHECKPOINT_PATH}"
    )
