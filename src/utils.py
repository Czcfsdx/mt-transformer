from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import (
    BOS_ID,
    EOS_ID,
    UNK_ID,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    D_FF,
    D_MODEL,
    DEVICE,
    DROPOUT,
    MAX_LEN,
    N_HEAD,
    N_LAYER,
    PAD_ID,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    DEFAULT_BEAM_SIZE,
)
from src.model import Transformer
from src.vocab import load_src_vocab, load_tgt_vocab, decode_tgt

def ensure_checkpoint_dir(checkpoint_dir: Path = CHECKPOINT_DIR) -> None:
    checkpoint_dir.mkdir(exist_ok=True)

def load_model(checkpoint_path: Path = CHECKPOINT_PATH, device=DEVICE) -> Transformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = Transformer(
        src_vocab_size=cfg["src_vocab_size"],
        tgt_vocab_size=cfg["tgt_vocab_size"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(
        f"loaded checkpoint from epoch {ckpt['epoch']} ",
        f"(val_loss={ckpt['val_loss']:.4f}, val_bleu={ckpt['val_bleu']:.2f})"
    )
    return model

def save_checkpoint(
    model: Transformer,
    config: dict[str, int | float] | None,
    epoch: int,
    val_loss: float,
    val_bleu: float,
) -> None:
    if config is None:
        config = {
            "d_model": D_MODEL,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "max_len": MAX_LEN,
            "src_vocab_size": SRC_VOCAB_SIZE,
            "tgt_vocab_size": TGT_VOCAB_SIZE,
        }

    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_bleu": val_bleu,
    }
    torch.save(payload, CHECKPOINT_PATH)


def make_src_pad_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    src    : (B, S) token-id tensor
    pad_id : int

    return:
        (B, 1, 1, S) bool tersor
        True at non-pad tokens.
    """
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    tgt    : (B, T) token-id tensor
    pad_id : int

    return:
        (B, 1, T, T) bool tensor
        True at tokens which are non-pad and not in the future
    """
    # Padding mask: (B, 1, 1, T)
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)

    # Causal mask: (1, 1, T, T)
    _, seq_len = tgt.size()
    causal = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=tgt.device)
    ).view(1, 1, seq_len, seq_len)

    return pad_mask & causal

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int = MAX_LEN,
    device: str = DEVICE,
) -> list[int]:
    """
    Greedily decoding.

    src is a (1, S) tensor of source token ids.
    """
    model.eval()
    src = src.to(device)
    src_mask = make_src_pad_mask(src, PAD_ID)
    memory = model.encode(src, src_mask)
    tgt_in = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

    generated: list[int] = []
    for _ in range(max_len):
        # compute log-probabilities of the next token for every position
        tgt_mask = make_tgt_mask(tgt_in, PAD_ID)
        decoded = model.decode(tgt_in, memory, src_mask, tgt_mask)
        log_probs = model.generator(decoded[:, -1:, :]) # only the last position

        # choose next token
        next_token = int(log_probs.argmax(dim=-1).item())
        if next_token == EOS_ID:
            break
        if next_token != PAD_ID:
            generated.append(next_token)

        # Append the chosen token to the decoder input for the next step
        tgt_in = torch.cat(
            [tgt_in, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )

    return generated

@dataclass(slots=True, kw_only=True)
class Beam:
    tokens: list[int]
    log_prob: float

    def length_penalised(self, alpha: float) -> float:
        """
        length-normalised score.
        lp(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}
        """
        lp = ((5.0 + len(self.tokens)) ** alpha) / ((5.0 + 1.0) ** alpha)
        return self.log_prob / lp

@torch.no_grad()
def beam_search(
    model: Transformer,
    src: torch.Tensor,
    beam_size: int = DEFAULT_BEAM_SIZE,
    max_len: int = MAX_LEN,
    length_penalty: float = 0.6,
    device: str = DEVICE,
) -> list[int]:
    """
    Beam-search decoding.

    src is a (1, S) tensor of source token ids.
    """
    assert beam_size >= 1, "beam_size must be positive"
    model.eval()
    src = src.to(device)
    src_mask = make_src_pad_mask(src, PAD_ID)
    memory = model.encode(src, src_mask)

    beams: list[Beam] = [Beam(tokens=[BOS_ID], log_prob=0.0)]
    completed: list[Beam] = []

    for _ in range(max_len):
        if not beams:
            break

        # Batch all currently active beams into a single decoder forward
        max_t = max(len(b.tokens) for b in beams)
        tgt_in = torch.full(
            (len(beams), max_t), PAD_ID, dtype=torch.long, device=device
        )
        for i, b in enumerate(beams):
            tgt_in[i, : len(b.tokens)] = torch.tensor(
                b.tokens, dtype=torch.long, device=device
            )

        # attend each beam with the same encoder output
        n_beams = tgt_in.size(0)
        memory_b = memory.expand(n_beams, -1, -1)
        src_mask_b = src_mask.expand(n_beams, -1, -1, -1)

        # compute log-probabilities of the next token for every position
        tgt_mask = make_tgt_mask(tgt_in, PAD_ID)
        decoded = model.decode(tgt_in, memory_b, src_mask_b, tgt_mask)
        log_probs = model.generator(decoded[:, -1, :])  # only the last position

        # choose next top-k token
        candidates: list[Beam] = []
        for i, beam in enumerate(beams):
            topk_log_probs, topk_ids = log_probs[i].topk(beam_size)
            for log_p, tok in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_tokens = beam.tokens + [tok]
                new_log_prob = beam.log_prob + log_p
                if tok == EOS_ID:
                    completed.append(Beam(tokens=new_tokens[1:-1], log_prob=new_log_prob))
                else:
                    candidates.append(Beam(tokens=new_tokens, log_prob=new_log_prob))

        # Keep the top-beam_size active beams
        candidates.sort(key=lambda b: b.length_penalised(length_penalty), reverse=True)
        beams = candidates[:beam_size]

    # pick the highest normalised log-probability among completed & remaining beams
    final_pool: list[Beam] = completed + [
        Beam(tokens=b.tokens[1:], log_prob=b.log_prob) for b in beams
    ]
    if not final_pool:
        return []
    final_pool.sort(key=lambda b: b.length_penalised(length_penalty), reverse=True)
    return final_pool[0].tokens

def generate_decoded(
    model: Transformer,
    src: list[int],
    beam_size: int = DEFAULT_BEAM_SIZE,
    max_len: int = MAX_LEN,
    device: str = DEVICE,
) -> list[int]:
    """
    Generate decoded token ids for src
    Choose Beam-search decoding or Greedily decoding based on beam_size

    src is a (1, S) tensor of source token ids.
    """
    src_tensor = torch.tensor(
        src, dtype=torch.long, device=device
    ).unsqueeze(0)
    if beam_size > 1:
        pred_ids = beam_search(
            model=model,
            src=src_tensor,
            beam_size=beam_size,
            max_len=max_len,
            device=DEVICE,
        )
    else:
        pred_ids = greedy_decode(
            model=model,
            src=src_tensor,
            max_len=max_len,
            device=DEVICE
        )
    return pred_ids

def translate_text(src: str, beam: int) -> str:
    """A warpper to translate the text."""
    src_vocab = load_src_vocab()
    tgt_vocab = load_tgt_vocab()
    model = load_model()

    src_tokens = src.strip().split()
    if not src_tokens:
        raise ValueError("empty input")
    src_ids = [
        src_vocab.word2int.get(tok, UNK_ID) for tok in src_tokens
    ]

    pred_ids = generate_decoded(model=model, src=src_ids, beam_size=beam)
    text = decode_tgt(pred_ids, tgt_vocab)
    return text

def interactive_translate(beam: int) -> None:
    """REPL translator."""
    print("Welcome to REPL Translator")
    import readline
    src_vocab = load_src_vocab()
    tgt_vocab = load_tgt_vocab()
    model = load_model()

    while(True):
        try:
            src = input("> ")
            src_tokens = src.strip().split()
            if not src_tokens:
                continue
            src_ids = [
                src_vocab.word2int.get(tok, UNK_ID) for tok in src_tokens
            ]
            pred_ids = generate_decoded(model=model, src=src_ids, beam_size=beam)
            text = decode_tgt(pred_ids, tgt_vocab)
            print(text)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        except Exception as e:
            print(f"[translate] {str(e)}")
