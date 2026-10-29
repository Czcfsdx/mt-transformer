import json
from dataclasses import dataclass

from src.config import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SRC_INV_VOCAB_PATH,
    SRC_VOCAB_PATH,
    TGT_INV_VOCAB_PATH,
    TGT_VOCAB_PATH,
    UNK_ID,
)


@dataclass
class Vocab:
    """
    mapping between tokens and ids.
    word2int: Token (str) -> ID (int)
    int2word: ID (int) -> Token (str)
    """
    word2int: dict[str, int]
    int2word: dict[int, str]

    @classmethod
    def load(cls, vocab_path: str, inv_vocab_path: str) -> "Vocab":
        """Load a vocabulary from a pair of JSON files."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            word2int = json.load(f)
        with open(inv_vocab_path, "r", encoding="utf-8") as f:
            # int2word stores ids as string keys for JSON compatibility
            int2word = {int(k): v for k, v in json.load(f).items()}
        return cls(word2int=word2int, int2word=int2word)

    def __len__(self) -> int:
        return len(self.word2int)


def load_src_vocab() -> Vocab:
    """Warpper to load the source language vocabulary define in config.py."""
    return Vocab.load(str(SRC_VOCAB_PATH), str(SRC_INV_VOCAB_PATH))

def load_tgt_vocab() -> Vocab:
    """Warpper to load the target language vocabulary define in config.py."""
    return Vocab.load(str(TGT_VOCAB_PATH), str(TGT_INV_VOCAB_PATH))


def encode_src(tokens: list[str], vocab: Vocab) -> list[int]:
    """Convert source tokens to ids, mapping unknowns to <UNK>."""
    return [vocab.word2int.get(tok, UNK_ID) for tok in tokens]


def encode_tgt(tokens: list[str], vocab: Vocab) -> list[int]:
    """Convert target tokens to ids and wrap with ``<BOS>`` / ``<EOS>``."""
    ids = [vocab.word2int.get(tok, UNK_ID) for tok in tokens]
    return [BOS_ID] + ids + [EOS_ID]

def decode_src(ids: list[int], vocab: Vocab) -> str:
    """
    Decode source ids into a string.

    The output will merging BPE ``@@`` continuations into single words.
    Example: ["bi", "@@", "ting"] → "biting".
    """
    tokens: list[str] = []
    special_tokens = {PAD_ID, BOS_ID, EOS_ID}
    for i in ids:
        if i in special_tokens:
            continue

        token = vocab.int2word.get(i, "")
        if not token:
            continue

        if token.endswith("@@"):
            tokens.append(token[:-2])  # strip @@ and no leading space
        else:
            tokens.append(" " + token)

    return "".join(tokens).strip()


def decode_tgt(ids: list[int], vocab: Vocab) -> str:
    """
    Decode target ids into a string.

    The output is separated by space and special tokens are stripped.
    """
    special_tokens = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
    tokens = [vocab.int2word.get(i, "") for i in ids if i not in special_tokens]
    return " ".join(tokens)
