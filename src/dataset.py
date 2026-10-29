from pathlib import Path
from typing import override

import torch
from torch.utils.data import Dataset

from src.config import MAX_LEN, PAD_ID
from src.vocab import Vocab, encode_src, encode_tgt


class TranslationDataset(Dataset):
    def __init__(
        self,
        path: Path,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int = MAX_LEN,
    ) -> None:
        super().__init__()
        self.src_vocab: Vocab = src_vocab
        self.tgt_vocab: Vocab = tgt_vocab
        self.max_len: int = max_len

        self.samples: list[tuple[list[int], list[int]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                # each line in file has form: <english> \t <chinese>
                src_part, tgt_part = line.split("\t")
                src_tokens = src_part.split()
                tgt_tokens = tgt_part.split()
                src_ids = encode_src(src_tokens, src_vocab)
                tgt_ids = encode_tgt(tgt_tokens, tgt_vocab)

                if len(src_ids) > max_len or len(tgt_ids) > max_len:
                    continue

                self.samples.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        src_ids, tgt_ids = self.samples[index]
        return src_ids, tgt_ids


def collate_fn(
    batch: list[tuple[list[int], list[int]]],
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_list, tgt_list = zip(*batch)

    src_max = max(len(s) for s in src_list)
    tgt_max = max(len(t) for t in tgt_list)

    src_padded = torch.full((len(batch), src_max), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
        src_padded[i, : len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_padded[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    # Teacher forcing: decoder input is target[:-1]; label is target[1:].
    tgt_in = tgt_padded[:, :-1]
    tgt_out = tgt_padded[:, 1:]

    # src_padded: encoder inputs
    # tgt_in:     decoder inputs
    # tgt_out:    target shifted left by one position
    return src_padded, tgt_in, tgt_out
