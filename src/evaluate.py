from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from src.config import MAX_LEN, TEST_FILE, DEFAULT_BEAM_SIZE
from src.dataset import TranslationDataset
from src.model import Transformer
from src.utils import generate_decoded, load_model
from src.vocab import decode_tgt, load_src_vocab, load_tgt_vocab

def translate_dataset(
    model: Transformer,
    dataset: TranslationDataset,
    beam_size: int = DEFAULT_BEAM_SIZE,
    max_len: int = MAX_LEN,
) -> tuple[list[str], list[str]]:
    src_vocab = load_src_vocab()
    tgt_vocab = load_tgt_vocab()

    sys: list[str] = []
    refs: list[str] = []
    for src_ids, tgt_ids in tqdm(dataset, desc="decoding"):
        pred_ids = generate_decoded(
            model=model,
            src=src_ids,
            beam_size=beam_size,
            max_len=max_len,
        )
        sys.append(decode_tgt(pred_ids, tgt_vocab))
        refs.append(decode_tgt(tgt_ids, tgt_vocab))
    return sys, refs


def evaluate(
    beam_size: int = DEFAULT_BEAM_SIZE,
    n_samples_preview: int = 0,
) -> None:
    """
    Evaluation on the test set.
    """
    src_vocab = load_src_vocab()
    tgt_vocab = load_tgt_vocab()
    test_ds = TranslationDataset(TEST_FILE, src_vocab, tgt_vocab)

    model = load_model()
    print(f"[evaluate] decoder: {'beam search (beam size = ' + str(beam_size) + ')' if beam_size > 1 else 'greedy'}")

    sys, refs = translate_dataset(model, test_ds, beam_size=beam_size)

    bleu = BLEU(tokenize="zh")
    score = bleu.corpus_score(sys, [refs])
    print(f"[evaluate] {score}")

    # sample preview
    print("\n[evaluate] sample translations:")
    for i in range(min(n_samples_preview, len(sys))):
        src_ids, _ = test_ds[i]
        src_text = " ".join(
            src_vocab.int2word.get(idx, "") for idx in src_ids
        )
        print(f"  src : {src_text}")
        print(f"  sys : {sys[i]}")
        print(f"  ref : {refs[i]}")
        print()
