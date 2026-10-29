from pathlib import Path
import torch

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "best.pt"

SRC_VOCAB_PATH: Path = DATA_DIR / "word2int_en.json"
SRC_INV_VOCAB_PATH: Path = DATA_DIR / "int2word_en.json"
TGT_VOCAB_PATH: Path = DATA_DIR / "word2int_cn.json"
TGT_INV_VOCAB_PATH: Path = DATA_DIR / "int2word_cn.json"

TRAIN_FILE: Path = DATA_DIR / "training.txt"
VAL_FILE: Path = DATA_DIR / "validation.txt"
TEST_FILE: Path = DATA_DIR / "testing.txt"

# Special token ids
PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
UNK_ID: int = 3

# Vocabulary sizes – derived from the provided dictionaries
SRC_VOCAB_SIZE: int = 3922
TGT_VOCAB_SIZE: int = 3775

# Model architecture
D_MODEL: int = 256 # 512
N_HEAD: int = 8
N_LAYER: int = 3 # 6
D_FF: int = 1024 # 2048
DROPOUT: float = 0.1
MAX_LEN: int = 50 # 512

assert D_MODEL % N_HEAD == 0, "D_MODEL must be divisible by N_HEAD"

# Optimisation
BATCH_SIZE: int = 64
EPOCHS: int = 50
LABEL_SMOOTHING: float = 0.1
WARMUP_STEPS: int = 4000  # Noam warmup
EARLY_STOP_PATIENCE: int = 10  # number of epochs with no val-loss improvement

# Others
DEFAULT_BEAM_SIZE: int = 4
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
