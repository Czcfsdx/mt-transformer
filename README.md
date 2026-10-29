# English-to-Chinese Machine Translation with Transformer

A machine translation project that implements a Transformer encoder–decoder
architecture. This project aims to demonstrate how to implement this using only
with basic PyTorch components and not using any high-level
`torch.nn.Transformer` wrapper.

- **Custom Transformer implementation:** Model components are implemented with `torch.nn.Module` using an encoder–decoder architecture.
- **Training pipeline:** Adam optimizer, Noam learning-rate scheduling, KL-form loss with label smoothing, and early stopping monitored on validation loss.
- **Multiple decoding strategies:** Greedy decoding and beam search.
- **Automatic evaluation:** Corpus-level BLEU computed with `sacreBLEU`.
- **Command-line interface:** Run training, evaluation, and translation through `main.py`.

## Requirements

- Python 3.13
- PyTorch
- sacreBLEU

## Getting Started

### 1. Prepare the code and data

Place the project code locally and prepare the training, validation, and test data, along with any vocabulary or model files required by the implementation, according to the configuration in `src/config.py`.

### 2. View CLI help

```bash
python main.py --help
```

The report identifies `train`, `evaluate`, and `translate` as the available CLI commands. Refer to the help output for each command's exact arguments.

### 3. Train the model

```bash
python main.py train --help
```

Review the available options, then launch training according to the project's configuration:

```bash
python main.py train
```

Training uses Adam with a Noam warmup learning-rate schedule. Validation loss is monitored for early stopping. Actual data paths, output locations, and configurable options are determined by `src/config.py` and the CLI help.

### 4. Evaluate the model

```bash
python main.py evaluate --help
```

Review the evaluation options and provide the test set and model checkpoint as required by the implementation. The evaluation metric is corpus-level BLEU, computed with `sacreBLEU`.

### 5. Translate English text

First inspect the available arguments:

```bash
python main.py translate --help
```

The command supports translation of specified text and interactive translation. Refer to the CLI help for the exact input arguments, checkpoint paths, and decoding options.
