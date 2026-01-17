# IMDGA: Interpretable Multi-Dimensional Graph Attack

[![Conference](https://img.shields.io/badge/WWW-2026-brightgreen?style=flat-square)](https://www2026.thewebconf.org/)
&ensp;
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](./LICENSE)
&ensp;
[![Arxiv](https://img.shields.io/badge/Arxiv-2510.12233-B31B1B?style=flat-square)](https://arxiv.org/abs/2510.12233)

## Introduction

This repository contains the official PyTorch implementation of the paper "Unveiling the Vulnerability of Graph-LLMs: An Interpretable Multi-Dimensional Graph Attack on TAGs" (IMDGA). IMDGA is a novel human-centric adversarial attack framework designed to evaluate the robustness of Text-Attributed Graphs (TAGs) that integrate Graph Neural Networks (GNNs) with Large Language Models (LLMs).

Unlike existing methods that target structure or features in isolation, IMDGA orchestrates **multi-level perturbations** across both graph topology and textual semantics. By leveraging interpretable modules, it uncovers underexplored vulnerabilities in Graph-LLM architectures, establishing a unified benchmark for Graph-LLM security.

## Quick Start

### 1. Installation

```bash
# Using conda (recommended)
conda create -n imdga python=3.10
conda activate imdga
pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Model Paths

Edit `config.py` to specify your trained GNN model paths:

```python
TRAINED_MODELS = {
    ('bert', 'cora'): 'data/model/your_cora_model.pt',
    ('bert', 'citeseer'): 'data/model/your_citeseer_model.pt',
    # Add more models as needed
}
```

Optionally, configure local LLM paths (otherwise models will auto-download from Hugging Face):

```python
LLM_MODELS = {
    'bert': 'bert-base-uncased',  # or '/path/to/local/bert'
    'roberta': 'roberta-base',
    'deberta': 'microsoft/deberta-base',
    'distilbert': 'distilbert-base-uncased'
}
```

### 3. Prepare Your Models

Train GNN models and place them in `data/model/`. The code includes a `train_model()` function in `main.py` (line 873) that you can uncomment and use.

### 4. Run Attack

```bash
python src/main.py \
    --dataset cora \
    --llm bert \
    --gpu 0 \
    --epochs 0 \
    --using_shap \
    --top_k 30
```

Or use the convenience script:

```bash
bash run_attack.sh --dataset cora --llm bert --gpu 0 --epochs 0
```

## Project Structure

```
IMDGA/
├── src/
│   ├── main.py              # Main attack script
│   ├── gnn.py               # GNN models (GCN, GAT, SAGE, etc.)
│   ├── data_utils.py        # Dataset loading and processing
│   ├── utils.py             # Utility functions
│   ├── logger.py            # Logging utilities
│   └── tag_pipeline.py      # SHAP analysis pipeline
├── gnnshap/                 # GNN explainer implementation
├── data/
│   ├── dataset/            # Pre-processed datasets (included)
│   │   ├── cora_bert_processed
│   │   ├── citeseer_bert_processed
│   │   └── pubmed_bert_processed
│   ├── model/              # Trained models (user-provided)
│   └── task/               # Attack nodes (auto-generated)
├── logs/                    # Experiment logs (auto-generated)
├── config.py               # Configuration file
├── prepare_data.py         # Optional: Generate new embeddings
├── run_attack.sh           # Convenience script
└── requirements.txt        # Dependencies
```

## Data

### Included Datasets

Three pre-processed datasets are included in `data/dataset/`:
- **cora_bert_processed**: Cora citation network with BERT embeddings
- **citeseer_bert_processed**: CiteSeer citation network with BERT embeddings  
- **pubmed_bert_processed**: PubMed citation network with BERT embeddings

Each dataset contains:
- Graph structure (edge indices)
- Node labels
- Raw text for each node
- Pre-computed text embeddings
- Train/validation/test splits

### Generating New Embeddings (Optional)

To use different language models (RoBERTa, DeBERTa, etc.):

```bash
python prepare_data.py --dataset cora --llm roberta --gpu 0
```

This loads the existing BERT-processed dataset and re-generates embeddings with your chosen LLM.

## Command-Line Arguments

### Main Arguments

- `--dataset`: Dataset name (`cora`, `citeseer`, `pubmed`)
- `--llm`: Language model (`bert`, `roberta`, `deberta`, `distilbert`)
- `--model`: GNN model type (`gcn`, `gat`, `sage`)
- `--gpu`: GPU device ID
- `--epochs`: Training epochs (0 = no training, load existing model)
- `--using_shap`: Enable SHAP-based word importance
- `--top_k`: Number of top important words to perturb (default: 30)

### Model Hyperparameters

- `--num_layers`: Number of GNN layers (default: 2)
- `--hidden_channels`: Hidden dimension size (default: 64)
- `--dropout`: Dropout rate (default: 0.5)
- `--lr`: Learning rate (default: 0.01)
- `--train_ratio`, `--val_ratio`, `--test_ratio`: Data split ratios

See `src/main.py` for all available arguments.

## Configuration

All paths and hyperparameters are centralized in `config.py`:

- **TRAINED_MODELS**: Map (llm, dataset) to model file paths
- **LLM_MODELS**: LLM paths (Hugging Face IDs or local paths)
- **DEFAULT_CONFIG**: Default hyperparameters
- **ATTACK_CONFIG**: Attack-specific parameters

## Output

Attack logs are saved to `logs/` with timestamps. Each log includes:
- Attack configuration
- Original and perturbed text for each node
- Success rate and accuracy metrics
- Node-wise attack results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
