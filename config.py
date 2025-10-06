"""
Configuration file for IMDGA
"""
import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
MODEL_DIR = os.path.join(DATA_DIR, 'model')
TASK_DIR = os.path.join(DATA_DIR, 'task')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Pretrained model paths
PRETRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'pretrained_models')

# Model configurations
LLM_MODELS = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'deberta': 'microsoft/deberta-base',
    'distilbert': 'distilbert-base-uncased'
}

# Dataset configurations
DATASETS = ['cora', 'citeseer', 'pubmed']

# GNN model configurations
GNN_MODELS = ['gcn', 'gat', 'sage']

# Trained GNN model paths (fill in your model paths here)
# Format: (llm, dataset): 'path/to/model.pt'
# Example:
# TRAINED_MODELS = {
#     ('bert', 'cora'): 'data/model/bert_cora_model.pt',
#     ('roberta', 'citeseer'): 'data/model/roberta_citeseer_model.pt',
# }
TRAINED_MODELS = {
    # Add your trained model paths here
}

# Default hyperparameters
DEFAULT_CONFIG = {
    'num_layers': 2,
    'hidden_channels': 64,
    'dropout': 0.5,
    'lr': 0.01,
    'l2decay': 0.0,
    'epochs': 200,
    'test_freq': 1,
    'top_k': 30,
    'train_ratio': 0.2,
    'val_ratio': 0.2,
    'test_ratio': 0.6,
    'batch_size': 512,
    'max_length': 512,
    'num_samples': 10000,  # For GNNShap
}

# Attack parameters
ATTACK_CONFIG = {
    'max_word_change_ratio': 0.3,  # Maximum 30% words can be changed
    'min_word_length': 2,  # Minimum word length to consider
    'substitute_threshold': 2.0,  # Threshold for word substitutes
}

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TASK_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)

def get_dataset_path(dataset, llm):
    """Get processed dataset path"""
    return os.path.join(DATASET_DIR, f"{dataset}_{llm}_processed")

def get_model_path(llm, dataset, num_layers=2, epochs=200, lr=0.01, 
                   train_ratio=0.2, val_ratio=0.2, test_ratio=0.6):
    """Get model weight path - checks TRAINED_MODELS first, then falls back to auto-generated path"""
    # First check if user has specified a path in TRAINED_MODELS
    if (llm, dataset) in TRAINED_MODELS:
        return TRAINED_MODELS[(llm, dataset)]
    # Fall back to auto-generated path
    return os.path.join(MODEL_DIR, 
        f"{llm}_{dataset}_{num_layers}layers_{epochs}_{lr}_{train_ratio}_{val_ratio}_{test_ratio}.pt")

def get_attack_nodes_path(llm, dataset, run_id=0):
    """Get attack nodes path"""
    return os.path.join(TASK_DIR, f"{llm}_{dataset}_attack_nodes_{run_id}")


def get_llm_path(llm_name):
    """Get LLM model path"""
    if llm_name in LLM_MODELS:
        return LLM_MODELS[llm_name]
    return llm_name

