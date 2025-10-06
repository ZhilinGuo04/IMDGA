"""
Data preparation script for IMDGA
Downloads and preprocesses standard datasets
"""
import os
import sys
import argparse
import torch
from torch_geometric.datasets import Planetoid, WikiCS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_utils import tag_dataset
from src.logger import create_logger
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer, DistilBertTokenizer, AutoModel

def prepare_dataset(args):
    """Prepare and process dataset"""
    
    print(f"Preparing {args.dataset} dataset with {args.llm} embeddings...")
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    print(f"Loading {args.llm} model...")
    if args.llm == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(config.get_llm_path('roberta'))
        llm_model = AutoModel.from_pretrained(config.get_llm_path('roberta')).to(device)
    elif args.llm == 'deberta':
        tokenizer = DebertaTokenizer.from_pretrained(config.get_llm_path('deberta'))
        llm_model = AutoModel.from_pretrained(config.get_llm_path('deberta')).to(device)
    elif args.llm == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(config.get_llm_path('distilbert'))
        llm_model = AutoModel.from_pretrained(config.get_llm_path('distilbert')).to(device)
    else:  # bert
        tokenizer = BertTokenizer.from_pretrained(config.get_llm_path('bert'))
        llm_model = AutoModel.from_pretrained(config.get_llm_path('bert')).to(device)
    
    logger = create_logger(args, f"{args.dataset}_{args.llm}_prep")
    
    base_path = config.get_dataset_path(args.dataset, 'bert') 
    output_path = config.get_dataset_path(args.dataset, args.llm)
    
    print(f"Loading dataset from: {base_path}")
    print(f"Regenerating embeddings with {args.llm}...")
    print(f"Will save to: {output_path}")
    
    dataset = tag_dataset(args, logger, device, base_path)
    data = dataset.process_data(tokenizer, llm_model, output_path)
    
    print(f"Dataset statistics:")
    print(f"  Number of nodes: {data.num_nodes}")
    print(f"  Number of edges: {data.num_edges}")
    print(f"  Number of features: {data.num_features}")
    print(f"  Number of classes: {data.num_classes}")
    print(f"  Train nodes: {len(data.train_indices)}")
    print(f"  Val nodes: {len(data.val_indices)}")
    print(f"  Test nodes: {len(data.test_indices)}")
    
    print(f"\nData saved to: {output_path}")
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for IMDGA')
    parser.add_argument('--dataset', type=str, default='cora', 
                       choices=['cora', 'citeseer', 'pubmed'],
                       help='Dataset to prepare')
    parser.add_argument('--llm', type=str, default='bert',
                       choices=['bert', 'roberta', 'deberta', 'distilbert'],
                       help='Language model for embeddings')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.6)
    parser.add_argument('--root_path', type=str, default='./',
                       help='Root path for logging')
    parser.add_argument('--re_split', type=bool, default=False,
                       help='Whether to re-split the dataset')
    
    args = parser.parse_args()
    
    prepare_dataset(args)

if __name__ == "__main__":
    main()

