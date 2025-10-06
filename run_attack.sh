#!/bin/bash

# IMDGA Attack Script
# This script runs the graph adversarial attack

# Default parameters
DATASET="cora"
LLM="bert"
GPU=0
EPOCHS=0
LR=0.01
TOP_K=30

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --llm)
            LLM="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running IMDGA attack with:"
echo "  Dataset: $DATASET"
echo "  LLM: $LLM"
echo "  GPU: $GPU"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Top-K: $TOP_K"
echo ""

# Run the attack
python src/main.py \
    --dataset $DATASET \
    --llm $LLM \
    --gpu $GPU \
    --epochs $EPOCHS \
    --lr $LR \
    --top_k $TOP_K \
    --model gcn \
    --num_layers 2 \
    --hidden_channels 64 \
    --dropout 0.5 \
    --using_shap \
    --train_ratio 0.1 \
    --val_ratio 0.1 \
    --test_ratio 0.8

echo ""
echo "Attack completed!"

