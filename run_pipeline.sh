#!/bin/bash

set -e

DATASET="cnn_dailymail"
MODEL="facebook/bart-large-cnn"
MAX_SAMPLES=100
OUTPUT_DIR="./output"
NUM_EPOCHS=3
BATCH_SIZE=4
DEVICE="cuda"

print_help() {
    echo "Usage: ./run_pipeline.sh [OPTIONS]"
    echo "Run the factual consistency improvement pipeline"
    echo ""
    echo "Options:"
    echo "  --dataset DATASET        Dataset to use (cnn_dailymail, xsum)"
    echo "  --model MODEL            Model to use (facebook/bart-large-cnn, t5-base, etc.)"
    echo "  --max_samples NUM        Maximum number of samples to use"
    echo "  --output_dir DIR         Output directory"
    echo "  --num_epochs NUM         Number of training epochs"
    echo "  --batch_size NUM         Batch size for training"
    echo "  --device DEVICE          Device to use (cuda, cpu)"
    echo "  --baseline_only          Only train the baseline model"
    echo "  --rl_only                Only train with RL enhancement"
    echo "  --postprocessing_only    Only evaluate post-processing"
    echo "  --decoding_only          Only evaluate factuality-guided decoding"
    echo "  --evaluate_only          Skip training and only evaluate"
    echo "  --help                   Show this help message"
    exit 0
}

BASELINE_ONLY=false
RL_ONLY=false
POSTPROCESSING_ONLY=false
DECODING_ONLY=false
EVALUATE_ONLY=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --baseline_only) BASELINE_ONLY=true; shift ;;
        --rl_only) RL_ONLY=true; shift ;;
        --postprocessing_only) POSTPROCESSING_ONLY=true; shift ;;
        --decoding_only) DECODING_ONLY=true; shift ;;
        --evaluate_only) EVALUATE_ONLY=true; shift ;;
        --help) print_help ;;
        *) echo "Unknown parameter: $1"; print_help ;;
    esac
done

echo "=== Factual Consistency Improvement Pipeline ==="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Max samples: $MAX_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "==============================================="

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_ID="${DATASET}_${MODEL##*/}_${TIMESTAMP}"
echo "Experiment ID: $EXPERIMENT_ID"

mkdir -p $OUTPUT_DIR
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/summaries"
mkdir -p "$OUTPUT_DIR/results"

MAIN_CMD="python main.py --dataset $DATASET --model $MODEL --max_samples $MAX_SAMPLES --output_dir $OUTPUT_DIR --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --device $DEVICE"

if [ "$EVALUATE_ONLY" = true ]; then
    MAIN_CMD="$MAIN_CMD --skip_baseline"
fi

if [ "$BASELINE_ONLY" = true ]; then
    MAIN_CMD="$MAIN_CMD --skip_rl --skip_postprocessing --skip_factuality_decoding"
fi

if [ "$RL_ONLY" = true ]; then
    MAIN_CMD="$MAIN_CMD --skip_postprocessing --skip_factuality_decoding"
fi

if [ "$POSTPROCESSING_ONLY" = true ]; then
    MAIN_CMD="$MAIN_CMD --skip_baseline --skip_rl --skip_factuality_decoding"
fi

if [ "$DECODING_ONLY" = true ]; then
    MAIN_CMD="$MAIN_CMD --skip_baseline --skip_rl --skip_postprocessing"
fi

echo "Running command: $MAIN_CMD"
eval $MAIN_CMD

echo "Generating analysis report..."
python -c "import analysis_notebook"

echo "Pipeline completed successfully!"
echo "Results saved to: $OUTPUT_DIR/results/report_${EXPERIMENT_ID}.html"
