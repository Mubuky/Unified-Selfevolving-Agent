#!/bin/bash

# ExpeL ALFWorld Full Pipeline Script
# This script runs the complete ExpeL pipeline:
# 1. Training Phase 1: Experience Gathering
# 2. Training Phase 2: Insights Extraction
# 3. Evaluation

set -e  # Exit on any error

# Configuration
RUN_NAME=${1:-"alfworld_run_$(date +%Y%m%d_%H%M%S)"}
LLM_MODEL=${2:-"gpt-4o-mini"}
TESTING=${3:-"false"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================================${NC}"
}

# Check if required files exist
check_dependencies() {
    print_status "Checking dependencies..."

    if [ ! -f "train.py" ]; then
        print_error "train.py not found!"
        exit 1
    fi

    if [ ! -f "insight_extraction.py" ]; then
        print_error "insight_extraction.py not found!"
        exit 1
    fi

    if [ ! -f "eval.py" ]; then
        print_error "eval.py not found!"
        exit 1
    fi

    if [ ! -f "configs/benchmark/alfworld.yaml" ]; then
        print_error "ALFWorld config not found!"
        exit 1
    fi

    print_success "All dependencies found"
}

# Function to check if process completed successfully
check_success() {
    if [ $? -eq 0 ]; then
        print_success "$1 completed successfully"
    else
        print_error "$1 failed"
        exit 1
    fi
}

# Main execution
main() {
    print_header "ExpeL ALFWorld Full Pipeline"
    echo "Run Name: $RUN_NAME"
    echo "LLM Model: $LLM_MODEL"
    echo "Testing Mode: $TESTING"
    echo ""

    check_dependencies

    # Phase 1: Experience Gathering (Training Phase 1)
    print_header "PHASE 1: EXPERIENCE GATHERING"
    print_status "Starting Training Phase 1..."

    python train.py \
        benchmark=alfworld \
        run_name="$RUN_NAME" \
        agent.llm="$LLM_MODEL" \
        testing="$TESTING" \
        resume=false

    check_success "Training Phase 1"

    # Phase 2: Insights Extraction (Training Phase 2)
    print_header "PHASE 2: INSIGHTS EXTRACTION"
    print_status "Starting Training Phase 2..."

    python insight_extraction.py \
        benchmark=alfworld \
        load_run_name="$RUN_NAME" \
        run_name="${RUN_NAME}_insights" \
        agent.llm="gpt-4" \
        agent.max_num_rules=10 \
        agent.success_critique_num=8 \
        testing="$TESTING" \
        resume=false

    check_success "Training Phase 2"

    # Phase 3: Evaluation
    print_header "PHASE 3: EVALUATION"
    print_status "Starting Evaluation..."

    python eval.py \
        benchmark=alfworld \
        load_run_name="extracted_insights/${RUN_NAME}_insights" \
        run_name="${RUN_NAME}_eval" \
        agent.fewshot_strategy=task_similarity \
        agent.retrieval_kwargs.max_fewshot_tokens=auto \
        testing="$TESTING" \
        resume=false

    check_success "Evaluation"

    # Summary
    print_header "PIPELINE COMPLETED SUCCESSFULLY"
    print_success "All phases completed for run: $RUN_NAME"
    echo ""
    echo "Results can be found in:"
    echo "  - Training Phase 1: logs/alfworld/expel/$RUN_NAME.pkl"
    echo "  - Training Phase 2: logs/alfworld/expel/extracted_insights/${RUN_NAME}_insights.pkl"
    echo "  - Evaluation: logs/alfworld/expel/eval/${RUN_NAME}_eval.pkl"
    echo ""
    echo "To view logs, check the logs/alfworld/expel/ directory"
}

# Help function
show_help() {
    echo "Usage: $0 [RUN_NAME] [LLM_MODEL] [TESTING]"
    echo ""
    echo "Parameters:"
    echo "  RUN_NAME    - Name for this experimental run (default: alfworld_run_TIMESTAMP)"
    echo "  LLM_MODEL   - LLM model to use (default: gpt-3.5-turbo)"
    echo "  TESTING     - Testing mode true/false (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use defaults"
    echo "  $0 my_experiment                      # Custom run name"
    echo "  $0 my_experiment gpt-4               # Custom run name and model"
    echo "  $0 my_experiment gpt-4 true          # Enable testing mode"
    echo ""
    echo "Note: Make sure you have set up your OpenAI API key in .env file or as environment variable"
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"