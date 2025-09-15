#!/bin/bash

# ExpeL ALFWorld Full Pipeline Script (Unified Training Version)
# This script runs the complete ExpeL pipeline:
# 1. Unified Two-Phase Training (Experience Gathering + Insights Extraction)
# 2. Evaluation

set -e  # Exit on any error

# Configuration
RUN_NAME=${1:-"alfworld_run_$(date +%Y%m%d_%H%M%S)"}
LLM_MODEL=${2:-"gpt-4o-mini"}
TESTING=${3:-"false"}
MAX_NUM_RULES=${4:-"10"}
SUCCESS_CRITIQUE_NUM=${5:-"8"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-WPZxkYp48tG3OMZcm5h2HwTrGicWCem0BkEZRCY6DaLDksXW"}
export BASE_URL=${BASE_URL:-"https://api.chatanywhere.tech/v1"}

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

    if [ ! -f "eval.py" ]; then
        print_error "eval.py not found!"
        exit 1
    fi

    if [ ! -f "configs/benchmark/alfworld.yaml" ]; then
        print_error "ALFWorld config not found!"
        exit 1
    fi

    print_success "All dependencies found"
    print_status "Using unified training system (train.py handles both phases)"
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
    print_header "ExpeL ALFWorld Full Pipeline (Unified Training)"
    echo "Run Name: $RUN_NAME"
    echo "LLM Model: $LLM_MODEL"
    echo "Testing Mode: $TESTING"
    echo "Max Num Rules: $MAX_NUM_RULES"
    echo "Success Critique Num: $SUCCESS_CRITIQUE_NUM"
    echo ""

    check_dependencies

    # Unified Two-Phase Training
    print_header "UNIFIED TWO-PHASE TRAINING"
    print_status "Starting unified training (Phase 1: Experience Gathering + Phase 2: Insights Extraction)..."

    python train.py \
        benchmark=alfworld \
        run_name="$RUN_NAME" \
        agent.llm="$LLM_MODEL" \
        agent.max_num_rules="$MAX_NUM_RULES" \
        agent.success_critique_num="$SUCCESS_CRITIQUE_NUM" \
        testing="$TESTING" \
        resume=false \
        run_phase_1=true \
        run_phase_2=true

    check_success "Unified Two-Phase Training"

    # Evaluation
    print_header "EVALUATION"
    print_status "Starting Evaluation..."

    python eval.py \
        benchmark=alfworld \
        load_run_name="extracted_insights/$RUN_NAME" \
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
    echo "  - Training (Both Phases): logs/alfworld/expel/$RUN_NAME.pkl"
    echo "  - Insights: logs/alfworld/expel/extracted_insights/$RUN_NAME.pkl"
    echo "  - Evaluation: logs/alfworld/expel/eval/${RUN_NAME}_eval.pkl"
    echo ""
    echo "To view logs, check the logs/alfworld/expel/ directory"
    echo ""
    print_success "🎉 ExpeL ALFWorld pipeline completed successfully with unified training!"
}

# Help function
show_help() {
    echo "ExpeL ALFWorld Full Pipeline Script (Unified Training Version)"
    echo "Usage: $0 [RUN_NAME] [LLM_MODEL] [TESTING] [MAX_NUM_RULES] [SUCCESS_CRITIQUE_NUM]"
    echo ""
    echo "Parameters:"
    echo "  RUN_NAME            - Name for this experimental run (default: alfworld_run_TIMESTAMP)"
    echo "  LLM_MODEL           - LLM model to use (default: gpt-4o-mini)"
    echo "  TESTING             - Testing mode true/false (default: false)"
    echo "  MAX_NUM_RULES       - Max number of insights to extract (default: 10)"
    echo "  SUCCESS_CRITIQUE_NUM - Number of success examples to analyze (default: 8)"
    echo ""
    echo "Features:"
    echo "  ✨ Unified Training: Runs both Experience Gathering and Insights Extraction in one command"
    echo "  🚀 Simplified Pipeline: No need to manage intermediate run names"
    echo "  📊 Automatic Progress: Clear visual indicators for each phase"
    echo "  💾 Checkpoint Support: Resume from any point in the pipeline"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Use all defaults"
    echo "  $0 my_experiment                            # Custom run name"
    echo "  $0 my_experiment gpt-4                     # Custom run name and model"
    echo "  $0 my_experiment gpt-4 false               # Specify testing mode"
    echo "  $0 my_experiment gpt-4 false 15 10         # Full customization"
    echo "  $0 test_run gpt-4o-mini true               # Testing mode (no API calls)"
    echo ""
    echo "Pipeline Phases:"
    echo "  1. 🏃 Unified Two-Phase Training:"
    echo "     • Phase 1: Experience Gathering (agent learns from environment)"
    echo "     • Phase 2: Insights Extraction (extract actionable insights)"
    echo "  2. 📈 Evaluation (test performance with extracted insights)"
    echo ""
    echo "Output Locations:"
    echo "  • Training Data: logs/alfworld/expel/[RUN_NAME].pkl"
    echo "  • Insights: logs/alfworld/expel/extracted_insights/[RUN_NAME].pkl"
    echo "  • Evaluation: logs/alfworld/expel/eval/[RUN_NAME]_eval.pkl"
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