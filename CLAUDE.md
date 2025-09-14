# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExpeL (LLM Agents are Experiential Learners) is a research implementation for AAAI 2024 that enables LLM agents to learn from experience autonomously. The agent gathers experiences from training tasks, extracts insights in natural language, and uses these insights for better decision-making during evaluation.

## Development Setup

### Prerequisites
- Python 3.9.17
- OpenAI API key (required for non-testing runs)

### Installation
```bash
conda create -n expel python=3.9.17
conda activate expel
pip install -r requirements.txt
```

### Environment Setup
OpenAI API key must be provided either:
- In `.env` file: `OPENAI_API_KEY=your_key_here`
- As environment variable
- Via command prompt when running scripts

## Core Commands

### Training Phase 1: Experience Gathering
```bash
python train.py benchmark=<benchmark-name> run_name=<train-run-name> testing=false resume=false

# Examples:
python train.py benchmark=alfworld run_name=my_train_run testing=false resume=false
python train.py benchmark=webshop run_name=my_train_run testing=false resume=false
python train.py benchmark=hotpotqa run_name=my_train_run testing=false resume=false
```

### Training Phase 2: Insights Extraction
```bash
python insight_extraction.py benchmark=<benchmark-name> load_run_name=<train-run-name> run_name=<insights-run-name> agent.llm=gpt-4 agent.max_num_rules=<num> agent.success_critique_num=<num> testing=false resume=false

# Examples:
python insight_extraction.py benchmark=alfworld load_run_name=my_train_run run_name=my_insights_run agent.llm=gpt-4 agent.max_num_rules=10 agent.success_critique_num=8 testing=false resume=false
```

### Evaluation
```bash
python eval.py benchmark=<benchmark-name> load_run_name=extracted_insights/<insights-run-name> run_name=<eval-run-name> agent.fewshot_strategy=task_similarity agent.retrieval_kwargs.max_fewshot_tokens=auto testing=false resume=false
```

### Testing Mode
Add `testing=true` to any command to run without OpenAI API calls (for development/testing).

## Architecture Overview

### Core Components

**Agent System (`agent/`)**:
- `ExpelAgent` (expel.py): Main agent implementing experiential learning
- `ReflectAgent` (reflect.py): Base agent with reflection capabilities
- `ReactAgent` (react.py): Basic ReAct-style agent
- Uses three-phase learning: experience gathering → insight extraction → evaluation

**Environment Support (`envs/`)**:
- ALFWorld: Interactive text-based household tasks
- WebShop: E-commerce environment (requires separate server setup)
- HotpotQA: Multi-hop question answering
- FEVER: Fact verification

**Memory & Retrieval (`memory/`)**:
- Trajectory storage and retrieval for experience learning
- Vector embeddings for similarity-based experience retrieval
- Episode-based memory management

**Prompt Engineering (`prompts/`)**:
- Structured prompts for different environments
- Template system for consistent agent interactions
- Reflection and critique prompt templates

### Key Data Flow
1. **Training Phase 1 (Experience Gathering)**: Agent interacts with environment, stores successful/failed trajectories
2. **Training Phase 2 (Insight Extraction)**: Analyzes collected experiences to extract actionable insights
3. **Evaluation**: Uses extracted insights and retrieved experiences for improved performance

### Configuration System
Uses Hydra for configuration management:
- `configs/train.yaml`: Training Phase 1 configuration
- `configs/insight_extraction.yaml`: Training Phase 2 configuration
- `configs/eval.yaml`: Evaluation phase configuration
- `configs/benchmark/`: Environment-specific settings

### Logging Structure
Results saved in `logs/<benchmark-name>/expel/`:
- Training Phase 1 logs in main directory
- Training Phase 2 results in `extracted_insights/` subdirectory
- Evaluation results in `eval/` subdirectory

## Environment-Specific Setup

### WebShop
Requires separate server installation and setup. The WebShop server must be running in parallel when using this environment. Update `envs/webshop/webshop.py` with the server URL.

### ALFWorld
Requires additional setup:
```bash
pip install alfworld[full]
export ALFWORLD_DATA="data/alfworld"
alfworld-download
```

## Key Parameters

- `agent.llm`: Model choice (gpt-3.5-turbo, gpt-4)
- `agent.max_num_rules`: Target number of insights to extract
- `agent.success_critique_num`: Number of experiences to analyze per iteration
- `agent.fewshot_strategy`: Retrieval strategy (task_similarity, thought_similarity, task_thought_similarity)
- `benchmark.eval_configs.k_folds`: Number of evaluation folds
- `resume=true/false`: Resume previous runs