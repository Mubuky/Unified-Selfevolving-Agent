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

**Modular Dataset System (`datasets/`)**:
- `BaseDataset` (base.py): Abstract base class for dataset handling
- `ExpelDataset` (expel_dataset.py): Concrete implementation supporting all benchmarks
- Handles data range configuration for training/evaluation splits
- Supports ALFWorld, WebShop, HotpotQA, and FEVER environments

**Modular Storage System (`storage/`)**:
- `BaseStorage` (base.py): Abstract base class for data persistence
- `ExpelStorage` (expel_storage.py): Concrete implementation managing complete data transfer chain
- Handles three-phase data flow: experience collection → insights extraction → evaluation
- Methods: `save_experience()`, `load_experience()`, `save_insights()`, `load_insights()`, `save_evaluation_results()`

**Modular Retrieval System (`retrieval/`)**:
- `BaseRetrieval` (base.py): Abstract base class for experience retrieval
- `ExpelRetrieval` (expel_retrieval.py): FAISS-based vector retrieval implementation
- Core methods: `setup_documents()`, `build_query_vectors()`, `create_filtered_vectorstore()`, `retrieve_topk_documents()`
- Supports multiple retrieval strategies: task_similarity, thought_similarity, rotation, etc.

**Environment Support (`envs/`)**:
- ALFWorld: Interactive text-based household tasks
- WebShop: E-commerce environment (requires separate server setup)
- HotpotQA: Multi-hop question answering
- FEVER: Fact verification
- Uses modular ExpelDataset for consistent data loading

**Memory & Trajectory (`memory/`)**:
- Trajectory storage and retrieval for experience learning
- Vector embeddings for similarity-based experience retrieval
- Episode-based memory management

**Prompt Engineering (`prompts/`)**:
- Structured prompts for different environments
- Template system for consistent agent interactions
- Reflection and critique prompt templates

### Key Data Flow
1. **Training Phase 1 (Experience Gathering)**:
   - Agent interacts with environment using ExpelDataset for task loading
   - ExpelStorage saves experiences via `save_experience()` method
   - Stores successful/failed trajectories in `succeeded_trial_history`/`failed_trial_history`

2. **Training Phase 2 (Insight Extraction)**:
   - ExpelStorage loads experiences via `load_experience()` method
   - Analyzes collected experiences to extract actionable insights and rules
   - ExpelStorage saves insights via `save_insights()` method

3. **Evaluation**:
   - ExpelStorage loads insights via `load_insights()` method
   - ExpelRetrieval system performs FAISS-based similarity search for relevant experiences
   - Uses extracted insights and retrieved experiences for improved performance
   - ExpelStorage saves results via `save_evaluation_results()` method

### Configuration System
Uses Hydra for configuration management:
- `configs/train.yaml`: Training Phase 1 configuration
- `configs/insight_extraction.yaml`: Training Phase 2 configuration
- `configs/eval.yaml`: Evaluation phase configuration
- `configs/benchmark/`: Environment-specific settings including data range configuration
  - `data_split.train_range`: Training data indices (e.g., [0, 10])
  - `data_split.eval_range`: Evaluation data indices (e.g., [10, 13])

### Modular Data Management
**ExpelDataset handles task loading**:
- Configurable training/evaluation data ranges via `data_split` configuration
- Supports mode-based loading ('train' vs 'eval')
- Uniform interface across all benchmark environments

**ExpelStorage manages data persistence**:
- Three-phase storage: experience → insights → evaluation
- Path structure: `logs/<benchmark>/<agent_type>/`, `extracted_insights/`, `eval/`
- Checkpoint support with automatic resume capabilities

**ExpelRetrieval provides experience search**:
- FAISS-based vector similarity search
- Multiple retrieval strategies: task_similarity, thought_similarity, rotation, etc.
- Dynamic few-shot example selection based on current context

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

### Core Agent Parameters
- `agent.llm`: Model choice (gpt-3.5-turbo, gpt-4)
- `agent.max_num_rules`: Target number of insights to extract
- `agent.success_critique_num`: Number of experiences to analyze per iteration
- `agent.fewshot_strategy`: Retrieval strategy (task_similarity, thought_similarity, task_thought_similarity, rotation, etc.)
- `agent.retrieval_kwargs.max_fewshot_tokens`: Token limit per few-shot example ('auto' or integer)
- `agent.retrieval_kwargs.embedder_type`: Embedding model (openai, huggingface, gpt4all)
- `agent.retrieval_kwargs.reranker`: Re-ranking strategy (none, len, thought, task)
- `agent.retrieval_kwargs.buffer_retrieve_ratio`: Over-retrieval factor for filtering

### Data Configuration Parameters
- `benchmark.data_split.train_range`: Training data indices (e.g., [0, 10])
- `benchmark.data_split.eval_range`: Evaluation data indices (e.g., [10, 13])
- `benchmark.eval_configs.k_folds`: Number of evaluation folds (for cross-validation, now typically 0)

### Execution Parameters
- `resume=true/false`: Resume previous runs
- `testing=true/false`: Enable testing mode (no API calls)
- `run_name`: Unique identifier for the current run
- `load_run_name`: Name of previous run to load data from

### Storage and Retrieval Configuration
- All modular components (ExpelDataset, ExpelStorage, ExpelRetrieval) are automatically initialized
- Storage paths are automatically managed based on benchmark and agent type
- Retrieval system is seamlessly integrated into the agent's decision-making process