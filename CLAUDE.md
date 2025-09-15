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

### Unified Two-Phase Training (Recommended)
Run both Training Phase 1 and Training Phase 2 sequentially with a single command:
```bash
python train.py benchmark=<benchmark-name> run_name=<run-name> agent.llm=gpt-4 agent.max_num_rules=<num> agent.success_critique_num=<num> testing=false resume=false

# Examples:
python train.py benchmark=alfworld run_name=my_experiment agent.llm=gpt-4 agent.max_num_rules=10 agent.success_critique_num=8 testing=false resume=false
python train.py benchmark=webshop run_name=my_experiment agent.llm=gpt-4 agent.max_num_rules=10 agent.success_critique_num=8 testing=false resume=false
python train.py benchmark=hotpotqa run_name=my_experiment agent.llm=gpt-4 agent.max_num_rules=10 agent.success_critique_num=8 testing=false resume=false
```

### Individual Phase Training (Advanced)
For fine-grained control, you can run phases separately:

**Training Phase 1 Only:**
```bash
python train.py benchmark=<benchmark-name> run_name=<run-name> run_phase_1=true run_phase_2=false testing=false resume=false

# Example:
python train.py benchmark=alfworld run_name=my_train_run run_phase_1=true run_phase_2=false testing=false resume=false
```

**Training Phase 2 Only:**
```bash
python train.py benchmark=<benchmark-name> run_name=<run-name> run_phase_1=false run_phase_2=true agent.llm=gpt-4 agent.max_num_rules=<num> agent.success_critique_num=<num> testing=false resume=false

# Example:
python train.py benchmark=alfworld run_name=my_train_run run_phase_1=false run_phase_2=true agent.llm=gpt-4 agent.max_num_rules=10 agent.success_critique_num=8 testing=false resume=false
```

### Legacy Commands (Deprecated)
The original separate scripts are still available but deprecated:

**Training Phase 1: Experience Gathering**
```bash
python train.py benchmark=<benchmark-name> run_name=<train-run-name> testing=false resume=false
```

**Training Phase 2: Insights Extraction**
```bash
python insight_extraction.py benchmark=<benchmark-name> load_run_name=<train-run-name> run_name=<insights-run-name> agent.llm=gpt-4 agent.max_num_rules=<num> agent.success_critique_num=<num> testing=false resume=false
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

**Modular Prompt Construction & Conversation Management System (`constructor/`)**:
- `BaseConstructor` (base.py): Abstract base class for prompt construction
- `ExpelConstructor` (expel_constructor.py): Complete prompt building and conversation management for ExpeL framework
- **Core Prompt Building**: `build_system_prompt()`, `build_fewshot_prompt()`, `build_rules_prompt()`, `build_task_prompt()`
- **Advanced Construction**: `build_complete_prompt()`, `build_complete_prompt_with_insertions()`, `update_prompt_with_fewshots()`
- **Conversation Management**: `initialize_conversation()`, `add_conversation_turn()`, `reset_conversation()`, `handle_observation()`
- **Dynamic Updates**: `update_dynamic_components()`, `update_fewshots_dynamically()`, `insert_rules_or_insights()`
- **LLM Integration**: `prompt_agent_for_llm()`, `handle_agent_step()`, `prepare_llm_input()`
- **Utility Methods**: `collapse_prompts()`, `remove_task_suffix()`, `get_conversation_statistics()`
- Handles complete pipeline from initial prompt construction to ongoing conversation management and LLM interaction

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

### Unified Training System
ExpeL now features a unified two-phase training system that simplifies the experiential learning workflow:

**Key Features**:
- **Single Command Execution**: Both phases run sequentially with automatic data handoff
- **Phase Control**: Individual phases can be enabled/disabled via `run_phase_1` and `run_phase_2` parameters
- **Automatic Configuration**: Phase 2 configuration is automatically prepared from Phase 1 settings
- **Seamless Data Transfer**: Experience data from Phase 1 is automatically loaded for Phase 2
- **Progress Tracking**: Clear visual indicators for phase transitions and completion status
- **Resume Support**: Can resume from either phase independently
- **100% Backward Compatibility**: Original separate scripts remain available

**Unified Training Command**:
```bash
python train.py benchmark=alfworld run_name=my_experiment agent.llm=gpt-4 agent.max_num_rules=10 agent.success_critique_num=8
```

**Phase Control Examples**:
```bash
# Run both phases (default from config)
python train.py benchmark=alfworld run_name=test

# Run only Phase 1
python train.py benchmark=alfworld run_name=test run_phase_1=true run_phase_2=false

# Run only Phase 2 (requires existing Phase 1 data)
python train.py benchmark=alfworld run_name=test run_phase_1=false run_phase_2=true agent.llm=gpt-4 agent.max_num_rules=10

# Use the simplified shell script (recommended)
./run_expel_alfworld.sh my_experiment gpt-4
```

### Key Data Flow
1. **Training Phase 1 (Experience Gathering)**:
   - Agent interacts with environment using ExpelDataset for task loading
   - ExpelStorage saves experiences via `save_experience()` method
   - Stores successful/failed trajectories in `succeeded_trial_history`/`failed_trial_history`

2. **Training Phase 2 (Insight Extraction)**:
   - **Unified Mode**: Automatically loads Phase 1 data using the same `run_name`
   - **Separate Mode**: ExpelStorage loads experiences via `load_experience()` method
   - ExpelManager analyzes collected experiences to extract actionable insights and rules
   - Uses success/failure trajectory comparison for critique generation
   - Applies rule parsing and updating logic with weighted prioritization
   - ExpelStorage saves insights via `save_insights()` method

3. **Evaluation**:
   - ExpelStorage loads insights via `load_insights()` method
   - ExpelRetrieval system performs FAISS-based similarity search for relevant experiences
   - ExpelConstructor builds complete prompts integrating insights, dynamic few-shots, and current context
   - Uses extracted insights and retrieved experiences for improved performance
   - ExpelStorage saves results via `save_evaluation_results()` method

### Configuration System
Uses Hydra for configuration management:
- `configs/train.yaml`: Unified training configuration (supports both phases)
- `configs/insight_extraction.yaml`: Legacy Training Phase 2 configuration (deprecated)
- `configs/eval.yaml`: Evaluation phase configuration
- `configs/benchmark/`: Environment-specific settings including data range configuration
  - `data_split.train_range`: Training data indices (e.g., [0, 10])
  - `data_split.eval_range`: Evaluation data indices (e.g., [10, 13])

**Unified Training Configuration**:
The new unified training system uses the `train.yaml` configuration and automatically adds Phase 2 parameters when needed:
- `run_phase_1`: Enable/disable Training Phase 1 (default: true)
- `run_phase_2`: Enable/disable Training Phase 2 (default: true)
- Phase 2 parameters (agent.llm, agent.max_num_rules, etc.) are automatically applied when Phase 2 runs

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

**ExpelConstructor handles prompt construction and conversation management**:
- Complete prompt building pipeline from static templates to final LLM input
- Full conversation state management with base prompt and dynamic turn tracking
- Supports all three phases: training experience gathering, insights extraction, and evaluation
- Dynamic integration of rules/insights during evaluation phase with conversation preservation
- Advanced LLM interaction workflow including error handling and retry mechanisms
- Unified few-shot replacement, message collapsing, and conversation state management
- Direct LLM integration with testing mode support and long context fallback

**ExpelManager orchestrates insights extraction process**:
- Core logic for Training Phase 2 (Insights Extraction) operations
- Critique generation from success/failure trajectory comparisons
- Rule parsing and updating with ADD/EDIT/REMOVE/AGREE operations
- Multi-type critique analysis: task comparison, success-only, failure-only
- Rule prioritization system with weighted scoring and automatic pruning
- Complete rule extraction workflow from raw experiences to formatted insights
- Integrated with LLM for automated critique generation and rule refinement

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

### Storage, Retrieval, and Constructor Configuration
- All modular components (ExpelDataset, ExpelStorage, ExpelRetrieval, ExpelConstructor) are automatically initialized
- Storage paths are automatically managed based on benchmark and agent type
- Retrieval system is seamlessly integrated into the agent's decision-making process
- Constructor system handles all prompt building with automatic benchmark-specific template selection

## Modular Architecture Design

### Four-Pillar Modular System
The ExpeL framework now features a comprehensive four-pillar modular architecture:

**1. Dataset Management (`ExpelDataset`)**:
- Handles task loading and data range configuration
- Unified interface across all benchmark environments
- Configurable training/evaluation splits

**2. Data Persistence (`ExpelStorage`)**:
- Complete three-phase data transfer chain management
- Experience → Insights → Evaluation data flow
- Automatic path management and checkpoint support

**3. Experience Retrieval (`ExpelRetrieval`)**:
- FAISS-based vector similarity search
- Dynamic few-shot selection based on current context
- Multiple retrieval strategies for different scenarios

**4. Prompt Construction & Conversation Management (`ExpelConstructor`)**:
- Complete prompt building and conversation management pipeline
- Integration of static templates, dynamic content, and contextual information
- Support for all learning phases with automatic rule/insight injection
- Full conversation state management with LLM interaction workflow
- Real-time dynamic updates while preserving conversation continuity

### Prompt Construction & Conversation Management Pipeline
The ExpelConstructor manages the complete prompt construction and conversation management process:

**Core Building Blocks**:
- System message construction with agent identity and instructions
- Few-shot example integration with instruction templates
- Rules/insights injection during evaluation phase
- Task description formatting with benchmark-specific cleaning
- Conversation state initialization and turn-by-turn management

**Advanced Features**:
- Dynamic few-shot replacement using retrieval system output
- Message collapsing for optimal prompt structure
- Incremental prompt building for conversation contexts
- Flexible content insertion for complex prompt requirements
- Full conversation history tracking with base prompt separation
- Direct LLM interaction with error handling and retry mechanisms

**Conversation Management**:
- **Initialization**: `initialize_conversation()` sets up base prompt structure
- **Turn Management**: `add_conversation_turn()` maintains conversation flow
- **Dynamic Updates**: Real-time few-shot and rules insertion while preserving conversation
- **LLM Integration**: `prompt_agent_for_llm()` handles complete interaction workflow
- **State Tracking**: Separation of base prompt and dynamic conversation turns

**Dependency Management**:
```
Static Config → Constructor Initialization → Conversation Setup
↓
Training Phase 1: Constructor + Conversation Management → Experience Data
↓
Training Phase 2: Constructor + Experience Data + Conversation → Insights/Rules
↓
Evaluation Phase: Constructor + Insights + Dynamic Retrieval + Conversation → Final Performance
```

### Integration Flow
All four modular components work together seamlessly:

1. **ExpelDataset** provides task data with configured ranges
2. **ExpelStorage** manages data persistence across all three phases
3. **ExpelRetrieval** dynamically selects relevant few-shot examples
4. **ExpelConstructor** manages complete prompt construction and conversation flow integrating all components

### Agent Simplification Through Constructor Integration

The modular design significantly simplifies agent implementation:

**Before Modularization**:
```python
# Agent classes contained complex prompt management logic
def step(self):
    # Complex prompt history management
    self.prompt_history.append(message)
    self.prompt_history = self.collapse_prompts(self.prompt_history)
    # Manual LLM calling with error handling
    try:
        response = self.llm(prompt_history, stop=['\n', '\n\n'])
    except openai.BadRequestError:
        # Complex error handling...
```

**After Modularization**:
```python
# Agent classes use clean constructor interface
def step(self):
    # Single constructor call handles everything
    message, message_type, others = self.constructor.handle_agent_step(
        llm_parser=self.llm_parser,
        llm_callable=self.llm,
        # ... other parameters
    )
    # Constructor manages all conversation state automatically
```

**Key Benefits**:
- **Zero Code Duplication**: All prompt logic centralized in ExpelConstructor
- **Automatic State Management**: Conversation history maintained automatically
- **Error Handling**: LLM retry logic and long context fallback built-in
- **Dynamic Updates**: Few-shot replacement and rules insertion seamlessly integrated
- **Testing Support**: Built-in testing mode with prompt debugging capabilities

This modular design maintains complete backward compatibility while providing clean separation of concerns and enhanced maintainability.