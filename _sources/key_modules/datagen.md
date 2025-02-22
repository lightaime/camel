# Data Generation

This document describes CAMEL's key data generation modules that enable high-quality training data creation through advanced reasoning and instruction tuning techniques. The modules include:

- Chain of Thought (CoT): Generates explicit reasoning paths
- Self-Instruct: Creates diverse instruction-following data
- Source2Synth: Converts source code into natural language
- Self-Improving CoT: Iteratively refines reasoning chains through self-critique

## Chain of Thought (CoT) Data Generation

### Overview

The Chain of Thought (CoT) data generation module implements a sophisticated system for generating high-quality reasoning paths through chat agent interactions. It combines several advanced algorithms to produce and validate reasoning chains.

### Key Features

- Monte Carlo Tree Search (MCTS) for solution exploration
- Binary Search Error Detection for precise error localization
- Dual-Agent Verification System for quality assurance
- Solution Tree Management for tracking reasoning paths

### Core Components

#### CoTDataGenerator Class

The main class that implements the CoT generation system with the following capabilities:

- **Dual-Agent Architecture**: Supports both single-agent (legacy) and dual-agent modes
- **Answer Generation**: Sophisticated answer generation with MCTS
- **Answer Verification**: Robust verification system using golden answers
- **Error Detection**: Binary search-based error detection in solutions
- **Solution Management**: Comprehensive solution tree management and export

### Usage

#### Basic Example

```python
from camel.agents import ChatAgent
from camel.datagen import CoTDataGenerator

# Initialize agents
generator_agent = ChatAgent("System message for generator")
verifier_agent = ChatAgent("System message for verifier")

# Define golden answers
golden_answers = {
    "question1": "answer1",
    "question2": "answer2"
}

# Create generator
cot_generator = CoTDataGenerator(
    generator_agent=generator_agent,
    verifier_agent=verifier_agent,
    golden_answers=golden_answers,
    search_limit=100
)

# Generate solution
solution = cot_generator.solve("question1")
```

#### Data Import/Export

```python
# Import QA pairs from JSON
cot_generator.import_qa_from_json("qa_pairs.json")

# Export solutions
cot_generator.export_solutions("solutions.json")
```

### Solution Generation Process

1. **Direct Solution Attempt**
   - First tries to solve the problem directly
   - Verifies against golden answer

2. **MCTS-based Exploration**
   - If direct solution fails, uses MCTS to explore solution space
   - Builds solution tree based on previous attempts

3. **Error Detection and Correction**
   - Uses binary search to locate errors in solutions
   - Generates new solutions based on correct parts

4. **Solution Verification**
   - Verifies solutions using dual-agent system or golden answers
   - Maintains solution quality through strict verification

### Configuration Options

- `search_limit`: Maximum number of search iterations (default: 100)
- `generator_agent`: Specialized agent for answer generation
- `verifier_agent`: Specialized agent for answer verification
- `golden_answers`: Pre-defined correct answers for validation

### Output Format

The solution tree is exported in JSON format containing:
- Solutions with intermediate steps
- Golden answers used for verification
- Export timestamp
- Solution scores and verification results


## Self-Instruct: Instruction Generation

### Overview

The Self-Instruct module implements a pipeline for generating and managing machine-generated instructions for tasks. It combines human-written seed instructions with machine-generated ones to create diverse, high-quality task instructions, while ensuring quality through configurable filtering mechanisms.

### Core Components

#### Self Instruct Pipeline

The main pipeline class that orchestrates the instruction generation process.

Key Features:
- Combines human-written and machine-generated instructions using configurable ratios
- Supports classification and non-classification task types
- Built-in instruction filtering and validation
- Automatic instance generation for tasks
- JSON-based data input/output

#### Instruction Filter

A comprehensive filtering system for validating and filtering generated instructions.

Features:
- Length-based filtering
- Keyword filtering
- Punctuation checks
- Non-English text detection
- ROUGE similarity filtering for deduplication
- Extensible filter registry for custom filters

### Usage

#### Basic Example

```python
from camel.agents import ChatAgent
from camel.datagen.self_instruct import SelfInstructPipeline

# Initialize agent
agent = ChatAgent()

# Create pipeline with default settings
pipeline = SelfInstructPipeline(
    agent=agent,
    seed='seed_tasks.jsonl',  # Path to human-written seed tasks
    num_machine_instructions=5,
    data_output_path='./data_output.json',
    human_to_machine_ratio=(6, 2)  # Use 6 human tasks and 2 machine tasks for generation
)

# Generate instructions
pipeline.generate()
```

#### Custom Filtering

```python
from camel.datagen.self_instruct import SelfInstructPipeline
from camel.datagen.self_instruct.filter import InstructionFilter

# Configure filters
filter_config = {
    "length": {},  # Default length constraints
    "keyword": {},  # Keyword-based filtering
    "punctuation": {},  # Punctuation checks
    "non_english": {},  # Non-English text detection
    "rouge_similarity": {  # ROUGE-based similarity filtering
        "threshold": 0.7,
        "metric": "rouge-l"
    }
}

# Create pipeline with custom filter configuration
pipeline = SelfInstructPipeline(
    agent=agent,
    seed='seed_tasks.jsonl',
    instruction_filter=InstructionFilter(filter_config),
    num_machine_instructions=5
)
```

### Configuration Options

#### Pipeline Parameters

- `agent`: ChatAgent instance for generating instructions
- `seed`: Path to human-written seed tasks in JSONL format
- `num_machine_instructions`: Number of machine-generated instructions to generate (default: 5)
- `data_output_path`: Path for saving generated data (default: './data_output.json')
- `human_to_machine_ratio`: Ratio of human to machine tasks for generation (default: (6, 2))
- `instruction_filter`: Custom InstructionFilter instance (optional)
- `filter_config`: Configuration dictionary for default filters (optional)

#### Filter Configuration

The default filter configuration includes:
- `length`: Configure length constraints for instructions
- `keyword`: Set up keyword-based filtering rules
- `punctuation`: Define punctuation validation rules
- `non_english`: Configure non-English text detection
- `rouge_similarity`: Set ROUGE similarity thresholds for deduplication

### Pipeline Stages

1. **Seed Loading**
   - Load human-written instructions from JSONL file
   - Validate seed format
   - Initialize task storage

2. **Instruction Generation**
   - Sample human and machine tasks based on ratio
   - Generate new instructions using ChatAgent
   - Apply instruction filters

3. **Task Classification**
   - Identify if tasks are classification or non-classification
   - Generate appropriate prompts based on task type

4. **Instance Generation**
   - Generate input-output pairs for each task
   - Parse and format instances based on task type
   - Apply quality filters

5. **Data Output**
   - Save generated tasks and instances to JSON
   - Include metadata and configuration details
   - Maintain structured output format

### Input/Output Format

#### Seed Tasks (Input)
```jsonl
{"instruction": "Classify the sentiment of this text as positive or negative."}
{"instruction": "Generate a summary of the given paragraph."}
```

#### Generated Data (Output)
```json
{
  "machine_instructions": [
    {
      "instruction": "...",
      "is_classification": true,
      "instances": [
        {
          "input": "...",
          "output": "..."
        }
      ]
    }
  ]
}
```


## Source2Synth: Multi-hop Question-Answer Generation

### Overview

Source2Synth is a sophisticated data generation system designed to create multi-hop question-answer pairs from source text data. It implements a pipeline that processes raw text, extracts information pairs, and generates complex, multi-hop reasoning questions with configurable complexity thresholds.

### Core Components

#### UserDataProcessor

The main orchestrator class that manages the entire pipeline from text processing to dataset generation.

Features:
- Single text and batch processing capabilities
- Configurable AI model or rule-based processing
- Integration with MultiHopGeneratorAgent for QA generation
- Random seed control for reproducibility

#### ExampleConstructor

Handles the construction of training examples from raw text data.

Features:
- Text preprocessing and quality validation
- Information pair extraction with premise-intermediate-conclusion relationships
- Multi-hop QA pair generation using AI or rule-based approaches
- Complexity scoring for generated examples

#### DataCurator

Manages and curates datasets of multi-hop question-answer pairs.

Features:
- Quality filtering based on configurable criteria
- Complexity threshold filtering
- Deduplication of similar examples
- Dataset sampling to achieve target size
- Random seed control for reproducible sampling

### Usage

#### Basic Example

```python
from camel.datagen.source2synth import (
    UserDataProcessor,
    ProcessorConfig
)

# Create configuration
config = ProcessorConfig(
    seed=42,
    min_length=50,
    max_length=1000,
    complexity_threshold=0.5,
    dataset_size=10,
    use_ai_model=True,
)

# Initialize processor
processor = UserDataProcessor(config)

# Process a single text
result = processor.process_text(
    "Your source text here",
    source="example_source"
)

# Process multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
sources = ["source1", "source2", "source3"]
batch_results = processor.process_batch(texts, sources)
```

### Configuration Options

#### ProcessorConfig

Key parameters:
- `seed`: Random seed for reproducibility
- `min_length`: Minimum text length for processing
- `max_length`: Maximum text length for processing
- `complexity_threshold`: Minimum complexity score (0.0-1.0)
- `dataset_size`: Target size for the final dataset
- `use_ai_model`: Toggle between AI model and rule-based processing
- `hop_generating_agent`: Custom MultiHopGeneratorAgent instance (optional)

### Pipeline Stages

1. **Text Preprocessing**
   - Length validation
   - Quality checks
   - Text standardization

2. **Information Extraction**
   - Premise identification
   - Intermediate relationship extraction
   - Conclusion formation

3. **QA Generation**
   - Multi-hop question generation
   - Answer validation
   - Complexity scoring

4. **Dataset Curation**
   - Quality filtering
   - Complexity thresholding
   - Deduplication
   - Target size sampling


## Self-Improving CoT Data Generation

### Overview

The Self-Improving CoT Data Generation pipeline implements an iterative approach to generate and improve reasoning traces for problem-solving tasks. This implementation is based on the methodology of self-taught reasoning, where an AI agent learns to improve its reasoning process through self-evaluation and feedback.

### Architecture

The pipeline consists of four main stages:
1. Initial reasoning trace generation
2. Self-evaluation
3. Feedback-based improvement
4. Iterative refinement

### Key Components

#### SelfImprovingCoTPipeline Class

The core class that implements the STaR methodology with the following features:
- Customizable reasoning and evaluation agents
- Support for both agent-based evaluation and external reward models
- Configurable quality thresholds for different evaluation dimensions
- Iterative improvement process with customizable maximum iterations
- Optional few-shot examples for better reasoning generation
- Flexible output formats and file saving options

### Usage

#### Basic Example

```python
from camel.agents import ChatAgent
from camel.datagen import SelfImprovingCoTPipeline

# Initialize agents
reason_agent = ChatAgent(
    """Answer my question and give your 
    final answer within \\boxed{}."""
)

evaluate_agent = ChatAgent(
    "You are a highly critical teacher who evaluates the student's answers "
    "with a meticulous and demanding approach."
)

# Prepare your problems
problems = [
    {"problem": "Your problem text here"},
    # Add more problems...
]

# Create and run the pipeline
pipeline = SelfImprovingCoTPipeline(
    reason_agent=reason_agent,
    evaluate_agent=evaluate_agent,
    problems=problems,
    max_iterations=3,
    output_path="star_output.json"
)

results = pipeline.generate()
```

#### Advanced Usage with External Reward Models

```python
from camel.models.reward import NemotronRewardModel

# Initialize reward model
reward_model = NemotronRewardModel(
    model_type=ModelType.NVIDIA_NEMOTRON_340B_REWARD,
    url="https://integrate.api.nvidia.com/v1",
    api_key="your_api_key"
)

# Create pipeline with reward model
pipeline = SelfImprovingCoTPipeline(
    reason_agent=reason_agent,
    evaluate_agent=evaluate_agent,
    problems=problems,
    reward_model=reward_model,
    score_threshold={
        "correctness": 0.8,
        "clarity": 0.7,
        "completeness": 0.7
    }
)
```

### Input/Output Formats

#### Input Format

The pipeline expects problems in a JSON format:

```json
{
    "problems": [
        {
            "problem": "Problem text here",
            "solution": "Optional solution text"
        }
    ]
}
```

#### Output Format

The pipeline generates output in JSON format containing:
- Original problem
- Final reasoning trace
- Improvement history with iterations
- Evaluation scores and feedback for each iteration

### Configuration Options

- `max_iterations`: Maximum number of improvement iterations (default: 3)
- `score_threshold`: Quality thresholds for evaluation dimensions (default: 0.7)
- `few_shot_examples`: Optional examples for few-shot learning
- `output_path`: Path for saving results (optional)


