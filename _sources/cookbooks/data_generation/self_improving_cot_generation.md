# Deep Dive into CAMEL’s Practices for Self-Improving CoT Generation 🚀

The field of AI is rapidly evolving, with reasoning models playing a crucial role in enhancing the problem-solving capabilities of large language models (LLMs). Recent developments, such as DeepSeek's R1 and OpenAI's o3-mini, demonstrate the industry's commitment to advancing reasoning through innovative approaches.

DeepSeek's R1 model, introduced in January 2025, has shown remarkable proficiency in tasks that require complex reasoning and code generation. Its exceptional performance in areas like mathematics, science, and programming is particularly noteworthy.

By distilling Chain-of-Thought (CoT) data from reasoning models, we can generate high-quality reasoning traces that are more accurate in solving complex problems. These generated data can be used to further fine-tune another LLM with less parameters, thereby enhancing its reasoning ability.

CAMEL developed an approach leverages iterative refinement, self-assessment, and efficient batch processing to enable the continuous improvement of reasoning traces. In this blog, we will delve into how CAMEL implements its self-improving CoT pipeline.

---

## 1. Overview of the End-to-End Pipeline 🔍

### 1.1 Why an Iterative CoT Pipeline? 

One-time CoT generation often leads to incomplete or suboptimal solutions. CAMEL addresses this challenge by employing a multi-step, iterative approach:

1. **Generate** an initial reasoning trace.
2. **Evaluate** the trace through either a dedicated evaluation agent or a specialized reward model.
3. **Refine** the trace based on the feedback provided.

This self-improving methodology ensures that the reasoning process improves progressively, meeting specific thresholds for correctness, clarity, and completeness. Each iteration enhances the model's ability to solve the problem by learning from the previous outputs and evaluations.

### 1.2 Core Components 

The self-improving pipeline consists of three key components:
1. **`reason_agent`:** This agent is responsible for generating or improving reasoning traces.
2. **`evaluate_agent`:** An optional agent that evaluates the quality of the reasoning trace. This can be replaced by a reward model if needed.
3. **`reward_model`:** An optional model that provides numerical feedback on the trace, evaluating dimensions such as correctness, coherence, complexity, and verbosity.

Here's a high-level diagram of the pipeline:

![Self-Improving CoT Pipeline](https://i.postimg.cc/DygTcWd6/download.png)

---

## 2. Generation of CoT Data: The Heart of the Pipeline 🤖

Generating CoT data is at the core of the pipeline. Below, we outline the process in detail.

### 2.1 Initial Trace Generation 🐣

The first step in the process is the generation of an initial reasoning trace. The **`reason_agent`** plays a central role here, creating a coherent and logical explanation of how to solve a given problem. The agent breaks down the problem into smaller steps, illustrating the thought process at each stage. We also support the use of non-reasoning LLMs to generate traces through prompt engineering.

The generation could also guided by **few-shot examples**, which provide context and help the agent understand the desired reasoning style. Here’s how this is accomplished:

- **Input**: The problem statement is provided to the **`reason_agent`**, we can optionally provide the ground truth to guide the reasoning process.
- **Output**: The agent generates a sequence of reasoning content.

This initial generation serves as a foundational reasoning process that can be directly useful or further refined.

### 2.2 Evaluation of the Initial Trace 📒

Once the reasoning trace is generated, it is evaluated for its quality. This evaluation serves two purposes:

- **Detecting weaknesses**: The evaluation identifies areas where the reasoning trace could be further improved.
- **Providing feedback**: The evaluation produces feedback that guides the agent in refining the reasoning trace. This feedback can come from either the **`evaluate_agent`** or a **`reward_model`**.

#### 2.2.1 Agent-Based Evaluation 

If an **`evaluate_agent`** is available, it examines the reasoning trace for:
1. **Correctness**: Does the trace logically solve the problem?
2. **Clarity**: Is the reasoning easy to follow and well-structured?
3. **Completeness**: Are all necessary steps included in the reasoning?

The feedback from the agent provides insights into areas for improvement, such as unclear reasoning or incorrect answers, offering a more generalized approach compared to rule-based matching.

#### 2.2.2 Reward Model Evaluation 

Alternatively, the pipeline supports using a **reward model** to evaluate the trace. The reward model outputs scores based on predefined dimensions such as correctness, coherence, complexity, and verbosity.

---

### 2.3 Iterative Refinement: The Self-Improving Cycle 🔁

The key to CAMEL's success in CoT generation is its **self-improving loop**. After the initial trace is generated and evaluated, the model refines the trace based on the evaluation feedback. This process is repeated in a loop.

#### How does this iterative refinement work?

1. **Feedback Integration**: The feedback from the evaluation phase is used to refine the reasoning. This could involve rewording unclear parts, adding missing steps, or adjusting the logic to make it more correct or complete.
   
2. **Improvement through Reasoning**: After receiving feedback, the **`reason_agent`** is used again to generate an improved version of the reasoning trace. This trace incorporates the feedback provided, refining the earlier steps and enhancing the overall reasoning.

3. **Re-evaluation**: Once the trace is improved, the new version is evaluated again using the same process (either agent-based evaluation or reward model). This new trace is assessed against the same criteria to ensure the improvements have been made.

4. **Threshold Check**: The iterative process continues until the desired quality thresholds are met or reached the maximum number of iterations.

---

## 3. Pipeline Setup in Code 💻

Below is a truncated version of our pipeline initialization. We encapsulate logic in a class called `SelfImprovingCoTPipeline`:

```python
class SelfImprovingCoTPipeline:
    def __init__(
        self,
        reason_agent: ChatAgent,
        problems: List[Dict],
        max_iterations: int = 3,
        score_threshold: Union[float, Dict[str, float]] = 0.7,
        evaluate_agent: Optional[ChatAgent] = None,
        reward_model: Optional[BaseRewardModel] = None,
        output_path: Optional[str] = None,
        few_shot_examples: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        solution_pattern: str = r'\\boxed{(.*?)}',
        trace_pattern: Optional[str] = None,
    ):
        r"""Initialize the STaR pipeline.

        Args:
            reason_agent (ChatAgent): The chat agent used for generating and
                improving reasoning traces.
            problems (List[Dict]): List of problem dictionaries to process.
            max_iterations (int, optional): Maximum number of improvement
                iterations. If set to `0`, the pipeline will generate an
                initial trace without any improvement iterations.
                (default: :obj:`3`)
            score_threshold (Union[float, Dict[str, float]], optional):
                Quality threshold. Can be either a single float value applied
                to average score, or a dictionary mapping score dimensions to
                their thresholds. For example: {"correctness": 0.8,
                "coherence": 0.7}. If using reward model and threshold for a
                dimension is not specified, will use the default value 0.7.
                (default: :obj:`0.7`)
            evaluate_agent (Optional[ChatAgent]): The chat agent used for
                evaluating reasoning traces. (default: :obj:`None`)
            reward_model (BaseRewardModel, optional): Model used to evaluate
                reasoning traces. If `None`, uses Agent self-evaluation.
                (default: :obj:`None`)
            output_path (str, optional): Output path for saving traces. If
                `None`, results will only be returned without saving to file.
                (default: :obj:`None`)
            few_shot_examples (str, optional): Examples to use for few-shot
                generation. (default: :obj:`None`)
            batch_size (int, optional): Batch size for parallel processing.
                (default: :obj:`None`)
            max_workers (int, optional): Maximum number of worker threads.
                (default: :obj:`None`)
            solution_pattern (str, optional): Regular expression pattern with
                one capture group to extract answers from solution text.
                (default: :obj:`r'\\boxed{(.*?)}'`)
            trace_pattern (str, optional): Regular expression pattern with one
                capture group to extract answers from trace text. If `None`,
                uses the same pattern as solution_pattern.
                (default: :obj:`None`)
        """
        ...
```

**Example usage:**

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

---

## 4. Batch Processing & API Request Handling 📦

### 4.1 The Need for Batch Processing ⏰

Early on, we tried generating CoT reasoning for each problem one by one. This approach quickly revealed two major issues:

1. **Time consumption**: Sequential processing doesn't scale to large problem sets.
2. **API request bottlenecks**: Slowdowns or occasional disconnections occurred when handling numerous calls.

Hence, we introduced a parallel **`BatchProcessor`** to:

- Split the tasks into manageable batches.
- Dynamically adjust batch size (`batch_size`) based on the success/failure rates and system resource usage (CPU/memory).
- Retry on transient errors or API timeouts to maintain a stable flow.

Below shows how we batch-process multiple problems:

```python
async def _batch_process_problems(
    self, problems: List[Dict], rationalization: bool
) -> List[ProblemResult]:
    results = []
    total_problems = len(problems)
    processed = 0

    while processed < total_problems:
        batch_size = self.batch_processor.batch_size
        batch = problems[processed : processed + batch_size]
        batch_start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.batch_processor.max_workers) as executor:
            futures = [
                executor.submit(
                    self.process_problem,
                    problem=problem,
                    rationalization=rationalization,
                )
                for problem in batch
            ]
            ...
        processed += len(batch)
        ...
        # Log progress & performance
```

### 4.2 Handling API Instability 🚨

Even with batching, API requests for LLMs can fail due to network fluctuations or remote server instability. We implemented a `retry_on_error` decorator:

```python
def retry_on_error(
    max_retries: int = 3, initial_delay: float = 1.0
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    time.sleep(delay)
                    delay *= 2
            raise
        return wrapper
    return decorator
```

Whenever we invoke LLM calls for generation, evaluation, or improvement, these decorated methods gracefully handle transient errors by retrying with exponential backoff (doubling the wait time after each failed attempt).

---

## 5. Model Switching & Dynamic File Writing 📝

### 5.1 Flexible Model Scheduling 🕒

In CAMEL's CoT pipeline, adding models to the `ChatAgent` is useful for handling errors and ensuring smooth operation. This setup allows the system to switch between models as needed, maintaining reasoning continuity.

To add models to a `ChatAgent`, you can create instances of models and include them in the agent's model list:

```python
model1 = ModelFactory.create(
    model_platform=ModelPlatformType.DEEPSEEK,
    model_type="deepseek-reasoner",
    ...
)

model2 = ModelFactory.create(
    model_platform=ModelPlatformType.TOGETHER,
    model_type="deepseek-reasoner",
    ...
)

agent = ChatAgent(
    system_message,
    model=[model1, model2]
)
```

By incorporating multiple models, CAMEL can effectively manage model availability and ensure robust error handling.

### 5.2 Real-Time JSON Updates 🔄

As soon as a problem’s results are ready, we lock the file (`output_path`) and update it in-place—rather than saving everything at the very end. This ensures data integrity if the process is interrupted partway through.

```python
def safe_write_json(self, file_path, data):
    temp_path = file_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, file_path)
```

This two-step write (to a `.tmp` file then replace) prevents partial writes from corrupting the output file.

---

## 6. CAMEL’s Next Steps in CoT Data Generation 🚀

1. **Real-Time Monitoring Dashboard**: Visualize throughput, error rates, running cost, data quality, etc. for smooth operational oversight.
2. **Performance Enhancements**: Further improve performance and add more error handling to make the system more robust.
3. **Cutting-Edge Research Solutions**: Integrate more cutting-edge research solutions for synthetic data generation.
4. **Rejection Sampling**: Integrate rejection sampling method to the SelfImprovingCoT pipeline.

---

## Conclusion 📚

CAMEL’s self-improving pipeline exemplifies a comprehensive approach to Chain-of-Thought data generation:

- **Flexible Evaluation**: Utilizing agent-based or reward-model-based evaluation provides adaptable scoring and feedback loops.
- **Continuous Improvement**: Iterative refinement ensures each reasoning trace is enhanced until it meets the desired quality.
- **Efficient Processing**: Batched concurrency increases throughput while maintaining system balance.
- **Robust Stability**: Error-tolerant mechanisms with retries enhance system reliability.
- **Consistent Output**: Dynamic file writing ensures partial results are consistently preserved and valid.

Looking ahead, CAMEL’s roadmap is dedicated to pioneering advanced synthetic data generation methods, integrating cutting-edge research and technology.

_Stay tuned for more updates on CAMEL's journey in advancing agentic synthetic data generation!_

---

**Further Reading & Resources**

- **CAMEL GitHub**: Explore our open-source projects on [GitHub](https://github.com/camel-ai/camel) and give us a 🌟star.

**Data Generation Cookbooks**

- [Self-Improving Math Reasoning Data Distillation](https://docs.camel-ai.org/cookbooks/data_generation/self_improving_math_reasoning_data_distillation_from_deepSeek_r1.html)
- [Generating High-Quality SFT Data with CAMEL](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)
- [Function Call Data Generation and Evaluation](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)
- [Agentic Data Generation, Evaluation & Filtering with Reward Models](https://docs.camel-ai.org/cookbooks/data_generation/synthetic_dataevaluation%26filter_with_reward_model.html)