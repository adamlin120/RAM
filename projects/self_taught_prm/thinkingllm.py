import textwrap
import re
from jinja2 import Template
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from datasets import Dataset, load_dataset
from typing import Dict, Any, Union
from distilabel.steps.tasks import Task
from distilabel.llms.vllm import vLLM
from distilabel.steps import RewardModelScore

from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import StepColumns

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

llm = vLLM(
    model=MODEL_ID,
    generation_kwargs={
        "temperature": 1.0,
        "max_new_tokens": 2048,
        "min_p": 0.1,
    },
)

template = """
Respond to the following user query in a comprehensive and detailed way. 

But first write down your internal thoughts. 

This must include your draft response and its evaluation. After this,

write your final response after “<R>”.

User query: {{instruction}}
"""

MATH_STEP_BY_STEP_COT_CHAT_PROMPT_TEMPLATE: Dict[str, str] = {
    "prompt": textwrap.dedent(
        """\
        Solve the following math problem efficiently and clearly:

        - For simple problems (2 steps or fewer):
        Provide a concise solution with minimal explanation.

        - For complex problems (3 steps or more):
        Use this step-by-step format:

        ## Step 1: [Concise description]
        [Brief explanation and calculations]

        ## Step 2: [Concise description]
        [Brief explanation and calculations]

        ...

        Regardless of the approach, always conclude with:

        Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

        Where [answer] is just the final number or expression that solves the problem.

        Problem: {{ instruction }}
    """
    ),
    "answer": "{{ solution }}. I hope it is correct.",
}

VERIFICATION_PROMPT = """
Let's verify step by step and provide a final verification. You are a math teacher. Grade the Solution, verifying correctness step by step. Use the Expected Answer to find any erroneous step in the Solution, but do not reveal or mention the Expected Answer in your verification process. At the end of the Solution verification, when you give your final grade, write it in the form "Verification: Is the answer correct (Yes/No)? X", where X is either Yes or No.

Question: {{instruction}}
Solution: {{solution}}
Expected Answer: {{expected_answer}}
"""

class ThinkingLLMGeneration(Task):
    """Text generation with thought optimization.

    This task generates a response for a given instruction, including the internal thoughts
    and a final response. The LLM will first write down its internal thoughts, including a
    draft response and its evaluation, before providing the final response.

    Input columns:
        - instruction (str): The instruction to generate the response for.

    Output columns:
        - thought (str): The internal thoughts, including draft response and evaluation.
        - output (str): The final response.
        - correct_format (bool): Whether the output is in the correct format.
    Categories:
        - text-generation
    """

    @property
    def inputs(self) -> StepColumns:
        return {"instruction": True}

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        template = Template(MATH_STEP_BY_STEP_COT_CHAT_PROMPT_TEMPLATE["prompt"])
        return [
            {"role": "user", "content": template.render(instruction=input["instruction"])}
        ]

    @property
    def outputs(self) -> StepColumns:
        return ["thought", "output", "correct_format", "parsed_output"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {"thought": None, "output": None, "correct_format": False, "parsed_output": None}

        # Split the output into thought and final response
        parts = output.rsplit("Therefore, the final answer is:", 1)

        if len(parts) < 2:
            return {"thought": output.strip(), "output": None, "correct_format": False, "parsed_output": None}

        thought = parts[0].strip()
        final_output = parts[1].strip()

        # Check if the format is correct (both thought and output are present)
        correct_format = bool(thought and final_output)

        # Extract the boxed answer or parse the final output
        boxed_answer = re.search(r'\$\\boxed{(.+?)}\$', final_output)
        if boxed_answer:
            parsed_output = boxed_answer.group(1).strip()
        else:
            parsed_output = final_output.replace("$", "").strip()

        return {"thought": thought, "output": final_output, "correct_format": correct_format, "parsed_output": parsed_output}


dataset = (
    load_dataset(
        "meta-math/MetaMathQA",
        split="train"
    )
    .rename_column("query", "instruction")
    .shuffle(seed=42)
    .select(range(50))
)

with Pipeline(name="thinking-llms") as pipeline:
    thought = ThinkingLLMGeneration(
        llm=llm,
        num_generations=8,
        group_generations=True
    )

if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset, use_cache=False)
    distiset.push_to_hub("yentinglin/ultrafeedback_binarized_thinking_llms-10", include_script=True)
    print(distiset['default']['train'][0]['output'])