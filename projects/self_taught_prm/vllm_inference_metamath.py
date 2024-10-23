import logging
from parser import extract_result_from_boxed

from datasets import load_dataset
from grading.grader import grade_answer
from prompt import LLAMA3_QUERY_TEMPLATE
from vllm import LLM, SamplingParams

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize vLLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)

# Load dataset
ds = load_dataset(
    "yentinglin/metamath-qwen2-math-alpaca", "unqiue_instruction", split="train"
)

ds = ds.shuffle(seed=42).select(range(100))

# Prepare prompts for batch inference
prompts = [
    [
        {
            "role": "user",
            "content": LLAMA3_QUERY_TEMPLATE.render(Question=example["instruction"]),
        }
    ]
    for example in ds
]

# Generate all responses in batch
outputs = llm.chat(prompts, sampling_params)
# Process results
results = []
correct = 0
total = len(ds)

from tqdm import tqdm

progress_bar = tqdm(enumerate(ds), total=total)
for i, example in progress_bar:
    response = outputs[i].outputs[0].text

    # Extract answer from model response
    extracted_answer = extract_result_from_boxed(response)

    # Grade the answer
    is_correct = False
    if extracted_answer:
        is_correct = grade_answer(extracted_answer, example["gold_ans"])
        if is_correct:
            correct += 1

    # Store result
    result = {
        "instruction": example["instruction"],
        "response": response,
        "target": example["gold_ans"],
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
    }
    results.append(result)

    # Update progress bar with running accuracy
    current_accuracy = correct / (i + 1)
    progress_bar.set_description(f"Accuracy: {current_accuracy:.2%}")

    if not is_correct:
        logger.info(f"\nIncorrect answer for example {i+1}:")
        logger.info(f"Question: {example['instruction']}")
        logger.info(f"Model response: {response}")
        logger.info(f"Model extracted answer: {extracted_answer}")
        logger.info(f"Target answer: {example['gold_ans']}")
        logger.info("---")

# Save results
import json

with open("metamath_results.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

final_accuracy = correct / total
logger.info(
    f"Completed inference and saved results. Final accuracy: {final_accuracy:.2%}"
)
