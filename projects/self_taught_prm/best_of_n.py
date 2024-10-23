import logging

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

DEBUG = False

good_token = "+"
bad_token = "-"
step_tag = "ки"

from parser import extract_result_from_boxed

from grading.grader import grade_answer
from prompt import LLAMA3_QUERY_TEMPLATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


tokenizer = AutoTokenizer.from_pretrained("peiyi9979/math-shepherd-mistral-7b-prm")
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902
model = AutoModelForCausalLM.from_pretrained(
    "peiyi9979/math-shepherd-mistral-7b-prm",
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()

# Initialize vLLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True,
    gpu_memory_utilization=0.5,
)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096, n=1)

# Load dataset
ds = load_dataset("yentinglin/MATH-prm800k", split="test")
if DEBUG:
    ds = ds.shuffle(seed=42).select(range(100))

# Prepare prompts for batch inference
prompts = [
    [
        {
            "role": "user",
            "content": LLAMA3_QUERY_TEMPLATE.render(Question=example["problem"]),
        }
    ]
    for example in ds
]

# Generate all responses in batch
outputs = llm.chat(prompts, sampling_params)
# Calculate average response length in tokens
total_tokens = 0
for output in outputs:
    for response in output.outputs:
        total_tokens += len(response.token_ids)

avg_tokens = total_tokens / (len(outputs) * len(outputs[0].outputs))
logger.info(f"Average response length: {avg_tokens:.2f} tokens")

# Process results
results = []
correct = 0
total = len(ds)
import torch
from openai import OpenAI
from tqdm import tqdm

# Initialize reward model
from transformers import AutoTokenizer

# model_name = "Qwen/Qwen2.5-Math-RM-72B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


progress_bar = tqdm(enumerate(ds), total=total)
for i, example in progress_bar:
    # Get all n responses for this example
    responses = outputs[i].outputs

    # Score each response with reward model
    scores = []
    valid_responses = []
    conversation_strs = []

    question = example["problem"]

    # First collect valid responses and create conversation strings
    for response in responses:
        # Extract answer and check if valid
        extracted_answer = extract_result_from_boxed(response.text)
        if not extracted_answer:
            continue

        # # Create conversation for reward model
        # chat = [
        #     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        #     {"role": "user", "content": example["problem"]},
        #     {"role": "assistant", "content": response.text}
        # ]

        # # Get conversation string
        # conversation_str = tokenizer.apply_chat_template(
        #     chat,
        #     tokenize=False,
        #     add_generation_prompt=False
        # )

        # output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
        attempt = response.text.strip()

        # Insert step tag before ## Step and at end
        # attempt = attempt.replace("\n\n## Step", f"{step_tag}\n\n## Step")
        # if not attempt.endswith(step_tag):
        attempt = attempt + f" {step_tag}"

        conversation_str = f"{question} {attempt}"

        conversation_strs.append(conversation_str)
        valid_responses.append(response.text)

    if conversation_strs:
        # # Score all conversations in one batch
        # client = OpenAI(
        #     api_key="EMPTY",
        #     base_url="http://cr6-p548xlarge-22:8080/v1"
        # )
        # embeddings = client.embeddings.create(
        #     input=conversation_strs,
        #     model=client.models.list().data[0].id
        # )

        # # Extract scores from embeddings
        # scores = [embedding.embedding[-1] for embedding in embeddings.data]

        for input_for_prm in conversation_strs:
            input_id = torch.tensor(
                [tokenizer.encode(input_for_prm)], device=model.device
            )

            with torch.no_grad():
                logits = model(input_id).logits[:, :, candidate_tokens]
                local_scores = logits.softmax(dim=-1)[:, :, 0]
                step_scores = local_scores[input_id == step_tag_id]
                # take the last step score as the overall score
                orm_score = step_scores[-1].item()
                scores.append(orm_score)

    if not valid_responses:
        continue

    # Select response with highest score
    best_idx = torch.tensor(scores).argmax().item()
    best_response = valid_responses[best_idx]
    best_extracted = extract_result_from_boxed(best_response)

    # Grade the answer
    is_correct = grade_answer(best_extracted, example["answer"])
    if is_correct:
        correct += 1

    # Store result
    result = {
        "problem": example["problem"],
        "response": best_response,
        "target": example["answer"],
        "extracted_answer": best_extracted,
        "is_correct": is_correct,
        "reward_score": scores[best_idx],
    }
    results.append(result)

    # Update progress bar with running accuracy
    current_accuracy = correct / (i + 1)
    progress_bar.set_description(f"Accuracy: {current_accuracy:.2%}")

    if not is_correct:
        logger.info(f"\nIncorrect answer for example {i+1}:")
        logger.info(f"Question: {example['problem']}")
        logger.info(f"Model extracted answer: {best_extracted}")
        logger.info(f"Target answer: {example['answer']}")
        logger.info(f"Reward score: {scores[best_idx]}")
        logger.info("---")

# Save results
import json

with open("prm_results.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

final_accuracy = correct / total
logger.info(
    f"Completed inference and saved results. Final accuracy: {final_accuracy:.2%}"
)
