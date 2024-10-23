# This file is a refactor of https://github.com/openai/simple-evals/blob/main/math_eval.py
"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import argparse
import logging
import os
from parser import extract_result_from_boxed
from typing import Callable, Dict, List

import pandas as pd
from grading.grader import grade_answer
from openai import OpenAI
from prompt import EQUALITY_TEMPLATE, LLAMA3_QUERY_TEMPLATE
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"  # for openai query prompt


def check_equality(
    equality_checker: Callable[[List[Dict[str, str]]], str], expr1: str, expr2: str
) -> bool:
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    logger.debug(f"Equality check prompt: {prompt}")
    response = equality_checker([dict(content=prompt, role="user")])
    logger.debug(f"Equality check response: {response}")
    return response.lower().strip() == "yes"


def main():
    parser = argparse.ArgumentParser(description="Evaluate MATH dataset")
    parser.add_argument(
        "--policy-model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model for policy decisions",
    )
    parser.add_argument(
        "--checker-model", default="gpt-4o-mini", help="Model for equality checking"
    )
    parser.add_argument(
        "--checker-base-url",
        default=None,
        help="Base URL for checker model API (optional)",
    )
    parser.add_argument(
        "--checker-api-key",
        help="API key for checker model",
        default=os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--use-openai-checker",
        action="store_true",
        help="Use OpenAI API for equality checking instead of grade_answer function",
    )
    args = parser.parse_args()

    # Initialize vLLM for policy model
    policy_llm = LLM(model=args.policy_model)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)

    checker_client_kwargs = {"api_key": args.checker_api_key}
    if args.checker_base_url:
        checker_client_kwargs["base_url"] = args.checker_base_url
    checker_client = OpenAI(**checker_client_kwargs)

    equality_checker = None
    if args.use_openai_checker:

        def equality_checker(messages):
            return (
                checker_client.chat.completions.create(
                    model=args.checker_model,
                    messages=messages,
                    temperature=0.0,
                    top_p=1.0,
                )
                .choices[0]
                .message.content
            )

        logger.info("Using OpenAI API for equality checking")
    else:
        logger.info("Using grade_answer function for equality checking")

    # Load test dataset
    df = pd.read_json(
        "https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/math_splits/test.jsonl",
        lines=True,
    )
    # {"problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$", "solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3.$  Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n\n[asy]\nunitsize(0.8 cm);\n\ndraw((-0.5,0)--(3.5,0));\ndraw((0,-0.5)--(0,3.5));\ndraw(arc((0,0),3,0,90),red,Arrow(6));\n\ndot((0,3), red);\nlabel(\"$(0,3)$\", (0,3), W);\ndot((3,0), red);\n[/asy]\n\nTherefore, the polar coordinates are $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.$", "answer": "\\left( 3, \\frac{\\pi}{2} \\right)", "subject": "Precalculus", "level": 2, "unique_id": "test/precalculus/807.json"}

    # Generate responses
    # response = policy_llm.generate(
    #     # prompts=[OPENAI_QUERY_TEMPLATE.format(Question=row["problem"]) for _, row in df.iterrows()],
    #     prompts=[LLAMA3_QUERY_TEMPLATE.render(Question=row["problem"]) for _, row in df.iterrows()],
    #     sampling_params=sampling_params,
    # )
    response = policy_llm.chat(
        messages=[
            [dict(role="user", content=row["problem"])] for _, row in df.iterrows()
        ],
        sampling_params=sampling_params,
    )
    response_text = [output.outputs[0].text for output in response]

    # Evaluate responses
    scores = []
    total = len(response_text)
    correct = 0

    progress_bar = tqdm(enumerate(zip(response_text, df.itertuples())), total=total)
    for i, (text, row) in progress_bar:
        # Extract answer from model response
        extracted_answer = extract_result_from_boxed(text)
        if not extracted_answer:
            scores.append(0.0)
            continue

        # Compare with ground truth
        correct_answer = row.answer
        if equality_checker:
            # Use equality checker if available
            is_correct = check_equality(
                equality_checker, extracted_answer, correct_answer
            )
        else:
            # Direct string comparison as fallback
            is_correct = grade_answer(extracted_answer, correct_answer)

        score = float(is_correct)
        scores.append(score)
        correct += score

        # Update progress bar with running accuracy
        current_accuracy = correct / (i + 1)
        progress_bar.set_description(f"Accuracy: {current_accuracy:.2%}")

        # Print problem details on incorrect answers
        if not is_correct:
            logger.info(f"\nIncorrect answer for problem {i+1}:")
            logger.info(f"Question: {row.problem}")
            logger.info(f"Model answer: {extracted_answer}")
            logger.info(f"Correct answer: {correct_answer}")
            logger.info("---")
    final_accuracy = correct / total
    logger.info(f"Final accuracy: {final_accuracy:.2%}")

    # Calculate accuracy by subject and difficulty
    df["score"] = scores

    # Subject breakdown
    subject_accuracy = df.groupby("subject")["score"].agg(["mean", "count"]).round(4)
    print("\nAccuracy by subject:")
    print(subject_accuracy)

    # Level breakdown
    level_accuracy = df.groupby("level")["score"].agg(["mean", "count"]).round(4)
    print("\nAccuracy by level:")
    print(level_accuracy)

    # Combined subject-level breakdown
    combined_accuracy = (
        df.groupby(["subject", "level"])["score"].agg(["mean", "count"]).round(4)
    )
    print("\nAccuracy by subject and level:")
    print(combined_accuracy)


if __name__ == "__main__":
    main()
