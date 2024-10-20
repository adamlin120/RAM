# This file is a refactor of https://github.com/openai/simple-evals/blob/main/math_eval.py
# python3 math_eval.py --policy-base-url "https://openrouter.ai/api/v1" --policy-model "meta-llama/llama-3.1-8b-instruct" --policy-api-key "$OPENROUTER_API_KEY"
# python3 math_eval.py --policy-base-url "https://openrouter.ai/api/v1" --policy-model "meta-llama/llama-3.1-405b-instruct" --policy-api-key "$OPENROUTER_API_KEY"
"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import os
import argparse
from openai import OpenAI

import random
from typing import Literal, Callable, List, Dict

import pandas as pd
from tqdm import tqdm
import logging
import jinja2
from grading.grader import grade_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

# prompt from https://huggingface.co/datasets/meta-llama/Llama-3.1-405B-Instruct-evals
LLAMA3_QUERY_TEMPLATE = jinja2.Template(
    """
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

Problem: {{ Question }}
""".strip()
)


ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"  # for openai query prompt


def extract_result_from_boxed(answer: str) -> str:
    box_start = "\\boxed"
    # format is `\\boxed <value>$` or `\\boxed{<value>}`, with potential white spaces framing `<value>`
    start = answer.rfind(box_start)
    if start < 0:
        return ""
    answer = answer[start + len(box_start) :].strip()
    ends_with_curly = answer.startswith("{")
    i = 0
    open_braces = 0
    while i < len(answer):
        if answer[i] == "{":
            open_braces += 1
        elif answer[i] == "}":
            open_braces -= 1
        if open_braces == 0:
            if ends_with_curly:
                answer = answer[: i + 1].strip()
                break
            elif answer[i] == "$":
                answer = answer[:i].strip()
                break
        i += 1
    else:
        return ""
    # remove extra curly braces
    while True:
        if answer.startswith("{") and answer.endswith("}"):
            answer = answer[1:-1].strip()
        else:
            break
    return answer


EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


def check_equality(
    equality_checker: Callable[[List[Dict[str, str]]], str], expr1: str, expr2: str
) -> bool:
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    logger.debug(f"Equality check prompt: {prompt}")
    response = equality_checker([dict(content=prompt, role="user")])
    logger.debug(f"Equality check response: {response}")
    return response.lower().strip() == "yes"


class MathEval:
    def __init__(
        self,
        equality_checker: Callable[[List[Dict[str, str]]], str] | None,
        num_examples: int | None = None,
        n_repeats: int = 16,
        split: Literal["math_test", "math_500_test"] = "math_500_test",
        use_openai_dataset: bool = False,
    ):
        logger.info(f"Loading {split} dataset...")
        if use_openai_dataset:
            df = pd.read_csv(
                f"https://openaipublic.blob.core.windows.net/simple-evals/{split}.csv"
            )
            """
            Question	Answer
            Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$	"We have that $r = \sqrt{0^2 + 3^2} = 3.$  Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\frac{\pi}{2}$ with the positive $x$-axis.

            [asy]
            unitsize(0.8 cm);

            draw((-0.5,0)--(3.5,0));
            draw((0,-0.5)--(0,3.5));
            draw(arc((0,0),3,0,90),red,Arrow(6));

            dot((0,3), red);
            label("$(0,3)$", (0,3), W);
            dot((3,0), red);
            [/asy]

            Therefore, the polar coordinates are $\boxed{\left( 3, \frac{\pi}{2} \right)}.$"
            """
            examples = [row.to_dict() for _, row in df.iterrows()]
        else:
            if split != "math_500_test":
                raise ValueError(
                    f"Unsupported split: {split}, use_openai_dataset=False"
                )
            df = pd.read_json(
                "https://github.com/openai/prm800k/raw/refs/heads/main/prm800k/math_splits/test.jsonl",
                lines=True,
            )
            # single row is: {"problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$", "solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3.$  Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n\n[asy]\nunitsize(0.8 cm);\n\ndraw((-0.5,0)--(3.5,0));\ndraw((0,-0.5)--(0,3.5));\ndraw(arc((0,0),3,0,90),red,Arrow(6));\n\ndot((0,3), red);\nlabel(\"$(0,3)$\", (0,3), W);\ndot((3,0), red);\n[/asy]\n\nTherefore, the polar coordinates are $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.$", "answer": "\\left( 3, \\frac{\\pi}{2} \\right)", "subject": "Precalculus", "level": 2, "unique_id": "test/precalculus/807.json"}
            # rename to match openai format, and keep the rest
            examples = [
                {
                    "Question": row["problem"],
                    "Answer": row["answer"],
                    "solution": row["solution"],
                    "subject": row["subject"],
                    "level": row["level"],
                    "unique_id": row["unique_id"],
                }
                for _, row in df.iterrows()
            ]

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)

        examples = examples * n_repeats
        examples_df = pd.DataFrame(examples)

        self.examples = examples_df
        self.equality_checker = equality_checker
        logger.info(f"Loaded {len(self.examples)} examples.")

    def __call__(self, sampler: Callable[[List[Dict[str, str]]], str]) -> List[float]:
        def fn(row: dict) -> tuple[float, str, str, str]:
            query_template = (
                LLAMA3_QUERY_TEMPLATE  # TODO: this is a hack to get llama3 working
            )
            # query_template.format(**row)
            content = query_template.render(**row)
            # print(content)
            prompt_messages = [dict(content=content, role="user")]
            logger.debug(f"Problem prompt: {prompt_messages}")
            response_text = sampler(prompt_messages)
            logger.debug(f"Problem response: {response_text}")
            # match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = extract_result_from_boxed(
                response_text
            )  # TODO: this is a hack to get llama3 working
            logger.debug(f"Extracted answer: {extracted_answer}")
            if self.equality_checker:
                score = float(
                    check_equality(
                        self.equality_checker, row["Answer"], extracted_answer
                    )
                )
            else:
                score = float(grade_answer(extracted_answer, row["Answer"]))
            # print(f"Response text: {response_text[-100:]}")
            # print(f"Extracted answer: {extracted_answer}")
            # print(f"Correct answer: {row['Answer']}")
            # print(f"Score: {score}")
            # print(f"Correct: {'CORRECT' if score == 1.0 else 'INCORRECT'}")
            logger.debug(f"Score: {score}")
            return score, content, response_text, extracted_answer

        logger.info("Evaluating examples...")
        scores = []
        total_examples = len(self.examples)
        with tqdm(
            self.examples.iterrows(), desc="Processing examples", total=total_examples
        ) as pbar:
            for i, (df_index, row) in enumerate(pbar, 1):
                score, content, response_text, extracted_answer = fn(row)
                scores.append(score)
                avg_score = sum(scores) / len(scores)
                pbar.set_postfix(
                    {"Average": f"{avg_score*100:.2f}% = {i}/{total_examples}"}
                )
                # write back to dataframe
                self.examples.at[df_index, "score"] = score
                self.examples.at[df_index, "content"] = content
                self.examples.at[df_index, "response_text"] = response_text
                self.examples.at[df_index, "extracted_answer"] = extracted_answer
                # print(self.examples.head())
        logger.info(
            f"Evaluation complete. Average score: {sum(scores)/len(scores):.2f}"
        )
        return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate MATH dataset")
    parser.add_argument(
        "--policy-model", default="gpt-4o-mini", help="Model for policy decisions"
    )
    parser.add_argument(
        "--checker-model", default="gpt-4o-mini", help="Model for equality checking"
    )
    parser.add_argument(
        "--policy-base-url",
        default=None,
        help="Base URL for policy model API (optional)",
    )
    parser.add_argument(
        "--checker-base-url",
        default=None,
        help="Base URL for checker model API (optional)",
    )
    parser.add_argument(
        "--policy-api-key",
        help="API key for policy model",
        default=os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--checker-api-key",
        help="API key for checker model",
        default=os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument("--split", default="math_500_test", help="Dataset split to use")
    parser.add_argument(
        "--n-repeats", type=int, default=1, help="Number of times to repeat the dataset"
    )
    parser.add_argument(
        "--use-openai-checker",
        action="store_true",
        help="Use OpenAI API for equality checking instead of grade_answer function",
    )
    args = parser.parse_args()

    policy_client_kwargs = {"api_key": args.policy_api_key}
    if args.policy_base_url:
        policy_client_kwargs["base_url"] = args.policy_base_url
    policy_client = OpenAI(**policy_client_kwargs)

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

    math_eval = MathEval(
        equality_checker=equality_checker,
        split=args.split,
        n_repeats=args.n_repeats,
    )

    scores = math_eval(
        lambda x: policy_client.chat.completions.create(
            model=args.policy_model, messages=x
        )
        .choices[0]
        .message.content
    )

    logger.info(f"Individual scores: {scores}")


if __name__ == "__main__":
    main()
