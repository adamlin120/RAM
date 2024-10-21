# This file is a refactor of https://github.com/openai/simple-evals/blob/main/math_eval.py
# python3 math_eval.py --policy-base-url "https://openrouter.ai/api/v1" --policy-model "meta-llama/llama-3.1-8b-instruct" --policy-api-key "$OPENROUTER_API_KEY"
# python3 math_eval.py --policy-base-url "https://openrouter.ai/api/v1" --policy-model "meta-llama/llama-3.1-405b-instruct" --policy-api-key "$OPENROUTER_API_KEY"
"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""


import argparse
import heapq
import logging
import os
import random
import textwrap
from typing import Callable, Dict, List, Literal

import jinja2
import pandas as pd
from grading.grader import grade_answer
from openai import OpenAI
from termcolor import colored
from tqdm import tqdm
from transformers import AutoTokenizer

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

STEP_VERIFICATION_PROMPT = jinja2.Template(
    textwrap.dedent(
        """
Given the problem and the partial solution, verify that the last step is correct.

Problem: {{ Question }}

Partial solution: {{ steps }}

FIRST RESPOND with "Yes" if the last step is correct or benefits the progress towards the solution, "No" otherwise.
THEN EXPLAIN why you chose "Yes" or "No" in 4 sentences or less.

For example:
"Yes, because ..."
"No, because ..."
""".strip()
    )
)


def check_equality(
    equality_checker: Callable[[List[Dict[str, str]]], str], expr1: str, expr2: str
) -> bool:
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    logger.debug(f"Equality check prompt: {prompt}")
    response = equality_checker([dict(content=prompt, role="user")])
    logger.debug(f"Equality check response: {response}")
    return response.lower().strip() == "yes"


class ProcessGuidedLLM:
    def __init__(self, client: OpenAI, beam_width: int = 3, model: str = "gpt-4o-mini"):
        self.client = client
        self.verifier = OpenAI()
        self.verifier_model = "gpt-4o-mini"
        self.beam_width = beam_width
        self.model = model  # Add this line
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def verify_step(self, question: str, steps: str) -> bool:
        prompt = STEP_VERIFICATION_PROMPT.render(Question=question, steps=steps)
        response = (
            self.verifier.chat.completions.create(
                model=self.verifier_model,
                messages=[{"content": prompt, "role": "user"}],
            )
            .choices[0]
            .message.content
        )
        return response.strip().lower().startswith("yes")

    def __call__(
        self,
        prompt_messages: List[Dict[str, str]],
        num_generations: int = 4,
        max_steps: int = 50,
    ) -> str:
        beams = [
            ([], 0, False)
        ]  # Each beam is a tuple of (steps_list, score, is_finished)
        for step_idx in range(1, max_steps + 1):
            all_candidates = []
            for steps, score, is_finished in beams:
                if is_finished:
                    # Beam is finished, carry it over
                    all_candidates.append((steps, score, True))
                    continue
                # Generate num_generations responses for the current beam
                if steps:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt_messages
                        + [
                            {"role": "assistant", "content": "\n\n".join(steps).strip()}
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                        continue_final_message=True,
                    )
                else:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                # print(formatted_prompt)
                responses = self.client.completions.create(
                    model=self.model,
                    prompt=formatted_prompt,
                    temperature=0.7,
                    top_p=0.95,
                    stop=[f"## Step {step_idx + 1}"],
                    n=num_generations,
                )
                generated_steps = [choice.text.strip() for choice in responses.choices]
                for step in generated_steps:
                    new_steps = steps + [step]
                    if self.verify_step(
                        prompt_messages[0]["content"], "\n".join(new_steps)
                    ):
                        # Check if 'final answer' is in the step
                        if "final answer" in step.lower():
                            # Beam is finished
                            all_candidates.append((new_steps, score + 1, True))
                            print(colored(step, "green"))
                        else:
                            all_candidates.append((new_steps, score + 1, False))
                            print(colored(step, "green"))
                    else:
                        print(colored(step, "red"))
            if not all_candidates:
                break  # No valid candidates, terminate early
            # Select top beam_width beams based on score
            beams = heapq.nlargest(self.beam_width, all_candidates, key=lambda x: x[1])
            # Check if all beams are finished
            if all(is_finished for (_, _, is_finished) in beams):
                break  # All beams finished, terminate early
        # Select the beam with the highest score
        best_steps, _, _ = beams[0]
        return "\n\n".join(best_steps)


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
            steps = [
                step.strip() for step in response_text.split("## Step") if step.strip()
            ]
            # using 4o-mini to verify steps are correct
            gpt4o_client = OpenAI()

            def gpt4o_mini_sampler(question: str, steps: str) -> str:

                return (
                    gpt4o_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            dict(
                                content=STEP_VERIFICATION_PROMPT.render(
                                    Question=question, steps=steps
                                ),
                                role="user",
                            )
                        ],
                    )
                    .choices[0]
                    .message.content
                )

            # for i, step in enumerate(steps):
            #     verified_text = gpt4o_mini_sampler(row["Question"], "\n\n".join(steps[:i+1]))
            #     if verified_text.strip().lower().startswith("yes"):
            #         print(f"\033[92mStep: {step}\033[0m")  # Green for correct step
            #         print(f"\033[92mStep verified response: {verified_text}\033[0m")
            #         print("\033[92m✓ Correct\033[0m")
            #     elif verified_text.strip().lower().startswith("no"):
            #         print(f"\033[91mStep: {step}\033[0m")  # Red for incorrect step
            #         print(f"\033[91mStep verified response: {verified_text}\033[0m")
            #         print("\033[91m✗ Incorrect\033[0m")
            #     else:
            #         print(f"\033[93mStep: {step}\033[0m")  # Yellow for uncertain step
            #         print(f"\033[93mStep verified response: {verified_text}\033[0m")
            #         print("\033[93m? Uncertain\033[0m")
            logger.debug(f"Problem response: {response_text}")
            # match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = extract_result_from_boxed(
                response_text
            )  # TODO: this is a hack to get llama3 working
            logger.debug(f"Extracted answer: {extracted_answer}")
            import ipdb

            ipdb.set_trace()
            if self.equality_checker:
                score = float(
                    check_equality(
                        self.equality_checker, row["Answer"], extracted_answer
                    )
                )
            else:
                score = float(grade_answer(extracted_answer, row["Answer"]))
            print(f"Response text: {response_text[-100:]}")
            print(f"Extracted answer: {extracted_answer}")
            print(f"Correct answer: {row['Answer']}")
            print(f"Score: {score}")
            print(f"Correct: {'CORRECT' if score == 1.0 else 'INCORRECT'}", end="")
            if score == 1.0:
                print("\033[92m ✓\033[0m")  # Green checkmark for correct
            else:
                print("\033[91m ✗\033[0m")  # Red X for incorrect
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
                # import ipdb; ipdb.set_trace()
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
    parser.add_argument(
        "--beam-width", type=int, default=2, help="Beam width for beam search"
    )
    parser.add_argument(
        "--num-generations", type=int, default=2, help="Number of generations per step"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Maximum number of steps for beam search",
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

    # # Using simple policy client
    # simple_scores = math_eval(
    #     lambda x: policy_client.chat.completions.create(
    #         model=args.policy_model, messages=x
    #     )
    #     .choices[0]
    #     .message.content
    # )

    # logger.info(f"Individual scores with simple policy client: {simple_scores}")

    # Using beam search
    process_guided_llm = ProcessGuidedLLM(
        client=policy_client,
        beam_width=args.beam_width,
        model=args.policy_model,  # Add this line
    )

    beam_search_scores = math_eval(
        lambda x: process_guided_llm(
            prompt_messages=x,
            num_generations=args.num_generations,
            max_steps=args.max_steps,
        )
    )

    logger.info(f"Individual scores with beam search: {beam_search_scores}")


if __name__ == "__main__":
    main()
