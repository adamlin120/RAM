import jinja2


# prompt from https://huggingface.co/datasets/meta-llama/Llama-3.1-405B-Instruct-evals
"""
# Example usage of LLAMA3_QUERY_TEMPLATE
example_question = "What is the sum of the first 100 positive integers?"
rendered_prompt = LLAMA3_QUERY_TEMPLATE.render(Question=example_question)
print(rendered_prompt)
"""
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


# From OpenAI simple eval repo
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
"""
Given the problem and the partial solution, verify that the last step is correct.

Problem: {{ Question }}

Partial solution: {{ steps }}

FIRST RESPOND with "Yes" if the last step is correct or benefits the progress towards the solution, "No" otherwise.
THEN EXPLAIN why you chose "Yes" or "No" in 4 sentences or less.

For example:
"Yes, because ..."
"No, because ..."
""".strip() # TODO: should first CoT then YES/NO
)


THINKING_LLM_PROMPT = """
Respond to the following user query in a comprehensive and detailed way. But first write down
your internal thoughts. This must include your draft response and its evaluation. After this,
write your final response after "<R>".
User query: {instruction}
""".strip()

THINKING_LLM_MATH_PROMPT = """
Respond to the following user query in a comprehensive and detailed way. But first write down
your internal thoughts. This must include your draft response and its evaluation. After this,
write your final response after "<R>".

In your final response after "<R>", conclude with:

Therefore, the final answer is: $\\boxed{answer}$

Where "answer" is just the final number or expression that solves the problem.

Problem: {instruction}
""".strip()