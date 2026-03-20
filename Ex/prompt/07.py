import os
import json
import ast
import re
from statistics import mean
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model = "claude-haiku-4-5"


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    response = client.messages.create(**params)
    return response.content[0].text


def validate_json(text):
    """Validates if text is valid JSON"""
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    """Validates if text is valid Python syntax"""
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    """Validates if text is a valid regex pattern"""
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(output, test_case):
    """Grades the syntax validity of the output based on expected format"""
    expected_format = test_case.get("format", "").lower()

    if expected_format == "json":
        return validate_json(output)
    elif expected_format == "python":
        return validate_python(output)
    elif expected_format == "regex":
        return validate_regex(output)
    else:
        # If format is not specified, return neutral score
        return 5


def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}

Respond only with {test_case.get("format", "code")} and no explanation or commentary.
"""
    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```")
    output = chat(messages, stop_sequences=["```"])
    return output


def grade_by_model(test_case, output):
    """Uses Claude to evaluate the quality of the generated output"""
    eval_prompt = f"""
You are an expert code reviewer. Evaluate this AI-generated solution.

Task: {test_case["task"]}
Solution: {output}

Provide your evaluation as a structured JSON object with:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your assessment
- "score": A number between 1-10
"""

    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")

    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)


def run_test_case(test_case):
    """Calls run_prompt, grades syntax and quality, returns comprehensive evaluation"""
    output = run_prompt(test_case)

    # Grade the output using both code and model graders
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    syntax_score = grade_syntax(output, test_case)

    # Combine scores (equal weight)
    combined_score = (model_score + syntax_score) / 2

    return {
        "output": output,
        "test_case": test_case,
        "model_score": model_score,
        "syntax_score": syntax_score,
        "combined_score": combined_score,
        "reasoning": model_grade.get("reasoning", ""),
        "strengths": model_grade.get("strengths", []),
        "weaknesses": model_grade.get("weaknesses", []),
    }


def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case, calculates average scores"""
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    model_avg = mean([result["model_score"] for result in results])
    syntax_avg = mean([result["syntax_score"] for result in results])
    combined_avg = mean([result["combined_score"] for result in results])

    print(f"Model score average: {model_avg:.2f}")
    print(f"Syntax score average: {syntax_avg:.2f}")
    print(f"Combined score average: {combined_avg:.2f}")

    return results


if __name__ == "__main__":
    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    print(f"Running evaluation on {len(dataset)} test cases...")
    results = run_eval(dataset)

    print("\n" + "=" * 50)
    print(json.dumps(results, indent=2))
