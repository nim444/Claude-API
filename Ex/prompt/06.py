import os
import json
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


def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""
    messages = []
    add_user_message(messages, prompt)
    output = chat(messages)
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
    """Calls run_prompt, grades the result, and returns comprehensive evaluation"""
    output = run_prompt(test_case)

    # Grade the output
    model_grade = grade_by_model(test_case, output)
    score = model_grade["score"]
    reasoning = model_grade.get("reasoning", "")

    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning,
        "strengths": model_grade.get("strengths", []),
        "weaknesses": model_grade.get("weaknesses", []),
    }


def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case, calculates average score"""
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    print(f"Average score: {average_score:.2f}")

    return results


if __name__ == "__main__":
    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    print(f"Running evaluation on {len(dataset)} test cases...")
    results = run_eval(dataset)

    print(json.dumps(results, indent=2))
