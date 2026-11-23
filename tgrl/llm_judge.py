#!/usr/bin/env python3
import os
import json
import re
import argparse
from typing import List, Dict, Any

from dotenv import load_dotenv
from jsonschema import validate, ValidationError
import openai
from tqdm import tqdm

# Load environment variables from .env file, specify API key in .env file
load_dotenv()

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
TEMPERATURE = 0.0  # deterministic
TIMEOUT_S = 120

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "string"}, "minItems": 0},
        "#claims": {"type": "integer", "minimum": 0},
        "faithfulness": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "supported": {"type": "array", "items": {"type": "string"}},
                "unsupported": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "reason": {"type": "string", "enum": ["Contradicted", "Not-in-g"]},
                            "pointer": {"type": "string"}
                        },
                        "required": ["claim", "reason", "pointer"]
                    }
                },
                "totals": {
                    "type": "object",
                    "properties": {
                        "supported": {"type": "integer", "minimum": 0},
                        "claims": {"type": "integer", "minimum": 1}
                    },
                    "required": ["supported", "claims"]
                }
            },
            "required": ["score", "supported", "unsupported", "totals"]
        },
        "logic": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 2},
                "rationale": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},

                    },
                    "required": ["summary"]
                }
            },
            "required": ["score", "rationale"]
        },
        "alignment": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "justified_answers": {"type": "array", "items": {"type": "integer"}},
                "unjustified_answers": {"type": "array", "items": {"type": "integer"}},

                "justification_notes": {"type": "string"}
            },
            "required": ["score", "justified_answers", "unjustified_answers", "justification_notes"]
        }
    },
    "required": ["claims", "faithfulness", "logic", "alignment"]
}

EVALUATION_PROCEDURE = """
---
### Evaluation Procedure

1. Parse response
- Extract the answer list A as the JSON array inside <answer> ... </answer>. If parsing fails, set A=[] and record a note in alignment.notes.
- Extract the explanation E as the natural-language content inside <think> ... </think>. Judge only what is explicitly stated in E.

2. Break explanation into atomic claims
- Split E into minimal atomic claims about edges, nodes, times, paths, counts, or membership related to graph.
- Produce a list of claims = [c1, c2, ...]. Use short, verifiable sentences.
- Also return the number of claims as #claims.

3. Faithfulness to g
- For each claim ci, label one of:
  "Supported" (entailed by g),
  "Contradicted" (g states the opposite),
  "Not-in-g" (cannot be verified from g; count as unsupported).
- faithfulness.score = #Supported / max(1, #claims).
- Return all Supported claims. For non-Supported claims, return objects with fields: claim, reason ("Contradicted"|"Not-in-g"), and pointer (cite/summarize lines in g).

4. Logic Consistency (internal reasoning soundness; independent of g’s truth)
- Use 0–2 scale:
  2 Excellent — steps are necessary & sufficient; no contradictions; valid transitions; no major gaps.
  1 Good — slight gap or mild unstated assumption; mostly valid.
  0 Poor/Invalid — The reasoning is unconvincing or fundamentally flawed. It may have significant gaps, make speculative leaps, contain inconsistencies, or include clear formal fallacies like contradictions or circular reasoning.
- Return a rationale with a concise one-sentence summary.

5. Answer–Explanation Alignment
- An answer a ∈ A is justified iff:
  (1) E explicitly argues for a, and
  (2) those supporting claims are Supported in step 3.
- alignment.score = |justified_answers| / max(1, |A|).
- Return the justified_answers.
- Return the justification_notes that explicitly indicates why the answers are justified. This part will be used to classify the reasoning patterns of models, so be clear and concise.
- Return the unjustified_answers (in A but not justified).

6. Output
- Return ONLY a JSON object with fields: claims, faithfulness, logic, alignment.
- Do not include any text outside the JSON object.
---
"""

SYSTEM_PROMPT = (
    "You are a meticulous evaluator for temporal graph QA with explanations.\n"
    "You will receive: (q) the question, (g) a temporal subgraph as lines of (src, dst, ts) strictly before the query timestamp, "
    "and a model response R that contains an explanation inside <think>...</think> and a final answer list inside <answer>...</answer>.\n"
    "Your job is to output ONLY valid JSON matching the JSON Schema provided in the instructions. "
    "You should follow the evaluation procedure as follows:\n"
    f"{EVALUATION_PROCEDURE}\n"
    "Score three aspects: (1) Faithfulness to g, (2) Logic Consistency, (3) Answer–Explanation Alignment.\n\n"
    "IMPORTANT INSTRUCTIONS:\n"
    "1. Please be VERY CAUTIOUS when you are asked to extract claims and calculate the number of claims.\n"
    "2. When you are asked to extract claims, DO NOT include any claim making conclusions about the final answer.\n"
    "3. In many cases, model will correct its previous claims with new claims during reasoning. When you are asked to extract claims, ALWAYS consider this situation and ONLY include the claims that are not corrected by the model in later steps.\n"
    "4. When you are asked to evaluate logic consistency, you should evaluate the explanation as a whole reagrdless of the result of faithfulness.\n"
    "5. The timestamps with larger numbers are later than the ones with smaller numbers.\n"
    "6. When judging whether answers are justified or writing justification_notes, remain strictly objective and evaluate only against the model’s own explanation. Consider an answer justified if the explanation explicitly supports it, even if you personally disagree with the reasoning. DO NOT mark an answer as unjustified simply because you think it should be justified in another way.\n\n"
)

USER_PROMPT_TEMPLATE = """### JSON Schema
Your output must be a single JSON object that validates against this schema:
{schema_json}

### Inputs
- q:
{question}

- g (historical interactions; all timestamps < {ts}):
{graph_lines}

- Metadata:
  - Query Source Node: {src}
  - Query Timestamp: {ts}
  - Ground-truth answers: {ground_truth_json}
  - Model's final answer: {final_answer}

- Model response R:
{model_response}
"""

def _build_user_prompt(
    question: str,
    graph_lines: str,
    src: int,
    ts: int,
    ground_truth: List[int],
    model_response: str,
    schema: Dict[str, Any],
    final_answer: str
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        schema_json=json.dumps(schema, indent=2),
        question=question.strip(),
        graph_lines=graph_lines.strip(),
        src=src,
        ts=ts,
        ground_truth_json=json.dumps(ground_truth),
        model_response=model_response.strip(),
        final_answer=final_answer.strip()
    )

def judge(
    question: str,
    graph_lines: str,
    src: int,
    ts: int,
    ground_truth: List[int],
    model_response: str,
    final_answer: str,
) -> Dict[str, Any]:
    """Run the judge with an OpenAI model and return validated JSON."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = openai.OpenAI(api_key=api_key)

    user_prompt = _build_user_prompt(question, graph_lines, src, ts, ground_truth, model_response, SCHEMA, final_answer)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Make the call
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        timeout=TIMEOUT_S,
    )
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")

    # Parse JSON
    try:
        text = resp.choices[0].message.content
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        raw_content = resp.choices[0].message.content if resp.choices else ""
        raise RuntimeError(f"Model did not return valid JSON: {e}\nRaw: {raw_content}")

    # Validate schema
    try:
        validate(instance=data, schema=SCHEMA)
    except ValidationError as ve:
        raise RuntimeError(f"Judge JSON failed schema validation: {ve.message}\nAt: {list(ve.path)}\nFull: {json.dumps(data, indent=2)}")

    return data

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file, LLM generation result")
    args = parser.parse_args()

    input_file = args.input
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = []
    output_file = f"{input_file}_judged_gpt4.1mini.jsonl"

    # Clear the output file before starting if it's a new run
    with open(output_file, 'a') as f:
        pass

    length = len(data)

    for i, item in enumerate(tqdm(data)):
        # print(f"--- Judging Record {i} ---")
        try:
            # Extract data from the JSON structure
            prompt_content = item['prompt'][1]['content']
            # model_response = item['responses'][0]
            model_response = item['responses']
            ground_truth = item['reward_model']['ground_truth']
            src, ts = item['extra_info']['link']

            # Parse question and graph_lines from the prompt
            question_match = re.search(r"Question:\n(.*?)$", prompt_content, re.S)
            if not question_match:
                raise ValueError("Could not parse question from prompt")
            question = question_match.group(1).strip()

            graph_lines_match = re.search(r"historical interactions:\n(.*?)\nCould you list", prompt_content, re.S)
            if not graph_lines_match:
                raise ValueError("Could not parse graph lines from prompt")
            graph_lines = graph_lines_match.group(1).strip()

            final_answer_match = re.search(r"<answer>(.*?)</answer>", model_response, re.S)
            if not final_answer_match:
                raise ValueError("Could not parse final answer from model response")
            final_answer = final_answer_match.group(1).strip()

            # Call the judge function
            judgement = judge(
                question=question,
                graph_lines=graph_lines,
                src=src,
                ts=ts,
                ground_truth=ground_truth,
                model_response=model_response,
                final_answer=final_answer,
            )

            # Add judgement to the original item
            item['judgement'] = judgement
            # results.append(item)
            with open(output_file, 'a') as f:
                f.write(json.dumps(item) + '\n')
            # print(f"Successfully judged record {i}.")

        except (KeyError, IndexError, ValueError, RuntimeError) as e:
            # print(f"Could not process record {i}: {e}")
            # Optionally, append the item with an error message
            item['judgement'] = {'error': str(e)}
            with open(output_file, 'a') as f:
                f.write(json.dumps(item) + '\n')
            # results.append(item)
        # print("\n")

    print(f"Finished judging. Results saved to {output_file}")