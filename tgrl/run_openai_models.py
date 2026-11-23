from openai import OpenAI
import json
import os
import argparse
from tqdm import tqdm

def main():
    """
    Main function to run OpenAI generation on a JSONL file.
    """
    parser = argparse.ArgumentParser(description="Run OpenAI generation on a JSONL file.")
    parser.add_argument("--input_file", default="test.jsonl", help="Path to the input JSONL file.")
    parser.add_argument("--output_file", default="test_gpt5_mini_coin.json", help="Path to the output JSONL file.")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07", help="OpenAI model to use for generation.")
    args = parser.parse_args()

    # It's recommended to set the OpenAI API key as an environment variable
    # for security purposes. You can do this by running:
    # export OPENAI_API_KEY='your_api_key_here'
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it to your OpenAI API key.")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Starting generation with model: {args.model}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            output_data = []
            for line in tqdm(lines, desc="Processing lines"):
                try:
                    data = json.loads(line)
                    messages = data.get("messages")

                    if not messages:
                        # If there are no messages, append the original data and continue
                        output_data.append(data)
                        continue

                    # Call the OpenAI API
                    response = client.responses.create (
                        model=args.model,
                        reasoning={"effort": "medium", "summary": "auto"},
                        input=[
                            {
                                "role": "user", 
                                "content": "You are a temporal graph learning expert.\n" + messages[1]["content"]
                            }
                        ]
                    )
                    info = str(response)
                    response_text = response.output_text.strip()
                    # Add the generation result to the data as a list
                    data["responses"] = [response_text]
                    data["openai info"] = info

                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSON decode error: {e}")
                    continue # Or handle as you see fit
                except Exception as e:
                    print(f"An error occurred while processing a line: {e}")
                    data["generation_error"] = str(e)
                
                # Append the updated data to the output list
                output_data.append(data)

            # Write all output as a single JSON array
            with open(args.output_file, 'w', encoding='utf-8') as f_out:
                json.dump(output_data, f_out, ensure_ascii=False, indent=2)

        print("Generation complete.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")

if __name__ == "__main__":
    main()
