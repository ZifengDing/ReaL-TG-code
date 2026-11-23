import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it", help="Hugging Face model name")
    parser.add_argument("--input_file", type=str, default="dataset/realtg/test.parquet", help="Path to the input parquet file")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Name of the column containing the prompts")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--hf_token", type=str, help="Your Hugging Face API token")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=args.hf_token,
    )
    model.eval()

    print(f"Loading data from: {args.input_file}")
    df = pd.read_parquet(args.input_file)

    df.dropna(subset=[args.prompt_column], inplace=True)
    prompt_conversations = df[args.prompt_column].tolist()

    prompts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in prompt_conversations]
    
    all_responses = []

    print("Generating responses...")
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i+args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,
                temperature=0.0,
            )

        input_token_lengths = [len(x) for x in inputs.input_ids]

        for j, output_tokens in enumerate(outputs):
            prompt_length = input_token_lengths[j]
            generated_tokens = output_tokens[prompt_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_responses.append(response.strip())

    if len(all_responses) == len(df):
        df['responses'] = all_responses
    else:
        processed_df = df.head(len(all_responses)).copy()
        processed_df['responses'] = all_responses
        df = processed_df

    print(f"Saving results to: {args.output_file}")
    df.to_json(args.output_file, orient='records', indent=4)

    print("Done!")

if __name__ == "__main__":
    main()