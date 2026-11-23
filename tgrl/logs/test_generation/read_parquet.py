import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Convert a verl generation parquet file to JSON for evaluation.")
    parser.add_argument("--input", default="realtg/test_realtg_4b.parquet", help="parquet file written by run_qwen3-4b_realtg.sh")
    parser.add_argument("--output", default=None, help="output JSON file (default: the input path with a .json extension)")
    args = parser.parse_args()
    data_path = args.input

    # Read the parquet file
    df = pd.read_parquet(data_path)
    
    # Save DataFrame as JSON in the same directory as the parquet file
    json_path = args.output or os.path.splitext(data_path)[0] + ".json"
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"Saved DataFrame as JSON to {json_path}")

if __name__ == "__main__":
    main()
