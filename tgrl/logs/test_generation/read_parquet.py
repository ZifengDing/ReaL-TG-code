import pandas as pd

# Path to the parquet file
data_path = LLM_GENERATION_PARQUET_PATH

def main():
    # Read the parquet file
    df = pd.read_parquet(data_path)
    
    # Save DataFrame as JSON in the same directory as the parquet file
    json_path = LLM_GENERATION_JSON_OUTPUT_PATH
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"Saved DataFrame as JSON to {json_path}")

if __name__ == "__main__":
    main()
