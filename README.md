# Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning

This repository is the official implementation of *Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning*. It contains the ReaL-TG QA dataset, the scripts that build it from [TGB](https://github.com/shenyangHuang/TGB) datasets, and a [verl](https://github.com/volcengine/verl) recipe for training, response generation and evaluation.

## Repository structure

```text
data_creator/                       dataset construction from TGB, incl. T-CGS
tgrl/                               verl recipe, copied to PATH_TO_VERL/recipe/tgrl
├── dataset/realtg/                 train / val / test splits in parquet (training, generation, evaluation)
├── dataset_raw/realtg/             the same splits in jsonl, for easy inspection
├── train_qwen3-4b_realtg.sh        GRPO training of Qwen3-4B
├── reward_tgrl.py                  reward: F1 between predicted and ground-truth node sets
├── run_qwen3-4b_realtg.sh          response generation with the trained model
├── logs/test_generation/           generation outputs; read_parquet.py converts them to JSON
├── eval_mrr_realtg.py              MRR, penalized MRR and accuracy
├── utils.py                        metric helpers
├── llm_judge.py                    LLM-as-a-judge evaluation of reasoning quality
├── run_openai_models.py            baseline generation with OpenAI models
└── run_hf_model_model_parallel.py  baseline generation with Hugging Face models
```

## Dataset

We provide the ReaL-TG QA dataset adapted from TGB datasets in `tgrl/dataset_raw/realtg`, in jsonl format for easy inspection. We also provide the processed dataset in parquet format in `tgrl/dataset/realtg`, which can be used directly for LLM training, generation and evaluation.

| Split | Questions |
| --- | --- |
| train | 1,000 |
| val | 500 |
| test | 6,000: 1,000 each from `tgbl-wiki`, `tgbl-subreddit`, `tgbl-coin`, `tgbl-flight`, `tgbl-uci` and `tgbl-enron` |

Each question gives the historical interactions of a temporal subgraph as `(source, destination, timestamp)` tuples and asks for the destination nodes of a query source node at a query timestamp; the ground-truth nodes are in `answer`.

The dataset construction scripts are in `data_creator`. T-CGS is implemented in the function `get_neighbor_topk_tppr` (line 331 of `data_creator/neighbor_tracker_analysis.py`).

## Requirements and environment setup

Create a virtual environment:
```bash
conda create -n graphrl python==3.10
conda activate graphrl
```

Install verl (developed with verl==0.3.1.dev):
```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
```

Install vllm==0.8.5.post1, from the verl repository folder:
```bash
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

Install TGB from source to download datasets and run evaluation:
```bash
git clone https://github.com/shenyangHuang/TGB.git
cd TGB
pip install -e .
```

Install tqdm and wandb:
```bash
pip install tqdm==4.67.1 wandb
```

Other packages the scripts import:
```bash
pip install math-verify                       # reward_tgrl.py
pip install openai python-dotenv jsonschema   # LLM judge and OpenAI baselines
pip install outlines pydantic numba           # dataset construction (data_creator)
```

## Model training, generation and evaluation

### Copy the source code
Copy the `tgrl` folder into the verl recipes:
```bash
cp -r tgrl PATH_TO_VERL/recipe/
```

The training and generation scripts activate the `graphrl` conda environment and log to Weights & Biases, so run `wandb login` (or export `WANDB_API_KEY`) first.

### Model training
You can freely change training parameters according to verl's documentation.
```bash
cd PATH_TO_VERL/recipe/tgrl/
./train_qwen3-4b_realtg.sh
```

### Response generation
Before running the generation script, merge the model checkpoint if the model was trained on multiple GPUs:
```bash
cd PATH_TO_VERL/scripts
python model_merger.py merge --backend fsdp --local_dir PATH_TO_CHECKPOINT_DIR --target_dir PATH_TO_VERL/recipe/tgrl/merged_checkpoints/verl_grpo_tgrl/realtg_4b
```
Then generate responses with the trained model. You can freely change generation parameters according to verl's documentation.
```bash
cd PATH_TO_VERL/recipe/tgrl/
./run_qwen3-4b_realtg.sh
```
The responses are saved to `logs/test_generation/realtg/test_realtg_4b.parquet`.

### Evaluation
Convert the responses to JSON, then compute the metrics:
```bash
cd PATH_TO_VERL/recipe/tgrl/logs/test_generation/
python read_parquet.py --input realtg/test_realtg_4b.parquet
cd PATH_TO_VERL/recipe/tgrl/
python eval_mrr_realtg.py --input logs/test_generation/realtg/test_realtg_4b.json --output logs/test_generation/realtg/test_realtg_4b_eval.txt
```
`eval_mrr_realtg.py` loads `tgbl-wiki` through TGB (downloaded into `datasets/` on first use). The penalized MRR is implemented in the function `predict_link_complete_discount` in `tgrl/utils.py`.

## LLM-as-a-judge

The judge (GPT-4.1 mini) scores each explanation for faithfulness to the subgraph, logical consistency, and answer–explanation alignment. Put your OpenAI key in a `.env` file (`OPENAI_API_KEY=...`) or export it, then run:
```bash
cd PATH_TO_VERL/recipe/tgrl/
python llm_judge.py --input PATH_TO_GENERATIONS_JSON
```
The judgements are written to `PATH_TO_GENERATIONS_JSON_judged_gpt4.1mini.jsonl`.

## Baselines

`run_openai_models.py` (OpenAI models; reads `OPENAI_API_KEY` from the environment) and `run_hf_model_model_parallel.py` (Hugging Face models) generate responses with baseline models. Run either with `--help` to see its options.

## Acknowledgements

This code builds on [verl](https://github.com/volcengine/verl) and [TGB](https://github.com/shenyangHuang/TGB).
