import sys
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset
import json
import re
import numpy as np
import ast
from utils import predict_link, predict_link_complete, predict_link_complete_discount
from tqdm import tqdm
from collections import defaultdict as ddict

def extract_jsonl_info(jsonl_path):
    """
    Reads the JSONL file and extracts 'responses', 'source node', and 'timestamp' from each entry.
    Returns a list of dicts with these fields.
    """
    def extract_solution(solution_str):
        solution = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
        try:
            final_solution = solution.group(1).strip()
            # # Normalize any non-Python empty list representations
            # if final_solution.replace(' ', '') == "[]":
            #     final_solution = "[-1]"
        except Exception as e:
            print(e)
            final_solution = "[-1]" # just set to -1 to mark incorrect prediction
        return final_solution

    results = []
    results_compact = []
    multiple_ans_cnt = 0
    with open(jsonl_path, 'r') as f:
        data = json.load(f)  # Load the whole file as a JSON array
        for entry in data:
            responses = entry.get('responses', None)
            content = entry.get('prompt', [])[1]['content']
            m = re.search(r'for `Query Source Node` (\d+) at `Query Timestamp` (\d+)\?', content)
            source_node = int(m.group(1)) if m else None
            timestamp = int(m.group(2)) if m else None
            destination_nodes = entry.get('reward_model', None)['ground_truth']
            idx = entry['extra_info']['index']
            dataset = entry['extra_info']['dataset']
            hop = entry['extra_info']['hop']
            answer_nonexistent = entry['extra_info']['answer_nonexistent']
            if len(destination_nodes) > 1:
                print("more than 1 destination node")
                print("link info:", source_node, timestamp, destination_nodes)
                multiple_ans_cnt += 1
            # predicted = extract_solution(responses[0])
            # if predicted == 'empty list':
            #     predicted = '[-1]'
            # Check if predicted is a string representation of a list
            # predicted = extract_solution(responses)
            predicted = extract_solution(responses[0])
            predicted_str = predicted.strip()
            if predicted_str.startswith("[") and predicted_str.endswith("]"):
                try:
                    predicted_list = ast.literal_eval(predicted_str)
                    if not isinstance(predicted_list, list):
                        predicted_list = [-1]
                except Exception:
                    predicted_list = [-1]
            else:
                predicted_list = [-1]

            results_compact.append({
                'predicted': predicted_list,
                'ground_truth': destination_nodes, # the ground truths for this query
                'index': idx,
                'dataset': dataset,
                'hop': hop,
                'answer_nonexistent': answer_nonexistent,
            })

            for destination_node in destination_nodes:
                results.append({
                    'responses': responses,
                    'source_node': source_node,
                    'timestamp': timestamp,
                    'destination_node': destination_node, # the ground truth for this link
                    'predicted': predicted_list,
                    'ground_truth': destination_nodes, # the ground truths for this query
                    'index': idx,
                    'dataset': dataset,
                    'hop': hop,
                    'answer_nonexistent': answer_nonexistent,
                })
    print("multiple answer count:", multiple_ans_cnt)
    return results, results_compact

name = "tgbl-wiki"
dataset = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
neg_sampler = dataset.negative_sampler
dataset.load_test_ns()
metric = dataset.eval_metric
evaluator = Evaluator(name="tgbl-wiki")

jsonl_file = YOUR_PATH_TO_LLM_GENERATION
info, info_compact = extract_jsonl_info(jsonl_file)

scores = []
scores_complete = []
scores_complete_exist = []
scores_complete_discount = []
scores_complete_discount_exist = []
scores_dataset = ddict(list)
scores_dataset_discount = ddict(list)
scores_dataset_exist = ddict(list)
scores_dataset_discount_exist = ddict(list)

num_nodes = dataset.num_nodes
all_nodes = list(range(num_nodes))
acc = []
acc_exist = []
acc_dataset = ddict(list)
acc_dataset_exist = ddict(list)
correct_idx = []

answer_num = []
gt_answer_num = []
for item in tqdm(info_compact):
    print(item['predicted'], item['ground_truth'])
    answer_num.append(len(item['predicted']))
    gt_answer_num.append(len(item['ground_truth']))
    acc.append(int(item['predicted'] == item['ground_truth']))
    acc_dataset[item['dataset']].append(int(item['predicted'] == item['ground_truth']))
    if len(item['answer_nonexistent']) == 0:
        acc_exist.append(int(item['predicted'] == item['ground_truth']))
        acc_dataset_exist[item['dataset']].append(int(item['predicted'] == item['ground_truth']))

# specify a list of test data sources
test_data_sources = ["tgbl-wiki", "tgbl-subreddit", "tgbl-coin", "tgbl-flight", "tgbl-uci", "tgbl-enron"]

for item in tqdm(info):
    # complete MRR
    neg_nodes = np.array(list(set(all_nodes) - set(item['ground_truth'])))
    y_pred = predict_link_complete(np.array([int(node) for node in item['ground_truth']]), item['predicted'], item['destination_node'], num_nodes)
    input_dict = {
            "y_pred_pos": np.array([y_pred[0]]),
            "y_pred_neg": np.array(y_pred[1:]),
            "eval_metric": [metric],
        }
    y_pred_discount = predict_link_complete_discount(np.array([int(node) for node in item['ground_truth']]), item['predicted'], item['destination_node'], num_nodes)
    input_dict_discount = {
            "y_pred_pos": np.array([y_pred_discount[0]]),
            "y_pred_neg": np.array(y_pred_discount[1:]),
            "eval_metric": [metric],
        }
    scores_complete.append(evaluator.eval(input_dict)[metric])
    scores_complete_discount.append(evaluator.eval(input_dict_discount)[metric])
    scores_dataset[item['dataset']].append(evaluator.eval(input_dict)[metric])
    scores_dataset_discount[item['dataset']].append(evaluator.eval(input_dict_discount)[metric])
    if len(item['answer_nonexistent']) == 0:
        scores_complete_exist.append(evaluator.eval(input_dict)[metric])
        scores_complete_discount_exist.append(evaluator.eval(input_dict_discount)[metric])
        scores_dataset_exist[item['dataset']].append(evaluator.eval(input_dict)[metric])
        scores_dataset_discount_exist[item['dataset']].append(evaluator.eval(input_dict_discount)[metric])


# Open a file to write the output
sys.stdout = open(YOUR_PATH_TO_EVAL_OUTPUT, 'w')

# print("correct idx:", correct_idx)
print("###################")
print("Complete Results:")
print(f"Final average tgb MRR: {np.mean(scores):.4f} over {len(scores)} edges")
print(f"Final average complete MRR: {np.mean(scores_complete):.4f} over {len(scores_complete)} edges")
print(f"Final average complete MRR (exist answer): {np.mean(scores_complete_exist):.4f} over {len(scores_complete_exist)} edges")
print(f"Final average complete MRR with discount: {np.mean(scores_complete_discount):.4f} over {len(scores_complete_discount)} edges")
print(f"Final average complete MRR with discount (exist answer): {np.mean(scores_complete_discount_exist):.4f} over {len(scores_complete_discount_exist)} edges")
print(f"Final average accuracy: {np.mean(acc):.4f} over {len(acc)} edges")
print(f"Final average accuracy (exist answer): {np.mean(acc_exist):.4f} over {len(acc_exist)} edges")
print(f"Average number of predicted answers: {np.mean(answer_num):.4f} over {len(answer_num)} edges")
print(f"Average number of ground truth answers: {np.mean(gt_answer_num):.4f} over {len(gt_answer_num)} edges")
print("###################")
for dataset in test_data_sources:
    print(f"Dataset: {dataset}")
    print(f"Final average complete MRR: {np.mean(scores_dataset[dataset]):.4f} over {len(scores_dataset[dataset])} edges")
    print(f"Final average complete MRR (exist answer): {np.mean(scores_dataset_exist[dataset]):.4f} over {len(scores_dataset_exist[dataset])} edges")
    print(f"Final average complete MRR with discount: {np.mean(scores_dataset_discount[dataset]):.4f} over {len(scores_dataset_discount[dataset])} edges")
    print(f"Final average complete MRR with discount (exist answer): {np.mean(scores_dataset_discount_exist[dataset]):.4f} over {len(scores_dataset_discount_exist[dataset])} edges")
    print(f"Final average accuracy: {np.mean(acc_dataset[dataset]):.4f} over {len(acc_dataset[dataset])} edges")
    print(f"Final average accuracy (exist answer): {np.mean(acc_dataset_exist[dataset]):.4f} over {len(acc_dataset_exist[dataset])} edges")
    print("###################")

