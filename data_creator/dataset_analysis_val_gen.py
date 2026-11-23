import torch
import numpy as np
import argparse
from enum import Enum
from typing import Dict, Tuple
from pydantic import BaseModel, constr
from transformers import AutoTokenizer
from outlines import models, generate, samplers
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset
from neighbor_tracker_analysis import NeighborTrackerv2, NeighborTrackerTPPR
import json
import os
import timeit
from collections import defaultdict as ddict
from tqdm import tqdm

parser = argparse.ArgumentParser('*** arguments for running data construction ***')
parser.add_argument('--data', type=str, help='specify which dataset', default='tgbl-wiki')
parser.add_argument('--nbr', type=int, help='how many timestamps to retrieve from the past to form the sampled neighbors', default=100)
parser.add_argument('--logfile', type=str, help='where to save the output', default=None)
parser.add_argument('--train_ts_length', type=int, help='how many timestamps in training set to form the training set', default=1000)
parser.add_argument('--val_ts_length', type=int, help='how many timestamps in validation set to form the validation set', default=500)
parser.add_argument('--resume', action="store_true", default=False, help="resume from the last construction state")


args = parser.parse_args()
DATA = args.data
print ("using the following arguments: ")
print (args)

# load dataset from tgb
dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
data = dataset.full_data  

metric = dataset.eval_metric
evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

src = data['sources']
dst = data['destinations']
link_feature = data['edge_feat']
ts = data['timestamps']
rows = np.stack((src, dst, ts), axis=1)
print(rows.shape[0])
tracker = NeighborTrackerTPPR(src.tolist(),
                              dst.tolist(),
                              ts.tolist(),
                              max_size=args.nbr,
                              dataname=DATA)

print(ts[dataset.train_mask][-1], ts[dataset.val_mask][0])
train_last_ts = ts[dataset.train_mask][-1]
print(ts[dataset.val_mask][-1], ts[dataset.test_mask][0])
val_last_ts = ts[dataset.val_mask][-1]

# collect training data, from the last timestamp of the training set
train_ts_list = []
for t in sorted([ts_ for ts_ in tracker.ts_dict.keys()]):
    if t == train_last_ts:
        train_ts_list.append(t)
        break
    elif t < train_last_ts:
        train_ts_list.append(t)

train_ts_list = train_ts_list[-args.train_ts_length:]
train_size = 0
for t in train_ts_list:
    if t in tracker.ts_dict:
        train_size += len(tracker.ts_dict[t])
# print(train_ts_list)
print("training data size: ", train_size)

# collect validation data, from the last timestamp of the validation set
val_ts_list = []
for t in sorted([ts_ for ts_ in tracker.ts_dict.keys()]):
    if t == val_last_ts:
        val_ts_list.append(t)
        break
    elif t < val_last_ts:
        val_ts_list.append(t)

val_ts_list = val_ts_list[-args.val_ts_length:]
assert val_ts_list[0] > train_ts_list[-1]
val_size = 0
for t in val_ts_list:
    if t in tracker.ts_dict:
        val_size += len(tracker.ts_dict[t])
print("validation data size: ", val_size)

# collect test data, from all of the test set
test_ts_list = []
for t in sorted([ts_ for ts_ in tracker.ts_dict.keys()]):
    if t > val_last_ts:
        test_ts_list.append(t)

assert test_ts_list[0] > val_ts_list[-1]
test_size = 0
for t in test_ts_list:
    if t in tracker.ts_dict:
        test_size += len(tracker.ts_dict[t])
print("test data size: ", test_size)

def make_user_prompt(src: int, ts: int, nbrs: Dict[int, Tuple[int, int]] = None, split: str = "train") -> str:

    user_prompt = (
        f"Question:\nGiven the following historical interactions:\n"
    )

    if len(nbrs[src]) == 0 and split in ["train", "val"]: # cannot find any neighbor, directly skip
        return None

    for _, v in nbrs.items():
        for source, destination, timestamp in v:
            source = int(source)
            dst = int(destination)
            timestamp = int(timestamp)
            user_prompt += f"({source}, {dst}, {timestamp}) \n"
    user_prompt += (f"Could you list all plausible `Query Destination Node`s for `Query Source Node` {src} at `Query Timestamp` {ts}?")

    return user_prompt

def make_task_prompt(task: str='link_prediction') -> str:
    if (task == 'link_prediction'):
        return ("You will be asked to predict the next interaction (i.e. `Query Destination Node`) given the `Query Source Node` and `Query Timestamp`.\n"
        "You will also be given a number of historical interactions extracted from a temporal subgraph, where each of them is represented as a tuple of (`Source Node`, `Destination Node`, `Timestamp`). Use this information to predict the most likely `Query Destination Node`s for `Query Source Node` at `Query Timestamp`.\n"
        "You will only receive information available before `Query Timestamp`. No information at or after this timestamp will be provided. The user instruction is correct and contains no mistakes or typos.\n"
        "INSTRUCTIONS:\n"
        "1. You must FIRST think about the reasoning process as an internal monologue and then provide the final answer.\n"
        "2. The reasoning process MUST BE enclosed within <think> </think> tags.\n"
        "3. The final answer MUST BE put within <answer> </answer> tags.\n"
        "4. If the answer contains multiple `Query Destination Node`s, please provide all of them and put them in a list in soreted order, e.g., <answer>[0, 1, 2]</answer>, otherwise, please show the answer as a list with only one element, e.g., <answer>[0]</answer>.\n\n"
        )
    else:
        return "Your task is to predict the next interaction (i.e. `Destination Node`) given the `Source Node` and `Timestamp`."

system_prompt = (
            "You are a temporal graph learning expert."
            )
# construct prompt for each training data; link prediction
task_prompt = make_task_prompt(task='link_prediction')

# check the distance between the query node and the ground truth node
# we take the examples with richer history first
hop2data = ddict(list)
# use a file to store the status of construction
if os.path.exists(f"processed_TG/{DATA}/status_val.json"):
    with open(f"processed_TG/{DATA}/status_val.json", "r") as f:
        status = json.load(f)
    resume = True
else:
    # initialize a status file
    status = {"time": 0}
    resume = False
    
for t_id, t in enumerate(val_ts_list[::-1]):
    if t in tracker.ts_dict:
        if resume and t >= status["time"]:
            continue

        for src, ts in tqdm(tracker.ts_dict[t].keys(), desc=f"Getting data from {t_id} out of {len(val_ts_list)} timestamps"):
            answer = list(tracker.ts_dict[t][(src, ts)])
            nbrs, involved_nodes, tppr_neighbors, involved_nodes_subgraph = tracker.get_neighbor_topk_tppr(src, ts)
            print("Involved nodes: ", involved_nodes, len(involved_nodes))
            print("Involved nodes in subgraph: ", involved_nodes_subgraph, len(involved_nodes_subgraph))
            print("Subgraph size: ", len(nbrs[src]))
            # print("nbrs[src]: ", nbrs[src])
            print("Answer: ", answer)
            if len(nbrs[src]) > 600:
                # we avoid excessively large subgraphs
                print("Subgraph size too large, skip")
                continue

            one_hop = []
            multi_hop = []
            for _, v in nbrs.items():
                for source, destination, timestamp in v:
                    source = int(source)
                    dst = int(destination)
                    if source == src:
                        # the current source is the query src, it means dst 1 hop of query src
                        one_hop.append(dst)
                    elif dst == src:
                        # the current dst is the query src, it means source 1 hop of query src
                        one_hop.append(source)
            for n in involved_nodes_subgraph:
                # we found 1 hop neighbors, then we take the rest of involved nodes from subgraph as multi hop; can be 2 or 3 hop
                if n not in one_hop:
                    multi_hop.append(n)

            # get the nearest hop of the answer
            nearest_hop = [0] * len(answer) # initialize to a dummy number for all answer
            for i, n in enumerate(answer):
                if n in one_hop:
                    nearest_hop[i] = 1
                elif n in multi_hop:
                    nearest_hop[i] = 2
                
            if 0 in nearest_hop:
                # current answer not in the subgraph, skip
                print("One or more answers not included in the subgraph, skip in training and validation")
                continue
            else:
                nearest_hop = min(nearest_hop) # nearest_hop == 1 -> answer is 1 hop; nearest_hop == 2 -> answer is multi hop
                print("Nearest hop: ", nearest_hop)
            
            # retain two buckets to store every hop-wise category; we only take examples where answer appears in the sampled subgraph
            if len(hop2data[2]) < 50 or len(hop2data[1]) < 100: # we sample until 2 hop answers have 50 examples
                hop2data[nearest_hop].append((src, ts, nbrs, involved_nodes, tppr_neighbors))
                user_prompt = make_user_prompt(src, ts, nbrs)
                prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task_prompt + user_prompt}]
                answer = sorted(list(tracker.ts_dict[t][(src, ts)]))
                # Store hop-wise data in a jsonl file
                if not os.path.exists(f"processed_TG/{DATA}/hop{nearest_hop}_val.jsonl"):
                    with open(f"processed_TG/{DATA}/hop{nearest_hop}_val.jsonl", "w") as f:
                        json.dump({"messages": prompt, "answer": answer, "hop": nearest_hop}, f)
                        f.write("\n")
                else:
                    with open(f"processed_TG/{DATA}/hop{nearest_hop}_val.jsonl", "a") as f:
                        json.dump({"messages": prompt, "answer": answer, "hop": nearest_hop}, f)
                        f.write("\n")
            else:
                break
        if len(hop2data[2]) >= 50 and len(hop2data[1]) >= 100:
            break
status["time"] = t
with open(f"processed_TG/{DATA}/status_val.json", "w") as f:
    json.dump(status, f)