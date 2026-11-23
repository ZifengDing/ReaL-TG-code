import csv
import numpy as np
import random
import torch
from typing import List

def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.
    Args:
        seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def readcsv2list(file_path):
    r"""
    Read a CSV file and return its content as a list of lists.
    Args:
        file_path: Path to the CSV file.
    Returns:
        List of lists containing the CSV data.
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        pred_list = []
        true_list = []
        for row in reader:
            pred_list.append(int(row[0]))
            true_list.append(int(row[1]))
    return pred_list, true_list


def append2csv(file_path, data):
    r"""
    Append data to a CSV file.
    Args:
        file_path: Path to the CSV file.
        data: Data to append (list of lists).
    """
    with open(file_path, 'a', newline='\n') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)



def load_txt_from_edgelist(file_path):
    out_str = ""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            out_str += f"{row[0]},{row[1]},{row[2]}\n"
    return out_str


def load_rows_from_edgelist(file_path):
    out_rows = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            out_rows.append([row[0],row[1],row[2]])
    return out_rows


def row2text(rows):
    out_text = ""
    for row in rows:
        out_text += f"({row[0]},{row[1]},{row[2]})"
    return out_text


def predict_link(query_dst: np.ndarray, llm_dst: List[int]) -> np.ndarray:
    r"""
    convert LLM prediction into MRR format, just check if the LLM prediction is within the possible destinations; designed for tgb negative samples
    """
    pred = np.zeros(len(query_dst))
    idx = 0
    for dst in query_dst:
        for dst_ in llm_dst:
            # print(str(dst), dst_)
            # print(type(dst), type(dst_))
            # assert 0
            if str(dst) == str(dst_):
                pred[idx] = 1.0
                break
        idx += 1
    return pred

def predict_link_complete(gt_dst: np.ndarray, llm_dst: List[int], dst_example: int, num_nodes: int) -> np.ndarray:
    r"""
    convert LLM prediction into MRR format, just check if the LLM prediction is within the possible destinations; designed for all negative samples
    """
    pred = np.zeros(num_nodes)
    idx1 = 0 # idx for correct labels
    idx2 = -1 # idx for incorrect labels
    # print(llm_dst, gt_dst)
    try:
        for dst_ in llm_dst:
            if dst_ in gt_dst:  # if the current node belongs to ground truth
                if dst_ == dst_example: # if the current node is the node that we ask for
                    pred[idx1] = 1.0
                    idx1 += 1
            else:
                pred[idx2] = 1.0
                idx2 -= 1
    except Exception as e:
        # meets error during evaluation, set to an incorrect answer
        print(e)
        pred[idx2] = 1.0
        idx2 -= 1
        
    return pred

def predict_link_complete_discount(gt_dst: np.ndarray, llm_dst: List[int], dst_example: int, num_nodes: int) -> np.ndarray:
    r"""
    convert LLM prediction into MRR format, just check if the LLM prediction is within the possible destinations; designed for all negative samples; also impose penalty on predicted answers not existing in ground truth
    """
    pred = np.zeros(num_nodes)
    idx1 = 0 # idx for correct labels
    idx2 = -1 # idx for incorrect labels
    # print(llm_dst, gt_dst)
    try:
        for dst_ in llm_dst:
            if dst_ in gt_dst:  # if the current node belongs to ground truth
                if dst_ == dst_example: # if the current node is the node that we ask for
                    pred[idx1] = 1.0
                    idx1 += 1
            else:
                pred[idx2] = 1.1
                idx2 -= 1
    except Exception as e:
        # meets error during evaluation, set to an incorrect answer
        print(e)
        pred[idx2] = 1.1
        idx2 -= 1
        
    return pred