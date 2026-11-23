from math_verify import parse, verify
import ast
import re

def extract_solution(solution_str):
    solution = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    final_solution = solution.group(1).strip()
    return final_solution

def set_f1(pred, truth):
    pred_set = set(pred)
    truth_set = set(truth)

    tp = len(pred_set & truth_set)  # true positives

    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(truth_set)
    print("precision:", precision)
    print("recall:", recall)
    return 2 * precision * recall / (precision + recall)

def compute_score_list(data_source, solution_str, ground_truth, extra_info=None) -> float:
    retval = 0.0
    # print("solution_str:", extract_solution(solution_str), type(extract_solution(solution_str)))
    # print("ground_truth:", ground_truth, type(ground_truth))
    try:
        solution = ast.literal_eval(extract_solution(solution_str))
        # gt = ground_truth.tolist()
        gt = ground_truth
        print(solution, gt)
        # set reward to the f1 score
        retval = set_f1(solution, gt)
    except Exception as e:
        print(e)

    return retval