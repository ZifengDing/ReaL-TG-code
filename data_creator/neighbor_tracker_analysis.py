from typing import Dict, List, Literal, Tuple
import numpy as np
import numba as nb
import os
import pickle
from collections import defaultdict as ddict

nb_key_type = nb.typeof((1, 1))


class NeighborTracker:
    def __init__(self,
                src: np.ndarray,
                dst: np.ndarray,
                ts: np.ndarray,
                max_size: int = 20,
            ) -> None:
        r"""initialize the neighbor tracker"""
        self.node_dict = {}
        self._check_input(src, dst, ts)
        self.max_size = max_size
        
        #! might not be efficient for large datasets
        for i in range(src.shape[0]):
            if src[i] not in self.node_dict:
                self.node_dict[src[i]] = [(dst[i], ts[i])]
            else:
                self.node_dict[src[i]].append((dst[i], ts[i]))

    def _check_input(self,
                    src: np.ndarray,
                    dst: np.ndarray,
                    ts: np.ndarray,
                    )-> None:
            r"""check if the input is valid"""
            if (src.shape[0] or dst.shape[0] or ts.shape[0]) == 0:
                raise ValueError("Empty input")
            if (src.shape[0] != dst.shape[0] or src.shape[0] != ts.shape[0] or dst.shape[0] != ts.shape[0]):
                raise ValueError("Input shapes do not match")


    def get_neighbor(self, 
                     nodes: np.ndarray,
                     )-> Dict[int, Tuple[int, int]]:
        r"""return the current neighbors for the given nodes"""
        neighbors = {k: self.node_dict[k][-self.max_size:] for k in nodes if k in self.node_dict}
        return neighbors
    
    def update(self,
               src: np.ndarray,
               dst: np.ndarray,
               ts: np.ndarray,
               )-> None:
        r"""update the neighbor tracker with new links"""
        self._check_input(src, dst, ts)
        
        for i in range(src.shape[0]):
            if src[i] not in self.node_dict:
                self.node_dict[src[i]] = [(dst[i], ts[i])]
            else:
                self.node_dict[src[i]].append((dst[i], ts[i]))

class NeighborTrackerv2:
    def __init__(self,
                src: List[int],
                dst: List[int],
                ts: List[int],
                max_size: int = 20,
            ) -> None:
        r"""initialize the neighbor tracker"""
        self.node_dict = {}
        self.ts_dict = {}
        # self._check_input(src, dst, ts)
        self.max_size = max_size
        cnt = 0
        # src id -> ts id -> dst id(s)
        for i in range(len(src)):
            if src[i] not in self.node_dict:
                self.node_dict[src[i]] = {}
                self.node_dict[src[i]][ts[i]] = [dst[i]]
                cnt += 1
            else:
                if ts[i] not in self.node_dict[src[i]]:
                    self.node_dict[src[i]][ts[i]] = [dst[i]]
                    cnt += 1
                else:
                    self.node_dict[src[i]][ts[i]].append(dst[i])
                    cnt += 1
        print (f"Number of nodes: {len(self.node_dict)}")
        print (f"Number of edges: {cnt}")

        self._check_node_dict_completeness()

        # ts id -> (queries + answer_lists)
        # ts_dict stores ground truth and the final data format, split by timestamps
        for i in range(len(src)):
            if ts[i] not in self.ts_dict:
                self.ts_dict[ts[i]] = ddict(set)
                self.ts_dict[ts[i]][(src[i], ts[i])].add(dst[i])
            else:
                self.ts_dict[ts[i]][(src[i], ts[i])].add(dst[i])

        self._check_ts_dict_completeness()

    def _check_node_dict_completeness(self) -> None:
        num_links = 0
        for node in self.node_dict.keys():
            for ts in self.node_dict[node].keys():
                num_links += len(self.node_dict[node][ts])
        print("Completeness check node_dict: EDGE NUM ", num_links)

    def _check_ts_dict_completeness(self) -> None:
        # only for dynamic graphs and cannot adapt to temporal knowledge graphs
        num_links = 0
        for ts in self.ts_dict.keys():
            for _, val in self.ts_dict[ts].items():
                num_links += len(val)
        print("Completeness check ts_dict: EDGE NUM ", num_links)

    def get_neighbor(self, 
                     nodes: List[int],
                     timestamps: List[int],
                     )-> Dict[int, Tuple[int, int]]:
        r"""return the current neighbors for the given nodes"""
        neighbors = {}
        for i, node in enumerate(nodes):
            # print(timestamps[i])
            # print([ts for ts in self.node_dict[node].keys()])
            sorted_ts = sorted([ts for ts in self.node_dict[node].keys()])
            # neighbor_ts = sorted_ts[:self.max_size]
            # print(sorted_ts)
            for j, t in enumerate(sorted_ts):
                if t == timestamps[i]:
                    # print(t)
                    neighbor_ts = sorted_ts[j-self.max_size:j] if j-self.max_size >= 0 else sorted_ts[:j]
                    # print(neighbor_ts)
                    break
            for ts in neighbor_ts:
                # print(ts, node)
                if node not in neighbors:
                    neighbors[node] = [(dst, ts) for dst in self.node_dict[node][ts]]
                else:
                    neighbors[node].extend([(dst, ts) for dst in self.node_dict[node][ts]])
        return neighbors
    
    def update(self,
               src: np.ndarray,
               dst: np.ndarray,
               ts: np.ndarray,
               )-> None:
        r"""update the neighbor tracker with new links"""
        # self._check_input(src, dst, ts)
        
        for i in range(src.shape[0]):
            if src[i] not in self.node_dict:
                self.node_dict[src[i]] = [(dst[i], ts[i])]
            else:
                self.node_dict[src[i]].append((dst[i], ts[i]))

class NeighborTrackerTPPR:
    def __init__(self,
                src: List[int],
                dst: List[int],
                ts: List[int],
                max_size: int = 20,
                mode: str = "link",
                dataname: str = "tgbl-wiki",
            ) -> None:
        r"""initialize the neighbor tracker"""
        self.node_dict = {}
        self.ts_dict = {}
        self.node_dict_inv = {}
        # self._check_input(src, dst, ts)
        self.max_size = max_size
        self.mode = mode

        path = f"processed_TG/{dataname}/"
        if not os.path.exists(path):
            os.makedirs(path)
        node_dict_path = f"{path}node_dict.pkl"

        if os.path.exists(node_dict_path):
            print(f"Loading node_dict from {node_dict_path}")
            with open(node_dict_path, 'rb') as f:
                self.node_dict = pickle.load(f)
        else:
            print(f"Creating node_dict and saving to {node_dict_path}")
            # src id -> ts id -> dst id(s)
            for i in range(len(src)):
                if src[i] not in self.node_dict:
                    self.node_dict[src[i]] = {}
                    self.node_dict[src[i]][ts[i]] = [dst[i]]
                else:
                    if ts[i] not in self.node_dict[src[i]]:
                        self.node_dict[src[i]][ts[i]] = [dst[i]]
                    else:
                        self.node_dict[src[i]][ts[i]].append(dst[i])
            with open(node_dict_path, 'wb') as f:
                pickle.dump(self.node_dict, f)

        # self._check_node_dict_completeness()

        # ts id -> (queries + answer_lists)
        # ts_dict stores ground truth and the final data format, split by timestamps
        ts_dict_path = f"{path}ts_dict.pkl"

        if os.path.exists(ts_dict_path):
            print(f"Loading ts_dict from {ts_dict_path}")
            with open(ts_dict_path, 'rb') as f:
                self.ts_dict = pickle.load(f)
        else:
            print(f"Creating ts_dict and saving to {ts_dict_path}")
            for i in range(len(src)):
                if ts[i] not in self.ts_dict:
                    self.ts_dict[ts[i]] = ddict(set)
                    self.ts_dict[ts[i]][(src[i], ts[i])].add(dst[i])
                else:
                    self.ts_dict[ts[i]][(src[i], ts[i])].add(dst[i])
            with open(ts_dict_path, 'wb') as f:
                pickle.dump(self.ts_dict, f)

        # self._check_ts_dict_completeness()

        # dst id -> ts id -> src id(s); this is to get the temporal neighbor where the target node is a dst node
        node_dict_inv_path = f"{path}node_dict_inv.pkl"

        if os.path.exists(node_dict_inv_path):
            print(f"Loading node_dict_inv from {node_dict_inv_path}")
            with open(node_dict_inv_path, 'rb') as f:
                self.node_dict_inv = pickle.load(f)
        else:
            print(f"Creating node_dict_inv and saving to {node_dict_inv_path}")
            for i in range(len(dst)):
                if dst[i] not in self.node_dict_inv:
                    self.node_dict_inv[dst[i]] = {}
                    self.node_dict_inv[dst[i]][ts[i]] = [src[i]]
                else:
                    if ts[i] not in self.node_dict_inv[dst[i]]:
                        self.node_dict_inv[dst[i]][ts[i]] = [src[i]]
                    else:
                        self.node_dict_inv[dst[i]][ts[i]].append(src[i])
            with open(node_dict_inv_path, 'wb') as f:
                pickle.dump(self.node_dict_inv, f)

    def _check_node_dict_completeness(self) -> None:
        num_links = 0
        for node in self.node_dict.keys():
            for ts in self.node_dict[node].keys():
                num_links += len(self.node_dict[node][ts])
        print("Completeness check node_dict: EDGE NUM ", num_links)

    def _check_ts_dict_completeness(self) -> None:
        # only for dynamic graphs and cannot adapt to temporal knowledge graphs
        num_links = 0
        for ts in self.ts_dict.keys():
            for _, val in self.ts_dict[ts].items():
                num_links += len(val)
        print("Completeness check ts_dict: EDGE NUM ", num_links)

    def get_neighbor(self, 
                     nodes: List[int],
                     timestamps: List[int],
                     )-> Dict[int, Tuple[int, int]]:
        r"""return the latest 1-hop neighbors for the given nodes, sorted according to time"""
        neighbors = {}
        for i, node in enumerate(nodes):
            if node not in self.node_dict:
                return {node: []}
            sorted_ts = sorted([ts for ts in self.node_dict[node].keys()])
            # print(sorted_ts)
            # print(nodes, timestamps)
            for j, t in enumerate(sorted_ts):
                if t == timestamps[i]:
                    neighbor_ts = sorted_ts[j-self.max_size:j] if j-self.max_size >= 0 else sorted_ts[:j]
                    break
                elif t > timestamps[i]:
                    # only when we do node feature prediction will we meet this case
                    neighbor_ts = sorted_ts[j-self.max_size:j] if j-self.max_size >= 0 else sorted_ts[:j]
                    break
                else:
                    continue

            if t < timestamps[i]:
                neighbor_ts = sorted_ts

            for ts in neighbor_ts:
                # print(ts, node)
                if node not in neighbors:
                    neighbors[node] = [(dst, ts) for dst in self.node_dict[node][ts]]
                else:
                    neighbors[node].extend([(dst, ts) for dst in self.node_dict[node][ts]])
        return neighbors

    def get_neighbor_inv(self, 
                         nodes: List[int],
                         timestamps: List[int],
                        )-> Dict[int, Tuple[int, int]]:
        r"""return the latest 1-hop neighbors for the given nodes, sorted according to time, inverse mode (given node as dst node)"""
        neighbors = {}
        for i, node in enumerate(nodes):
            if node not in self.node_dict_inv:
                return {node: []}
            sorted_ts = sorted([ts for ts in self.node_dict_inv[node].keys()])
            # print(sorted_ts)
            for j, t in enumerate(sorted_ts):
                # print(sorted_ts)
                # print(t, timestamps[i])
                if t == timestamps[i]:
                    neighbor_ts = sorted_ts[j-self.max_size:j] if j-self.max_size >= 0 else sorted_ts[:j]
                    break
                elif t > timestamps[i]:
                    # only when we do node feature prediction will we meet this case
                    neighbor_ts = sorted_ts[j-self.max_size:j] if j-self.max_size >= 0 else sorted_ts[:j]
                    break
                else:
                    continue

            if t < timestamps[i]:
                neighbor_ts = sorted_ts

            # print(t, timestamps[i])
            # print(neighbor_ts)
            for ts in neighbor_ts:
                # print(ts, node)
                if node not in neighbors:
                    neighbors[node] = [(src, ts) for src in self.node_dict_inv[node][ts]]
                else:
                    neighbors[node].extend([(src, ts) for src in self.node_dict_inv[node][ts]])
        return neighbors

    def get_neighbor_topk_tppr(self, target_node: int, target_timestamp: int, depth: int=2, alpha: int=0.3, beta: int=0.6, k: int=100):
        #initialize a dictionary to store TPPR scores for each (node, timestamp)
        tppr_dict = {}

        # get dictionary of neighbors
        query_list = []
        # initialize the score of the target node as 1.0, depth as 0 because it is 0 hop
        query_list.append((target_node, target_timestamp, 1.0, 0))

        # initialize a dictionary to keep track of the links connected to each (node, timestamp)
        temp_node2link = ddict(set)

        for dep in range(depth):
            # print("query_list: ", query_list)
            new_query_list = []

            # traverse the query list
            for query_node, query_timestamp, query_weight, query_depth in query_list:
                neighbors_ = self.get_neighbor([query_node], [query_timestamp])
                neighbors_inv = self.get_neighbor_inv([query_node], [query_timestamp])
                # print(neighbors_)

                # flattern the neighbors
                neighbors, edge_times = [], []
                for _, val in neighbors_.items(): # target node as src
                    neighbors.extend([pair[0] for pair in val])
                    edge_times.extend([pair[1] for pair in val])
                    for n, t in val:
                        temp_node2link[(n, t)].add((query_node, n, t))

                for _, val in neighbors_inv.items(): # target node as dst
                    neighbors.extend([pair[0] for pair in val])
                    edge_times.extend([pair[1] for pair in val])
                    for n, t in val:
                        temp_node2link[(n, t)].add((n, query_node, t))

                # print(neighbors)
                # print(edge_times)

                # compute total number of 1-hop neighbors of the target node
                n_ngh = len(neighbors)

                if n_ngh == 0:
                    continue
                else:
                    norm = beta / (1 - beta) * (1 - pow(beta, n_ngh))
                    weight = query_weight * (1 - alpha) * beta / norm * alpha if alpha != 0 and dep==0 else query_weight * (1 - alpha) * beta / norm

                    for z in range(n_ngh):
                        node = neighbors[-(z+1)]

                        # a temporal walk

                        timestamp = edge_times[-(z+1)]
                        state = (node, timestamp, dep+1)

                        # update dictionary
                        if state in tppr_dict:
                            tppr_dict[state] = tppr_dict[state] + weight
                        else:
                            tppr_dict[state] = weight

                        # update query list; it will be used later for the next step of walk
                        new_query = (node, timestamp, weight, dep+1)
                        new_query_list.append(new_query)

                        # update weight; weight is the TPPR score after we finish the walk
                        weight = weight * beta

            if len(new_query_list) == 0:
                break
            else:
                query_list = new_query_list
    
        # sort and get the top-k neighbors after the walk finishes
        tppr_neighbors = []
        tppr_size = len(tppr_dict)
        print("tppr_size: ", tppr_size)
        if tppr_size == 0:
            return {target_node: []}, set(), [], set()

        keys = list(tppr_dict.keys())
        values = np.array(list(tppr_dict.values()))
        # print(tppr_size, k)
        if tppr_size <= k:
            inds = np.arange(tppr_size)
        else:
            inds = np.argsort(values)[-k:]

        # also keep track of the nodes involved in tppr neighbors, regardless of timestamps
        involved_nodes = set()
        # get the temporal neighbors in the form of (node, timestamp)
        for j, ind in enumerate(inds):
            key = keys[ind]
            weight = values[ind]
            node = key[0]
            timestamp = key[1]
            depth = key[2]

            tppr_neighbors.append((node, timestamp, depth))
            involved_nodes.add(int(node))

        # get links according to temporal nodes
        tppr_links = []
        for temp_node in tppr_neighbors:
            tppr_links.extend(list(temp_node2link[(temp_node[0], temp_node[1])]))

        tppr_links_sorted = sorted(tppr_links, key=lambda x: x[2])
        # keep track of the nodes involved in the subgraph, not that some of them do not belong to tppr neighbors
        involved_nodes_subgraph = set()
        for link in tppr_links_sorted:
            involved_nodes_subgraph.add(int(link[0]))
            involved_nodes_subgraph.add(int(link[1]))

        # assert 0
        return {target_node: tppr_links_sorted}, involved_nodes, tppr_neighbors, involved_nodes_subgraph
    
    def update(self,
               src: np.ndarray,
               dst: np.ndarray,
               ts: np.ndarray,
               )-> None:
        r"""update the neighbor tracker with new links"""
        # self._check_input(src, dst, ts)
        
        for i in range(src.shape[0]):
            if src[i] not in self.node_dict:
                self.node_dict[src[i]] = [(dst[i], ts[i])]
            else:
                self.node_dict[src[i]].append((dst[i], ts[i]))