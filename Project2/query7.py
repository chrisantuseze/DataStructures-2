#!/bin/env python

import sys
import os
import math
import time
import pandas as pd 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import List

@dataclass(order=True)
class NFTTransaction:
    txn_hash: str
    time_stamp: str
    date_time: str
    buyer: str
    token_id: str
    price: float
    price_str: str

def currency_converter(data):
  for row in data.itertuples():
    price = data.at[row.Index, 'Price']

    if type(price) is not str:
      data.at[row.Index, 'Price'] = float(float(price) * 1.00)
      continue

    try:
      price, currency, _ = price.split(" ")
      price = price.replace(",", "")

      if currency == "ETH":
        data.at[row.Index, 'Price'] = float(float(price) * 1309.97)
      
      elif currency == "WETH":
        data.at[row.Index, 'Price'] = float(float(price) * 1322.16)

      elif currency == "ASH":
        data.at[row.Index, 'Price'] = float(float(price) * 0.9406)
      
      elif currency == "GALA":
        data.at[row.Index, 'Price'] = float(float(price) * 0.03748)
        
      elif currency == "TATR":
        data.at[row.Index, 'Price'] = float(float(price) * 0.012056)
        
      elif currency == "USDC":
        data.at[row.Index, 'Price'] = float(float(price) * 1.00)
        
      elif currency == "MANA":
        data.at[row.Index, 'Price'] = float(float(price) * 0.64205)
        
      elif currency == "SAND":
        data.at[row.Index, 'Price'] = float(float(price) * 0.7919)
        
      elif currency == "RARI":
        data.at[row.Index, 'Price'] = float(float(price) * 2.18)
        
      elif currency == "CTZN":
        data.at[row.Index, 'Price'] = float(float(price) * 0.00321)
        
      elif currency == "APE":
        data.at[row.Index, 'Price'] = float(float(price) * 4.62)

      else:
        data.at[row.Index, 'Price'] = float(float(price) * 1.00)

    except ValueError:
      None
      
  return data

def get_and_prepare_data():
    data = pd.read_csv("dataset.csv")
    data = data.dropna()
    data = data.drop_duplicates()

    nft_txns = data[['Txn Hash', 'UnixTimestamp', 'Date Time (UTC)', 'Buyer', 'Token ID', 'NFT', 'Price']]

    # nft_txns = nft_txns.iloc[0:50000]

    nft_txns = currency_converter(nft_txns)

    nft_txns = nft_txns.sort_values(by=['NFT', 'Token ID', 'UnixTimestamp'], ascending=[False, False, True])

    nft_txns = convert_to_object_list(nft_txns)

    return nft_txns

def convert_to_object_list(data) -> List[NFTTransaction]:
    transactions = []
    for i, row in data.iterrows():
        transactions.append(NFTTransaction(
            txn_hash=row['Txn Hash'], 
            time_stamp=row['UnixTimestamp'],
            date_time=row['Date Time (UTC)'],
            buyer=row['Buyer'],
            token_id=row['Token ID'],
            price_str=row['Price'],
            price=0.0))
        
    return transactions

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_7.png", rows=92):
    x_axis = [i for i in range(rows+1)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Query 7 Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x5000)', 'Actual Runtime'], loc='upper left')

    plt.savefig(filename)
    # plt.show()

class Node:
    def __init__(self, buyer_id, token_id, date_time, indexloc = None):
        self.buyer_id = buyer_id
        self.token_id = token_id
        self.date_time = date_time
        self.index = indexloc


class DijkstraNodeDecorator:
    def __init__(self, node: Node):
        self.node = node
        self.prov_dist = float('inf')
        self.hops = []

    def index(self):
        return self.node.index

    def buyer_id(self):
        return self.node.buyer_id

    def token_id(self):
        return self.node.token_id

    def date_time(self):
        return self.node.date_time
    
    def update_data(self, data):
        self.prov_dist = data['prov_dist']
        self.hops = data['hops']
        return self


class BinaryTree:

    def __init__(self, nodes = []):
        self.nodes = nodes

    def root(self):
        return self.nodes[0]
    
    def iparent(self, i):
        return (i - 1) // 2
    
    def ileft(self, i):
        return 2*i + 1

    def iright(self, i):
        return 2*i + 2

    def left(self, i):
        return self.node_at_index(self.ileft(i))
    
    def right(self, i):
        return self.node_at_index(self.iright(i))

    def parent(self, i):
        return self.node_at_index(self.iparent(i))

    def node_at_index(self, i):
        return self.nodes[i]

    def size(self):
        return len(self.nodes)

class MinHeap(BinaryTree):
    def __init__(self, nodes: List[DijkstraNodeDecorator], is_less_than = lambda a, b: a < b, get_index = None, update_node = lambda node, newval: newval):
        BinaryTree.__init__(self, nodes)
        self.order_mapping = list(range(len(nodes)))
        self.is_less_than, self.get_index, self.update_node = is_less_than, get_index, update_node
        self.min_heapify()

    # Heapify at a node assuming all subtrees are heapified
    def min_heapify_subtree(self, i):
        size = self.size()
        ileft = self.ileft(i)
        iright = self.iright(i)
        imin = i
        if( ileft < size and self.is_less_than(self.nodes[ileft], self.nodes[imin])):
            imin = ileft
        if( iright < size and self.is_less_than(self.nodes[iright], self.nodes[imin])):
            imin = iright
        if( imin != i):
            self.nodes[i], self.nodes[imin] = self.nodes[imin], self.nodes[i]
            # If there is a lambda to get absolute index of a node
            # update your order_mapping array to indicate where that index lives in the nodes array (so lookup by this index is O(1))
            if self.get_index is not None:
                self.order_mapping[self.get_index(self.nodes[imin])] = imin
                self.order_mapping[self.get_index(self.nodes[i])] = i
            self.min_heapify_subtree(imin)


    # Heapify an un-heapified array
    def min_heapify(self):
        for i in range(len(self.nodes), -1, -1):
            self.min_heapify_subtree(i)

    def min(self):
        return self.nodes[0]

    def pop(self):
        min = self.nodes[0]
        if self.size() > 1:
            self.nodes[0] = self.nodes[-1]
            self.nodes.pop()

            # Update order_mapping if applicable
            if self.get_index is not None:
                self.order_mapping[self.get_index(self.nodes[0])] = 0
            self.min_heapify_subtree(0)
        elif self.size() == 1: 
            self.nodes.pop()
        else:
            return None

        # If self.get_index exists, update self.order_mapping to indicate the node of this index is no longer in the heap
        if self.get_index is not None:
            # Set value in self.order_mapping to None to indicate it is not in the heap
            self.order_mapping[self.get_index(min)] = None
        return min

    # Update node value, bubble it up as necessary to maintain heap property
    def decrease_key(self, i, val):
        self.nodes[i] = self.update_node(self.nodes[i], val)
        iparent = self.iparent(i)
        while( i != 0 and not self.is_less_than(self.nodes[iparent], self.nodes[i])):
            self.nodes[iparent], self.nodes[i] = self.nodes[i], self.nodes[iparent]
            # If there is a lambda to get absolute index of a node
            # update your order_mapping array to indicate where that index lives
            # in the nodes array (so lookup by this index is O(1))
            if self.get_index is not None:
                self.order_mapping[self.get_index(self.nodes[iparent])] = iparent
                self.order_mapping[self.get_index(self.nodes[i])] = i
            i = iparent
            iparent = self.iparent(i) if i > 0 else None

    def index_of_node_at(self, i):
        return self.get_index(self.nodes[i])

class Graph(): 
    def __init__(self):
        self.graph = {}

        self.adjacency_graph = []
        self.buyers = {}

        self.src_node = None
        self.is_first = True


    def connect_dir(self, node1: Node, node2: Node, weight = 1):
        if node1.buyer_id not in self.graph:
            if node1.buyer_id not in self.buyers:
                node1.index = len(self.buyers)
                self.buyers[node1.buyer_id] = node1

            if node2.buyer_id not in self.buyers:
                node2.index = len(self.buyers)
                self.buyers[node2.buyer_id] = node2

            if self.is_first:
                self.src_node = node1
                self.is_first = False
                
            self.graph[node1.buyer_id] = [(node1, 0), (node2, weight)]
        else:
            if node2.buyer_id not in self.buyers:
                node2.index = len(self.buyers)
                self.buyers[node2.buyer_id] = node2

            if node2.index is not None:
                self.graph[node1.buyer_id].append((node2, weight))

    def connections(self, node: Node):
        if node.buyer_id not in self.graph:
            return [(node, 0)]

        return self.graph[node.buyer_id]
            
    def build(self, data: List[NFTTransaction]) -> None:
        for i in range(1, len(data)):
            if data[i-1].token_id == data[i].token_id and data[i-1].buyer != data[i].buyer:
                node1 = Node(data[i-1].buyer, data[i-1].token_id, data[i].date_time)
                node2 = Node(data[i].buyer, data[i].token_id, data[i].date_time)
                self.connect_dir(node1, node2, weight=data[i].price_str)

                self.adjacency_graph.append(f"{data[i-1].buyer} - {data[i].buyer} -> [{data[i].token_id, data[i].price_str, data[i].date_time}] \n")

            elif data[i-1].token_id == data[i].token_id and data[i-1].buyer == data[i].buyer and i + 1 < len(data) and data[i].buyer != data[i+1].buyer:
                node1 = Node(data[i].buyer, data[i].token_id, data[i+1].date_time)
                node2 = Node(data[i+1].buyer, data[i+1].token_id, data[i+1].date_time)
                self.connect_dir(node1, node2, weight=data[i+1].price_str)

                self.adjacency_graph.append(f"{data[i].buyer} - {data[i+1].buyer} -> [{data[i+1].token_id, data[i+1].price_str, data[i+1].date_time}] \n")
                
                
    def dijkstras_shortest_path(self):
        src_index = self.src_node.index

        # Map nodes to DijkstraNodeDecorators. This will initialize all provisional distances to infinity
        dnodes = []
        for node_edges in self.buyers.values():
            if node_edges.index is not None:
                dnodes.append(DijkstraNodeDecorator(node_edges))
        
        # Set the source node provisional distance to 0 and its hops array to its node
        dnodes[src_index].prov_dist = 0
        dnodes[src_index].hops.append(dnodes[src_index].node)
        
        # Set up all heap customization methods
        is_less_than = lambda a, b: a.prov_dist < b.prov_dist
        get_index = lambda node: node.index()
        update_node = lambda node, data: node.update_data(data)

        #Instantiate heap to work with DijkstraNodeDecorators as the hep nodes
        heap = MinHeap(dnodes, is_less_than, get_index, update_node)

        min_dist_list = []

        while heap.size() > 0:
            # Get node in heap that has not yet been "seen"
            # that has smallest distance to starting node
            min_decorated_node = heap.pop()
            min_dist = min_decorated_node.prov_dist
            hops = min_decorated_node.hops
            if not math.isinf(min_dist):
                min_dist_list.append([min_dist, hops])
            
            # Get all next hops. This is no longer an O(n^2) operation
            connections = self.connections(min_decorated_node.node)
            # For each connection, update its path and total distance from 
            # starting node if the total distance is less than the current distance
            # in dist array
            for (inode, weight) in connections: 
                # node = self.graph[inode.buyer_id]
                if inode.index is None:
                    # print(inode.index, inode.buyer_id)
                    continue

                heap_location = heap.order_mapping[inode.index]
                if(heap_location is not None):
                    tot_dist = weight + min_dist
                    if tot_dist < heap.nodes[heap_location].prov_dist:
                        hops_cpy = list(hops)
                        hops_cpy.append(inode)
                        data = {'prov_dist': tot_dist, 'hops': hops_cpy}
                        heap.decrease_key(heap_location, data)

        return min_dist_list 

def run_query(nft_txns):
    graph = Graph()

    start_time = time.time_ns()
    graph.build(nft_txns)
    end_time = time.time_ns()
    elapsed_time1 = (end_time - start_time)

    start_time = time.time_ns()

    shortest_paths = []
    for (weight, node) in graph.dijkstras_shortest_path():
        node_path = []
        for n in node:
            item = {
                "Buyer ID": n.buyer_id,
                "Token ID": n.token_id
            }
            node_path.append(item)

        entry = {
            "Weight": weight,
            "Path": node_path
        }
        shortest_paths.append(f"{entry} \n")

    end_time = time.time_ns()
    elapsed_time2 = (end_time - start_time)

    return elapsed_time1, elapsed_time2, graph.adjacency_graph, shortest_paths

if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    global root_path
    root_path = os.getcwd()

    # Output path to store results
    global output_path
    output_path = root_path + "/output"

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    nft_txns = get_and_prepare_data()

    elapsed_times = []
    asymptotic_times = []
    rows = int(len(nft_txns) / 1000)

    # Run in intervals of 1000, 2000, 3000, ......
    for i in range(rows + 1):
        n = (i + 1) * 1000
        build_elapsed_time_ns, run_elapsed_time_ns, adjacency_graph, shortest_paths = run_query(nft_txns[0: n])
        elapsed_times.append(run_elapsed_time_ns)

        #  O(ElogV)
        asymptotic_run_time = (n + len(adjacency_graph) * np.log10(n))
        asymptotic_run_time *= 1000 # This ensures that both are on the same scale
        asymptotic_times.append(asymptotic_run_time) 
        print(f'The total time taken to build graph and perform SCC for {n} transactions is {run_elapsed_time_ns/1e9} secs\n')

        if i == rows:
          with open(output_path + "/query7_original_adjacency_matrix.txt", "w") as file:
            file.writelines("Buyer 1 ------------------------------------> Buyer 2 ------------------------------------> Token ID ------> Price -------> Timestamp\n\n")
            file.writelines(adjacency_graph)

          with open(output_path + "/shortest_paths_adjacency_matrix.txt", "w") as file:
            file.writelines(f"The shortest paths of the connnected nodes from the source are: \n\n")
            file.writelines(shortest_paths)

    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_times, filename=output_path+"/query_7.png", rows=rows)