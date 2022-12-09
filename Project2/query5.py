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

    # nft_txns = nft_txns.iloc[0:5000]

    nft_txns = currency_converter(nft_txns)
    unique_buyer_txns = nft_txns.groupby('Buyer', as_index=False).first()
    nft_txns = nft_txns.sort_values(by=['NFT', 'Token ID', 'UnixTimestamp'], ascending=[False, False, True])
    nft_txns = convert_to_object_list(nft_txns)

    unique_buyers = unique_buyer_txns['Buyer']
    n_unique_buyers = len(unique_buyers)
    buyer_timestamps = unique_buyer_txns['Date Time (UTC)']
    token_ids = unique_buyer_txns['Token ID']

    return nft_txns, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids

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

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_5.png", rows=92):
    x_axis = [i for i in range(rows+1)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Query 5 Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x5000)', 'Actual Runtime'], loc='upper left')

    plt.savefig(filename)
    # plt.show()

class Graph:
    def __init__(self, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids, save=False) -> None:
        self.unique_buyers = unique_buyers
        self.n_unique_buyers = n_unique_buyers
        self.buyer_timestamps = buyer_timestamps
        self.token_ids = token_ids

        self.n_buyers = 0
        self.buyers = []
        self.graph = defaultdict(list)

        self.scc_output = []
        self.save = save
        self.adjacency_graph = []

    def addEdge(self, node1, node2):
        self.graph[node1].append(node2)

    def add_to_buyers_list(self, buyer):
        if buyer not in self.buyers:
            self.buyers.append(buyer)

    def build(self, data: List[NFTTransaction]) -> None:
        for i in range(1, len(data)):
          if data[i-1].token_id == data[i].token_id and data[i-1].buyer != data[i].buyer:
            self.add_to_buyers_list(data[i-1].buyer)
            self.add_to_buyers_list(data[i].buyer)

            # Create an edge to build the graph
            self.addEdge(self.buyers.index(data[i-1].buyer), self.buyers.index(data[i].buyer))
            self.adjacency_graph.append(f"{data[i-1].buyer} - {data[i].buyer} -> [{data[i].token_id, data[i].price_str, data[i].date_time}] \n")

          elif data[i-1].token_id == data[i].token_id and data[i-1].buyer == data[i].buyer and i + 1 < len(data) and data[i].buyer != data[i+1].buyer:
            self.add_to_buyers_list(data[i].buyer)
            self.add_to_buyers_list(data[i+1].buyer)

            # Create an edge to build the graph
            self.addEdge(self.buyers.index(data[i].buyer), self.buyers.index(data[i+1].buyer))
            self.adjacency_graph.append(f"{data[i].buyer} - {data[i+1].buyer} -> [{data[i+1].token_id, data[i+1].price_str, data[i+1].date_time}] \n")
            
        self.n_buyers = len(self.buyers)

    def top_sort_dfs(self, start, visited, order):
        visited[start] = True

        for i in self.graph[start]:
            if not visited[i]:
                self.top_sort_dfs(i, visited, order)

        order.insert(0, start)

    def topological_sort(self) -> List[int]:
        visited = [False] * self.n_buyers
        order = [0] * self.n_buyers

        for i in range(self.n_buyers):
            if not visited[i]:
                self.top_sort_dfs(i, visited, order)
        
        return order

    def dfs(self, start, visited):
        visited[start] = True
        
        self.scc_output.append(f"({self.unique_buyers[start]}, {self.buyer_timestamps[start], self.token_ids[start]})")

        for i in self.graph[start]:
            if not visited[i]:
                self.dfs(i, visited)

    def get_transpose(self):
        graph = Graph(self.unique_buyers, self.n_unique_buyers, self.buyer_timestamps, self.token_ids)

        for i in self.graph:
            for j in self.graph[i]:
                graph.addEdge(j, i)

        return graph

    def scc_dfs(self, start, visited, order):
        visited[start] = True

        for i in self.graph[start]:
            if not visited[i]:
                self.scc_dfs(i, visited, order)
        
        order.append(start)

    def strongly_connected_components(self) -> list:
        order = [] * self.n_buyers

        visited = [False] * self.n_buyers

        for i in range(self.n_buyers):
            if not visited[i]:
                self.scc_dfs(i, visited, order)

        transposed_graph = self.get_transpose()
        visited = [False] * self.n_buyers

        count = 0
        while order:
            i = order.pop()
            if not visited[i]:
                transposed_graph.scc_output.append(f"SCC {count}")
                transposed_graph.dfs(i, visited)
                transposed_graph.scc_output.append("")

                count += 1

        sccs = []
        for row in transposed_graph.scc_output:
          sccs.append(f"{row}\n")

        return sccs

def run_query(nft_txns, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids, save):
    graph = Graph(unique_buyers, n_unique_buyers, buyer_timestamps, token_ids, save)

    start_time = time.time_ns()
    graph.build(nft_txns)
    end_time = time.time_ns()
    elapsed_time1 = (end_time - start_time)

    start_time = time.time_ns()
    sccs = graph.strongly_connected_components()
    end_time = time.time_ns()
    elapsed_time2 = (end_time - start_time)

    return elapsed_time1, elapsed_time2, graph.adjacency_graph, sccs

if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    global root_path
    root_path = os.getcwd()

    # Output path to store results
    global output_path
    output_path = root_path + "/output"

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    nft_txns, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids = get_and_prepare_data()

    elapsed_times = []
    asymptotic_times = []
    rows = int(len(nft_txns) / 1000)

    # Run in intervals of 1000, 2000, 3000, ......
    for i in range(rows + 1):
        n = (i + 1) * 1000
        build_elapsed_time_ns, run_elapsed_time_ns, adjacency_graph, sccs = run_query(
            nft_txns[0: n], unique_buyers[0: n], 
            n_unique_buyers=len(unique_buyers[0: n]), buyer_timestamps=buyer_timestamps[0: n], 
            token_ids=token_ids[0: n], save=(i == rows)
          )
        elapsed_times.append(run_elapsed_time_ns)

        #  O(V + E)
        asymptotic_run_time = (n + len(adjacency_graph))
        asymptotic_run_time *= 5000 # This ensures that both are on the same scale
        asymptotic_times.append(asymptotic_run_time)
        print(f'The total time taken to build graph and perform SCC for {n} transactions is {run_elapsed_time_ns/1e9} secs\n')

        if i == rows:
          with open(output_path + "/query5_original_adjacency_matrix.txt", "w") as file:
            file.writelines("Buyer 1 ------------------------------------> Buyer 2 ------------------------------------> Token ID ------> Price -------> Timestamp\n\n")
            file.writelines(adjacency_graph)

          with open(output_path + "/scc_adjacency_matrix.txt", "w") as file:
            file.writelines(sccs)

    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_times, filename=output_path+"/query_5.png", rows=rows)
