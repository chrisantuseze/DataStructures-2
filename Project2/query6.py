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

    nft_txns = data[['Txn Hash', 'UnixTimestamp', 'Date Time (UTC)', 'Buyer', 'Token ID', 'Price']]

    # nft_txns = nft_txns.iloc[0:5000]

    nft_txns = currency_converter(nft_txns)
    unique_buyer_txns = nft_txns.groupby('Buyer', as_index=False).first()
    nft_txns = nft_txns.sort_values(by=['Token ID', 'Date Time (UTC)'], ascending=[False, True])
    nft_txns = convert_to_object_list(nft_txns)

    unique_buyers = unique_buyer_txns['Buyer']
    n_unique_buyers = len(unique_buyers)
    buyer_timestamps = unique_buyer_txns['Date Time (UTC)']
    token_ids = unique_buyer_txns['Token ID']

    return nft_txns, unique_buyer_txns, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids

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

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_6.png", rows=92):
    x_axis = [i for i in range(rows+1)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes[0], color ='blue')
    plt.plot(x_axis, actual_runtimes[1], color ='orange')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Query 6 Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime', 'Min ST Runtime', 'Max ST Runtime'], loc='upper left')

    plt.savefig(filename)
    plt.show()


class Graph:
    def __init__(self, unique_buyer_txns, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids, save=False) -> None:
        self.unique_buyer_txns = unique_buyer_txns

        self.unique_buyers = unique_buyers
        self.n_unique_buyers = n_unique_buyers
        self.buyer_timestamps = buyer_timestamps
        self.token_ids = token_ids

        self.buyers = []
        self.graph = []
        self.save = save

        self.adjacency_matrix = np.zeros((self.n_unique_buyers, self.n_unique_buyers), dtype=tuple)

    def addEdge(self, node1, node2, price):
        self.graph.append([node1, node2, price])

    def add_to_buyers_list(self, buyer):
        if buyer not in self.buyers:
            self.buyers.append(buyer)

    def build(self, data: List[NFTTransaction]) -> None:
        count = 0
        adjacency_graph = []
        for i in range(1, len(data)):
          if data[i-1].token_id == data[i].token_id and data[i-1].buyer != data[i].buyer:
            self.buyers.insert(count, data[i-1].buyer)
            self.buyers.insert(count + 1, data[i].buyer)
            self.addEdge(count, count + 1, float(data[i].price_str))
            count += 2

            adjacency_graph.append(f"{data[i-1].buyer} - {data[i].buyer} -> [{data[i].token_id, data[i].price_str, data[i].date_time}] \n")

          elif data[i-1].token_id == data[i].token_id and data[i-1].buyer == data[i].buyer and i + 1 < len(data) and data[i].buyer != data[i+1].buyer:
            self.buyers.insert(count, data[i].buyer)
            self.buyers.insert(count + 1, data[i+1].buyer)
            self.addEdge(count, count + 1, float(data[i+1].price_str))
            count += 2

            adjacency_graph.append(f"{data[i].buyer} - {data[i+1].buyer} -> [{data[i+1].token_id, data[i+1].price_str, data[i+1].date_time}] \n")

        if self.save:
          with open(output_path + "/original_adjacency_matrix.txt", "w") as file:
            file.writelines(adjacency_graph)
                
     # Search function
    def find(self, parent, i):
        try:
          if parent[i] != i:
            return self.find(parent, parent[i])
        except:
              pass

        return i

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        try:
          if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot

          elif rank[xroot] > rank[yroot]:
              parent[yroot] = xroot

          else:
              parent[yroot] = xroot
              rank[xroot] += 1

        except:
          pass

    #  Applying Kruskal algorithm
    def kruskal(self, filename, is_reverse):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2], reverse=is_reverse)
        parent = []
        rank = []
        for node in range(len(self.graph) + 1):
            parent.append(node)
            rank.append(0)

        while e < len(self.graph) - 1:
            try:
              node1, node2, price = self.graph[i]
            except:
              pass

            i = i + 1
            x = self.find(parent, node1)
            y = self.find(parent, node2)

            if x != y:
                e = e + 1
                result.append([node1, node2, price])
                self.apply_union(parent, rank, x, y)


        output = []
        tree = "maximum" if is_reverse else "minimum"
        output.append(f"\nThe following are vertices and edges of the constructed {tree} spanning tree:\n")
        output.append("------------------------------------------------------------------------------\n\n")
        cost = 0
        for node1, node2, price in result:
            # print("%d - %d: %d" % (node1, node2, price))

            buyer1 = self.buyers[node1]
            buyer2 = self.buyers[node2]
            buyer2_row = self.unique_buyer_txns.loc[self.unique_buyer_txns['Buyer'] == buyer2]

            # Save the entry in an adjacency matrix
            cost += price
            try:
              output.append(f"({buyer1} - {buyer2} -> [{buyer2_row.iloc[0]['Token ID']}, {buyer2_row.iloc[0]['Price']}, {buyer2_row.iloc[0]['Date Time (UTC)']}])\n\n")
        
            except:
              pass

        output.append("------------------------------------------------------------------------------\n")
        output.append(f"The {tree} cost is {cost}\n")
        output.append("------------------------------------------------------------------------------\n")

        if self.save:
          with open(output_path + "/" + filename, "w") as file:
            file.writelines(output)
          

    def kruskal_min_st(self):
        self.kruskal("min_st_adjacency_matrix.txt", False)

    def kruskal_max_st(self):
        self.kruskal("max_st_adjacency_matrix.txt", True)

def run_query(nft_txns, unique_buyer_txns, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids, save):
    graph = Graph(unique_buyer_txns, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids, save)

    start_time = time.time_ns()
    graph.build(nft_txns)
    end_time = time.time_ns()
    elapsed_time1 = (end_time - start_time)/1e9
    # print(f'The time taken to build the graph is {elapsed_time1} secs\n')

    start_time = time.time_ns()
    graph.kruskal_min_st()
    end_time = time.time_ns()
    elapsed_time2 = (end_time - start_time)/1e9

    start_time = time.time_ns()
    graph.kruskal_max_st()
    end_time = time.time_ns()
    elapsed_time3 = (end_time - start_time)/1e9

    return elapsed_time1, elapsed_time2, elapsed_time3

if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    global root_path
    root_path = os.getcwd()

    # Output path to store results
    global output_path
    output_path = root_path + "/output"

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    nft_txns, unique_buyer_txns, unique_buyers, n_unique_buyers, buyer_timestamps, token_ids = get_and_prepare_data()

    min_st_elapsed_times = []
    max_st_elapsed_times = []
    asymptotic_times = []
    rows = int(len(nft_txns) / 1000)

    # Run in intervals of 1000, 2000, 3000, ......
    for i in range(rows + 1):
        n = (i + 1) * 1000
        build_elapsed_time, min_st_run_elapsed_time, max_st_run_elapsed_time = run_query(
            nft_txns[0: n], unique_buyer_txns[0: n], unique_buyers[0: n], 
            n_unique_buyers=len(unique_buyers[0: n]), buyer_timestamps=buyer_timestamps[0: n], 
            token_ids=token_ids[0: n], save=(i == rows)
          )
        min_st_elapsed_times.append(min_st_run_elapsed_time)
        max_st_elapsed_times.append(max_st_run_elapsed_time)

        asymptotic_times.append(n * 2) #TODO change this to the actual asymptotic time (build time + SCC time)
        print(f'The total time taken to for both minimum and maximum spanning tree for {n} transactions is {min_st_run_elapsed_time + max_st_run_elapsed_time} secs\n')

    #Note since the two runtimes would have close values, one many superimpose on the other. It is not a bug. Although it won't happen once everything is done
    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=[min_st_elapsed_times, max_st_elapsed_times], filename=output_path + "/query_6.png", rows=rows)