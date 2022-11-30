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
    nft: str
    price: float
    price_str: str

def currency_converter(data: List[NFTTransaction]) -> List[NFTTransaction]:
  for row in data:
    price = row.price_str

    if type(price) is not str:
      row.price = float(float(price) * 1.00)
      continue

    try:
      price, currency, _ = price.split(" ")
      price = price.replace(",", "")

      if currency == "ETH":
        row.price = float(float(price) * 1309.97)
      
      elif currency == "WETH":
        row.price = float(float(price) * 1322.16)

      elif currency == "ASH":
        row.price = float(float(price) * 0.9406)
      
      elif currency == "GALA":
        row.price = float(float(price) * 0.03748)
        
      elif currency == "TATR":
        row.price = float(float(price) * 0.012056)
        
      elif currency == "USDC":
        row.price = float(float(price) * 1.00)
        
      elif currency == "MANA":
        row.price = float(float(price) * 0.64205)
        
      elif currency == "SAND":
        row.price = float(float(price) * 0.7919)
        
      elif currency == "RARI":
        row.price = float(float(price) * 2.18)
        
      elif currency == "CTZN":
        row.price = float(float(price) * 0.00321)
        
      elif currency == "APE":
        row.price = float(float(price) * 4.62)

      else:
        row.price = float(float(price) * 1.00)

    except ValueError:
      None
      
  return data

def merge_sort_by_nft(A: List[NFTTransaction]) -> List[NFTTransaction]:
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_nft(B)
    R = merge_sort_by_nft(C)
    return merge_by_nft(L, R)

def merge_by_nft(L: List[NFTTransaction], R: List[NFTTransaction]) -> List[NFTTransaction]:
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and L[i].nft >= R[j].nft):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B

def prepare_data(data) -> List[NFTTransaction]:
    # data = data.reset_index()  # make sure indexes pair with number of rows

    transactions = []
    for i, row in data.iterrows():
        transactions.append(NFTTransaction(
            txn_hash=row['Txn Hash'], 
            time_stamp=row['UnixTimestamp'],
            date_time=row['Date Time (UTC)'],
            buyer=row['Buyer'],
            nft=row['NFT'],
            price_str=row['Price'],
            price=0.0))
        
    return transactions

def get_dataframe(data: List[NFTTransaction]):
  txns_list = []

  for row in data:
    dic = {
        'Txn Hash': row.txn_hash,
        'UnixTimestamp': row.time_stamp,
        'Date Time (UTC)': row.date_time,
        'Buyer': row.buyer,
        'NFT': row.nft,
        'Price': row.price
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  df.to_excel("sorted_data.xlsx")
  return df

def get_unique_buyers(data: List[NFTTransaction]) -> List[str]:
        unique_count = 0
        unique_buyers = []
        buyer_timestamps = []
        for row in data:
            if row.buyer not in unique_buyers:
                unique_count += 1
                unique_buyers.append(row.buyer)
                buyer_timestamps.append(row.time_stamp)

        return unique_count, unique_buyers, buyer_timestamps

class Graph:
    def __init__(self, temp_n_buyers, temp_buyers, buyer_timestamps) -> None:
        self.temp_n_buyers = temp_n_buyers
        self.temp_buyers = temp_buyers
        self.buyer_timestamps = buyer_timestamps

        self.n_buyers = 0
        self.buyers = []
        self.graph = defaultdict(list)

        self.adjacency_matrix = np.zeros((temp_n_buyers, temp_n_buyers), dtype=tuple)

        self.scc_adjacency_matrix = np.zeros((temp_n_buyers, temp_n_buyers), dtype=tuple)

    def addEdge(self, node1, node2):
        self.graph[node1].append(node2)

    def add_to_buyers_list(self, buyer):
        if buyer not in self.buyers:
            self.buyers.append(buyer)

    def build(self, data: List[NFTTransaction]) -> None:
        self.data = data

        with open("original_adjacency_matrix.txt", "w") as file:
            for i in range(1, len(self.data)): #also consider adding a logic to prevent a scenario where a relationship would exist for two buyers for the a different txns
                if self.data[i-1].nft == self.data[i].nft and self.data[i-1].buyer != self.data[i].buyer:
                    # index1 = self.temp_buyers.index(self.data[i-1].buyer)
                    # index2 = self.temp_buyers.index(self.data[i].buyer)

                    # Saves the transaction in the adjacency matrix
                    # self.adjacency_matrix[index1][index2] = (self.data[i].nft, self.data[i].price)
                    
                    self.add_to_buyers_list(self.data[i-1].buyer)
                    self.add_to_buyers_list(self.data[i].buyer)

                    # Create an edge to build the graph
                    self.addEdge(self.buyers.index(self.data[i-1].buyer), self.buyers.index(self.data[i].buyer))

                    # file.writelines(f'{self.data[i-1].buyer} - {self.data[i].buyer} -> {str(self.adjacency_matrix[index1][index2])} \n')

                elif self.data[i-1].nft == self.data[i].nft and self.data[i-1].buyer == self.data[i].buyer and i + 1 < len(self.data) and self.data[i].buyer != self.data[i+1].buyer:
                    # index1 = self.temp_buyers.index(self.data[i].buyer)
                    # index2 = self.temp_buyers.index(self.data[i+1].buyer)

                    # Saves the transaction in the adjacency matrix
                    # self.adjacency_matrix[index1][index2] = (self.data[i+1].nft, self.data[i+1].price)

                    self.add_to_buyers_list(self.data[i].buyer)
                    self.add_to_buyers_list(self.data[i+1].buyer)

                    # Create an edge to build the graph
                    self.addEdge(self.buyers.index(self.data[i].buyer), self.buyers.index(self.data[i+1].buyer))

                    # file.writelines(f'{self.data[i].buyer} - {self.data[i+1].buyer} -> {str(self.adjacency_matrix[index1][index2])} \n')

                
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

    def dfs(self, start, visited, file):
        visited[start] = True

        # Save the entry in an adjacency matrix
        file.writelines(f"({self.temp_buyers[start]}, {self.buyer_timestamps[start]})\n")

        for i in self.graph[start]:
            if not visited[i]:
                self.scc_adjacency_matrix[start][i] = 1
                self.dfs(i, visited, file)

    def get_transpose(self):
        graph = Graph(self.temp_n_buyers, self.temp_buyers, self.buyer_timestamps)

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

    def strongly_connected_components(self):
        order = [] * self.n_buyers

        visited = [False] * self.n_buyers

        for i in range(self.n_buyers):
            if not visited[i]:
                self.scc_dfs(i, visited, order)

        transposed_graph = self.get_transpose()
        visited = [False] * self.n_buyers

        with open("scc_adjacency_matrix.txt", "w") as file:
          while order:
              i = order.pop()
              if not visited[i]:
                  transposed_graph.dfs(i, visited, file)
                  # print("")


if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    data = pd.read_csv("dataset.csv")
    nft_txns = prepare_data(data)

    nft_txns = currency_converter(nft_txns)

    nft_txns = merge_sort_by_nft(nft_txns)

    temp_n_buyers, temp_buyers, buyer_timestamps = get_unique_buyers(nft_txns)
    graph = Graph(temp_n_buyers, temp_buyers, buyer_timestamps)

    start_time = time.time_ns()
    graph.build(nft_txns)
    end_time = time.time_ns()
    elapsed_time = (end_time - start_time)/1e9
    print(f'The time taken to build the graph is {elapsed_time} secs\n')

    # start_time = time.time_ns()
    # graph.topological_sort()
    # end_time = time.time_ns()
    # elapsed_time = (end_time - start_time)/1e9
    # print(f'The time taken to perform topological sorting is {elapsed_time} secs\n')

    start_time = time.time_ns()
    graph.strongly_connected_components()
    end_time = time.time_ns()
    elapsed_time = (end_time - start_time)/1e9
    print(f'The time taken to perform SCC is {elapsed_time} secs\n')