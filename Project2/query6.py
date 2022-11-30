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

# Kruskal's algorithm in Python
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

        # self.n_buyers = 0
        # self.buyers = []
        self.buyers = []
        self.graph = []

        self.adjacency_matrix = np.zeros((temp_n_buyers, temp_n_buyers), dtype=tuple)

        self.scc_adjacency_matrix = np.zeros((temp_n_buyers, temp_n_buyers), dtype=tuple)

    def addEdge(self, node1, node2, price):
        self.graph.append([node1, node2, price])

    def add_to_buyers_list(self, buyer):
        if buyer not in self.buyers:
            self.buyers.append(buyer)

    def build(self, data: List[NFTTransaction]) -> None:
        self.data = data

        with open("original_adjacency_matrix.txt", "w") as file:
            count = 0
            for i in range(1, len(self.data)): #also consider adding a logic to prevent a scenario where a relationship would exist for two buyers for the a different txns
                if self.data[i-1].nft == self.data[i].nft and self.data[i-1].buyer != self.data[i].buyer:
                    # index1 = self.temp_buyers.index(self.data[i-1].buyer)
                    # index2 = self.temp_buyers.index(self.data[i].buyer)

                    # self.adjacency_matrix[index1][index2] = (self.data[i].nft, self.data[i].price)
                    
                    # self.add_to_buyers_list(self.data[i-1].buyer)
                    # self.add_to_buyers_list(self.data[i].buyer)

                    self.buyers.insert(count, self.data[i-1].buyer)
                    self.buyers.insert(count + 1, self.data[i].buyer)
                    self.addEdge(count, count + 1, self.data[i].price)

                    count += 1

                    # file.writelines(f'{self.data[i-1].buyer} - {self.data[i].buyer} -> {str(self.adjacency_matrix[index1][index2])} \n')

                elif self.data[i-1].nft == self.data[i].nft and self.data[i-1].buyer == self.data[i].buyer and i + 1 < len(self.data) and self.data[i].buyer != self.data[i+1].buyer:
                    # index1 = self.temp_buyers.index(self.data[i].buyer)
                    # index2 = self.temp_buyers.index(self.data[i+1].buyer)

                    # self.adjacency_matrix[index1][index2] = (self.data[i].nft, self.data[i].price)

                    # self.add_to_buyers_list(self.data[i-1].buyer)
                    # self.add_to_buyers_list(self.data[i].buyer)

                    self.buyers.insert(count, self.data[i-1].buyer)
                    self.buyers.insert(count + 1, self.data[i].buyer)
                    self.addEdge(count, count + 1, self.data[i].price)

                    count += 1

                    # file.writelines(f'{self.data[i].buyer} - {self.data[i+1].buyer} -> {str(self.adjacency_matrix[index1][index2])} \n')

                
        # self.n_buyers = len(self.buyers)

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
    def kruskal_min_st(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
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



        with open("min_st_adjacency_matrix.txt", "w") as file:
          for node1, node2, price in result:
              # print("%d - %d: %d" % (node1, node2, price))

              buyer1 = self.buyers[node1]
              buyer2 = self.buyers[node2]

              index1 = self.temp_buyers.index(buyer1)
              index2 = self.temp_buyers.index(buyer2)

              # Save the entry in an adjacency matrix
              file.writelines(f"({buyer1}, {self.buyer_timestamps[index1]})\n")
              file.writelines(f"({buyer2}, {self.buyer_timestamps[index2]})\n")

    #  Applying Kruskal algorithm
    def kruskal_max_st(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2], reverse=True)
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


        with open("max_st_adjacency_matrix.txt", "w") as file:
          for node1, node2, price in result:
              # print("%d - %d: %d" % (node1, node2, price))

              buyer1 = self.buyers[node1]
              buyer2 = self.buyers[node2]

              index1 = self.temp_buyers.index(buyer1)
              index2 = self.temp_buyers.index(buyer2)

              # Save the entry in an adjacency matrix
              file.writelines(f"({buyer1}, {self.buyer_timestamps[index1]})\n")
              file.writelines(f"({buyer2}, {self.buyer_timestamps[index2]})\n")


if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    data = pd.read_csv("dataset.csv")
    nft_txns = prepare_data(data[0:80000])

    nft_txns = currency_converter(nft_txns)

    nft_txns = merge_sort_by_nft(nft_txns)

    temp_n_buyers, temp_buyers, buyer_timestamps = get_unique_buyers(nft_txns)
    graph = Graph(temp_n_buyers, temp_buyers, buyer_timestamps)

    start_time = time.time_ns()
    graph.build(nft_txns)
    end_time = time.time_ns()
    elapsed_time = (end_time - start_time)/1e9
    print(f'The time taken to build the graph is {elapsed_time} secs\n')

    start_time = time.time_ns()
    graph.kruskal_min_st()
    end_time = time.time_ns()
    elapsed_time = (end_time - start_time)/1e9
    print(f'The time taken to perform kruskal MST is {elapsed_time} secs')

    start_time = time.time_ns()
    graph.kruskal_max_st()
    end_time = time.time_ns()
    elapsed_time = (end_time - start_time)/1e9
    print(f'The time taken to perform kruskal Max ST is {elapsed_time} secs')