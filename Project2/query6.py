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

def merge_sort_by_nft(A):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A.iloc[:q]
    C = A.iloc[q:]

    L = merge_sort_by_nft(B)
    R = merge_sort_by_nft(C)
    return merge_by_nft(L, R)

def merge_by_nft(L, R):
    n = len(L) + len(R)
    i = j = 0
    B = pd.DataFrame()
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and L.iloc[i]['NFT'] >= R.iloc[j]['NFT']):
            pd.concat([B, L.iloc[i]])
            i = i + 1
        else:
            pd.concat([B, R.iloc[j]])
            j = j + 1

    return B

def get_and_prepare_data():
    data = pd.read_csv("dataset.csv")
    data = data.dropna()
    data = data.drop_duplicates()

    nft_txns = data[['Txn Hash', 'UnixTimestamp', 'Date Time (UTC)', 'Buyer', 'NFT', 'Price']]

    nft_txns = nft_txns.iloc[0:5000]

    nft_txns = currency_converter(nft_txns)
    unique_buyer_txns = nft_txns.groupby('Buyer', as_index=False).first()
    nft_txns = nft_txns.sort_values(by=['NFT'], ascending=False)
    nft_txns = convert_to_object_list(nft_txns)

    return nft_txns, unique_buyer_txns

def convert_to_object_list(data) -> List[NFTTransaction]:
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
    def __init__(self, unique_buyer_txns) -> None:
        self.unique_buyer_txns = unique_buyer_txns
        self.temp_buyers = unique_buyer_txns['Buyer']

        self.temp_n_buyers = len(self.temp_buyers)
        self.buyer_timestamps = unique_buyer_txns['Date Time (UTC)']

        self.buyers = []
        self.graph = []

        self.adjacency_matrix = np.zeros((self.temp_n_buyers, self.temp_n_buyers), dtype=tuple)

    def addEdge(self, node1, node2, price):
        self.graph.append([node1, node2, price])

    def add_to_buyers_list(self, buyer):
        if buyer not in self.buyers:
            self.buyers.append(buyer)

    def build(self, data: List[NFTTransaction]) -> None:

        with open(output_path + "/original_adjacency_matrix.txt", "w") as file:
            count = 0
            for i in range(1, len(data)):
                if data[i-1].nft == data[i].nft and data[i-1].buyer != data[i].buyer:
                    
                    self.buyers.insert(count, data[i-1].buyer)
                    self.buyers.insert(count + 1, data[i].buyer)
                    self.addEdge(count, count + 1, float(data[i].price_str))

                    count += 2

                    file.writelines(f"{data[i-1].buyer} - {data[i].buyer} -> [{data[i].nft, data[i].price_str, data[i].date_time}] \n")

                elif data[i-1].nft == data[i].nft and data[i-1].buyer == data[i].buyer and i + 1 < len(data) and data[i].buyer != data[i+1].buyer:
                   
                    self.buyers.insert(count, data[i].buyer)
                    self.buyers.insert(count + 1, data[i+1].buyer)
                    self.addEdge(count, count + 1, float(data[i+1].price_str))

                    count += 2

                    file.writelines(f"{data[i].buyer} - {data[i+1].buyer} -> [{data[i+1].nft, data[i+1].price_str, data[i+1].date_time}] \n")

                
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


        with open(output_path + "/" + filename, "w") as file:
          tree = "maximum" if is_reverse else "minimum"
          file.writelines(f"\nThe following are vertices and edges of the constructed {tree} spanning tree:\n")
          file.writelines("------------------------------------------------------------------------------\n\n")
          cost = 0
          for node1, node2, price in result:
              # print("%d - %d: %d" % (node1, node2, price))

              buyer1 = self.buyers[node1]
              buyer2 = self.buyers[node2]

              buyer1_row = self.unique_buyer_txns.loc[self.unique_buyer_txns['Buyer'] == buyer1]
              buyer2_row = self.unique_buyer_txns.loc[self.unique_buyer_txns['Buyer'] == buyer2]

              # Save the entry in an adjacency matrix
              cost += price
              try:
                file.writelines(f"({buyer1} - {buyer2} -> [{buyer2_row.iloc[0]['NFT']}, {buyer2_row.iloc[0]['Price']}, {buyer2_row.iloc[0]['Date Time (UTC)']}])\n\n")
          
              except:
                pass


          file.writelines("------------------------------------------------------------------------------\n")
          file.writelines(f"The {tree} cost is {cost}\n")
          file.writelines("------------------------------------------------------------------------------\n")

    def kruskal_min_st(self):
        self.kruskal("min_st_adjacency_matrix.txt", False)

    def kruskal_max_st(self):
        self.kruskal("max_st_adjacency_matrix.txt", True)

def run_query(nft_txns, unique_buyer_txns):
    graph = Graph(unique_buyer_txns)

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

    nft_txns, unique_buyer_txns = get_and_prepare_data()

    min_st_elapsed_times = []
    max_st_elapsed_times = []
    asymptotic_times = []
    rows = int(len(nft_txns) / 1000)

    # Run in intervals of 1000, 2000, 3000, ......
    for i in range(rows + 1):
        n = (i + 1) * 1000
        build_elapsed_time, min_st_run_elapsed_time, max_st_run_elapsed_time = run_query(nft_txns[0: n], unique_buyer_txns[0: n])
        min_st_elapsed_times.append(min_st_run_elapsed_time)
        max_st_elapsed_times.append(max_st_run_elapsed_time)

        asymptotic_times.append(n * 2) #TODO change this to the actual asymptotic time (build time + SCC time)
        print(f'The total time taken to for both minimum and maximum st for {n} transactions is {min_st_run_elapsed_time + max_st_run_elapsed_time} secs\n')

    #Note since the two runtimes would have close values, one many superimpose on the other. It is not a bug. Although it won't happen once everything is done
    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=[min_st_elapsed_times, max_st_elapsed_times], filename=output_path + "/query_6.png", rows=rows)