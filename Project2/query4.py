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

def get_dataframe(data: List[NFTTransaction]):
  txns_list = []

  for row in data:
    dic = {
        'Txn Hash': row.txn_hash,
        'UnixTimestamp': row.time_stamp,
        'Date Time (UTC)': row.date_time,
        'Buyer': row.buyer,
        'Token ID': row.token_id,
        'Price': row.price
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  df.to_excel("sorted_data.xlsx")
  return df

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_4.png", rows=92):
    x_axis = [i for i in range(rows+1)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Query 4 Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x500)', 'Actual Runtime'], loc='upper left')

    plt.savefig(filename)
    # plt.show()

class Graph:
  def __init__(self) -> None:
    self.adjacency_graph = []

  def build(self, data: List[NFTTransaction]) -> None:
    for i in range(1, len(data)):
        if data[i-1].token_id == data[i].token_id and data[i-1].buyer != data[i].buyer:
          self.adjacency_graph.append(f"{data[i-1].buyer} - {data[i].buyer} -> [{data[i].token_id, data[i].price_str, data[i].date_time}] \n")

        elif data[i-1].token_id == data[i].token_id and data[i-1].buyer == data[i].buyer and i + 1 < len(data) and data[i].buyer != data[i+1].buyer:
          self.adjacency_graph.append(f"{data[i].buyer} - {data[i+1].buyer} -> [{data[i+1].token_id, data[i+1].price_str, data[i+1].date_time}] \n")

def run_query(nft_txns):
    graph = Graph()

    start_time = time.time_ns()
    graph.build(nft_txns)
    end_time = time.time_ns()
    elapsed_time = (end_time - start_time)

    return elapsed_time, graph.adjacency_graph

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
        run_elapsed_time_ns, adjacency_graph = run_query(nft_txns[0: n])
        elapsed_times.append(run_elapsed_time_ns)

        asymptotic_run_time = n
        asymptotic_run_time *= 500 # This ensures that both are on the same scale
        asymptotic_times.append(asymptotic_run_time)
        print(f'The total time taken to build graph for {n} transactions is {run_elapsed_time_ns/1e9} secs\n')

        if i == rows:
          with open(output_path + "/query4_adjacency_matrix.txt", "w") as file:
            file.writelines("Buyer 1 ------------------------------------> Buyer 2 ------------------------------------> Token ID ------> Price -------> Timestamp\n\n")
            file.writelines(adjacency_graph)

    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_times, filename=output_path+"/query_4.png", rows=rows)
