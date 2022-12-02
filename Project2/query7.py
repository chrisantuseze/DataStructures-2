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

    nft_txns = nft_txns.iloc[0:1000]

    nft_txns = currency_converter(nft_txns)

    unique_buyer_txns = nft_txns.groupby('Buyer', as_index=False).first()

    nft_txns = nft_txns.sort_values(by=['NFT'], ascending=False)

    nft_txns = convert_to_object_list(nft_txns)

    # txns_list = []
    # for row in nft_txns:
    #     dic = {
    #         'Buyer': row.buyer,
    #         'NFT': row.nft,
    #     }
    #     txns_list.append(dic)

    # df = pd.DataFrame.from_records(txns_list)
    # print(df.head(20))

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

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_7.png", rows=92):
    x_axis = [i for i in range(rows+1)]
    print(len(actual_runtimes))
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Query 7 Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime', 'Actual Runtime'], loc='upper left')

    plt.savefig(filename)
    plt.show()

class Node:
    def __init__(self, buyer_id, nft, indexloc = None):
        self.buyer_id = buyer_id
        self.nft = nft
        self.index = indexloc

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
    def __init__(self, nodes, is_less_than = lambda a, b: a < b, get_index = None, update_node = lambda node, newval: newval):
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
            # update your order_mapping array to indicate where that index lives
            # in the nodes array (so lookup by this index is O(1))
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
        # If self.get_index exists, update self.order_mapping to indicate
        # the node of this index is no longer in the heap
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

class DijkstraNodeDecorator:
    def __init__(self, node):
        self.node = node
        self.prov_dist = float('inf')
        self.hops = []

    def index(self):
        return self.node.index

    def buyer_id(self):
        return self.node.buyer_id

    def nft(self):
        return self.node.nft
    
    def update_data(self, data):
        self.prov_dist = data['prov_dist']
        self.hops = data['hops']
        return self

class Graph(): 
    def __init__(self, unique_buyer_txns):
        self.unique_buyer_txns = unique_buyer_txns
        self.unique_buyers = unique_buyer_txns['Buyer']
        self.n_buyers = len(self.unique_buyers)
        self.timestamps = unique_buyer_txns['Date Time (UTC)']

        nodes = [Node(unique_buyer_txns.at[row.Index, 'Buyer'], unique_buyer_txns.at[row.Index, 'NFT']) for row in unique_buyer_txns.itertuples()]

        # self.graph = [ [node, []] for node in nodes ]
        # for i in range(len(nodes)):
        #     nodes[i].index = i

        self.graph = [[None, []]]
        # print(len(self.graph))
        self.src_buyer = None


    def connect_dir(self, node1, node2, weight = 1):
        node1, node2 = self.get_index_from_node(node1), self.get_index_from_node(node2)
        # Note that the below doesn't protect from adding a connection twice

        self.graph[node1][1].append((node2, weight))

    def connect(self, node1, node2, weight = 1):
        self.connect_dir(node1, node2, weight)
        self.connect_dir(node2, node1, weight)

    
    def connections(self, node):
        node = self.get_index_from_node(node)
        return self.graph[node][1]
    
    def get_index_from_node(self, node):
        if not isinstance(node, Node) and not isinstance(node, int):
            raise ValueError("node must be an integer or a Node object")
        if isinstance(node, int):
            return node
        else:
            return node.index


    def build(self, data) -> None:

        with open(output_path + "/original_adjacency_matrix.txt", "w") as file:
            count = 0
            for i in range(1, len(data)):
                if data[i-1].nft == data[i].nft and data[i-1].buyer != data[i].buyer:

                    node1 = Node(data[i-1].buyer, data[i-1].nft)
                    node1.index = count
                    count += 1

                    node2 = Node(data[i].buyer, data[i].nft)
                    node2.index = count
                    count += 1

                    # try:
                    #     self.connect_dir(node1, node2, weight=data[i].price_str)
                    # except:
                    #     print(len(self.graph), count)
                    #     self.connect_dir(node1, node2, weight=data[i].price_str)

                    file.writelines(f"{data[i-1].buyer} - {data[i].buyer} -> [{data[i].nft, data[i].price_str, data[i].date_time}] \n")

                elif data[i-1].nft == data[i].nft and data[i-1].buyer == data[i].buyer and i + 1 < len(data) and data[i].buyer != data[i+1].buyer:

                    # print(data.at[row.Index-1, 'Buyer'], data.at[row.Index, 'Buyer'], data.at[row.Index, 'NFT'])

                    node1 = Node(data[i].buyer, data[i].price_str)
                    node1.index = count
                    count += 1

                    node2 = Node(data[i+1].buyer, data[i+1].nft)
                    node2.index = count
                    count += 1

                    # try:
                    #     self.connect_dir(node1, node2, weight=data[i+1].price_str)
                    # except:
                    #     print(len(self.graph), count)

                    #     self.connect_dir(node1, node2, weight=data[i+1].price_str)

                    file.writelines(f"{data[i].buyer} - {data[i+1].buyer} -> [{data[i+1].nft, data[i+1].price_str, data[i+1].date_time}] \n")
                    
                    # skip to row.Index + 2 since we have processed the txn at row.Index + 1
                    jump = True

        self.src_buyer = self.graph[0][0]
                
    def dijkstras_shortest_path(self):        
        src_index = self.get_index_from_node(self.src_buyer)

        # Map nodes to DijkstraNodeDecorators
        # This will initialize all provisional distances to infinity
        dnodes = [ DijkstraNodeDecorator(node_edges[0]) for node_edges in self.graph ]
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
            min_dist_list.append([min_dist, hops])

            # print(min_decorated_node.prov_dist)
            
            # Get all next hops. This is no longer an O(n^2) operation
            connections = self.connections(min_decorated_node.node)
            # For each connection, update its path and total distance from 
            # starting node if the total distance is less than the current distance
            # in dist array
            for (inode, weight) in connections: 
                node = self.graph[inode][0]
                heap_location = heap.order_mapping[inode]
                if(heap_location is not None):
                    tot_dist = weight + min_dist
                    if tot_dist < heap.nodes[heap_location].prov_dist:
                        hops_cpy = list(hops)
                        hops_cpy.append(node)
                        data = {'prov_dist': tot_dist, 'hops': hops_cpy}
                        heap.decrease_key(heap_location, data)

        return min_dist_list 

def run_query(nft_txns, unique_buyer_txns):
    graph = Graph(unique_buyer_txns)

    start_time = time.time_ns()
    graph.build(nft_txns)
    end_time = time.time_ns()
    elapsed_time1 = (end_time - start_time)/1e9

    start_time = time.time_ns()
    print([(weight, [n.buyer_id for n in node]) for (weight, node) in graph.dijkstras_shortest_path()])
    end_time = time.time_ns()
    elapsed_time2 = (end_time - start_time)/1e9

    return elapsed_time1, elapsed_time2

if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    global root_path
    root_path = os.getcwd()

    # Output path to store results
    global output_path
    output_path = root_path + "/output"

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # nft_txns, unique_buyer_txns = get_and_prepare_data()

    # elapsed_times = []
    # asymptotic_times = []
    # rows = int(len(nft_txns) / 1000)

    # # Run in intervals of 1000, 2000, 3000, ......
    # for i in range(rows + 1):
    #     n = (i + 1) * 1000
    #     build_elapsed_time, run_elapsed_time = run_query(nft_txns[0: n], unique_buyer_txns[0: n])
    #     elapsed_times.append(run_elapsed_time)

    #     asymptotic_times.append(n * 2) #TODO change this to the actual asymptotic time 
    #     print(f'The total time taken to build graph and perform SCC for {n} transactions is {run_elapsed_time} secs\n')

    # plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_times, filename=output_path+"/query_7.png", rows=rows)


    nft_txns, unique_buyer_txns = get_and_prepare_data()

    graph = Graph(unique_buyer_txns)

    start_time = time.time_ns()
    graph.build(nft_txns)
    end_time = time.time_ns()
    elapsed_time = (end_time - start_time)/1e9
    print(f'The time taken to build the graph is {elapsed_time} secs\n')

    # start_time = time.time_ns()
    # graph.dijkstras_shortest_path()
    # end_time = time.time_ns()
    # elapsed_time = (end_time - start_time)/1e9
    # print(f'The time taken to perform dijkstras shortest path is {elapsed_time} secs')

    # a = Node('a')
    # b = Node('b')
    # c = Node('c')
    # d = Node('d')
    # e = Node('e')
    # f = Node('f')

    # g = Graph([a,b,c,d,e,f])

    # g.connect(a,b,5)
    # g.connect(a,c,10)
    # g.connect(a,e,2)
    # g.connect(b,c,2)
    # g.connect(b,d,4)
    # g.connect(c,d,7)
    # g.connect(c,f,10)
    # g.connect(d,e,3)

    print([(weight, [n.buyer_id for n in node]) for (weight, node) in graph.dijkstras_shortest_path()])