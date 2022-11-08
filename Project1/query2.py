#imports
from typing import List
import pandas as pd 
import numpy as np
import glob

import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field
import os

##################################################### Data ###################################################

@dataclass(order=True)
class Query2Data:
    token_id: int
    avg: float

@dataclass(order=True)
class NFTTransaction:
    txn_hash: str
    time_stamp: str
    date_time: str
    action: str
    buyer: str
    nft: str
    token_id: int
    type_: int
    quantity: int
    price: float
    price_str: str
    market: str

def merge_sort(A: List[Query2Data]):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort(B)
    R = merge_sort(C)
    return merge_by_ntxn(L, R)

def merge_by_ntxn(L: List[Query2Data], R: List[Query2Data]):
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and L[i].avg >= R[j].avg):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B


############################################ Main Program ######################################################

def main():
    # getting all the input files
    files = glob.glob(root_path + "/*.csv")
    # Read all the files placed in the current wokring directory with csv extenion
    data = [pd.read_csv(f) for f in files]
    data = pd.concat(data, ignore_index=True)
    # drop all null records
    data = data.dropna()
    # drop all duplicate records
    data = data.drop_duplicates()

    # prepare data accordingly as per data types and get required columns
    transactions = prepare_data(data)
    transactions = currency_converter(transactions)

    elapsed_time_averages = []
    asymptotic_times = []
    rows = int(len(transactions) / 1000)
    # Run the sorting in batches of 1000, 2000, 3000, ......
    for i in range(rows + 1):
        print(f"{(i + 1) * 1000} transactions")

        n = (i + 1) * 1000

        # run the query for a specified number of runs
        aveg_elapsed_time_ns = run_n_times(transactions[0: n], no_of_runs, save= i == rows)
        elapsed_time_averages.append(aveg_elapsed_time_ns)

        # this is used to ensure both the asymptotic and actual run time have the same scale while plotting the graph
        n *= 1000

        asymptotic_times.append(n * np.log10(n))

    # plot graphs for the collected asymptotic run times
    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_time_averages,
               filename=output_path + "/query_2.png", rows=rows)

    # run_query(transactions, save=True)

def run_n_times(transactions, n, save=False):
    elapsed_times = []
    for i in range(n):
        elapsed_time, sorted_txns = run_query(transactions, run=i+1, save=save)
        elapsed_times.append(elapsed_time)

    aveg_elapsed_time_ns = sum(elapsed_times)/len(elapsed_times)
    aveg_elapsed_time_s = aveg_elapsed_time_ns/1e9
    print(f"\nThe average elapsed time is {aveg_elapsed_time_ns} nano secs (i.e {aveg_elapsed_time_s} secs)\n")

    return aveg_elapsed_time_ns

def run_query(transactions, run=1, save=False):
    data = process_data(transactions)

    # collect start_time
    start_time = time.time_ns()
    # sort using merge sort algorithm
    # merge_sort(sorted_txns, 0, len(sorted_txns)-1)
    sorted_txns = merge_sort(data)
    # collect end_time
    end_time = time.time_ns()

    # calculate elapsed time
    elapsed_time = end_time - start_time

    if save:
        save_result(sorted_txns, transactions, elapsed_time)
        
    return elapsed_time, sorted_txns

########################################## Utils #####################################################
def process_data(transactions):
  hash = {}

  for item in transactions:
    if item.token_id in hash:
      hash[item.token_id].append(item)
    else:
      hash[item.token_id] = [item]
  
  new_transactions = []
  for key in hash:
    cur = hash[key]
    sum = 0
    count = 0
    for item in cur:
      sum = sum + (float(item.price) * float(item.quantity))
      count = count + item.quantity
    
    avg = sum / count
    
    new_transactions.append(Query2Data(token_id=key, avg=avg))  

  return new_transactions

def get_all_transactions(data: List[NFTTransaction]):
    hash = {}

    for row in data:
        if row.token_id in hash:
            hash[row.token_id].append(row)
        else:
            hash[row.token_id] = [row]

    return hash

def save_result(data: List[Query2Data], all_txns, elapsed_time):
    all_txns = get_all_transactions(all_txns)

    with open(output_path + "/query2_out.txt", "w") as file:
        file.writelines(f"The execution time is {elapsed_time} nano secs\n\n")

        for row in data:
            file.writelines(f"{row.token_id} (average = {row.avg})\n")
            file.writelines("Token ID,\t Txn hash,\t Date Time (UTC),\t Buyer,\t NFT,\t Type,\t Quantity,\t Price (USD)\n")
            file.writelines("\n")
            for value in all_txns[row.token_id]:
                file.writelines(f"{value.token_id},\t\t {value.txn_hash},\t {value.date_time},\t {value.buyer},\t {value.nft},\t {value.type_},\t {value.quantity},\t {value.price}\n")

            file.writelines("\n\n")

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

def prepare_data(data) -> List[NFTTransaction]:
  data = data.reset_index()  # make sure indexes pair with number of rows

  transactions = []
  for i, row in data.iterrows():
    transactions.append(NFTTransaction(
        txn_hash=row['Txn Hash'], 
        time_stamp=row['UnixTimestamp'],
        date_time=row['Date Time (UTC)'],
        action=row['Action'],
        buyer=row['Buyer'],
        nft=row['NFT'],
        token_id=row['Token ID'],
        type_=row['Type'],
        quantity=row['Quantity'],
        price_str=row['Price'],
        price=0.0,
        market=row['Market']))
    
  return transactions

def get_dataframe(data: List[Query2Data]):
  txns_list = []

  for row in data:
    dic = {
        'Token ID': row.token_id,
        'Average': row.avg,
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  df.to_excel(output_path + "/query2_out.xlsx")
  return df

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_2.png", rows=92):
    x_axis = [i for i in range(rows+1)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Query 2 Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x1000)', 'Actual Runtime ((x10000))'], loc='upper left')

    plt.savefig(filename)

    plt.show()


if __name__ == "__main__":
    # declare root path
    global root_path
    root_path = os.getcwd()

    print("Kindly specify the number of runs needed (suggested runs is 1), Example : 1")
    # No of times the script need to be run
    global no_of_runs
    no_of_runs = int(input())

    # Output path to store results
    global output_path
    output_path = root_path + "/output"

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    main()
