from typing import List
import pandas as pd 
import numpy as np
import glob

import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field

##################################################### Data ###################################################

@dataclass(order=True)
class Query2Data:
    txn_hash: str
    token_id: int
    quantity: int
    price: str
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
    n_unique_buyers: int

# #################################################### Merge sort ##############################################

# def merge(A, l, m, r):
#     n1 = m - l + 1
#     n2 = r - m
 
#     # create temp arrays
#     L = [0] * (n1)
#     R = [0] * (n2)
 
#     # Copy data to temp arrays L[] and R[]
#     for i in range(0, n1):
#         L[i] = A[l + i]
 
#     for j in range(0, n2):
#         R[j] = A[m + 1 + j]
 
#     # Merge the temp arrays back into arr[l..r]
#     i = 0     # Initial index of first subarray
#     j = 0     # Initial index of second subarray
#     k = l     # Initial index of merged subarray
 
#     while i < n1 and j < n2:
#         if L[i].avg >= R[j].avg:
#             A[k] = L[i]
#             i += 1
#         else:
#             A[k] = R[j]
#             j += 1
#         k += 1
 
#     # Copy the remaining elements of L[], if there
#     # are any
#     while i < n1:
#         A[k] = L[i]
#         i += 1
#         k += 1
 
#     # Copy the remaining elements of R[], if there
#     # are any
#     while j < n2:
#         A[k] = R[j]
#         j += 1
#         k += 1
 
# # l is for left index and r is right index of the
# # sub-array of arr to be sorted
 
 
# def merge_sort(A, l, r):
#     if l < r:
 
#         # Same as (l+r)//2, but avoids overflow for
#         # large l and h
#         m = l+(r-l)//2
 
#         # Sort first and second halves
#         merge_sort(A, l, m)
#         merge_sort(A, m+1, r)
#         merge(A, l, m, r)

#################################################### Merge sort ##############################################
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

    # please replace this with the path where the dataset file is
    files = glob.glob("/Users/chrisantuseze/VSCodeProjects/DataStructures 2/Project1/*.csv")

    data = [pd.read_csv(f) for f in files]
    data = pd.concat(data,ignore_index=True)
    data = data.dropna()

    transactions = prepare_data(data)
    transactions = currency_converter(transactions)

    elapsed_time_averages = []
    asymptotic_times = []
    rows = int(len(transactions)/1000)
    for i in range(rows):
        print(f"{(i + 1) * 1000} transactions")

        n = (i + 1) * 1000
        aveg_elapsed_time_ns = run_n_times(transactions[0: n], 100)
        elapsed_time_averages.append(aveg_elapsed_time_ns)

        # this is used to ensure both the asymptotic and actual run time have the same scale
        n *= 5000

        asymptotic_times.append(n * np.log10(n))

    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_time_averages, rows=rows)

    # run_query(transactions, run=1)

def run_n_times(transactions, n):
    elapsed_times = []
    for i in range(n):
        elapsed_time, sorted_txns = run_query(transactions, run=i+1)
        elapsed_times.append(elapsed_time)

    aveg_elapsed_time_ns = sum(elapsed_times)/len(elapsed_times)
    aveg_elapsed_time_s = aveg_elapsed_time_ns/1e9
    print(f"\nThe average elapsed time is {aveg_elapsed_time_ns} nano secs (i.e {aveg_elapsed_time_s} secs)\n")

    return aveg_elapsed_time_ns

def run_query(transactions, run=1):
    data = process_data(transactions)

    start_time = time.time_ns()
    sorted_txns = merge_sort(data)
    end_time = time.time_ns()

    elapsed_time = end_time - start_time

    if run == 1:
        save_result(sorted_txns, transactions)

        df = get_dataframe(sorted_txns)
        print(df.head(10))
        
        print(f"Run - {run} Sorting took {elapsed_time} nano secs ({elapsed_time/1e9} secs)")

    return elapsed_time, sorted_txns

########################################## Utils #####################################################
def process_data(transactions):
  hash = {}

  for item in transactions:
    if item.token_id in hash:
      hash[item.token_id].append(item)
    else:
      hash[item.token_id] = [item]
  
  for key in hash:
    cur = hash[key]
    sum = 0
    count = 0
    for item in cur:
      sum = sum + (float(item.price) * float(item.quantity))
      count = count + item.quantity
    
    avg = sum / count

    cur_with_avg = []
    for item in cur:
      item.avg = avg
      cur_with_avg.append(item)
    
    hash[key] = cur_with_avg      

  transactions_with_avg_price = []
  for key in hash:
    transactions_with_avg_price = np.concatenate((transactions_with_avg_price, hash[key]))

  return transactions_with_avg_price

def get_all_transactions(data: List[NFTTransaction]):
    hash = {}

    for row in data:
        if row.token_id in hash:
            hash[row.token_id].append(row)
        else:
            hash[row.token_id] = [row]

    return hash

def save_result(data: List[Query2Data], all_txns):
    all_txns = get_all_transactions(all_txns)

    with open("query2_out.txt", "w") as file:
        for row in data:
            file.writelines(f"{row.token_id} (frequency = {row.avg})\n")
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
        market=row['Market'],
        n_unique_buyers=0))
    
  return transactions

def get_dataframe(data):
  txns_list = []

  for row in data:
    dic = {
        'Txn Hash': row.txn_hash, 
        'Token ID': row.token_id,
        'Quantity': row.quantity,
        'Price': row.price,
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  df.to_excel('query2_out.xlsx') 
  return df

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_3.png", rows=92):
    x_axis = [i for i in range(rows)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x5000)', 'Actual Runtime ((x10000))'], loc='upper left')

    plt.savefig(filename)

    plt.show()


if __name__ == "__main__":
    main()
