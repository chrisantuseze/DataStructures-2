# imports
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt 
import time
import glob
import os

##################################################### Data ###################################################

from dataclasses import dataclass, field

@dataclass(order=True)
class Query5Data:
    buyer: str
    nft: str
    n_txns_for_nft: int
    total_unique_nft: int
    total_txns: int

@dataclass(order=True)
class Query5Input:
    buyer: str
    nft: str
    token_id: int

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

def aux_sort_by_txns(A: List[Query5Data]) -> List[Query5Data]:
    for i in range(1, len(A)):
        x = A[i]
        j = i-1

        while j >= 0 and A[j].total_txns < x.total_txns:
            A[j + 1] = A[j]
            j = j - 1

        A[j + 1] = x
    return A

def sort_by_txns(A: List[Query5Data]) -> List[Query5Data]:
    B = []
    start, end = 0, 0
    flag = True

    for i in range(len(A)):
        B.append(A[i])

        if i >= len(A) - 1:
            continue

        if flag and A[i].total_unique_nft != A[i+1].total_unique_nft:
            start = i+1
            continue

        if flag and (A[i].total_unique_nft == A[i+1].total_unique_nft):
            flag = False
            continue

        if not flag and (A[i].total_unique_nft != A[i+1].total_unique_nft) :
            end = i

            C = aux_sort_by_txns(A[start:end+1])
            k = 0
            for j in range(start, end+1):
                B[j] = C[k]
                k += 1


            start = i+1
            flag = True

    return B

#################################################### Radix sort ##############################################

def counting_sort_by_n_nft(A: List[Query5Data], exp) -> List[Query5Data]:
    n = len(A)
 
    # The output array elements that will have sorted arr
    output = [0] * (n)
 
    # initialize count array as 0
    count = [0] * (10)
 
    # Store count of occurrences in count[]
    for i in range(0, n):
        index = int(A[i].total_unique_nft) // exp
        index = int(index)

        count[index % 10] += 1
 
    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]
 
    # Build the output array
    i = n - 1
    while i >= 0:
        index = int(A[i].total_unique_nft) // exp
        index = int(index)

        output[count[index % 10] - 1] = A[i]
        count[index % 10] -= 1
        i -= 1
 
    return output
 
# Method to do Radix Sort
def radix_sort_by_n_nft(A: List[Query5Data]) -> List[Query5Data]:
    # Find the maximum number to know number of digits
    max1 = max_by_n_nft(A)
 
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp >= 1:
        A = counting_sort_by_n_nft(A, exp)
        exp *= 10

    B = []
    n = len(A)
    for i in range(n):
        B.append(A[n - i - 1])

    return B

def max_by_n_nft(A: List[Query5Data]) -> List[Query5Data]:
    max = 0
    for value in A:
        total_unique_nft = int(value.total_unique_nft)
        if  total_unique_nft > max:
            max = total_unique_nft

    return max

########################################## Utils #####################################################

def get_all_transactions(data: List[NFTTransaction]):
    hash = {}

    for row in data:
        if row.buyer in hash:
            hash[row.buyer].append(row)
        else:
            hash[row.buyer] = [row]

    return hash

def save_result(data: List[Query5Data], all_txns, elapsed_time):
    all_txns = get_all_transactions(all_txns)

    with open(output_path + "/query5_out.txt", "w") as file:
        file.writelines(f"The execution time is {elapsed_time} nano secs\n\n")

        for row in data:
            file.writelines(f"{row.buyer} (frequency (unique NFTs = {row.total_unique_nft}, total NFTs = {row.total_txns})\n")
            file.writelines("Token ID,\t Txn hash,\t Date Time (UTC),\t Buyer,\t NFT,\t Type,\t Quantity,\t Price (USD)\n")
            file.writelines("\n")
            for value in all_txns[row.buyer]:
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

def get_dataframe(data: List[Query5Data]):
  txns_list = []

  for row in data:
    dic = {
        'Buyer': row.buyer,
        'NFT': row.nft,
        'Transactions Per NFT': row.n_txns_for_nft,
        'Total NFT': row.total_unique_nft,
        'Total Transactions': row.total_txns
    }

    # dic = {
    #     'Buyer': row.buyer,
    #     'NFT': row.nft,
    # }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  df.to_excel(output_path + "/query5_out.xlsx")
  return df

def update_with_n_unique_nfts(sorted_txns: List[Query5Input]) -> List[Query5Data]:
  # Lets sort the token ids by the number of unique buyers
  txns_count = 0
  n = len(sorted_txns)

  new_txn_list = []

  nft_txn_dic = {}
  
  for i, row in enumerate(sorted_txns):
      txns_count += 1

      if row.nft in nft_txn_dic.keys():
        nft_txn_dic[row.nft] += 1

      else:
        nft_txn_dic[row.nft] = 1

      # This is for the scenario when the transaction is the last in the array (spreadsheet)
      if i == n - 1:
          for k, v in nft_txn_dic.items():
              data = Query5Data(
                sorted_txns[i].buyer, 
                nft=k, n_txns_for_nft=v, 
                total_unique_nft=len(nft_txn_dic.keys()), 
                total_txns=txns_count)

              new_txn_list.append(data)

          nft_txn_dic = {}
          txns_count = 0

      elif sorted_txns[i].buyer != sorted_txns[i+1].buyer:
          for k, v in nft_txn_dic.items():
              data = Query5Data(
                sorted_txns[i].buyer, 
                nft=k, n_txns_for_nft=v, 
                total_unique_nft=len(nft_txn_dic.keys()), 
                total_txns=txns_count)

              new_txn_list.append(data)

          nft_txn_dic = {}
          txns_count = 0

  return new_txn_list

def update_with_n_unique_nfts_without_nft_names(sorted_txns: List[Query5Input]) -> List[Query5Data]:
  unique_nft_count = 0
  txns_count = 0
  unique_nfts = []
  n = len(sorted_txns)

  new_txn_list = []
  
  for i, row in enumerate(sorted_txns):
      txns_count += 1

      if row.nft not in unique_nfts:
          unique_nft_count += 1
          unique_nfts.append(row.nft)

      # This is for the scenario when the transaction is the last in the array (spreadsheet)
      if i == n - 1:
          data = Query5Data(
            sorted_txns[i].buyer, 
            nft=None, n_txns_for_nft=None, 
            total_unique_nft=unique_nft_count, 
            total_txns=txns_count)

          new_txn_list.append(data)

      elif sorted_txns[i].buyer != sorted_txns[i+1].buyer:
          data = Query5Data(
            sorted_txns[i].buyer, 
            nft=None, n_txns_for_nft=None, 
            total_unique_nft=unique_nft_count, 
            total_txns=txns_count)

          new_txn_list.append(data)

          unique_nfts = []
          txns_count = 0
          unique_nft_count = 0

  return new_txn_list


def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_5.png", rows=92):
  x_axis = [i for i in range(rows+1)]
  plt.plot(x_axis, asymptotic_runtimes, color ='red')
  plt.plot(x_axis, actual_runtimes, color ='blue')
  plt.xlabel("Transaction Batch (x1000)")
  plt.ylabel("Runtime")
  plt.title("Query 5 Runtime vs Transaction batch size")
  plt.legend(['Asymptotic Runtime (x1000)', 'Actual Runtime ((x10000))'], loc='upper left')

  plt.savefig(filename)

  plt.show()


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

        k = 2
        asymptotic_times.append(n * k)

    # plot graphs for the collected asymptotic run times
    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_time_averages,
               filename=output_path + "/query_5.png", rows=rows)

    # elapsed_time, sorted_txns = run_query(transactions, save=True)


def run_n_times(transactions, n, save=False):
    elapsed_times = []
    for i in range(n):
        elapsed_time, sorted_txns = run_query(transactions, save=save)
        elapsed_times.append(elapsed_time)

    aveg_elapsed_time_ns = sum(elapsed_times)/len(elapsed_times)
    aveg_elapsed_time_s = aveg_elapsed_time_ns/1e9
    print(f"\nThe average elapsed time is {aveg_elapsed_time_ns} nano secs (i.e {aveg_elapsed_time_s} secs)\n")

    return aveg_elapsed_time_ns
    
def run_query(transactions, save=False):
    data = process_data(transactions)
    
    # collect start time
    start_time1 = time.time_ns()
    # sort using radix sort
    sorted_txns = radix_sort_by_n_nft(data)
    # collect end time
    end_time1 = time.time_ns()

    start_time2 = time.time_ns()
    sorted_txns = sort_by_txns(sorted_txns)
    end_time2 = time.time_ns()

    elapsed_time = (end_time1 - start_time1) + (end_time2 - start_time2)

    if save:
        save_result(sorted_txns, transactions, elapsed_time)

    return elapsed_time, sorted_txns

def process_data(A: List[Query5Input]):
    hash = {}
    for row in A:
        if row.buyer in hash:
            hash[row.buyer].append(row)
        else:
            hash[row.buyer] = [row]
        
    transactions = []
    for key in hash:
        transactions = np.concatenate((transactions, hash[key]))

    A = update_with_n_unique_nfts_without_nft_names(transactions)
    return A

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