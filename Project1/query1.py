import pandas as pd 
import numpy as np
import time
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt 

##################################################### Data ###################################################
from dataclasses import dataclass, field

@dataclass(order=True)
class Query1Data:
    token_id: int
    n_txns: int

@dataclass(order=True)
class Query1Input:
    txn_hash: str
    token_id: int


#################################################### Merge sort ##############################################
def merge_sort_by_ntxn(A: List[Query1Data]):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_ntxn(B)
    R = merge_sort_by_ntxn(C)
    return merge_by_ntxn(L, R)

def merge_by_ntxn(L: List[Query1Data], R: List[Query1Data]):
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and L[i].n_txns >= R[j].n_txns):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B

#################################################### Radix sort ##############################################

def counting_sort_by_n_txns(A: List[Query1Data], exp) -> List[Query1Data]:
    n = len(A)
 
    # The output array elements that will have sorted arr
    output = [0] * (n)
 
    # initialize count array as 0
    count = [0] * (10)
 
    # Store count of occurrences in count[]
    for i in range(0, n):
        index = int(A[i].n_txns) // exp
        index = int(index)

        count[index % 10] += 1
 
    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]
 
    # Build the output array
    i = n - 1
    while i >= 0:
        index = int(A[i].n_txns) // exp
        index = int(index)

        output[count[index % 10] - 1] = A[i]
        count[index % 10] -= 1
        i -= 1
 
    return output
 
# Method to do Radix Sort
def radix_sort_by_n_txns(A: List[Query1Data]) -> List[Query1Data]:
    # Find the maximum number to know number of digits
    max1 = max_by_n_txns(A)
 
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp >= 1:
        A = counting_sort_by_n_txns(A, exp)
        exp *= 10

    B = []
    n = len(A)
    for i in range(n):
        B.append(A[n - i - 1])

    return B

def max_by_n_txns(A: List[Query1Data]) -> List[Query1Data]:
    max = 0
    for value in A:
        n_txns = int(value.n_txns)
        if  n_txns > max:
            max = n_txns

    return max

########################################## Utils #####################################################

def prepare_data(data) -> List[Query1Input]:
  data = data.reset_index()  # make sure indexes pair with number of rows

  transactions = []
  for i, row in data.iterrows():
    transactions.append(Query1Input(
        txn_hash=row['Txn Hash'], 
        token_id=row['Token ID']))
    
  return transactions


def get_dataframe(data: List[Query1Data]):
  txns_list = []

  for row in data:
    dic = {
        'Token ID': row.token_id,
        'Number of Transactions': row.n_txns,
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  df.to_excel('query1_out.xlsx') 
  return df

def update_with_n_txns(sorted_txns: List[Query1Input]) -> List[Query1Data]:
  # Lets sort the token ids by the number of unique buyers
  unique_count = 0
  unique_txn_hash = []
  n = len(sorted_txns)

  new_txns = []
  
  for i, row in enumerate(sorted_txns):
      if row.txn_hash not in unique_txn_hash:
          unique_count += 1
          unique_txn_hash.append(row.txn_hash)

      # This is for the scenario when the transaction is the last in the array (spreadsheet)
      if i == n - 1:
          data = Query1Data(token_id=sorted_txns[i].token_id, n_txns=unique_count)
          new_txns.append(data)

      elif sorted_txns[i].token_id != sorted_txns[i+1].token_id:
          data = Query1Data(token_id=sorted_txns[i].token_id, n_txns=unique_count)
          new_txns.append(data)

          unique_count = 0
          unique_txn_hash = []

  return new_txns

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_1.png"):
    x_axis = [i for i in range(92)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x5000)', 'Actual Runtime ((x10000))'], loc='upper left')

    plt.savefig(filename)

    plt.show()

############################################ Main Program ######################################################

def main():
    data = pd.read_csv("full_dataset.csv")
    transactions = prepare_data(data)

    elapsed_time_averages = []
    asymptotic_times = []
    for i in range(int(len(transactions)/1000)):
        print(f"{(i + 1) * 1000} transactions")

        n = (i + 1) * 1000
        aveg_elapsed_time_ns = run_n_times(transactions[0: n], 100)
        elapsed_time_averages.append(aveg_elapsed_time_ns)

        # this is used to ensure both the asymptotic and actual run time have the same scale
        n *= 5000

        # we figured out that the token with the most number of transactions had 1209 transactions, hence the exponent, k = 4
        k = 4
        asymptotic_times.append(n * k)

    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_time_averages)

    # This is used to print out the sorted records
    # run_query(transactions, run=0)

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
    sorted_txns = sort_query1(transactions)

    start_time = time.time_ns()
    sorted_txns = radix_sort_by_n_txns(sorted_txns)
    # sorted_txns = merge_sort_by_ntxn(sorted_txns)
    end_time = time.time_ns()

    elapsed_time = (end_time - start_time)

    if run == 0:
        df = get_dataframe(sorted_txns)
        print(df.head(10))

    # print(f"Run - {run} Sorting took {elapsed_time} nano secs ({elapsed_time/1e9} secs)")

    return elapsed_time, sorted_txns

def sort_query1(A: List[Query1Input]) -> List[Query1Data]:
    hash = {}
    for row in A:
        if row.token_id in hash:
            hash[row.token_id].append(row)
        else:
            hash[row.token_id] = [row]
        
    transactions = []
    for key in hash:
        transactions = np.concatenate((transactions, hash[key]))

    A = update_with_n_txns(transactions)
    return A

if __name__ == "__main__":
    main()