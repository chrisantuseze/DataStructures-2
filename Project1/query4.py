import pandas as pd 
import numpy as np
from typing import List
import matplotlib.pyplot as plt 
import time
import glob

##################################################### Data ###################################################
from dataclasses import dataclass

@dataclass(order=True)
class Query4Data:
    token_id: str
    # buyer: str
    n_unique_buyers: int


@dataclass(order=True)
class Query4Input:
    txn_hash: str
    token_id: int
    buyer: str

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

#################################################### Merge sort ##############################################
def merge_sort_by_tokenid(A):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_tokenid(B)
    R = merge_sort_by_tokenid(C)
    return merge_by_tokenid(L, R)

def merge_by_tokenid(L, R):
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and (int(L[i].token_id) >= int(R[j].token_id))):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B

def merge_sort_by_nbuyer(A: List[Query4Data]) -> List[Query4Data]:
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_nbuyer(B)
    R = merge_sort_by_nbuyer(C)
    return merge_by_nbuyer(L, R)

def merge_by_nbuyer(L: List[Query4Data], R: List[Query4Data]) -> List[Query4Data]:
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and L[i].n_unique_buyers >= R[j].n_unique_buyers):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B

#################################################### Radix sort ##############################################

def counting_sort_by_nbuyer(A: List[Query4Data], exp) -> List[Query4Data]:
    n = len(A)
 
    # The output array elements that will have sorted arr
    output = [0] * (n)
 
    # initialize count array as 0
    count = [0] * (10)
 
    # Store count of occurrences in count[]
    for i in range(0, n):
        index = int(A[i].n_unique_buyers) // exp
        index = int(index)

        count[index % 10] += 1
 
    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]
 
    # Build the output array
    i = n - 1
    while i >= 0:
        index = int(A[i].n_unique_buyers) // exp
        index = int(index)

        output[count[index % 10] - 1] = A[i]
        count[index % 10] -= 1
        i -= 1
 
    return output
 
# Method to do Radix Sort
def radix_sort_by_nbuyer(A: List[Query4Data]) -> List[Query4Data]:
    # Find the maximum number to know number of digits
    max1 = max_by_nbuyer(A)
 
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp >= 1:
        A = counting_sort_by_nbuyer(A, exp)
        exp *= 10

    B = []
    n = len(A)
    for i in range(n):
        B.append(A[n - i - 1])

    return B

def max_by_nbuyer(A: List[Query4Data]) -> List[Query4Data]:
    max = 0
    for value in A:
        n_unique_buyers = int(value.n_unique_buyers)
        if  n_unique_buyers > max:
            max = n_unique_buyers

    return max

########################################## Utils #####################################################

def get_all_transactions(data: List[NFTTransaction]):
    hash = {}

    for row in data:
        if row.token_id in hash:
            hash[row.token_id].append(row)
        else:
            hash[row.token_id] = [row]

    return hash

def save_result(data: List[Query4Data], all_txns):
    all_txns = get_all_transactions(all_txns)

    with open("query4_out.txt", "w") as file:
        for row in data:
            file.writelines(f"{row.token_id} (frequency = {row.n_unique_buyers})\n")
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

def get_dataframe(data: List[Query4Data]):
  txns_list = []

  for row in data:
    dic = {
        'Token ID': row.token_id,
        'Unique Buyers': row.n_unique_buyers,
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  df.to_excel('query4_out.xlsx') 
  return df

def update_with_n_unique_buyers(sorted_txns: List[Query4Input]) -> List[Query4Data]:
  # Lets sort the token ids by the number of unique buyers
  unique_count = 0
  unique_buyers = []
  n = len(sorted_txns)

  new_txns = []
  
  for i, row in enumerate(sorted_txns):
      if row.buyer not in unique_buyers:
          unique_count += 1
          unique_buyers.append(row.buyer)

      # This is for the scenario when the transaction is the last in the array (spreadsheet)
      if i == n - 1:
          data = Query4Data(token_id=sorted_txns[i].token_id, n_unique_buyers=unique_count)
          new_txns.append(data)

      elif sorted_txns[i].token_id != sorted_txns[i+1].token_id:
          data = Query4Data(token_id=sorted_txns[i].token_id, n_unique_buyers=unique_count)
          new_txns.append(data)

          unique_count = 0
          unique_buyers = []

  return new_txns

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_4.png", rows=92):
    x_axis = [i for i in range(rows)]
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

        n *= 5000
        
        # we figured out that the token with the most number of buyers had 1201 unique buyers, hence the exponent, k = 4
        k = 4
        asymptotic_times.append(n * k)
        
    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_time_averages, rows=rows)

    # This is used to print out the sorted records
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
    sorted_txns = sort_query4(transactions)

    start_time = time.time_ns()
    # sorted_txns = timsort(sorted_txns)
    sorted_txns = radix_sort_by_nbuyer(sorted_txns)
    # sorted_txns = merge_sort_by_nbuyer(sorted_txns)
    end_time = time.time_ns()

    elapsed_time = (end_time - start_time)

    if run == 1:
        save_result(sorted_txns, transactions)

        df = get_dataframe(sorted_txns)
        print(df.head(10))

    # print(f"Run - {run} Sorting took {elapsed_time} nano secs ({elapsed_time/1e9} secs)")

    return elapsed_time, sorted_txns

def sort_query4(A: List[Query4Input]) -> List[Query4Data]:
    hash = {}
    for row in A:
        if row.token_id in hash:
            hash[row.token_id].append(row)
        else:
            hash[row.token_id] = [row]
        
    transactions = []
    for key in hash:
        transactions = np.concatenate((transactions, hash[key]))

    A = update_with_n_unique_buyers(transactions)
    return A

if __name__ == "__main__":
    main()