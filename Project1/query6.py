import pandas as pd 
import numpy as np
from typing import List
import matplotlib.pyplot as plt 
import time
from datetime import datetime
import glob

##################################################### Data ###################################################
from dataclasses import dataclass, field

@dataclass(order=True)
class Query6Data:
    first_buy_date: str
    last_buy_date: str
    second_last_buy_date: str
    third_last_buy_date: str
    # nft: str
    token_id: int
    n_txns: int
    n_unique_buyers: int
    fraudulent: str
    fraudulent_ascii: int

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
def merge_sort_by_ntxn(A: List[Query6Data]):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_ntxn(B)
    R = merge_sort_by_ntxn(C)
    return merge_by_ntxn(L, R)

def merge_by_ntxn(L: List[Query6Data], R: List[Query6Data]):
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

def merge_sort_by_fraudulent(A: List[Query6Data]) -> List[Query6Data]:
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_fraudulent(B)
    R = merge_sort_by_fraudulent(C)
    return merge_by_fraudulent(L, R)

def merge_by_fraudulent(L: List[Query6Data], R: List[Query6Data]) -> List[Query6Data]:
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and ((L[i].fraudulent_ascii) >= (R[j].fraudulent_ascii))):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B

#################################################### Radix sort ##############################################

def counting_sort_by_fraudulent(A: List[Query6Data], exp) -> List[Query6Data]:
    n = len(A)
 
    # The output array elements that will have sorted arr
    output = [0] * (n)
 
    # initialize count array as 0
    count = [0] * (10)
 
    # Store count of occurrences in count[]
    for i in range(0, n):
        index = int(A[i].fraudulent_ascii) // exp
        index = int(index)

        count[index % 10] += 1
 
    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]
 
    # Build the output array
    i = n - 1
    while i >= 0:
        index = int(A[i].fraudulent_ascii) // exp
        index = int(index)

        output[count[index % 10] - 1] = A[i]
        count[index % 10] -= 1
        i -= 1
 
    return output
 
# Method to do Radix Sort
def radix_sort_by_fraudulent(A: List[Query6Data]) -> List[Query6Data]:
    # Find the maximum number to know number of digits
    max1 = max_by_fraudulent(A)
 
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp >= 1:
        A = counting_sort_by_fraudulent(A, exp)
        exp *= 10

    B = []
    n = len(A)
    for i in range(n):
        B.append(A[n - i - 1])

    return B

def max_by_fraudulent(A: List[Query6Data]) -> List[Query6Data]:
    max = 0
    for value in A:
        fraudulent_ascii = int(value.fraudulent_ascii)
        if  fraudulent_ascii > max:
            max = fraudulent_ascii

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

def save_result(data: List[Query6Data], all_txns):
    all_txns = get_all_transactions(all_txns)

    with open("query6_out.txt", "w") as file:
        for row in data:
            file.writelines(f"{row.token_id} (frequency (number of transactions = {row.n_txns}, number of unique buyers = {row.n_unique_buyers}, status = {row.fraudulent})\n")
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

def get_dataframe(data: List[Query6Data]):
  txns_list = []

  for row in data:
    dic = {
        'First Buy Time': row.first_buy_date, 
        'Last Buy Time': row.last_buy_date,
        'Second Last Buy Time': row.second_last_buy_date,
        'Third Last Buy Time': row.third_last_buy_date,
        'Token ID': row.token_id,
        'Number of Txns': row.n_txns,
        'Number of Unique Buyers': row.n_unique_buyers,
        'Fraudulent': row.fraudulent,
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  df.to_excel('query6_out.xlsx') 
  return df

def update_with_n_unique_txns(sorted_txns: List[NFTTransaction]) -> List[Query6Data]:
  # Lets sort the token ids by the number of txns
  unique_txn_count = 0
  unique_txns = []

  unique_buyer_count = 0
  unique_buyers = []
  n = len(sorted_txns)
  
  new_txns_list = []

  first_buy_date = ""
  last_buy_date = ""
  for i, row in enumerate(sorted_txns):
      if i == 0:
        first_buy_date = sorted_txns[i].date_time

      if row.txn_hash not in unique_txns:
          unique_txn_count += 1
          unique_txns.append(row.txn_hash)

      if row.buyer not in unique_buyers:
          unique_buyer_count += 1
          unique_buyers.append(row.buyer)


      if i == n - 1:
          last_buy_date = sorted_txns[i].date_time

          second_last_buy_date = None
          if sorted_txns[i].token_id == sorted_txns[i-1].token_id:
            second_last_buy_date = sorted_txns[i-1].date_time

          third_last_buy_date = None
          if sorted_txns[i].token_id == sorted_txns[i-2].token_id:
            third_last_buy_date = sorted_txns[i-2].date_time

          new_txns_list.append(get_txn(first_buy_date, last_buy_date, second_last_buy_date, third_last_buy_date, sorted_txns[i].token_id, unique_txn_count, unique_buyer_count))
          
      elif sorted_txns[i].token_id != sorted_txns[i+1].token_id:
          last_buy_date = sorted_txns[i].date_time

          second_last_buy_date = None
          if sorted_txns[i].token_id == sorted_txns[i-1].token_id:
            second_last_buy_date = sorted_txns[i-1].date_time

          third_last_buy_date = None
          if sorted_txns[i].token_id == sorted_txns[i-2].token_id:
            third_last_buy_date = sorted_txns[i-2].date_time

          new_txns_list.append(get_txn(first_buy_date, last_buy_date, second_last_buy_date, third_last_buy_date, sorted_txns[i].token_id, unique_txn_count, unique_buyer_count))
          
          unique_txn_count = 0
          unique_txns = []

          unique_buyer_count = 0
          unique_buyers = []

          first_buy_date = sorted_txns[i+1].date_time

  return new_txns_list

def get_txn(first_buy_date, last_buy_date, second_last_buy_date, third_last_buy_date, tokenid, n_txns, n_buyers):
  fraudulent = "No"
  interval_threshold_hr = 1
  ntxn_nbuyer_ratio = 1.8

  if float(n_txns/n_buyers) > ntxn_nbuyer_ratio and (
    hours_between(last_buy_date, first_buy_date) <= interval_threshold_hr or 
    hours_between(second_last_buy_date, first_buy_date) <= interval_threshold_hr or 
    hours_between(third_last_buy_date, first_buy_date) <= interval_threshold_hr
    ):

    fraudulent = "Yes"

  elif n_txns < n_buyers:
      fraudulent = "Suspicious"

  return Query6Data (
      first_buy_date=first_buy_date, 
      last_buy_date=last_buy_date, 
      second_last_buy_date=second_last_buy_date,
      third_last_buy_date=third_last_buy_date,
      token_id=tokenid, 
      n_txns=n_txns, 
      n_unique_buyers=n_buyers, 
      fraudulent=fraudulent,
      fraudulent_ascii=convert_string_to_ascii(fraudulent)
    )

def hours_between(last_date, start_date):
  last_date = str(last_date)
  start_date = str(start_date)

  f = "%m/%d/%y %H:%M"
  t1 = datetime.strptime(last_date, f)
  t2 = datetime.strptime(start_date, f)

  diff_in_hours = (t1 - t2).seconds/360
  return diff_in_hours

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_6.png", rows=92):
    x_axis = [i for i in range(rows)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x5000)', 'Actual Runtime ((x10000))'], loc='upper left')

    plt.savefig(filename)

    plt.show()

def convert_string_to_ascii(input):
  a = list(input.encode('ascii'))
  return int("".join(map(str, a)))

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

    start_time1 = time.time_ns()
    sorted_txns = merge_sort_by_fraudulent(data)
    end_time1 = time.time_ns()

    elapsed_time = (end_time1 - start_time1)

    if run == 1:
        save_result(sorted_txns, transactions)

        df = get_dataframe(sorted_txns)
        print(df.head(10))

        print(f"Run - {run} Sorting took {elapsed_time} nano secs ({elapsed_time/1e9} secs)")

    return elapsed_time, sorted_txns

def process_data(A: List[NFTTransaction]):
    hash = {}
    for row in A:
        if row.token_id in hash:
            hash[row.token_id].append(row)
        else:
            hash[row.token_id] = [row]
        
    transactions = []
    for key in hash:
        transactions = np.concatenate((transactions, hash[key]))

    A = update_with_n_unique_txns(transactions)
    return A

if __name__ == "__main__":
    main()
