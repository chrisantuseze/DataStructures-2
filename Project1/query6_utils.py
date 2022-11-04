import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt 

from typing import List
from ml_data import MLData
from query6_data import Query6Data, Query6Input
from nfttransaction_data import NFTTransaction

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
        price=row['Price'],
        market=row['Market'],
        n_unique_buyers=0))
    
  return transactions

def get_dataframe(data: List[MLData]):
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

def update_with_n_unique_txns(sorted_txns: List[NFTTransaction]) -> List[MLData]:
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

  return MLData (
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

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_6.png"):
    x_axis = [i for i in range(92)]
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