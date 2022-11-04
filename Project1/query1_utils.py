import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt 

from typing import List
from query1_data import Query1Data, Query1Input

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