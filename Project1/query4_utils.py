import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt 

from typing import List
from ml_data import MLData
from query4_data import Query4Data, Query4Input

def prepare_data(data) -> List[Query4Input]:
  data = data.reset_index()  # make sure indexes pair with number of rows

  transactions = []
  for i, row in data.iterrows():
    transactions.append(Query4Input(
        txn_hash=row['Txn Hash'], 
        token_id=row['Token ID'],
        buyer=row['Buyer']))
    
  return transactions


def get_dataframe(data: List[Query4Data]):
  txns_list = []

  for row in data:
    dic = {
        'Token ID': row.token_id,
        'Unique Buyers': row.n_unique_buyer,
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
          data = Query4Data(token_id=sorted_txns[i].token_id, n_unique_buyer=unique_count)
          new_txns.append(data)

      elif sorted_txns[i].token_id != sorted_txns[i+1].token_id:
          data = Query4Data(token_id=sorted_txns[i].token_id, n_unique_buyer=unique_count)
          new_txns.append(data)

          unique_count = 0
          unique_buyers = []

  return sorted_txns

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_4.png"):
    x_axis = [i for i in range(92)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x5000)', 'Actual Runtime ((x10000))'], loc='upper left')

    plt.savefig(filename)

    plt.show()