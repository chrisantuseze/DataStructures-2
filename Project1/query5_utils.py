from typing import List
import pandas as pd
import matplotlib.pyplot as plt 

from query5_data import Query5Data, Query5Input

def prepare_data(data) -> List[Query5Input]:
  data = data.reset_index()  # make sure indexes pair with number of rows

  transactions = []
  for i, row in data.iterrows():
    transactions.append(Query5Input(
        buyer=row['Buyer'],
        nft=row['NFT'],
        token_id=row['Token ID']))
    
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
  df.to_excel('query5_out.xlsx') 

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


def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_5.png"):
  x_axis = [i for i in range(92)]
  plt.plot(x_axis, asymptotic_runtimes, color ='red')
  plt.plot(x_axis, actual_runtimes, color ='blue')
  plt.xlabel("Transaction Batch (x1000)")
  plt.ylabel("Runtime")
  plt.title("Runtime vs Transaction batch size")
  plt.legend(['Asymptotic Runtime (x5000)', 'Actual Runtime ((x10000))'], loc='upper left')

  plt.savefig(filename)

  plt.show()