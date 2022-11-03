import pandas as pd
import matplotlib.pyplot as plt

from query3_data import Query3Data
import numpy as np
from query3_merge_sort import mergeSort


def sort_by_avg_price(transactions):
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

  print(len(transactions_with_avg_price))
  mergeSort(transactions_with_avg_price, 0, len(transactions_with_avg_price)-1)

  #for i in transactions_with_avg_price:
   # print(i)
    #print(count)
    #print("------------")
  #return transactions_with_avg_price

def prepare_data(data) -> list[Query3Data]:
  data = data.reset_index()  # make sure indexes pair with number of rows

  transactions = []
  for i, row in data.iterrows():
    transactions.append(Query3Data  (
        txn_hash=row['Txn Hash'], 
        token_id=row['Token ID'],
        quantity=row['Quantity'],
        price=row['Price'],
        avg=0
        ))
    
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
  return df

  
def currency_converter(data) -> list[Query3Data]:
  data = prepare_data(data)

  for row in data:
    price = row.price

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

def plot_graph(asymptotic_runtimes, actual_runtimes, filename="query_3.png"):
    x_axis = [i for i in range(92)]
    plt.plot(x_axis, asymptotic_runtimes, color ='red')
    plt.plot(x_axis, actual_runtimes, color ='blue')
    plt.xlabel("Transaction Batch (x1000)")
    plt.ylabel("Runtime")
    plt.title("Runtime vs Transaction batch size")
    plt.legend(['Asymptotic Runtime (x5000)', 'Actual Runtime ((x10000))'], loc='upper left')

    plt.savefig(filename)

    plt.show()