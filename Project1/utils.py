import pandas as pd
from datetime import datetime

from typing import Optional, Tuple, List
from ml_data import MLData
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


def get_nft_dataframe(data):
  txns_list = []

  for row in data:
    dic = {
        'Txn Hash': row.txn_hash, 
        # 'UnixTimestamp': row.time_stamp,
        # 'Date Time (UTC)': row.date_time,
        # 'Action': row.action,
        'Buyer': row.buyer,
        # 'NFT': row.nft,
        'Token ID': row.token_id,
        # 'Type': row.type_,
        # 'Quantity': row.quantity,
        'Price': row.price,
        # 'Market': row.market,
        'Unique Buyers': row.n_unique_buyers
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  return df

def update_with_n_unique_buyers(sorted_txns) -> List[NFTTransaction]:
  # Lets sort the token ids by the number of unique buyers
  unique_count = 0
  unique_buyers = []
  n = len(sorted_txns)
  
  for i, row in enumerate(sorted_txns):
      if row.buyer not in unique_buyers:
          unique_count += 1
          unique_buyers.append(row.buyer)

      # This is for the scenario when the transaction is the last in the array (spreadsheet)
      if i == n - 1:
          sorted_txns[i].n_unique_buyers = unique_count

      elif sorted_txns[i].token_id != sorted_txns[i+1].token_id:
          sorted_txns[i].n_unique_buyers = unique_count
          unique_count = 0
          unique_buyers = []

  return sorted_txns

def currency_converter(data) -> List[NFTTransaction]:
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
          new_txns_list.append(get_txn(first_buy_date, last_buy_date, sorted_txns[i].token_id, unique_txn_count, unique_buyer_count))

      elif sorted_txns[i].token_id != sorted_txns[i+1].token_id:
          last_buy_date = sorted_txns[i].date_time

          new_txns_list.append(get_txn(first_buy_date, last_buy_date, sorted_txns[i].token_id, unique_txn_count, unique_buyer_count))
          
          unique_txn_count = 0
          unique_txns = []

          unique_buyer_count = 0
          unique_buyers = []

          first_buy_date = sorted_txns[i+1].date_time

  return new_txns_list

def get_txn(first_buy_date, last_buy_date, tokenid, n_txns, n_buyers):
  fraudulent = 0
  if hours_between(str(last_buy_date), str(first_buy_date)) <= 2 and n_txns > 100:
    fraudulent = 1

  return MLData(
      first_buy_date=first_buy_date, 
      last_buy_date=last_buy_date, 
      token_id=tokenid, 
      n_txns=n_txns, 
      n_unique_buyers=n_buyers, 
      fraudulent=fraudulent
    )

def hours_between(last_date, start_date):
  f = "%Y-%m-%d %H:%M:%S"
  t1 = datetime.strptime(last_date, f)
  t2 = datetime.strptime(start_date, f)

  diff = (t1 - t2).seconds/360
  return diff

def get_ml_dataframe(data: List[MLData]):
  txns_list = []

  for row in data:
    dic = {
        'First Buy Date Time': row.first_buy_date, 
        'Last Buy Date Time': row.last_buy_date,
        'Token ID': row.token_id,
        'Number of Transactions': row.n_txns,
        'Number of Unique Buyers': row.n_unique_buyers,
        'Fraudulent': row.fraudulent,
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  return df