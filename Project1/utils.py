import pandas as pd
from nfttransaction_data import NFTTransaction

def prepare_data(data):
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


def get_dataframe(data):
  txns_list = []

  for row in data:
    dic = {
        'Txn Hash': row.txn_hash, 
        'UnixTimestamp': row.time_stamp,
        'Date Time (UTC)': row.date_time,
        'Action': row.action,
        'Buyer': row.buyer,
        'NFT': row.nft,
        'Token ID': row.token_id,
        'Type': row.type_,
        'Quantity': row.quantity,
        'Price': row.price,
        'Market': row.market,
        'Unique Buyers': row.n_unique_buyers
    }
    txns_list.append(dic)

  df = pd.DataFrame.from_records(txns_list)
  return df

def update_with_n_unique_buyers(sorted_txns):
    # Lets sort the token ids by the number of unique buyers
    unique_count = 0
    unique_buyers = []
    n = len(sorted_txns)
    
    for i, row in enumerate(sorted_txns):
        if row.buyer not in unique_buyers:
            unique_count += 1
            unique_buyers.append(row.buyer)

        if i == n - 1:
            sorted_txns[i].n_unique_buyers = unique_count

        elif sorted_txns[i].token_id != sorted_txns[i+1].token_id:
            sorted_txns[i].n_unique_buyers = unique_count
            unique_count = 0
            unique_buyers = []

    return sorted_txns
