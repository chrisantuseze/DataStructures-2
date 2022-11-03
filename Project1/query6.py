import pandas as pd 
import numpy as np
from typing import List

import time
from Project1.query5_merge_sort import sort_by_txns
from query6_utils import get_ml_dataframe, prepare_data, update_with_n_unique_txns
from query6_data import Query6Input
from query6_merge_sort import merge_sort_by_ntxn

def main():
    data = pd.read_csv("full_dataset.csv")
    transactions = prepare_data(data)
    test_ml_util(transactions)


def test_ml_util(transactions):
    sorted_txns = sort_query6(transactions)

    start_time1 = time.time_ns()
    sorted_txns = merge_sort_by_ntxn(sorted_txns)
    end_time1 = time.time_ns()

    df = get_ml_dataframe(sorted_txns)
    print(df.head(6))

def sort_query6(A: List[Query6Input]):
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