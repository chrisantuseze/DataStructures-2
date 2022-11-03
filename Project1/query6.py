import pandas as pd 
import numpy as np
import time
from query6_utils import get_ml_dataframe, prepare_data, update_with_n_unique_buyers

from query6_merge_sort import merge_sort_by_nbuyer, merge_sort_by_tokenid, merge_sort_by_ntxn, sort_query6

def main():
    data = pd.read_csv("full_dataset.csv")
    transactions = prepare_data(data)
    test_ml_util(transactions)


def test_ml_util(transactions):
    sorted_txns = sort_query6(transactions)
    # sorted_txns = update_with_n_unique_txns(sorted_txns)

    # sorted_txns = merge_sort_by_nbuyer(sorted_txns)
    # sorted_txns = merge_sort_by_ntxn(sorted_txns)

    df = get_ml_dataframe(sorted_txns)
    print(df.head(6))

if __name__ == "__main__":
    main()