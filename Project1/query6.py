import pandas as pd 
import numpy as np
import time
from utils import plot_graph
from utils import update_with_n_unique_txns, currency_converter, get_ml_dataframe, prepare_data, update_with_n_unique_buyers

from merge_sort import merge_sort_by_nbuyer, merge_sort_by_tokenid, merge_sort_by_ntxn

def main():
    data = pd.read_csv("full_dataset.csv")
    transactions = prepare_data(data)


def test_ml_util(transactions):
    sorted_txns = merge_sort_by_tokenid(transactions)
    sorted_txns = update_with_n_unique_txns(sorted_txns)

    sorted_by_nbuyer = merge_sort_by_nbuyer(sorted_txns)
    sorted_by_ntxns = merge_sort_by_ntxn(sorted_by_nbuyer)

    df = get_ml_dataframe(sorted_by_ntxns)
    print(df.head(6))

if __name__ == "__main__":
    main()