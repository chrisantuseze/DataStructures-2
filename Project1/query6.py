import pandas as pd 
import numpy as np
from typing import List

import time
from query6_utils import prepare_data, update_with_n_unique_txns, plot_graph, get_dataframe
from query6_data import Query6Input
from query6_merge_sort import merge_sort_by_ntxn, merge_sort_by_fraudulent

def main():
    data = pd.read_csv("full_dataset.csv")
    transactions = prepare_data(data)

    elapsed_time_averages = []
    asymptotic_times = []
    for i in range(int(len(transactions)/1000)):
        print(f"{(i + 1) * 1000} transactions")

        n = (i + 1) * 1000
        aveg_elapsed_time_ns = run_n_times(transactions[0: n], 100)
        elapsed_time_averages.append(aveg_elapsed_time_ns)

        n *= 5000
        asymptotic_times.append(n * np.log10(n))

    plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_time_averages)

    # This is used to print out the sorted records
    # run_query(transactions, run=0)

def run_n_times(transactions, n):
    elapsed_times = []
    for i in range(n):
        elapsed_time, sorted_txns = run_query(transactions, run=i+1)
        elapsed_times.append(elapsed_time)

    aveg_elapsed_time_ns = sum(elapsed_times)/len(elapsed_times)
    aveg_elapsed_time_s = aveg_elapsed_time_ns/1e9
    print(f"\nThe average elapsed time is {aveg_elapsed_time_ns} nano secs (i.e {aveg_elapsed_time_s} secs)\n")

    return aveg_elapsed_time_ns


def run_query(transactions, run=1):
    sorted_txns = sort_query6(transactions)

    start_time1 = time.time_ns()
    sorted_txns = merge_sort_by_fraudulent(sorted_txns)
    end_time1 = time.time_ns()

    elapsed_time = (end_time1 - start_time1)

    if run == 0:
        df = get_dataframe(sorted_txns)
        print(df.head(10))

    print(f"Run - {run} Sorting took {elapsed_time} nano secs ({elapsed_time/1e9} secs)")

    return elapsed_time, sorted_txns

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
