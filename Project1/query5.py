import pandas as pd 
import numpy as np
from typing import List

import time
from query5_utils import get_dataframe
from query5_utils import plot_graph, update_with_n_unique_nfts_without_nft_names, update_with_n_unique_nfts
from query5_data import Query5Data, Query5Input

from query5_merge_sort import merge_sort_by_n_nft, sort_by_txns
from query5_utils import prepare_data

def main():
    data = pd.read_csv("full_dataset.csv")
    transactions = prepare_data(data)

    elapsed_time_averages = []
    asymptotic_times = []
    # for i in range(int(len(transactions)/1000)):
    #     print(f"{(i + 1) * 1000} transactions")

    #     n = (i + 1) * 1000
    #     aveg_elapsed_time_ns = run_n_times(transactions[0: n], 2)
    #     elapsed_time_averages.append(aveg_elapsed_time_ns)

    #     n *= 5000
    #     asymptotic_times.append(n * np.log10(n))

    # plot_graph(asymptotic_runtimes=asymptotic_times, actual_runtimes=elapsed_time_averages)

    # This is used to print out the sorted records
    run_query(transactions, run=0)

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
    sorted_txns = sort_query5(transactions)
    
    start_time1 = time.time_ns()
    sorted_txns = merge_sort_by_n_nft(sorted_txns)
    end_time1 = time.time_ns()

    start_time2 = time.time_ns()
    sorted_txns = sort_by_txns(sorted_txns)
    end_time2 = time.time_ns()

    elapsed_time = (end_time1 - start_time1) + (end_time2 - start_time2)

    if run == 0:
        df = get_dataframe(sorted_txns)
        print(df.head(10))

    # print(f"Run - {run} Sorting took {elapsed_time} nano secs")

    return elapsed_time, sorted_txns

def sort_query5(A: List[Query5Input]):
    hash = {}
    for row in A:
        if row.buyer in hash:
            hash[row.buyer].append(row)
        else:
            hash[row.buyer] = [row]
        
    transactions = []
    for key in hash:
        transactions = np.concatenate((transactions, hash[key]))

    A = update_with_n_unique_nfts(transactions)
    return A

if __name__ == "__main__":
    main()