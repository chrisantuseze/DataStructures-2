import pandas as pd 
import numpy as np
import time
from utils import plot_graph
from utils import prepare_data, update_with_n_unique_buyers

from merge_sort import merge_sort_by_nbuyer, merge_sort_by_tokenid

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
    start_time1 = time.time_ns()
    sorted_txns = merge_sort_by_tokenid(transactions)
    end_time1 = time.time_ns()

    sorted_txns = update_with_n_unique_buyers(sorted_txns)

    start_time2 = time.time_ns()
    nbuyer_sorted_txns = merge_sort_by_nbuyer(sorted_txns)
    end_time2 = time.time_ns()

    elapsed_time = (end_time1 - start_time1) + (end_time2 - start_time2)

    # print(f"Run - {run} Sorting took {elapsed_time} nano secs")

    return elapsed_time, nbuyer_sorted_txns


if __name__ == "__main__":
    main()