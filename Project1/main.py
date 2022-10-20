import pandas as pd 
import time

from merge_sort import merge_sort_by_nbuyer, merge_sort_by_tokenid
from quick_sort import quick_sort_by_nbuyer, quick_sort_by_tokenid
from radix_sort import radix_sort_by_nbuyer, radix_sort_by_token_id
from utils import get_dataframe, prepare_data, update_with_n_unique_buyers

def main():
    data = pd.read_excel("Consolidated file for Aug 5 - Aug 13.xlsx")
    # print(data.head(2))

    transactions = prepare_data(data)

    elapsed_times = []
    for i in range(10):
        elapsed_time, sorted_txns = run_query(transactions, run=i+1)
        elapsed_times.append(elapsed_time)

    aveg_elapsed_time_ns = sum(elapsed_times)/len(elapsed_times)
    aveg_elapsed_time_s = aveg_elapsed_time_ns/1e9
    print(f"\nThe average elapsed time is {aveg_elapsed_time_ns} nano secs (i.e {aveg_elapsed_time_s} secs)\n")


def run_query(transactions, run=1):
    start_time1 = time.perf_counter_ns()
    # sorted_txns = radix_sort_by_token_id(transactions)
    sorted_txns = merge_sort_by_tokenid(transactions)
    # sorted_txns = quick_sort_by_tokenid(transactions) 
    end_time1 = time.perf_counter_ns()

    sorted_txns = update_with_n_unique_buyers(sorted_txns)

    start_time2 = time.perf_counter_ns()
    # nbuyer_sorted_txns = radix_sort_by_nbuyer(sorted_txns)
    nbuyer_sorted_txns = merge_sort_by_nbuyer(sorted_txns)
    # nbuyer_sorted_txns = quick_sort_by_nbuyer(sorted_txns)
    end_time2 = time.perf_counter_ns()

    elapsed_time = (end_time1 - start_time1) + (end_time2 - start_time2)

    print(f"Run - {run} Sorting took {elapsed_time} nano secs")

    # df = get_dataframe(nbuyer_sorted_txns)
    # print(df.head(10))

    return elapsed_time, nbuyer_sorted_txns


if __name__ == "__main__":
    main()