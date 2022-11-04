from typing import List
from query5_data import Query5Data, Query5Input
from query5_utils import update_with_n_unique_nfts_without_nft_names

def merge_sort_by_n_nft(A: List[Query5Data]) -> List[Query5Data]:
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_n_nft(B)
    R = merge_sort_by_n_nft(C)
    return merge_by_n_nft(L, R)

def merge_by_n_nft(L: List[Query5Data], R: List[Query5Data]) -> List[Query5Data]:
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and (( L[i].total_unique_nft ) >= ( R[j].total_unique_nft ))):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B

def aux_sort_by_txns(A: List[Query5Data]) -> List[Query5Data]:
    for i in range(1, len(A)):
        x = A[i]
        j = i-1

        while j >= 0 and A[j].total_txns < x.total_txns:
            A[j + 1] = A[j]
            j = j - 1

        A[j + 1] = x
    return A

def sort_by_txns(A: List[Query5Data]) -> List[Query5Data]:
    B = []
    start, end = 0, 0
    flag = True

    for i in range(len(A)):
        B.append(A[i])

        if i >= len(A) - 1:
            continue

        if flag and A[i].total_unique_nft != A[i+1].total_unique_nft:
            start = i+1
            continue

        if flag and (A[i].total_unique_nft == A[i+1].total_unique_nft):
            flag = False
            continue

        if not flag and (A[i].total_unique_nft != A[i+1].total_unique_nft) :
            end = i

            C = aux_sort_by_txns(A[start:end+1])
            k = 0
            for j in range(start, end+1):
                B[j] = C[k]
                k += 1


            start = i+1
            flag = True

    return B