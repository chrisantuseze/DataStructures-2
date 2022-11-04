from typing import List

from query1_data import Query1Data, Query1Input


def merge_sort_by_ntxn(A: List[Query1Data]):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_ntxn(B)
    R = merge_sort_by_ntxn(C)
    return merge_by_ntxn(L, R)

def merge_by_ntxn(L: List[Query1Data], R: List[Query1Data]):
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and L[i].n_txns >= R[j].n_txns):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B