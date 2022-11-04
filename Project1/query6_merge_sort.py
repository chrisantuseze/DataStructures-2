from typing import List
import numpy as np

from nfttransaction_data import NFTTransaction
from ml_data import MLData
from query6_utils import update_with_n_unique_txns
from query6_data import Query6Data, Query6Input

def merge_sort_by_ntxn(A: List[MLData]):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_ntxn(B)
    R = merge_sort_by_ntxn(C)
    return merge_by_ntxn(L, R)

def merge_by_ntxn(L: List[MLData], R: List[MLData]):
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

def merge_sort_by_fraudulent(A: List[MLData]) -> List[MLData]:
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_fraudulent(B)
    R = merge_sort_by_fraudulent(C)
    return merge_by_fraudulent(L, R)

def merge_by_fraudulent(L: List[MLData], R: List[MLData]) -> List[MLData]:
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and (str(L[i].fraudulent) >= str(R[j].fraudulent))):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B