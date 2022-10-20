import random

def quick_sort_by_tokenid(A):
    p, r = 0, len(A) - 1
    return quick_sort_by_tokenid_(A, p, r)

def quick_sort_by_tokenid_(A, p, r):
    if p >= r:
        return A

    q = partition_by_tokenid(A, p, r)
    A = quick_sort_by_tokenid_(A, p, q - 1)
    A = quick_sort_by_tokenid_(A, q + 1, r)
    return A


def partition_by_tokenid(A, p, r):
    i = p - 1
    # pivot_index = random.randint(p, r)
    x = A[r]

    for q in range(p, r):
        if int(A[q].token_id) >= int(x.token_id):
            i = i + 1
            A[i], A[q] = A[q], A[i]

    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1

def quick_sort_by_nbuyer(A):
    p, r = 0, len(A) - 1
    return quick_sort_by_nbuyer_(A, p, r)

def quick_sort_by_nbuyer_(A, p, r):
    if p >= r:
        return A

    q = partition_by_nbuyer(A, p, r)
    A = quick_sort_by_nbuyer_(A, p, q - 1)
    A = quick_sort_by_nbuyer_(A, q + 1, r)
    return A


def partition_by_nbuyer(A, p, r):
    i = p - 1
    # pivot_index = random.randint(p, r)
    x = A[r]

    for q in range(p, r):
        # print(A[q])
        # print()
        if int(A[q].n_unique_buyers) >= int(x.n_unique_buyers):
            i = i + 1
            A[i], A[q] = A[q], A[i]

    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1
