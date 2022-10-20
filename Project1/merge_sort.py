def merge_sort_by_tokenid(A):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_tokenid(B)
    R = merge_sort_by_tokenid(C)
    return merge_by_tokenid(L, R)

def merge_by_tokenid(L, R):
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and (int(L[i].token_id) >= int(R[j].token_id))):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B


def merge_sort_by_nbuyer(A):
    if len(A) == 1:
        return A

    q = int(len(A)/2)
    B = A[:q]
    C = A[q:]

    L = merge_sort_by_nbuyer(B)
    R = merge_sort_by_nbuyer(C)
    return merge_by_nbuyer(L, R)

def merge_by_nbuyer(L, R):
    n = len(L) + len(R)
    i = j = 0
    B = []
    for k in range(0, n):
        if j >= len(R) or (i < len(L) and L[i].n_unique_buyers >= R[j].n_unique_buyers):
            B.append(L[i])
            i = i + 1
        else:
            B.append(R[j])
            j = j + 1

    return B