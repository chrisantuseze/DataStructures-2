# Python program for implementation of Radix Sort
# A function to do counting sort of arr[] according to
# the digit represented by exp.

from typing import List
from query6_data import Query6Data, Query6Input
from ml_data import MLData

def counting_sort_by_fraudulent(A: List[MLData], exp) -> List[MLData]:
    n = len(A)
 
    # The output array elements that will have sorted arr
    output = [0] * (n)
 
    # initialize count array as 0
    count = [0] * (10)
 
    # Store count of occurrences in count[]
    for i in range(0, n):
        index = int(A[i].fraudulent_ascii) // exp
        index = int(index)

        count[index % 10] += 1
 
    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]
 
    # Build the output array
    i = n - 1
    while i >= 0:
        index = int(A[i].fraudulent_ascii) // exp
        index = int(index)

        output[count[index % 10] - 1] = A[i]
        count[index % 10] -= 1
        i -= 1
 
    return output
 
# Method to do Radix Sort
def radix_sort_by_fraudulent(A: List[MLData]) -> List[MLData]:
    # Find the maximum number to know number of digits
    max1 = max_by_fraudulent(A)
 
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp >= 1:
        A = counting_sort_by_fraudulent(A, exp)
        exp *= 10

    B = []
    n = len(A)
    for i in range(n):
        B.append(A[n - i - 1])

    return B

def max_by_fraudulent(A: List[MLData]) -> List[MLData]:
    max = 0
    for value in A:
        fraudulent_ascii = int(value.fraudulent_ascii)
        if  fraudulent_ascii > max:
            max = fraudulent_ascii

    return max