from typing import List
from query4_data import Query4Data

def insertion_sort(A: List[Query4Data], left=0, right=None):

    if right is None:

        right = len(A) - 1


    # Loop from the element indicated by

    # `left` until the element indicated by `right`

    for i in range(left + 1, right + 1):

        # This is the element we want to position in its

        # correct place

        key_item = A[i]


        # Initialize the variable that will be used to

        # find the correct position of the element referenced

        # by `key_item`

        j = i - 1


        # Run through the list of items (the left

        # portion of the array) and find the correct position

        # of the element referenced by `key_item`. Do this only

        # if the `key_item` is smaller than its adjacent values.

        while j >= left and A[j].n_unique_buyers > key_item.n_unique_buyers:

            # Shift the value one position to the left

            # and reposition `j` to point to the next element

            # (from right to left)

            A[j + 1] = A[j]

            j -= 1


        # When you finish shifting the elements, position

        # the `key_item` in its correct location

        A[j + 1] = key_item


    return A


def timsort(A: List[Query4Data]) -> List[Query4Data]:

    min_run = 32

    n = len(A)


    # Start by slicing and sorting small portions of the

    # input array. The size of these slices is defined by

    # your `min_run` size.

    for i in range(0, n, min_run):

        insertion_sort(A, i, min((i + min_run - 1), n - 1))


    # Now you can start merging the sorted slices.

    # Start from `min_run`, doubling the size on

    # each iteration until you surpass the length of

    # the array.

    size = min_run

    while size < n:

        # Determine the arrays that will

        # be merged together

        for start in range(0, n, size * 2):

            # Compute the `midpoint` (where the first array ends

            # and the second starts) and the `endpoint` (where

            # the second array ends)

            midpoint = start + size - 1

            end = min((start + size * 2 - 1), (n-1))


            # Merge the two subarrays.

            # The `left` array should go from `start` to

            # `midpoint + 1`, while the `right` array should

            # go from `midpoint + 1` to `end + 1`.

            merged_array = merge(

                L=A[start:midpoint + 1],

                R=A[midpoint + 1:end + 1])


            # Finally, put the merged array back into

            # your array

            A[start:start + len(merged_array)] = merged_array


        # Each iteration should double the size of your arrays

        size *= 2


    return A

def merge(L: List[Query4Data], R: List[Query4Data]) -> List[Query4Data]:

    # If the first array is empty, then nothing needs

    # to be merged, and you can return the second array as the result

    if len(L) == 0:

        return R


    # If the second array is empty, then nothing needs

    # to be merged, and you can return the first array as the result

    if len(R) == 0:

        return L


    result = []

    index_left = index_right = 0


    # Now go through both arrays until all the elements

    # make it into the resultant array

    while len(result) < len(L) + len(R):

        # The elements need to be sorted to add them to the

        # resultant array, so you need to decide whether to get

        # the next element from the first or the second array

        if L[index_left].n_unique_buyers <= R[index_right].n_unique_buyers:

            result.append(L[index_left])

            index_left += 1

        else:

            result.append(R[index_right])

            index_right += 1


        # If you reach the end of either array, then you can

        # add the remaining elements from the other array to

        # the result and break the loop

        if index_right == len(R):

            result += L[index_left:]

            break


        if index_left == len(L):

            result += R[index_right:]

            break


    return result


def merge_sort(A):

    # If the input array contains fewer than two elements,

    # then return it as the result of the function

    if len(A) < 2:

        return A


    midpoint = len(A) // 2


    # Sort the array by recursively splitting the input

    # into two equal halves, sorting each half and merging them

    # together into the final result

    return merge(

        left=merge_sort(A[:midpoint]),

        right=merge_sort(A[midpoint:]))