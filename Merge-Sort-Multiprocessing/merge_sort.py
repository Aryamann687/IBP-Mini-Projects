import time
import random
import multiprocessing

# -----------------------------
# NORMAL MERGE SORT
# -----------------------------
def merge(arr, left, mid, right):
    L = arr[left:mid+1]
    R = arr[mid+1:right+1]

    i = j = 0
    k = left

    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1

    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1


def merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)


# -----------------------------
# HELPER (normal sort for subprocess)
# -----------------------------
def sort_subarray(arr):
    merge_sort(arr, 0, len(arr) - 1)
    return arr


# -----------------------------
# PARALLEL MERGE SORT (FIXED)
# -----------------------------
def parallel_merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    with multiprocessing.Pool(2) as pool:
        left, right = pool.map(sort_subarray, [left, right])

    return merge_lists(left, right)


def merge_lists(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    arr = [random.randint(1, 10000000) for _ in range(1000000)]

    # Normal
    arr1 = arr.copy()
    start = time.time()
    merge_sort(arr1, 0, len(arr1) - 1)
    end = time.time()
    print("Normal Merge Sort Time:", end - start)

    # Parallel
    arr2 = arr.copy()
    start = time.time()
    sorted_arr = parallel_merge_sort(arr2)
    end = time.time()
    print("Parallel Merge Sort Time:", end - start)

    print("Sorting Correct:", arr1 == sorted_arr)
