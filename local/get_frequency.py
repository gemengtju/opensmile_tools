#/bin/python
import numpy as np
from collections import Counter

def counter(arr):
    return Counter(arr).most_common(2)

def single_list(arr, target):
    return arr.count(target)

def all_list(arr):
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result

def single_np(arr, target):
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size

def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result





if __name__ == "__main__":
    array = [1, 2, 3, 3, 2, 1, 0, 2]
    print(counter(array))
    print(single_list(array, 2))
    print(all_list(array))
    print(single_np(array, 2))
    print(all_np(array))
