# https://medium.com/swlh/sorting-algorithms-demystified-efc6f0a5bf8f
import math
import numpy as np

# O(nk)
# O(n log10(range))
# that is because every radix position takes O(n)
# This counting sort takes O(n+10)
# In general O(n + range) 

def radix(a):
    # find the largest radix
    b = max(a)
    r = int(math.log(b)/math.log(10))
    for pos in range(r+1):
        count_sort(a, pos)

def count_sort(a, pos):
    pos = 10**pos
    count = [0]*10
    for el in a:
        count[el//pos % 10] += 1
    # buckets
    idx = [0]*10
    for i in range(1, 10):
        idx[i] = idx[i-1] + count[i-1]
    output = [0] * len(a)
    for el in a:
        bucket = el//pos % 10
        output[idx[bucket]] = el
        idx[bucket]+=1
    for i in range(len(a)):
        a[i] = output[i]

if __name__=="__main__":
    to_sort = list(np.random.randint(1,1000,size=1000))
    check_sort = sorted(to_sort)
    radix(to_sort)
    for i in range(len(to_sort)):
        if check_sort[i] != to_sort[i]:
            print("Error")
            break