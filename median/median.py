from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        sht, lon = nums1, nums2
        if len(sht) > len(lon):
            sht = nums2
            lon = nums1
        n = len(sht) + len(lon)
        half = n//2
        
        l, r = -1, len(sht)-1
        while True:
            sht_mid = (l+r)//2
            lon_mid = half - sht_mid - 2

            sht_val1 = sht[sht_mid] if sht_mid>=0 else -float('inf')
            sht_val2 = sht[sht_mid + 1] if sht_mid<=len(sht)-2 else float('inf')
            lon_val1 = lon[lon_mid] if lon_mid>=0 else -float('inf')
            lon_val2 = lon[lon_mid + 1] if lon_mid<=len(lon)-2 else float('inf')

            if sht_val2 < lon_val1:
                l = sht_mid + 1
            elif lon_val2 < sht_val1:
                r = sht_mid - 1
            elif n%2:
                return min(sht_val2, lon_val2)
            else:
                return (
                    max(sht_val1, lon_val1) +
                    min(sht_val2, lon_val2)
                )/2