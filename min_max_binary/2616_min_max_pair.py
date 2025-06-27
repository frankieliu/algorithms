from typing import List
class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        def can_form_p_pairs_with(max_allowed_diff):
            pairs = 0
            i = 0
            while i + 1 < N:
                if nums[i + 1] - nums[i] <= max_allowed_diff:
                    pairs += 1
                    i += 1
                i += 1
            return pairs >= p
        N = len(nums)
        nums.sort()
        left = 0
        right = nums[-1] - nums[0]
        while left < right:
            mid = (left + right) // 2
            if can_form_p_pairs_with(mid):
                right = mid
            else:
                left = mid + 1
        return left