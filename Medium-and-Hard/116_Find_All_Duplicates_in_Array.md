# Find All Duplicates in an Array

This markdown file provides a concise and optimized C++ solution to the LeetCode problem ["Find All Duplicates in an Array"](https://leetcode.com/problems/find-all-duplicates-in-an-array/description/).

## Problem Statement
Given an integer array `nums` of length `n` where all the integers of `nums` are in the range `[1, n]` and each integer appears **once** or **twice**, return an array of all the integers that appear twice.

You must write an algorithm that runs in `O(n)` time and uses only `O(1)` extra space.

---

## Solution
The solution uses the input array to mark visited elements, ensuring the solution is efficient in both time and space complexity.

### C++ Code
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> duplicates;
        for (int i = 0; i < nums.size(); ++i) {
            int index = abs(nums[i]) - 1;
            if (nums[index] < 0)
                duplicates.push_back(index + 1);
            else
                nums[index] = -nums[index];
        }
        return duplicates;
    }
};
```

---

## Explanation
### Key Idea
- Use the input array to mark visited elements by negating the value at the index corresponding to each number.
- If a number's corresponding index is already negative, it indicates a duplicate.

### Steps
1. Traverse the array.
2. For each number `nums[i]`:
   - Calculate the index: `abs(nums[i]) - 1`.
   - Check if the value at `nums[index]` is negative:
     - If yes, the number is a duplicate; add it to the result.
     - Otherwise, mark the value at `nums[index]` as visited by negating it.
3. Return the list of duplicates.

### Complexity
- **Time Complexity**: `O(n)`
  - The array is traversed once.
- **Space Complexity**: `O(1)`
  - No extra data structures are used; the input array is modified in place.

---

## Example
### Input
```text
nums = [4,3,2,7,8,2,3,1]
```
### Output
```text
[2, 3]
```
### Explanation
- After processing the array, the duplicates are identified as `2` and `3`.

---

This approach is efficient and adheres to the problem constraints.
