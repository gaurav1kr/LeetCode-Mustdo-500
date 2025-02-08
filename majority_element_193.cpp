# Majority Element II - LeetCode Solution

This markdown file provides the optimized C++ solution for the [Majority Element II](https://leetcode.com/problems/majority-element-ii/) problem using the **Boyer-Moore Voting Algorithm**.

## Problem Description

Given an integer array `nums`, find all elements that appear more than `\lfloor n / 3 \rfloor` times.

### Constraints:
- `1 <= nums.length <= 5 * 10^4`
- `-10^9 <= nums[i] <= 10^9`

The problem requires solving in linear time `O(n)` and constant space `O(1)`.

---

## Optimized C++ Solution

Below is the concise and efficient implementation:

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        int n = nums.size();
        int candidate1 = 0, candidate2 = 1, count1 = 0, count2 = 0;

        // Phase 1: Find potential candidates
        for (int num : nums) {
            if (num == candidate1) count1++;
            else if (num == candidate2) count2++;
            else if (count1 == 0) candidate1 = num, count1 = 1;
            else if (count2 == 0) candidate2 = num, count2 = 1;
            else count1--, count2--;
        }

        // Phase 2: Verify candidates
        count1 = count2 = 0;
        for (int num : nums) {
            if (num == candidate1) count1++;
            else if (num == candidate2) count2++;
        }

        vector<int> result;
        if (count1 > n / 3) result.push_back(candidate1);
        if (count2 > n / 3) result.push_back(candidate2);
        return result;
    }
};
```

---

## Explanation

### **Phase 1: Identify Potential Candidates**
1. Use two variables `candidate1` and `candidate2` to store potential majority elements.
2. Use counters `count1` and `count2` to track their respective counts.
3. Traverse through the array:
   - Increment the counter if the current number matches a candidate.
   - Replace a candidate if its count drops to zero.
   - Decrease both counters if no match is found.

### **Phase 2: Verify Candidates**
1. Reset the counters to zero.
2. Traverse the array again to count the occurrences of the two candidates.
3. Add a candidate to the result if its count exceeds `n/3`.

---

## Complexity Analysis

- **Time Complexity**: `O(n)`
  - The algorithm traverses the array twice (Phase 1 and Phase 2).
- **Space Complexity**: `O(1)`
  - Only a few variables are used for tracking candidates and their counts.

---

## Example Usage

```cpp
int main() {
    Solution solution;
    vector<int> nums = {3, 2, 3};
    vector<int> result = solution.majorityElement(nums);

    for (int num : result) {
        cout << num << " ";
    }
    return 0;
}
```

### Input:
`nums = [3, 2, 3]`

### Output:
`[3]`

---

## Key Insights

1. At most, there can only be two majority elements greater than `n/3`.
2. The Boyer-Moore Voting Algorithm ensures efficient computation with minimal space usage.

This solution is both time and space efficient, making it ideal for handling large arrays within the problem's constraints.

