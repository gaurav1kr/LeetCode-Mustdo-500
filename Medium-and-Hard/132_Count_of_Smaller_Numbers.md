# Solution to "Count of Smaller Numbers After Self"

This document provides a concise and optimized C++ solution for the LeetCode problem ["Count of Smaller Numbers After Self"](https://leetcode.com/problems/count-of-smaller-numbers-after-self).

---

## Explanation

The solution uses the following concepts:

1. **Coordinate Compression:**  
   This technique maps numbers to a smaller range, allowing us to handle negative numbers and large ranges efficiently.

2. **Fenwick Tree (Binary Indexed Tree):**  
   A Fenwick Tree is utilized for efficient prefix sum queries and updates in logarithmic time.

---

## Code

```cpp
#include <vector>
#include <algorithm>
#include <map>

using namespace std;

class Solution {
public:
    vector<int> countSmaller(vector<int>& nums) {
        int n = nums.size();
        vector<int> result(n), sortedNums(nums);

        // Coordinate compression
        sort(sortedNums.begin(), sortedNums.end());
        map<int, int> ranks;
        for (int i = 0; i < sortedNums.size(); ++i) {
            ranks[sortedNums[i]] = i + 1;
        }

        // Fenwick Tree
        vector<int> fenwickTree(n + 1, 0);

        auto update = [&](int index, int value) {
            while (index <= n) {
                fenwickTree[index] += value;
                index += index & -index;
            }
        };

        auto query = [&](int index) -> int {
            int sum = 0;
            while (index > 0) {
                sum += fenwickTree[index];
                index -= index & -index;
            }
            return sum;
        };

        // Process from right to left
        for (int i = n - 1; i >= 0; --i) {
            int rank = ranks[nums[i]];
            result[i] = query(rank - 1);
            update(rank, 1);
        }

        return result;
    }
};
```

---

## Key Features

1. **Time Complexity:**  
   The overall time complexity is \(O(n \log n)\), where \(n\) is the size of the input array. This is achieved using coordinate compression and Fenwick Tree operations.

2. **Space Complexity:**  
   The space complexity is \(O(n)\), which includes space for the Fenwick Tree and the ranks mapping.

---

## How It Works

1. **Coordinate Compression:**  
   By sorting the input array and assigning ranks to each unique value, we reduce the input space to indices ranging from 1 to \(n\).

2. **Fenwick Tree Operations:**  
   - `update(index, value)` adds a value to the tree at a specific index.
   - `query(index)` computes the prefix sum up to a given index, which is used to count smaller numbers.

3. **Processing from Right to Left:**  
   This ensures that at each step, we consider only numbers that come after the current index.

---

## Example

Input: `nums = [5, 2, 6, 1]`  
Output: `[2, 1, 1, 0]`

**Explanation:**  
- For 5, there are 2 smaller numbers (2, 1) after it.  
- For 2, there is 1 smaller number (1) after it.  
- For 6, there is 1 smaller number (1) after it.  
- For 1, there are no smaller numbers after it.

