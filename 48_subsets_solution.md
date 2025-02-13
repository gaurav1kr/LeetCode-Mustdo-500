## Optimal C++ Solutions for LeetCode "Subsets" Problem

### **Problem Statement:**
Given an integer array `nums` of **unique** elements, return all possible subsets (the power set).

**Example:**
```cpp
Input: nums = [1,2,3]
Output: [[],[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]
```

---

## **Approach 1: Backtracking**
### **Idea:**
- Use **backtracking (DFS)** to explore all possible subsets.
- Start with an empty subset and iteratively add elements while exploring further.
- **Time Complexity:** \( O(2^n) \) (each subset is generated once).
- **Space Complexity:** \( O(2^n) \) (storing all subsets).

### **C++ Code:**
```cpp
#include <vector>

class Solution {
public:
    void backtrack(int start, std::vector<int>& nums, std::vector<int>& current, std::vector<std::vector<int>>& result) {
        result.push_back(current);  // Store current subset

        for (int i = start; i < nums.size(); ++i) {
            current.push_back(nums[i]);      // Include nums[i]
            backtrack(i + 1, nums, current, result);  // Recurse with next elements
            current.pop_back();              // Backtrack
        }
    }

    std::vector<std::vector<int>> subsets(std::vector<int>& nums) {
        std::vector<std::vector<int>> result;
        std::vector<int> current;
        backtrack(0, nums, current, result);
        return result;
    }
};
```

---

## **Approach 2: Bit Manipulation**
### **Idea:**
- Each subset can be represented using a **binary mask** from `0` to `2^n - 1`.
- If the `i`th bit is `1`, include `nums[i]` in the subset.
- **Time Complexity:** \( O(2^n \cdot n) \) (iterating over `n` bits for `2^n` subsets).
- **Space Complexity:** \( O(2^n) \) (storing all subsets).

### **C++ Code:**
```cpp
#include <vector>

class Solution {
public:
    std::vector<std::vector<int>> subsets(std::vector<int>& nums) {
        int n = nums.size();
        int totalSubsets = 1 << n;  // 2^n subsets
        std::vector<std::vector<int>> result;

        for (int mask = 0; mask < totalSubsets; ++mask) {
            std::vector<int> subset;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {  // Check if i-th element is in subset
                    subset.push_back(nums[i]);
                }
            }
            result.push_back(subset);
        }
        return result;
    }
};
```

---

## **Comparison of Approaches**
| Approach       | Time Complexity | Space Complexity |
|---------------|----------------|-----------------|
| **Backtracking** | \( O(2^n) \) | \( O(2^n) \) |
| **Bit Masking** | \( O(2^n \cdot n) \) | \( O(2^n) \) |

**Which One to Use?**
- **Backtracking** is preferable when subsets are needed in lexicographical order.
- **Bit Masking** is often slightly faster for small `n` due to no recursion overhead.

---

## **Conclusion**
- Both approaches generate all subsets efficiently.
- Use **backtracking** when order matters and **bit manipulation** for a simpler, non-recursive method.
