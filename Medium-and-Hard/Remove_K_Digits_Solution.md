# Remove K Digits - Optimized C++ Solution

## Problem Description
The problem can be found on LeetCode: [Remove K Digits](https://leetcode.com/problems/remove-k-digits/)

**Objective:**
Given a string `num` representing a non-negative integer and an integer `k`, remove `k` digits from the number so that the resulting number is the smallest possible.

Return the resulting smallest number as a string.

---

## Solution
### C++ Code
```cpp
class Solution {
public:
    string removeKdigits(string num, int k) {
        if (k == num.size()) return "0";

        string result; // Acts as a stack
        for (char c : num) {
            while (!result.empty() && result.back() > c && k > 0) {
                result.pop_back();
                --k;
            }
            result.push_back(c);
        }

        // Remove remaining digits from the end if k > 0
        while (k > 0) {
            result.pop_back();
            --k;
        }

        // Remove leading zeros
        int nonZeroIndex = 0;
        while (nonZeroIndex < result.size() && result[nonZeroIndex] == '0') {
            ++nonZeroIndex;
        }
        
        result = result.substr(nonZeroIndex);

        return result.empty() ? "0" : result;
    }
};
```

---

### Explanation
1. **Stack Simulation with String:**
   - Use `result` as a stack to build the smallest number.
   - Pop characters from `result` if the current character (`c`) is smaller than the top of the stack and `k > 0`.

2. **Trim Remaining Digits:**
   - If `k > 0` after iterating through `num`, remove digits from the end of the stack.

3. **Remove Leading Zeros:**
   - Traverse the result to skip leading zeros using `substr`.

4. **Edge Cases:**
   - If `k == num.size()`, return "0".
   - Handle cases with leading zeros or empty results.

---

### Complexity Analysis
- **Time Complexity:**
  - **O(n):** Each digit is pushed and popped from the `result` stack at most once, where `n` is the size of `num`.

- **Space Complexity:**
  - **O(n):** Space required for the `result` string, which acts as a stack.

---

### Example Walkthrough
#### Input:
```plaintext
num = "1432219", k = 3
```
#### Process:
- Iteration 1: Push '1' into `result`.
- Iteration 2: Push '4'.
- Iteration 3: '3' is smaller than '4', so pop '4'. Then push '3'.
- Iteration 4: Push '2'.
- Iteration 5: Push '2'.
- Iteration 6: '1' is smaller than '2', so pop '2' twice. Then push '1'.
- Iteration 7: Push '9'.
- Remove remaining leading zeros if necessary.

#### Output:
```plaintext
"1219"
```

---

### Edge Cases
1. **`k == num.size()`**:
   - Input: `num = "10", k = 2`
   - Output: `"0"`

2. **Leading Zeros After Removal:**
   - Input: `num = "10200", k = 1`
   - Output: `"200"`

3. **Already Smallest Number:**
   - Input: `num = "1234", k = 0`
   - Output: `"1234"`
