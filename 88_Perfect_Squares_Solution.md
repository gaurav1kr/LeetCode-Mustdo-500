
# Solution to LeetCode Problem: Perfect Squares

## Problem Description
Given a positive integer `n`, find the least number of perfect square numbers (e.g., `1, 4, 9, 16, ...`) which sum to `n`.

You can assume that `1 <= n <= 10^4`.

---

## Optimized C++ Solution

```cpp
#include <vector>
#include <cmath>
#include <algorithm>

class Solution {
public:
    int numSquares(int n) {
        // DP array to store the minimum number of perfect squares for each number
        std::vector<int> dp(n + 1, INT_MAX);
        
        // Base case: 0 can be represented as the sum of 0 perfect squares
        dp[0] = 0;

        // Precompute all perfect squares less than or equal to n
        std::vector<int> perfectSquares;
        for (int i = 1; i * i <= n; ++i) {
            perfectSquares.push_back(i * i);
        }

        // Fill the DP array
        for (int i = 1; i <= n; ++i) {
            for (int square : perfectSquares) {
                if (i < square) break; // No need to check further if the square exceeds i
                dp[i] = std::min(dp[i], dp[i - square] + 1);
            }
        }

        return dp[n];
    }
};
```

---

## Explanation

### Approach
This problem is solved using **Dynamic Programming (DP)**. The key idea is to find the minimum number of perfect square numbers that add up to the given number `n`. We use a DP array where `dp[i]` represents the minimum number of perfect squares that sum up to `i`.

### Steps:
1. **Initialization**:
   - A `dp` vector of size `n+1` is initialized with `INT_MAX` to represent large values.
   - `dp[0]` is set to 0 since 0 can be represented as the sum of 0 perfect squares.

2. **Perfect Squares Precomputation**:
   - All perfect squares less than or equal to `n` are precomputed and stored in the `perfectSquares` vector to reduce redundant calculations.

3. **Dynamic Programming Transition**:
   - For each number `i` from 1 to `n`, iterate through all precomputed perfect squares. If the current square is less than or equal to `i`, update `dp[i]` using the relation:
     ```
     dp[i] = min(dp[i], dp[i - square] + 1)
     ```

4. **Return the Result**:
   - After the `dp` array is filled, `dp[n]` contains the minimum number of perfect squares that sum up to `n`.

---

## Complexity Analysis

### Time Complexity
- **Outer Loop**: `O(n)`
- **Inner Loop**: `O(sqrt(n))` (because there are roughly `sqrt(n)` perfect squares less than or equal to `n`)
- **Overall**: `O(n * sqrt(n))`

### Space Complexity
- `O(n)` for the DP array and the precomputed perfect squares.

---

## Example Usage

```cpp
#include <iostream>
int main() {
    Solution solution;
    int n = 12;
    std::cout << "Minimum perfect squares for " << n << ": " << solution.numSquares(n) << std::endl;
    return 0;
}
```

**Input**: `n = 12`  
**Output**: `3` (since `12 = 4 + 4 + 4`)

---

## Summary
This solution is efficient, utilizing Dynamic Programming to ensure optimal performance. The precomputation of perfect squares reduces redundant calculations, and the `dp` array keeps track of the minimum counts for each number from `1` to `n`. The approach has a time complexity of `O(n * sqrt(n))` and passes all test cases on LeetCode.
