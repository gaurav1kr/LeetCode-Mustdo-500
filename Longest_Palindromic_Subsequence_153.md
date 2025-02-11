# Longest Palindromic Subsequence

This file contains the explanation and solution for the [Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/) problem.

## Problem Description

Given a string `s`, find the length of the longest palindromic subsequence in `s`.

## Solution

### Approach

1. **Dynamic Programming**:
   - Use a space-optimized 1D DP approach to compute the longest palindromic subsequence.

2. **Key Points**:
   - If `s[i] == s[j]`, then `dp[j] = prev_dp[j-1] + 2`.
   - Otherwise, `dp[j] = max(prev_dp[j], dp[j-1])`.

3. **Optimization**:
   - Instead of a 2D DP table, we use two 1D arrays (`dp` and `prev_dp`) to reduce space complexity from \(O(n^2)\) to \(O(n)\).

### Code

```cpp
#include <vector>
#include <string>
#include <algorithm>

class Solution {
public:
    int longestPalindromeSubseq(const std::string& s) {
        int n = s.size();
        std::vector<int> dp(n, 0), prev_dp(n, 0);

        for (int i = n - 1; i >= 0; --i) {
            dp[i] = 1; // Single characters are palindromes of length 1
            for (int j = i + 1; j < n; ++j) {
                if (s[i] == s[j]) {
                    dp[j] = prev_dp[j - 1] + 2;
                } else {
                    dp[j] = std::max(prev_dp[j], dp[j - 1]);
                }
            }
            prev_dp.swap(dp); // Save the current row for the next iteration
        }

        return prev_dp[n - 1];
    }
};
```

### Complexity

- **Time Complexity**: \(O(n^2)\), where \(n\) is the length of the string.
- **Space Complexity**: \(O(n)\), due to the use of 1D DP arrays.

## Notes

This solution is concise and optimized for competitive programming scenarios.
