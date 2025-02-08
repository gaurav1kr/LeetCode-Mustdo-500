
# Wildcard Matching Solution in C++

This document provides a detailed explanation and implementation of the "Wildcard Matching" problem from LeetCode.

## Problem Statement

You are given two strings `s` and `p`. Implement wildcard pattern matching with support for `?` and `*`.

- `?` Matches any single character.
- `*` Matches any sequence of characters (including the empty sequence).

The goal is to determine if the string `s` matches the pattern `p`.

## Approach

The solution uses **dynamic programming (DP)** for efficiency. The DP approach avoids redundant computations and has a time complexity of \(O(n \cdot m)\), where:

- \(n\) is the length of the string `s`.
- \(m\) is the length of the pattern `p`.

### DP Table Definition

- Let `dp[i][j]` represent whether the substring `s[0..i-1]` matches the pattern `p[0..j-1]`.

### Base Cases

1. An empty string matches an empty pattern:
   ```
   dp[0][0] = true
   ```

2. For patterns that start with `*`, they can match an empty string:
   ```
   dp[0][j] = dp[0][j-1] (if p[j-1] == '*')
   ```

### Transition Rules

1. If `p[j-1]` is `?` or matches `s[i-1]`:
   ```
   dp[i][j] = dp[i-1][j-1]
   ```

2. If `p[j-1]` is `*`, it can match:
   - An empty sequence: `dp[i][j-1]`.
   - A sequence that includes `s[i-1]`: `dp[i-1][j]`.

   ```
   dp[i][j] = dp[i-1][j] || dp[i][j-1]
   ```

### Final Result

The result is stored in `dp[n][m]`, where \(n\) and \(m\) are the lengths of `s` and `p`, respectively.

## Implementation

Below is the C++ implementation of the solution:

```cpp
#include <vector>
#include <string>
#include <iostream>
using namespace std;

class Solution {
public:
    bool isMatch(string s, string p) {
        int n = s.size(), m = p.size();
        
        // DP table where dp[i][j] means if s[0..i-1] matches p[0..j-1]
        vector<vector<bool>> dp(n + 1, vector<bool>(m + 1, false));
        
        // Base case: empty string matches empty pattern
        dp[0][0] = true;

        // Handle cases where the pattern starts with '*' characters
        for (int j = 1; j <= m; ++j) {
            if (p[j - 1] == '*') {
                dp[0][j] = dp[0][j - 1];
            }
        }

        // Fill the DP table
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (p[j - 1] == s[i - 1] || p[j - 1] == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p[j - 1] == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
            }
        }

        return dp[n][m];
    }
};

// Example usage
int main() {
    Solution solution;
    string s = "adceb";
    string p = "*a*b";
    
    if (solution.isMatch(s, p)) {
        cout << "The string matches the pattern." << endl;
    } else {
        cout << "The string does not match the pattern." << endl;
    }

    return 0;
}
```

## Example

### Input:

```text
s = "adceb"
p = "*a*b"
```

### Output:

```text
The string matches the pattern.
```

## Complexity Analysis

### Time Complexity:

\(O(n \cdot m)\), where \(n\) is the length of `s` and \(m\) is the length of `p`. Each cell in the DP table is computed once.

### Space Complexity:

\(O(n \cdot m)\) for the DP table. This can be optimized to \(O(m)\) by using a single-dimensional array.

## Optimizations

The space complexity can be reduced to \(O(m)\) by using a rolling array. If you would like an optimized version, feel free to ask!
