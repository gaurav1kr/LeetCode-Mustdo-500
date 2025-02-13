
# Decode Ways - Optimal Solution in C++

## Problem Description
Given a string `s` containing only digits, return the number of ways to decode it. Each digit or pair of digits can represent a letter (`'1' -> 'A'`, `'2' -> 'B'`, ..., `'26' -> 'Z'`).

[LeetCode Problem Link](https://leetcode.com/problems/decode-ways/description/)

## Solution
Here is the optimal C++ solution using dynamic programming:

```cpp
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        if (n == 0 || s[0] == '0') return 0; // Edge case: invalid starting character
        
        vector<int> dp(n + 1, 0);
        dp[0] = 1; // Base case: empty string has one way to decode
        dp[1] = 1; // Single character (non-'0') has one way to decode
        
        for (int i = 2; i <= n; ++i) {
            // Check if single-digit decode is valid
            if (s[i - 1] != '0') {
                dp[i] += dp[i - 1];
            }
            
            // Check if two-digit decode is valid
            int twoDigit = stoi(s.substr(i - 2, 2));
            if (twoDigit >= 10 && twoDigit <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        
        return dp[n];
    }
};

int main() {
    Solution sol;
    string s = "226"; // Example input
    cout << "Number of ways to decode: " << sol.numDecodings(s) << endl;
    return 0;
}
```

## Explanation

### Dynamic Programming Approach:
- `dp[i]` represents the number of ways to decode the substring `s[0...i-1]`.
- **Base cases**:
  - `dp[0] = 1`: There is one way to decode an empty string.
  - `dp[1] = 1` if `s[0] != '0'`: A single non-zero digit can only be decoded in one way.
- **Transition**:
  - Add `dp[i - 1]` to `dp[i]` if the current character `s[i-1]` is not `'0'`.
  - Add `dp[i - 2]` to `dp[i]` if the last two characters form a valid two-digit number between 10 and 26.

### Complexity:
- **Time Complexity**: `O(n)`, where `n` is the length of the string `s`.
  - We iterate over the string once.
- **Space Complexity**: `O(n)` due to the `dp` array. This can be optimized further to `O(1)` by keeping track of only the last two `dp` states.

## Optimized Version

To save space, we can reduce the space complexity to `O(1)`:

```cpp
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        if (n == 0 || s[0] == '0') return 0;

        int prev2 = 1, prev1 = 1; // Initialize base cases
        for (int i = 1; i < n; ++i) {
            int current = 0;

            // Check if single-digit decode is valid
            if (s[i] != '0') {
                current += prev1;
            }

            // Check if two-digit decode is valid
            int twoDigit = stoi(s.substr(i - 1, 2));
            if (twoDigit >= 10 && twoDigit <= 26) {
                current += prev2;
            }

            prev2 = prev1;
            prev1 = current;
        }

        return prev1;
    }
};
```

This optimized version uses two variables `prev1` and `prev2` to store the results of the last two states, thus reducing space usage to constant `O(1)`.
