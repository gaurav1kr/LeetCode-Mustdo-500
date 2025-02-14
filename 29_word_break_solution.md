# Word Break - Optimal C++ Solution

## Problem Statement
Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

## C++ Solution

```cpp
#include <bits/stdc++.h>
using namespace std;

bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
    vector<bool> dp(s.size() + 1, false);
    dp[0] = true;

    for (int i = 1; i <= s.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (dp[j] && wordSet.count(s.substr(j, i - j))) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[s.size()];
}
```

## Complexity Analysis
- **Time Complexity:** \(O(n^2)\) (Nested loops for substring checking)
- **Space Complexity:** \(O(n)\) (DP array storage)

## Explanation
1. Convert `wordDict` into an `unordered_set` for **O(1) lookups**.
2. Create a DP array `dp`, where `dp[i]` represents whether `s[:i]` can be segmented.
3. Iterate over `s` and check all substrings `s[j:i]`:
   - If `dp[j]` is `true` and `s[j:i]` exists in `wordSet`, mark `dp[i]` as `true`.
   - Break early to optimize.

This solution efficiently determines whether `s` can be segmented using words from `wordDict`. 🚀
