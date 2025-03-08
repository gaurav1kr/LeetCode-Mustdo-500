## Regular Expression Matching - Optimal C++ Solution

### **Problem Description:**
Given an input string `s` and a pattern `p`, implement regular expression matching with support for `.` and `*`.

- `.` Matches any single character.
- `*` Matches zero or more of the preceding element.

The function should return `true` if the string matches the pattern, otherwise `false`.

---

### **Approach:**
We use **Dynamic Programming (DP) with Memoization** to efficiently match the string against the pattern.

1. **Recursive Matching with Memoization:**
   - Use a 2D `memo` table to store computed results for `(i, j)` states.
   - This avoids redundant calculations and speeds up execution.

2. **Handling `*` Character:**
   - If `p[j+1]` is `*`, consider two cases:
     - Ignore `p[j]` and move to `p[j+2]` (zero occurrences).
     - If `s[i]` matches `p[j]`, recursively check for `s[i+1]` with `p[j]` (one or more occurrences).
   
3. **Handling `.` Character:**
   - Directly matches any character at `s[i]`.

---

### **C++ Code Implementation:**
```cpp
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> memo;

    bool isMatch(string s, string p) {
        int m = s.size(), n = p.size();
        memo.assign(m + 1, vector<int>(n + 1, -1));
        return dp(0, 0, s, p);
    }

    bool dp(int i, int j, const string &s, const string &p) {
        if (memo[i][j] != -1) return memo[i][j];

        if (j == p.size()) return memo[i][j] = (i == s.size());

        bool first_match = (i < s.size() && (s[i] == p[j] || p[j] == '.'));

        if (j + 1 < p.size() && p[j + 1] == '*') {
            memo[i][j] = (dp(i, j + 2, s, p) || // Zero occurrence
                          (first_match && dp(i + 1, j, s, p))); // One or more occurrence
        } else {
            memo[i][j] = first_match && dp(i + 1, j + 1, s, p);
        }

        return memo[i][j];
    }
};
```

---

### **Complexity Analysis:**
- **Time Complexity:** \(O(m \times n)\), where `m` is the length of `s` and `n` is the length of `p`. Each `(i, j)` state is computed once.
- **Space Complexity:** \(O(m \times n)\) for memoization table.

---

### **Why This Solution?**
✅ **Efficient:** Avoids redundant computations using **top-down memoization**.
✅ **Handles Edge Cases:** Works for empty strings, multiple `*`, and `.` wildcards.
✅ **LeetCode Optimized:** Runs efficiently even for large inputs.

Let me know if you need further explanations! 🚀
