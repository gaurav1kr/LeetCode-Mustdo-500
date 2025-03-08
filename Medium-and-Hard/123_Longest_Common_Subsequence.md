
### Longest Common Subsequence - Optimized C++ Solution

The following solution uses **Dynamic Programming** with a **space-optimized approach** to solve the Longest Common Subsequence problem.

```cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int longestCommonSubsequence(string text1, string text2) {
    int m = text1.size(), n = text2.size();
    vector<int> prev(n + 1, 0), curr(n + 1, 0);

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (text1[i - 1] == text2[j - 1]) {
                curr[j] = 1 + prev[j - 1];
            } else {
                curr[j] = max(prev[j], curr[j - 1]);
            }
        }
        prev = curr;
    }

    return prev[n];
}

int main() {
    string text1 = "abcde";
    string text2 = "ace";
    cout << "Longest Common Subsequence Length: " << longestCommonSubsequence(text1, text2) << endl;
    return 0;
}
```

---

### Explanation

1. **Time Complexity**: 
   - \(O(m \times n)\), where \(m\) and \(n\) are the lengths of `text1` and `text2`.
   - This is because we iterate through all characters of both strings.

2. **Space Complexity**: 
   - \(O(n)\), since we only store the `prev` and `curr` rows of the DP table, reducing memory usage from \(O(m \times n)\) to \(O(n)\).

---

### Approach
- We maintain two rows: `prev` (previous row) and `curr` (current row).
- If the characters match (`text1[i-1] == text2[j-1]`), the value is updated as:
  ```
  curr[j] = 1 + prev[j-1];
  ```
- If they don't match, we take the maximum of the previous results:
  ```
  curr[j] = max(prev[j], curr[j-1]);
  ```
- At the end of each outer loop iteration, the current row becomes the previous row (`prev = curr`).

---

### Example

#### Input:
```text
text1 = "abcde"
text2 = "ace"
```

#### Output:
```text
Longest Common Subsequence Length: 3
```

---

This solution is efficient and suitable for large inputs due to its reduced space complexity.
