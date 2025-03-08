## Edit Distance Problem - Optimal C++ Solution

### Problem Statement:
Given two strings `word1` and `word2`, return the minimum number of operations required to convert `word1` to `word2`. Allowed operations:
1. Insert a character
2. Delete a character
3. Replace a character

### Optimal C++ Solution (1D DP - Space Optimized):

```cpp
#include <iostream>
#include <vector>
using namespace std;

int minDistance(string word1, string word2) {
    int m = word1.size(), n = word2.size();
    vector<int> prev(n + 1), curr(n + 1);

    // Initialize base case: converting an empty string to word2
    for (int j = 0; j <= n; j++) {
        prev[j] = j;
    }

    for (int i = 1; i <= m; i++) {
        curr[0] = i; // Converting word1[0..i] to an empty string requires 'i' deletions
        for (int j = 1; j <= n; j++) {
            if (word1[i - 1] == word2[j - 1]) {
                curr[j] = prev[j - 1]; // No operation needed
            } else {
                curr[j] = 1 + min({prev[j - 1], prev[j], curr[j - 1]});
                // prev[j-1] -> replace, prev[j] -> delete, curr[j-1] -> insert
            }
        }
        prev = curr; // Move to next row
    }

    return prev[n];
}

int main() {
    string word1 = "horse", word2 = "ros";
    cout << "Minimum Edit Distance: " << minDistance(word1, word2) << endl;
    return 0;
}
```

### Explanation:
1. **Dynamic Programming Table (1D Optimization)**:
   - Uses **two 1D arrays** (`prev` and `curr`) instead of a full `dp[m+1][n+1]` table.
   - Iteratively updates rows while keeping only the last row in memory.

2. **Base Case Initialization**:
   - `prev[j] = j` (converting empty string to `word2` requires `j` insertions).
   - `curr[0] = i` (converting `word1[0..i]` to empty requires `i` deletions).

3. **Recurrence Relation**:
   - If characters match: `curr[j] = prev[j-1]`
   - Otherwise, consider three operations:
     - **Insert** (`curr[j-1]` → move left)
     - **Delete** (`prev[j]` → move up)
     - **Replace** (`prev[j-1]` → diagonal move)
   - Take the minimum and add `1` operation.

### Complexity Analysis:
- **Time Complexity**: `O(m * n)` (Iterate through DP table)
- **Space Complexity**: `O(min(m, n))` (Using only 2 rows instead of full matrix)

This approach significantly **reduces space usage** while maintaining efficiency.
