# Permutation in String

This is a concise and optimized solution for the [Permutation in String](https://leetcode.com/problems/permutation-in-string/) problem using the sliding window technique in C++.

```cpp
#include <string>
#include <vector>
using namespace std;

class Solution 
{
public:
    bool checkInclusion(string s1, string s2) 
    {
        if (s1.size() > s2.size()) return false;

        vector<int> s1Count(26, 0), s2Count(26, 0);

        // Count frequency of characters in s1
        for (char c : s1) s1Count[c - 'a']++;

        // Sliding window on s2
        for (int i = 0; i < s2.size(); i++) 
	{
            s2Count[s2[i] - 'a']++;  // Add current character to the window
            if (i >= s1.size())     // Remove character out of window
                s2Count[s2[i - s1.size()] - 'a']--;
            if (s1Count == s2Count) // Compare counts
                return true;
        }

        return false;
    }
};
```
e
## Explanation

### Sliding Window
- Maintain a window of size `s1` on `s2`.
- Use `s1Count` and `s2Count` vectors to track character frequencies for `s1` and the current window in `s2`.

### Efficiency
- **Time complexity**: **O(n)** where `n` is the size of `s2` (the sliding window traverses `s2` once).
- **Space complexity**: **O(1)** since the frequency arrays are fixed at size 26 (for lowercase English letters).

### Key Optimization
- Instead of recalculating counts for every substring, adjust the frequency vector dynamically as the window slides.
