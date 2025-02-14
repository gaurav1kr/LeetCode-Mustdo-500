## Minimum Window Substring - C++ Solution

### **Approach:**
- Use **two pointers** (left and right) to maintain a sliding window.
- Use a **hash map** to store the frequency of characters in `t`.
- Expand the window to include required characters.
- Once all characters are included, try to shrink the window while maintaining the required condition.
- Keep track of the minimum window found.

### **Code:**
```cpp
#include <bits/stdc++.h>
using namespace std;

string minWindow(string s, string t) {
    if (s.size() < t.size()) return "";

    unordered_map<char, int> freq;
    for (char c : t) freq[c]++;

    int left = 0, right = 0, minLen = INT_MAX, count = t.size(), start = 0;
    
    while (right < s.size()) {
        if (freq[s[right]]-- > 0) count--; // Found a needed character
        right++;

        while (count == 0) { // Valid window
            if (right - left < minLen) {
                minLen = right - left;
                start = left;
            }
            if (++freq[s[left]] > 0) count++; // Shrink from left
            left++;
        }
    }
    
    return minLen == INT_MAX ? "" : s.substr(start, minLen);
}
```

### **Complexity Analysis:**
- **Time Complexity:** \(O(N)\), where \(N\) is the length of `s` (each character is processed at most twice).
- **Space Complexity:** \(O(1)\) (since the character set is limited to 256 in ASCII).

This solution efficiently finds the minimum window substring in **linear time** using the sliding window technique. 🚀
