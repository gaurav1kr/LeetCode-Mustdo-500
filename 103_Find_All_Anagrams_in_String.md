# Find All Anagrams in a String (LeetCode)

This is a concise and optimized solution for the [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/) problem using the sliding window technique.

### Problem Description
Given two strings `s` and `p`, return an array of all the start indices of `p`'s anagrams in `s`. You may return the answer in any order.

### Solution Code
```cpp
#include <vector>
#include <string>
using namespace std;

vector<int> findAnagrams(string s, string p) {
    vector<int> result, pCount(26, 0), sCount(26, 0);
    int pLen = p.size(), sLen = s.size();

    if (sLen < pLen) return result;

    for (int i = 0; i < pLen; ++i) {
        ++pCount[p[i] - 'a'];
        ++sCount[s[i] - 'a'];
    }

    if (pCount == sCount) result.push_back(0);

    for (int i = pLen; i < sLen; ++i) {
        ++sCount[s[i] - 'a'];
        --sCount[s[i - pLen] - 'a'];
        if (pCount == sCount) result.push_back(i - pLen + 1);
    }

    return result;
}
```

### Explanation
1. **Initialization**:
   - `pCount` and `sCount` are frequency vectors for `p` and the current sliding window in `s`, respectively.
   - Count character frequencies in the first `pLen` characters of both `p` and `s`.

2. **Sliding Window**:
   - Slide the window one character at a time:
     - Add the next character in `s` to `sCount`.
     - Remove the character that’s sliding out of the window from `sCount`.
   - Check if `sCount` matches `pCount`. If yes, add the starting index of the window to `result`.

3. **Return Result**:
   - Return the list of starting indices of anagrams.

### Complexity Analysis
- **Time Complexity**: 
  - Building the initial count: \(O(pLen)\)
  - Sliding the window: \(O(sLen)\)
  - Overall: \(O(sLen + pLen)\)

- **Space Complexity**:
  - The space used for frequency vectors is \(O(26) = O(1)\).

### Example Usage
```cpp
int main() {
    string s = "cbaebabacd";
    string p = "abc";
    vector<int> result = findAnagrams(s, p);
    // Output: [0, 6]
    return 0;
}
```

This approach is both efficient and easy to understand, making it suitable for competitive programming and interviews.
