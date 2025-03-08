
# Longest Substring Without Repeating Characters - C++ Solution

## Problem Description
Given a string `s`, find the length of the longest substring without repeating characters.

### Example Input and Output

- Input: `s = "abcabcbb"`
- Output: `3` (The answer is "abc", with the length of 3.)

## Optimal C++ Solution
Below is a concise and efficient solution using the **sliding window technique**:

```cpp
#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;

int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> lastIndex;
    int maxLength = 0, start = 0;

    for (int i = 0; i < s.size(); ++i) {
        if (lastIndex.count(s[i]) && lastIndex[s[i]] >= start) {
            start = lastIndex[s[i]] + 1;  // Move start to avoid repeating
        }
        lastIndex[s[i]] = i;  // Update the index of the character
        maxLength = max(maxLength, i - start + 1);  // Update the max length
    }

    return maxLength;
}

int main() {
    string s = "abcabcbb";
    cout << "Length of the longest substring: " << lengthOfLongestSubstring(s) << endl;
    return 0;
}
```

## Explanation of the Code

### Sliding Window Technique:
- A **start** pointer keeps track of the beginning of the current substring without repeating characters.
- The **unordered_map `lastIndex`** stores the last index of each character in the string.

### Algorithm Steps:
1. Loop through each character in the string using an index `i`.
2. If the character exists in `lastIndex` and its last occurrence is within the current window (`start <= lastIndex[s[i]]`), move the start pointer to `lastIndex[s[i]] + 1`.
3. Update the `lastIndex` of the current character to its current index.
4. Calculate the current substring length as `i - start + 1` and update `maxLength` if it's greater.

### Complexity:
- **Time Complexity**: \(O(n)\), where \(n\) is the length of the string. Each character is processed at most twice (once added, once removed from the sliding window).
- **Space Complexity**: \(O(min(n, m))\), where \(m\) is the size of the character set (e.g., 26 for lowercase English letters).

## Example Run

Input: `s = "abcabcbb"`

| i   | char | start | lastIndex | maxLength |
|-----|------|-------|-----------|-----------|
| 0   | 'a'  | 0     | {'a': 0}  | 1         |
| 1   | 'b'  | 0     | {'a': 0, 'b': 1} | 2         |
| 2   | 'c'  | 0     | {'a': 0, 'b': 1, 'c': 2} | 3         |
| 3   | 'a'  | 1     | {'a': 3, 'b': 1, 'c': 2} | 3         |
| 4   | 'b'  | 2     | {'a': 3, 'b': 4, 'c': 2} | 3         |
| 5   | 'c'  | 3     | {'a': 3, 'b': 4, 'c': 5} | 3         |
| 6   | 'b'  | 5     | {'a': 3, 'b': 6, 'c': 5} | 3         |
| 7   | 'b'  | 7     | {'a': 3, 'b': 7, 'c': 5} | 3         |

Output: `3`

## Usage
To test with your own input, replace the string `s` in the `main()` function and run the program.
