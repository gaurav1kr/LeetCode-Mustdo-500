# Palindromic Substrings Solution

Here is an optimized solution for the [LeetCode Problem: Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/description/) written in C++.

### Problem Description
Given a string `s`, your task is to count how many substrings of `s` are palindromic. A string is palindromic if it reads the same backward as forward.

### Optimized Solution
We can use the **Expand Around Center** approach to solve this problem in \(O(n^2)\) time and \(O(1)\) space.

```cpp
class Solution {
public:
    int countSubstrings(string s) {
        int n = s.size();
        int count = 0;

        // Helper lambda function to expand around center
        auto expandAroundCenter = [&](int left, int right) {
            while (left >= 0 && right < n && s[left] == s[right]) {
                count++; // Count the palindrome
                left--;
                right++;
            }
        };

        // Iterate through each possible center
        for (int i = 0; i < n; i++) {
            // Odd-length palindromes (single center)
            expandAroundCenter(i, i);

            // Even-length palindromes (two-center)
            expandAroundCenter(i, i + 1);
        }

        return count;
    }
};
```

### Explanation

1. **Expand Around Center**:
   - A palindrome can expand from a "center." The center can either be:
     - A single character (odd-length palindrome).
     - A pair of characters (even-length palindrome).
   - We expand outward while the characters on both sides match.

2. **Iterate Over All Centers**:
   - There are \(2n - 1\) possible centers (every character as a single center and between every pair of characters as a two-character center).
   - For each center, expand outward and count palindromic substrings.

3. **Time Complexity**:
   - For each center, we may expand up to the length of the string in the worst case.
   - Total complexity: \(O(n^2)\).

4. **Space Complexity**:
   - We use no extra data structures, so space complexity is \(O(1)\).

### Example
#### Input:
```text
s = "abc"
```
#### Output:
```text
3
```
#### Explanation:
The palindromic substrings are "a", "b", and "c".

#### Input:
```text
s = "aaa"
```
#### Output:
```text
6
```
#### Explanation:
The palindromic substrings are "a", "a", "a", "aa", "aa", and "aaa".

### Conclusion
This solution is concise and efficient, making it suitable for large input strings. The use of a helper function to expand around centers ensures readability and clarity.
