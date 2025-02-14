```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        if (s.empty()) return "";
        int n = s.size();
        int start = 0, maxLength = 1;

        for (int i = 0; i < n; i++) {
            // Odd-length palindromes (center at i)
            expandAroundCenter(i, i, s, start, maxLength);
            // Even-length palindromes (center between i and i+1)
            expandAroundCenter(i, i + 1, s, start, maxLength);
        }

        return s.substr(start, maxLength);
    }

private:
    void expandAroundCenter(int left, int right, const string& s, int& start, int& maxLength) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            left--;
            right++;
        }
        // Adjust back to the last valid indices
        left++;
        right--;
        if (right - left + 1 > maxLength) {
            start = left;
            maxLength = right - left + 1;
        }
    }
};
```
