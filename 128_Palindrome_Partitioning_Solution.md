
# Palindrome Partitioning - Optimized C++ Solution

This solution provides an optimized approach to solving the Palindrome Partitioning problem from LeetCode using backtracking and memoization.

## C++ Solution

```cpp
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> result;
        vector<string> current;
        backtrack(0, s, current, result);
        return result;
    }

private:
    // Memoization for isPalindrome
    unordered_map<string, bool> palindromeCache;

    void backtrack(int start, const string& s, vector<string>& current, vector<vector<string>>& result) {
        if (start == s.size()) {
            result.push_back(current);
            return;
        }

        for (int end = start; end < s.size(); ++end) {
            string substring = s.substr(start, end - start + 1);
            if (isPalindrome(substring)) {
                current.push_back(substring);
                backtrack(end + 1, s, current, result);
                current.pop_back();
            }
        }
    }

    bool isPalindrome(const string& s) {
        if (palindromeCache.count(s)) return palindromeCache[s];
        int left = 0, right = s.size() - 1;
        while (left < right) {
            if (s[left++] != s[right--]) return palindromeCache[s] = false;
        }
        return palindromeCache[s] = true;
    }
};
```

## Explanation

1. **Backtracking**:
   - The function `backtrack` explores all possible partitions starting from the current position.
   - For every substring, it checks if it is a palindrome. If yes, it proceeds to the next part of the string.

2. **Palindrome Check with Memoization**:
   - The `isPalindrome` function checks whether a string is a palindrome.
   - A cache (`palindromeCache`) is used to store results of previously checked substrings to avoid redundant checks, significantly improving performance.

3. **Complexity**:
   - The time complexity is improved compared to brute force due to memoization. However, since it's backtracking, the worst case could still explore all possible partitions.

## Usage

```cpp
int main() {
    Solution solution;
    string input = "aab";
    vector<vector<string>> result = solution.partition(input);
    
    for (const auto& partition : result) {
        for (const auto& s : partition) {
            cout << s << " ";
        }
        cout << endl;
    }
    return 0;
}
```

### Output

For the input `"aab"`, the output will be:
```
a a b
aa b
```

This solution is optimal for most practical inputs while remaining easy to understand.
