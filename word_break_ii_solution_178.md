# LeetCode Problem: Word Break II

## Problem Description
Given a string `s` and a dictionary of strings `wordDict`, add spaces in `s` to construct a sentence where each word is a valid dictionary word. Return all possible sentences. Each word in `wordDict` can be used multiple times in the segmentation.

[Problem Link](https://leetcode.com/problems/word-break-ii/description/)

---

## Optimized and Concise C++ Solution
The following is an optimized and concise C++ solution using Depth-First Search (DFS) with memoization to avoid redundant computations.

```cpp
class Solution {
public:
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        unordered_map<string, vector<string>> memo;
        return dfs(s, wordSet, memo);
    }

private:
    vector<string> dfs(const string& s, unordered_set<string>& wordSet, unordered_map<string, vector<string>>& memo) {
        if (memo.count(s)) return memo[s];
        if (s.empty()) return {""};
        
        vector<string> result;
        for (int i = 1; i <= s.size(); ++i) {
            string word = s.substr(0, i);
            if (wordSet.count(word)) {
                vector<string> subResults = dfs(s.substr(i), wordSet, memo);
                for (const string& sub : subResults) {
                    result.push_back(word + (sub.empty() ? "" : " ") + sub);
                }
            }
        }
        return memo[s] = result;
    }
};
```

---

## Explanation

### Key Components:
1. **Word Set:**
   - Convert the `wordDict` into an `unordered_set` for O(1) lookup time for each word.

2. **DFS with Memoization:**
   - The `dfs` function recursively partitions the string `s` into valid words from the dictionary.
   - Use `memo` to store results for substrings we've already computed, avoiding redundant work.

3. **Base Cases:**
   - If the substring `s` is empty, return `{""}` (an empty sentence).

4. **Recursive Splitting:**
   - Iterate through prefixes of `s` (`s.substr(0, i)`).
   - If the prefix exists in the dictionary, recursively call `dfs` for the remainder of the string (`s.substr(i)`).
   - Combine the prefix with results from the recursive call.

5. **Combining Results:**
   - Append the current word to all valid sentences from the recursive calls and handle spaces properly.

6. **Memoization:**
   - Store the result of each substring in `memo` to avoid recomputing.

---

## Complexity Analysis

### Time Complexity:
- **Worst-Case:** O(n^2 * 2^n)
  - \(n\) is the length of the string. The recursion explores all possible substrings, but memoization reduces the actual number of calls.

### Space Complexity:
- O(n * k), where \(k\) is the average length of a word in the dictionary. This accounts for the memoization storage.

---

## Example Usage
### Input:
```cpp
string s = "catsanddog";
vector<string> wordDict = {"cat", "cats", "and", "sand", "dog"};
```

### Output:
```cpp
["cats and dog", "cat sand dog"]
```

---

## Advantages of This Solution:
- Efficient through memoization.
- Clear and concise implementation.
- Handles all edge cases, such as empty strings or no valid partitions.
