# Optimal and Concise C++ Solution for Leetcode 22: Generate Parentheses

## **Approach:**
- Use a recursive backtracking approach.
- Maintain two counters: `open` for open parentheses used and `close` for closed parentheses used.
- Add an open parenthesis if `open < n`.
- Add a close parenthesis if `close < open`.
- When `open == close == n`, add the generated string to the result.

## **Code:**
```cpp
class Solution {
public:
    void backtrack(int open, int close, int n, string& path, vector<string>& result) {
        if (path.size() == 2 * n) {
            result.push_back(path);
            return;
        }
        if (open < n) {
            path.push_back('(');
            backtrack(open + 1, close, n, path, result);
            path.pop_back();
        }
        if (close < open) {
            path.push_back(')');
            backtrack(open, close + 1, n, path, result);
            path.pop_back();
        }
    }

    vector<string> generateParenthesis(int n) {
        vector<string> result;
        string path;
        backtrack(0, 0, n, path, result);
        return result;
    }
};
```

## **Time Complexity:**
- The number of valid sequences follows the **n-th Catalan number**:

  \[
  C_n = \frac{1}{n+1} \binom{2n}{n} = \frac{(2n)!}{(n+1)!n!}
  \]

- Using Stirling's approximation for factorials:

  \[
  C_n \approx \frac{4^n}{n^{3/2} \sqrt{\pi}}
  \]

- This gives the final complexity:

  \[
  O\left(\frac{4^n}{\sqrt{n}}\right)
  \]

## **Space Complexity:**
- \( O(n) \) for recursion stack.

## **Why Not \( O(2^{2n}) \)?**
- Pruning ensures that we only explore valid sequences, significantly reducing recursive calls.
- Instead of checking all `2^(2n)` sequences, we explore only **valid** ones, which are much fewer.

## **Conclusion:**
This solution is both **optimal** and **concise**, following an elegant **backtracking** approach. 🚀
