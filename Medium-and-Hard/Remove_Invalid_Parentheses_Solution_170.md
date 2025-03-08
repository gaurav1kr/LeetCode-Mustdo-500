
# Remove Invalid Parentheses Solution

This markdown file contains an optimized and concise C++ solution for the LeetCode problem [Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses).

## Problem Description
Given a string `s` that may contain invalid parentheses, remove the minimum number of invalid parentheses to make the input string valid. Return all possible results.

### Constraints:
- The input string `s` may contain `(`, `)`, and lowercase English characters.

## Solution: BFS Approach
Below is the C++ solution that uses Breadth-First Search (BFS) to solve the problem efficiently.

```cpp
#include <iostream>
#include <unordered_set>
#include <queue>
#include <vector>
using namespace std;

vector<string> removeInvalidParentheses(string s) {
    vector<string> result;
    unordered_set<string> visited;
    queue<string> q;
    bool found = false;

    q.push(s);
    visited.insert(s);

    while (!q.empty()) {
        string current = q.front();
        q.pop();

        if (isValid(current)) {
            result.push_back(current);
            found = true;
        }

        if (found) continue;

        for (int i = 0; i < current.size(); ++i) {
            if (current[i] != '(' && current[i] != ')') continue;
            string next = current.substr(0, i) + current.substr(i + 1);
            if (visited.find(next) == visited.end()) {
                q.push(next);
                visited.insert(next);
            }
        }
    }

    return result;
}

bool isValid(const string &s) {
    int count = 0;
    for (char c : s) {
        if (c == '(') ++count;
        else if (c == ')') {
            if (--count < 0) return false;
        }
    }
    return count == 0;
}

int main() {
    string input = "()())()";
    vector<string> result = removeInvalidParentheses(input);
    for (const string &str : result) {
        cout << str << endl;
    }
    return 0;
}
```

## Explanation

### BFS Traversal
- Use a queue to explore all possible strings obtained by removing one parenthesis at a time.
- Track visited strings using an unordered set to avoid redundant work.

### Validity Check
- Use the `isValid()` function to check if a string has balanced parentheses. This function iterates through the string and ensures:
  - The count of `(` and `)` remains non-negative at any point.
  - The final count is zero.

### Stopping Condition
- As soon as we find valid strings at a certain BFS level, we stop exploring deeper levels. This ensures minimum removals.

## Complexity Analysis

### Time Complexity
- **O(2^n)**: In the worst case, we explore all subsets of the string `s`.
- **O(n)** for each validity check.

### Space Complexity
- **O(n * 2^n)**: For the queue and visited set.

## Example

### Input
```plaintext
s = "()())()"
```

### Output
```plaintext
()()()
(())()
```

## Notes
- This solution ensures both correctness and efficiency.
- It explores all possibilities but stops early as soon as valid strings are found at a level.

Let me know if you have further questions or need additional explanations!
