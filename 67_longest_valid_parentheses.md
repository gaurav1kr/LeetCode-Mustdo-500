# Longest Valid Parentheses - C++ Solution

This Markdown file provides an optimal C++ solution for the [Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/) problem on LeetCode.

## Problem Description
The goal is to find the length of the longest valid (well-formed) parentheses substring in a given string.

### Solution Overview
We use a stack-based approach to solve this problem with:
- **Time Complexity**: \(O(n)\)
- **Space Complexity**: \(O(n)\)

---

## C++ Code Implementation
```cpp
#include <iostream>
#include <stack>
#include <string>
#include <algorithm>

using namespace std;

int longestValidParentheses(string s) {
    stack<int> st;
    int maxLength = 0;
    st.push(-1); // Initialize the stack with -1 as a base for the first valid substring

    for (int i = 0; i < s.length(); ++i) {
        if (s[i] == '(') {
            st.push(i); // Push the index of '('
        } else {
            st.pop(); // Pop the last '(' or unmatched index
            if (st.empty()) {
                st.push(i); // Push the current index as the base for the next substring
            } else {
                maxLength = max(maxLength, i - st.top());
            }
        }
    }

    return maxLength;
}

int main() {
    string s = "(()))())("; // Example input
    cout << "Longest Valid Parentheses: " << longestValidParentheses(s) << endl;
    return 0;
}
```

---

## Explanation of the Code
### 1. **Stack Initialization**
- Start with a stack initialized with `-1`.
- This `-1` acts as a base index for the first valid substring.

### 2. **Traverse the String**
- For each character in the string:
  - **If '('**: Push its index onto the stack.
  - **If ')'**:
    - Pop the stack. If the stack becomes empty after popping, push the current index as the new base.
    - Otherwise, calculate the length of the valid substring as the difference between the current index and the top of the stack.

### 3. **Update Maximum Length**
- Keep track of the maximum valid substring length during traversal.

---

## Example Walkthrough
### Input
```plaintext
s = "(()))())("
```

### Steps
| Character | Stack State         | Action                                   | maxLength |
|-----------|---------------------|------------------------------------------|-----------|
| `(`       | `[-1, 0]`           | Push index of `(`                       | 0         |
| `(`       | `[-1, 0, 1]`        | Push index of `(`                       | 0         |
| `)`       | `[-1, 0]`           | Pop and calculate valid length (2 - 0)  | 2         |
| `)`       | `[-1]`              | Pop and calculate valid length (3 - (-1)) | 4         |
| `)`       | `[3]`               | Stack empty; push current index          | 4         |
| `(`       | `[3, 5]`            | Push index of `(`                       | 4         |
| `)`       | `[3]`               | Pop and calculate valid length (6 - 3)  | 4         |
| `)`       | `[7]`               | Stack empty; push current index          | 4         |
| `(`       | `[7, 8]`            | Push index of `(`                       | 4         |

### Output
```plaintext
Longest Valid Parentheses: 4
```

---

## Complexity Analysis
### Time Complexity
- The string is traversed once, and each character is pushed or popped from the stack at most once.
- **Overall Time Complexity**: \(O(n)\)

### Space Complexity
- The stack stores indices of characters, which in the worst case is proportional to the length of the string.
- **Overall Space Complexity**: \(O(n)\)

---

## Conclusion
This solution effectively finds the longest valid parentheses substring in linear time using a stack, making it both intuitive and efficient.
