# Optimized C++ Solution for LeetCode Problem: Decode String

## Problem Description
The problem involves decoding an encoded string with the following format:
- A positive integer `k`, followed by a substring enclosed in square brackets `[ ]`, means the substring should be repeated `k` times.
- Nested patterns are allowed.

For example:
- Input: `"3[a2[c]]"`
- Output: `"accaccacc"`

## Optimized Solution
Here is an efficient solution using stacks in C++.

```cpp
#include <iostream>
#include <string>
#include <stack>

using namespace std;

string decodeString(const string &s) {
    stack<string> strStack;
    stack<int> numStack;
    string currentStr = "";
    int currentNum = 0;

    for (char c : s) {
        if (isdigit(c)) {
            // Build the number
            currentNum = currentNum * 10 + (c - '0');
        } else if (c == '[') {
            // Push the current string and number onto their respective stacks
            strStack.push(currentStr);
            numStack.push(currentNum);

            // Reset current string and number
            currentStr = "";
            currentNum = 0;
        } else if (c == ']') {
            // Pop from the stacks
            string tempStr = currentStr;
            int repeatCount = numStack.top();
            numStack.pop();
            currentStr = strStack.top();
            strStack.pop();

            // Repeat the string and append it
            while (repeatCount-- > 0) {
                currentStr += tempStr;
            }
        } else {
            // Append current character to the current string
            currentStr += c;
        }
    }

    return currentStr;
}

int main() {
    string input = "3[a2[c]]";
    cout << decodeString(input) << endl; // Output: "accaccacc"
    return 0;
}
```

## Explanation
1. **Stack Usage**:
   - `strStack` is used to store the string accumulated before encountering a `[` character.
   - `numStack` stores the multiplier (number) for the string pattern inside brackets.

2. **Flow**:
   - When a number is encountered, build the multiplier (`currentNum`).
   - When `[` is encountered, push the `currentStr` and `currentNum` onto their respective stacks and reset them.
   - When `]` is encountered:
     - Pop the top of `numStack` and `strStack`.
     - Repeat the string formed inside the brackets and append it to the popped string.
   - Otherwise, keep appending characters to `currentStr`.

3. **Complexity**:
   - **Time Complexity**: \(O(n)\), where \(n\) is the length of the input string. Each character is processed at most twice (once when encountered and once when popped from the stack).
   - **Space Complexity**: \(O(d)\), where \(d\) is the maximum depth of nested brackets, as this determines the size of the stack.

## Example Execution
Input: `"3[a2[c]]"`

### Step-by-Step Execution
| Step | Action                  | `currentStr` | `currentNum` | `strStack`       | `numStack` |
|------|-------------------------|--------------|--------------|------------------|------------|
| 1    | Read `3`               |              | 3            |                  |            |
| 2    | Read `[`               |              | 0            | ""               | [3]        |
| 3    | Read `a`               | "a"          | 0            |                  |            |
| 4    | Read `2`               | "a"          | 2            | ["a"]           | [3]        |
| 5    | Read `[`               | ""           | 0            | ["a", ""]       | [3, 2]     |
| 6    | Read `c`               | "c"          | 0            |                  |            |
| 7    | Read `]`               | "cc"         | 0            | ["a"]           | [3]        |
| 8    | Read `]`               | "accaccacc"  | 0            |                  |            |

Output: `"accaccacc"`

## Additional Test Cases
- Input: `"2[abc]3[cd]ef"`
  - Output: `"abcabccdcdcdef"`

- Input: `"3[a]2[bc]"`
  - Output: `"aaabcbc"`

- Input: `"10[a]"`
  - Output: `"aaaaaaaaaa"`

## Conclusion
This approach is optimal for solving the "Decode String" problem, leveraging stacks for managing nested structures and ensuring efficient decoding of the input string.
# Optimized C++ Solution for LeetCode Problem: Decode String

## Problem Description
The problem involves decoding an encoded string with the following format:
- A positive integer `k`, followed by a substring enclosed in square brackets `[ ]`, means the substring should be repeated `k` times.
- Nested patterns are allowed.

For example:
- Input: `"3[a2[c]]"`
- Output: `"accaccacc"`

## Optimized Solution
Here is an efficient solution using stacks in C++.

```cpp
#include <iostream>
#include <string>
#include <stack>

using namespace std;

string decodeString(const string &s) {
    stack<string> strStack;
    stack<int> numStack;
    string currentStr = "";
    int currentNum = 0;

    for (char c : s) {
        if (isdigit(c)) {
            // Build the number
            currentNum = currentNum * 10 + (c - '0');
        } else if (c == '[') {
            // Push the current string and number onto their respective stacks
            strStack.push(currentStr);
            numStack.push(currentNum);

            // Reset current string and number
            currentStr = "";
            currentNum = 0;
        } else if (c == ']') {
            // Pop from the stacks
            string tempStr = currentStr;
            int repeatCount = numStack.top();
            numStack.pop();
            currentStr = strStack.top();
            strStack.pop();

            // Repeat the string and append it
            while (repeatCount-- > 0) {
                currentStr += tempStr;
            }
        } else {
            // Append current character to the current string
            currentStr += c;
        }
    }

    return currentStr;
}

int main() {
    string input = "3[a2[c]]";
    cout << decodeString(input) << endl; // Output: "accaccacc"
    return 0;
}
```

## Explanation
1. **Stack Usage**:
   - `strStack` is used to store the string accumulated before encountering a `[` character.
   - `numStack` stores the multiplier (number) for the string pattern inside brackets.

2. **Flow**:
   - When a number is encountered, build the multiplier (`currentNum`).
   - When `[` is encountered, push the `currentStr` and `currentNum` onto their respective stacks and reset them.
   - When `]` is encountered:
     - Pop the top of `numStack` and `strStack`.
     - Repeat the string formed inside the brackets and append it to the popped string.
   - Otherwise, keep appending characters to `currentStr`.

3. **Complexity**:
   - **Time Complexity**: \(O(n)\), where \(n\) is the length of the input string. Each character is processed at most twice (once when encountered and once when popped from the stack).
   - **Space Complexity**: \(O(d)\), where \(d\) is the maximum depth of nested brackets, as this determines the size of the stack.

## Example Execution
Input: `"3[a2[c]]"`

### Step-by-Step Execution
| Step | Action                  | `currentStr` | `currentNum` | `strStack`       | `numStack` |
|------|-------------------------|--------------|--------------|------------------|------------|
| 1    | Read `3`               |              | 3            |                  |            |
| 2    | Read `[`               |              | 0            | ""               | [3]        |
| 3    | Read `a`               | "a"          | 0            |                  |            |
| 4    | Read `2`               | "a"          | 2            | ["a"]           | [3]        |
| 5    | Read `[`               | ""           | 0            | ["a", ""]       | [3, 2]     |
| 6    | Read `c`               | "c"          | 0            |                  |            |
| 7    | Read `]`               | "cc"         | 0            | ["a"]           | [3]        |
| 8    | Read `]`               | "accaccacc"  | 0            |                  |            |

Output: `"accaccacc"`

## Additional Test Cases
- Input: `"2[abc]3[cd]ef"`
  - Output: `"abcabccdcdcdef"`

- Input: `"3[a]2[bc]"`
  - Output: `"aaabcbc"`

- Input: `"10[a]"`
  - Output: `"aaaaaaaaaa"`

## Conclusion
This approach is optimal for solving the "Decode String" problem, leveraging stacks for managing nested structures and ensuring efficient decoding of the input string.

