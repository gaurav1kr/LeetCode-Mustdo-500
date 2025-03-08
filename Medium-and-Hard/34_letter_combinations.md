# Letter Combinations of a Phone Number

## Solution Approach

The problem requires generating all possible letter combinations corresponding to a given phone number's digits. The optimal approach to solving this problem is **recursive backtracking**.

### **Steps to Solve:**
1. **Base Case:** If the input string is empty, return an empty result.
2. **Digit to Letters Mapping:** Use a vector to store the mapping of digits to corresponding letters (like on a phone keypad).
3. **Recursive Backtracking:**
   - Start from the first digit and explore all possible letter combinations by appending characters one by one.
   - Recursively call the function for the next digit.
   - Remove the last character (backtrack) before exploring the next possibility.
4. **Time Complexity:** The worst-case scenario is **O(4ⁿ)**, where `n` is the length of the input string (since digits 7 & 9 map to 4 letters).

---

## **C++ Code Implementation**

```cpp
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return {};
        vector<string> res;
        string cur;
        vector<string> map = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        
        function<void(int)> backtrack = [&](int i) {
            if (i == digits.size()) {
                res.push_back(cur);
                return;
            }
            for (char c : map[digits[i] - '0']) {
                cur.push_back(c);
                backtrack(i + 1);
                cur.pop_back();
            }
        };
        
        backtrack(0);
        return res;
    }
};
```

## **Complexity Analysis**
- **Time Complexity:** `O(4ⁿ)`, where `n` is the length of the input string.
- **Space Complexity:** `O(n)`, used for recursion stack.

This approach ensures an efficient and optimized way to generate letter combinations while keeping memory usage minimal.
