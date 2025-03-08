# Largest Number Problem Solution

This document provides an optimized C++ implementation for solving the "Largest Number" problem from LeetCode.

## Problem Description

Given a list of non-negative integers, arrange them such that they form the largest number. Since the result may be very large, you need to return a string instead of an integer.

### Example

**Input:**
```
nums = [3, 30, 34, 5, 9]
```
**Output:**
```
"9534330"
```

## Solution

Here is the optimized and concise C++ implementation:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    string largestNumber(vector<int>& nums) {
        // Convert integers to strings
        vector<string> strNums;
        for (int num : nums) {
            strNums.push_back(to_string(num));
        }

        // Custom comparator to sort strings in the desired order
        sort(strNums.begin(), strNums.end(), [](const string& a, const string& b) {
            return a + b > b + a;
        });

        // If the largest number is "0", the entire number is "0"
        if (strNums[0] == "0") {
            return "0";
        }

        // Concatenate sorted strings to form the largest number
        string result;
        for (const string& str : strNums) {
            result += str;
        }

        return result;
    }
};

int main() {
    Solution solution;
    vector<int> nums = {3, 30, 34, 5, 9};
    cout << solution.largestNumber(nums) << endl;  // Output: "9534330"
    return 0;
}
```

## Explanation

1. **String Conversion**: Convert each integer in the input array to a string for custom comparison.
2. **Custom Sort**: Sort the strings using a custom comparator where `a + b > b + a` ensures the largest concatenated number comes first.
3. **Edge Case**: If the largest number after sorting is `"0"`, return `"0"` as the result.
4. **Concatenation**: Concatenate the sorted strings to form the final largest number.

## Complexity Analysis

- **Time Complexity**: O(n log n), where `n` is the number of integers in the input. The complexity is dominated by the sorting step.
- **Space Complexity**: O(n), as we store the string representation of each integer.

## Edge Cases

- Input contains only zeros: `[0, 0, 0]` -> Output: `"0"`
- Single element in input: `[10]` -> Output: `"10"`
- Large numbers: `[999999991, 999999992]` -> Output: `"999999992999999991"`

## How to Run the Code

1. Copy the provided code into a C++ compiler or IDE.
2. Replace the `nums` array in the `main` function with your test case.
3. Compile and execute the code to see the result.

## Conclusion

This solution leverages a custom sorting strategy to efficiently arrange numbers into the largest possible combination. By handling edge cases and ensuring efficient concatenation, the implementation is both robust and optimized for competitive programming.
