
# Reverse Integer Solution

## Problem
Given a signed 32-bit integer `x`, reverse its digits. If reversing `x` causes the value to go outside the signed 32-bit integer range `[-2^31, 2^31 - 1]`, return 0.

## C++ Code Solution
```cpp
class Solution {
public:
    int reverse(int x) {
        int result = 0;
        
        while (x != 0) {
            int digit = x % 10;
            
            // Check for overflow/underflow before multiplying
            if (result > INT_MAX / 10 || (result == INT_MAX / 10 && digit > 7)) {
                return 0; // Overflow
            }
            if (result < INT_MIN / 10 || (result == INT_MIN / 10 && digit < -8)) {
                return 0; // Underflow
            }
            
            result = result * 10 + digit;
            x /= 10;
        }
        
        return result;
    }
};
```

## Explanation
1. **Digit Extraction**:
   - Extract the last digit of `x` using `x % 10`.

2. **Overflow/Underflow Check**:
   - Before updating `result`, check if multiplying `result` by 10 would cause it to exceed the bounds of a 32-bit signed integer (`INT_MAX` or `INT_MIN`).

3. **Update Result**:
   - Add the digit to `result` and remove the last digit from `x` using `x /= 10`.

4. **Return the Final Reversed Number**:
   - If no overflow/underflow occurs, return `result`.

## Complexity Analysis
- **Time Complexity**: `O(log_{10}(n))` where `n` is the absolute value of the input integer. This is because we process each digit of the number exactly once.
- **Space Complexity**: `O(1)` as no additional space is used apart from variables.

## Example
**Input**:
```
x = 123
```
**Output**:
```
321
```
**Input**:
```
x = -123
```
**Output**:
```
-321
```
**Input**:
```
x = 120
```
**Output**:
```
21
```
**Input**:
```
x = 0
```
**Output**:
```
0
```

## Edge Cases
1. Overflow or underflow when reversing:
   - Input: `x = 1534236469`
   - Output: `0` (overflow)

2. Negative numbers:
   - Input: `x = -123`
   - Output: `-321`

3. Zero:
   - Input: `x = 0`
   - Output: `0`
