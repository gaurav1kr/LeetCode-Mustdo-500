
# Coin Change II Solution in C++

This document presents a robust solution for the "Coin Change II" problem on LeetCode using Dynamic Programming.

## Problem Statement
Given an integer `amount` and an array of integers `coins` representing coin denominations, return the number of combinations that make up the given amount. If no combination is possible, return `0`.

Constraints:
- \(1 \leq coins.length \leq 300\)
- \(1 \leq coins[i] \leq 5000\)
- All the values of `coins` are distinct.
- \(0 \leq amount \leq 5000\)

---

## Optimized C++ Solution
The following solution uses a **1D Dynamic Programming (DP)** approach with overflow handling to ensure correctness for large inputs.

```cpp
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<long long> dp(amount + 1, 0); // Use long long to prevent overflow
        dp[0] = 1; // Base case: one way to make amount 0

        for (int coin : coins) {
            for (int i = coin; i <= amount; ++i) {
                dp[i] += dp[i - coin];
                // Handle overflow
                if (dp[i] > INT_MAX) dp[i] = INT_MAX;
            }
        }

        return dp[amount] > INT_MAX ? 0 : static_cast<int>(dp[amount]); // Return within int limits
    }
};
```

---

## Explanation

### Key Ideas:
1. **Dynamic Programming Definition**:
   - Let `dp[i]` represent the number of ways to make up the amount `i` using the given coins.

2. **Initialization**:
   - `dp[0] = 1`: There is only one way to make up the amount `0` (using no coins).

3. **Transition**:
   - For each coin, iterate through all possible amounts from the coin value to `amount`.
   - Update `dp[i]` as:
     ```cpp
     dp[i] += dp[i - coin];
     ```
     This means the number of ways to make `i` includes the ways to make `i - coin`.

4. **Overflow Handling**:
   - Use `long long` for intermediate results in the `dp` array.
   - Clamp values exceeding `INT_MAX` to prevent undefined behavior.

5. **Result**:
   - If `dp[amount]` exceeds `INT_MAX`, return `0` (indicating an invalid result due to overflow).

---

## Complexity Analysis

### Time Complexity:
- **Outer Loop**: Iterates through all coins: \(O(n)\), where `n` is the number of coins.
- **Inner Loop**: For each coin, iterates through amounts from `coin` to `amount`: \(O(m)\), where `m` is the target amount.
- Total: \(O(n \times m)\).

### Space Complexity:
- Uses a 1D `dp` array of size `amount + 1`.
- Total: \(O(m)\).

---

## Example

### Input:
```plaintext
amount = 5
coins = [1, 2, 5]
```

### Output:
```plaintext
4
```

### Explanation:
There are four ways to make up the amount 5:
1. 5 = 5
2. 5 = 2 + 2 + 1
3. 5 = 2 + 1 + 1 + 1
4. 5 = 1 + 1 + 1 + 1 + 1

---

## Edge Cases
1. **Large Amounts**:
   - Handle cases like `amount = 4681` and large coin denominations to avoid overflow.

2. **No Coins**:
   - If `coins` is empty, return `0` for any `amount > 0`.

3. **Zero Amount**:
   - Always return `1` when `amount = 0`, regardless of the coin denominations.

---

This implementation is efficient, concise, and robust, ensuring correctness for all valid inputs.
