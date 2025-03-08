# Coin Change Problem - Approach

## Problem Statement
Given an integer array `coins` representing coins of different denominations and an integer `amount`, return the **fewest number of coins** that you need to make up that amount. If that amount cannot be made up by any combination of the coins, return `-1`.

## Approach: Dynamic Programming (Bottom-Up)

### **Algorithm Explanation**
1. **Define a DP Array:**
   - Create a `dp` array where `dp[i]` represents the minimum number of coins required to get amount `i`.
   - Initialize `dp` with a large value (`amount + 1`) since the minimum coins required cannot exceed `amount`.
   
2. **Base Case:**
   - `dp[0] = 0` since no coins are needed to make an amount of `0`.

3. **Iterate Through Each Coin:**
   - For each coin, iterate through amounts from `coin` to `amount`.
   - Update `dp[i]` as `min(dp[i], dp[i - coin] + 1)`.

4. **Return the Result:**
   - If `dp[amount]` remains `amount + 1`, return `-1` (impossible case), otherwise return `dp[amount]`.

---

### **C++ Code Implementation**

```cpp
#include <vector>
#include <algorithm>

using namespace std;

int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;  // Base case: 0 coins needed for amount 0

    for (int coin : coins) {
        for (int i = coin; i <= amount; ++i) {
            dp[i] = min(dp[i], dp[i - coin] + 1);
        }
    }

    return dp[amount] == amount + 1 ? -1 : dp[amount];
}
```

### **Time Complexity Analysis**
- **Time Complexity:** `O(n * m)`, where `n` is the amount and `m` is the number of coins.
- **Space Complexity:** `O(n)`, as we use an array `dp` of size `amount + 1`.

### **Edge Cases Considered:**
- If `amount == 0`, return `0`.
- If all coins are larger than `amount`, return `-1`.
- If there is only one coin type, ensure it correctly calculates multiples.

### **Why is this the Optimal Solution?**
- Uses **Dynamic Programming** to efficiently solve the problem in `O(n * m)` time.
- Avoids redundant recalculations by storing results in the `dp` array.
- More efficient than naive recursion or brute-force methods.

---

This approach ensures an **optimal and efficient** solution to the Coin Change problem using **Dynamic Programming**. 🚀
