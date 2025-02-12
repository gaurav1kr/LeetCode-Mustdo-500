# Best Time to Buy and Sell Stock with Cooldown

## Problem Description
You are given an array `prices` where `prices[i]` is the price of a given stock on the `i-th` day. Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:

- After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

## Optimal C++ Solution
Here is an optimal and concise solution using dynamic programming with space optimization:

```cpp
#include <vector>
#include <algorithm>

using namespace std;

int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n <= 1) return 0;

    int sold = 0, hold = INT_MIN, rest = 0;

    for (int price : prices) {
        int prev_sold = sold;
        sold = hold + price;      // Sell stock
        hold = max(hold, rest - price); // Buy stock or keep holding
        rest = max(rest, prev_sold);   // Cooldown or stay in cooldown
    }

    return max(sold, rest);
}
```

## Explanation

### States:
1. **`sold`**: The maximum profit on the day you sell a stock.
2. **`hold`**: The maximum profit on the day you hold a stock (bought but not sold yet).
3. **`rest`**: The maximum profit on a cooldown day.

### State Transitions:
- `sold = hold + price`: If you sell the stock, your profit is what you held plus today's price.
- `hold = max(hold, rest - price)`: Either continue holding or buy a stock today.
- `rest = max(rest, prev_sold)`: Either stay in cooldown or enter cooldown after selling.

### Space Optimization:
Instead of maintaining a DP array for each state, the variables `sold`, `hold`, and `rest` are updated iteratively, reducing the space complexity to **O(1)**.

## Complexity
- **Time Complexity**: `O(n)` where `n` is the number of days (size of the `prices` array).
- **Space Complexity**: `O(1)` as we use constant space.

## Example Usage
```cpp
#include <iostream>
#include <vector>

int main() {
    vector<int> prices = {1, 2, 3, 0, 2};
    cout << "Max Profit: " << maxProfit(prices) << endl;
    return 0;
}
```

### Input:
```
prices = [1, 2, 3, 0, 2]
```

### Output:
```
Max Profit: 3
```

This solution is clean, efficient, and adheres to the problem's constraints.
