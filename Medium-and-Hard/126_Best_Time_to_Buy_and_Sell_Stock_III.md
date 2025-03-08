# Best Time to Buy and Sell Stock III - Optimized C++ Solution

## Problem Statement

You are given an integer array `prices` where `prices[i]` is the price of a given stock on the `i-th` day. Design an algorithm to find the maximum profit. You may complete at most two transactions.

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

---

## Optimized C++ Solution

```cpp
#include <vector>
#include <algorithm>
#include <limits.h>
using namespace std;

int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n == 0) return 0;

    // Initialize variables for two transactions
    int firstBuy = INT_MIN, firstSell = 0;
    int secondBuy = INT_MIN, secondSell = 0;

    for (int price : prices) {
        // Maximize profit for the first buy
        firstBuy = max(firstBuy, -price);

        // Maximize profit for the first sell
        firstSell = max(firstSell, firstBuy + price);

        // Maximize profit for the second buy
        secondBuy = max(secondBuy, firstSell - price);

        // Maximize profit for the second sell
        secondSell = max(secondSell, secondBuy + price);
    }

    return secondSell;
}
```

---

## Explanation

### Variables:
- `firstBuy`: The maximum profit after buying the first stock (negative since it represents spending money).
- `firstSell`: The maximum profit after selling the first stock.
- `secondBuy`: The maximum profit after buying the second stock (considering profit from the first transaction).
- `secondSell`: The maximum profit after selling the second stock.

### Logic:
1. Iterate over the `prices` array.
2. Update `firstBuy`, `firstSell`, `secondBuy`, and `secondSell` at each step based on the current price.

### Complexity:
- **Time Complexity**: \(O(n)\), where \(n\) is the number of days.
- **Space Complexity**: \(O(1)\), as only a few variables are used.

---

## Example Usage

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> prices = {3, 3, 5, 0, 0, 3, 1, 4};
    cout << "Maximum Profit: " << maxProfit(prices) << endl;
    return 0;
}
```

### Input:
`prices = [3,3,5,0,0,3,1,4]`

### Output:
`Maximum Profit: 6`

---

## Notes
This solution uses dynamic programming to efficiently compute the maximum profit with at most two transactions. The key is to maintain the state of each transaction while iterating through the array.
