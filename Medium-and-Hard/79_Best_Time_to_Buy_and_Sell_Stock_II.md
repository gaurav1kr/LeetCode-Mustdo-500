
# LeetCode Problem: Best Time to Buy and Sell Stock II

This solution provides an optimized approach to solve the problem using a greedy algorithm.

## Problem Description
The goal is to calculate the maximum profit you can achieve from as many transactions as you'd like, with the constraint that you can only hold one stock at a time.

### Optimized C++ Solution

```cpp
#include <vector>
#include <iostream>
using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int maxProfit = 0;
        for (int i = 1; i < prices.size(); ++i) {
            // Add to profit if the price is increasing
            if (prices[i] > prices[i - 1]) {
                maxProfit += prices[i] - prices[i - 1];
            }
        }
        return maxProfit;
    }
};

int main() {
    Solution solution;
    vector<int> prices = {7, 1, 5, 3, 6, 4};
    cout << "Max Profit: " << solution.maxProfit(prices) << endl;
    return 0;
}
```

## Explanation

1. **Greedy Choice**: 
   The solution adds the difference `prices[i] - prices[i - 1]` to `maxProfit` whenever `prices[i] > prices[i - 1]`. This captures the profit for every ascending segment in the price array.

2. **Complexity**:
   - **Time Complexity**: \(O(n)\), where \(n\) is the size of the `prices` array. We iterate through the array once.
   - **Space Complexity**: \(O(1)\). No extra space is used apart from a few variables.

## Example Walkthrough

For `prices = [7, 1, 5, 3, 6, 4]`:
- Day 2: Buy at 1, sell at 5 → profit = 4
- Day 4: Buy at 3, sell at 6 → profit = 3
- Total profit = 4 + 3 = **7**

This is the most efficient way to solve the problem.
