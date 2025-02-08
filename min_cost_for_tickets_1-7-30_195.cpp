You are given:
- An integer array `days` where each element represents a day you need to travel.
- An integer array `costs` where:
  - `costs[0]` is the cost of a 1-day pass.
  - `costs[1]` is the cost of a 7-day pass.
  - `costs[2]` is the cost of a 30-day pass.

You need to determine the **minimum cost** required to travel for all the given days.

---

## Approach: Dynamic Programming

### Key Idea

We use a dynamic programming approach where:
- Let `dp[i]` represent the minimum cost to travel up to day `i`.

### Steps

1. **Initialization**:
   - Define a DP array `dp` where `dp[i]` represents the minimum cost up to day `i`.
   - Use a `set` to store the travel days for quick lookup.

2. **Base Case**:
   - If there are no travel days (i.e., the array is empty), the cost is `0`.

3. **Transition**:
   - For each day `i`:
     - If day `i` is **not** in the travel days, `dp[i] = dp[i - 1]` (same cost as the previous day).
     - If day `i` **is** a travel day, calculate the minimum cost considering the three types of tickets:
       - **1-day pass**: `dp[i - 1] + costs[0]`.
       - **7-day pass**: `dp[max(0, i - 7)] + costs[1]`.
       - **30-day pass**: `dp[max(0, i - 30)] + costs[2]`.

4. **Result**:
   - The final result is stored in `dp[last_day]`, where `last_day` is the last travel day.

---

## Complexity Analysis

### Time Complexity
- **DP Array Iteration**: We iterate from day `1` to `last_day`, which takes `O(L)` time where `L = days.back()` (the last travel day).
- **Set Lookup**: Checking if a day is a travel day is `O(1)` for each day.

Thus, the overall **time complexity** is **`O(L)`**.

### Space Complexity
- **DP Array**: We use a DP array of size `O(L)` to store the minimum costs up to each day.
- **Travel Days Set**: We use an unordered set for travel days, which requires `O(N)` space where `N` is the number of travel days.

Thus, the overall **space complexity** is **`O(L)`**.

---

## Example Walkthrough

### Input
```plaintext
Days: [1, 4, 6, 7, 8, 20]
Costs: [2, 7, 15]
```

### Output
```plaintext
Minimum Cost: 11
```

### Explanation
- Day 1: Buy a 1-day pass for $2.
- Day 4: Buy a 1-day pass for $2.
- Days 6-8: Buy a 7-day pass for $7.
- Day 20: Buy a 1-day pass for $2.

**Total Cost** = $2 + $2 + $7 = $11.

---

## Code

### C++ Implementation
```cpp
#include <vector>
#include <unordered_set>
#include <algorithm>
using namespace std;

int mincostTickets(vector<int>& days, vector<int>& costs) {
    // Last travel day
    int lastDay = days.back();
    // Create a set for quick lookup of travel days
    unordered_set<int> travelDays(days.begin(), days.end());
    
    // DP array to store minimum cost up to each day
    vector<int> dp(lastDay + 1, 0);
    
    // Iterate through all days up to the last travel day
    for (int i = 1; i <= lastDay; ++i) {
        if (travelDays.count(i) == 0) {
            // If not a travel day, cost remains the same as the previous day
            dp[i] = dp[i - 1];
        } else {
            // Calculate minimum cost for a travel day
            dp[i] = min({
                dp[i - 1] + costs[0], // 1-day pass
                dp[max(0, i - 7)] + costs[1], // 7-day pass
                dp[max(0, i - 30)] + costs[2] // 30-day pass
            });
        }
    }
    
    // The answer is the cost up to the last travel day
    return dp[lastDay];
}

