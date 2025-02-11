
# Gas Station Problem Solution

## Problem Description
You are given two integer arrays `gas` and `cost`. The `gas[i]` represents the amount of gas at station `i`, and `cost[i]` represents the cost of gas required to travel from station `i` to station `i + 1`. Your task is to determine the starting gas station index to complete the circuit exactly once, or return `-1` if it is not possible.

### Constraints:
- If there exists a solution, it is guaranteed to be unique.
- If `totalGas < totalCost`, it is impossible to complete the circuit.

## Optimized C++ Solution
The following C++ solution uses a greedy algorithm with `O(n)` time complexity:

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int totalGas = 0, totalCost = 0, startIndex = 0, currentTank = 0;

        for (int i = 0; i < gas.size(); ++i) {
            totalGas += gas[i];
            totalCost += cost[i];
            currentTank += gas[i] - cost[i];

            if (currentTank < 0) {
                // Reset start index to the next station and reset tank
                startIndex = i + 1;
                currentTank = 0;
            }
        }

        // If total gas is less than total cost, it's not possible to complete the circuit
        return totalGas >= totalCost ? startIndex : -1;
    }
};
```

## Explanation

### Key Observations
1. If the total amount of gas (`totalGas`) is less than the total cost (`totalCost`), it is impossible to complete the circuit, so return `-1`.
2. If you can't reach station `i+1` from station `i`, then no station from `0` to `i` can be a valid starting point. Thus, reset the starting index and `currentTank` whenever the remaining gas becomes negative.

### Algorithm Steps
1. Calculate the total gas and total cost.
2. Traverse each gas station and track the `currentTank` (gas remaining).
3. If `currentTank` becomes negative, reset the starting index to the next station and reset `currentTank` to `0`.
4. After the loop, check if `totalGas >= totalCost`. If true, return `startIndex`. Otherwise, return `-1`.

## Complexity Analysis
- **Time Complexity**: `O(n)`, as we only iterate through the gas stations once.
- **Space Complexity**: `O(1)`, since no additional space is used.

## Example

### Input:
```text
gas = [1,2,3,4,5]
cost = [3,4,5,1,2]
```

### Output:
```text
3
```

### Explanation:
- Starting at station `3`:
  - Remaining gas: `4 - 1 = 3`
  - From station `3` to station `4`: `3 + 5 - 2 = 6`
  - From station `4` to station `0`: `6 + 1 - 3 = 4`
  - From station `0` to station `1`: `4 + 2 - 4 = 2`
  - From station `1` to station `2`: `2 + 3 - 5 = 0`
- The circuit is completed successfully.

## Notes
This solution is concise and follows a greedy strategy to identify the valid starting index in a single pass.
