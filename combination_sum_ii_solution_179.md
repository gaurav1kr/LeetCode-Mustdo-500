
# LeetCode Problem: Combination Sum II

## Problem Description
Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sum to `target`.

Each number in `candidates` may only be used **once** in the combination.

**Note**: The solution set must not contain duplicate combinations.

## Optimized C++ Solution
Here is the concise and optimized C++ solution for the problem using backtracking:

```cpp
class Solution {
public:
    void backtrack(vector<int>& candidates, int target, int start, vector<int>& current, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(current);
            return;
        }
        for (int i = start; i < candidates.size(); ++i) {
            if (i > start && candidates[i] == candidates[i - 1]) continue; // Skip duplicates
            if (candidates[i] > target) break; // No need to proceed further
            current.push_back(candidates[i]);
            backtrack(candidates, target - candidates[i], i + 1, current, result);
            current.pop_back();
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> result;
        vector<int> current;
        backtrack(candidates, target, 0, current, result);
        return result;
    }
};
```

## Explanation
### Approach
1. **Sorting**:
   - The input `candidates` array is sorted to enable skipping duplicates easily and to terminate recursion early when a candidate exceeds the target.

2. **Backtracking**:
   - A recursive function `backtrack` is used to explore all possible combinations.
   - At each recursive step:
     - If the current sum (`target`) becomes zero, the combination is added to the result.
     - If the candidate at the current index exceeds the remaining target, the recursion stops early.
     - To avoid duplicate combinations, we skip over consecutive identical numbers at the same recursion depth.

3. **Pruning**:
   - The recursion halts as soon as a candidate exceeds the remaining target value.

### Key Steps in the Code
- **Skip Duplicates**: The condition `if (i > start && candidates[i] == candidates[i - 1])` ensures that the same number is not reused at the same depth of recursion.
- **Recursive Call**: For each candidate, the function recurses with the target reduced by the current candidate's value, moving to the next index (`i + 1`) to ensure each number is used at most once.
- **Backtracking**: After exploring a candidate, it is removed from the current combination (`current.pop_back()`) to try other candidates.

## Complexity Analysis
- **Time Complexity**: 
  - Worst-case: \(O(2^n)\) due to the exploration of subsets, but sorting and pruning reduce unnecessary recursion.
- **Space Complexity**:
  - \(O(n)\) for the recursion stack and the `current` vector used to store the temporary combination.

## Example Input/Output
### Example 1:
**Input:**
```text
candidates = [10,1,2,7,6,1,5], target = 8
```
**Output:**
```text
[
  [1,1,6],
  [1,2,5],
  [1,7],
  [2,6]
]
```

### Example 2:
**Input:**
```text
candidates = [2,5,2,1,2], target = 5
```
**Output:**
```text
[
  [1,2,2],
  [5]
]
```

## Additional Notes
This solution leverages the backtracking approach effectively while avoiding duplicate combinations by:
- Sorting the input.
- Skipping duplicate numbers in consecutive indices.
- Terminating recursion early when the candidate exceeds the target.

This ensures both efficiency and correctness.
