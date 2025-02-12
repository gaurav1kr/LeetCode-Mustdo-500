
# Optimized C++ Solution for 4Sum Problem

This markdown file provides an optimized C++ solution for the [4Sum problem](https://leetcode.com/problems/4sum/description/) from LeetCode.

## Solution Code
```cpp
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

vector<vector<int>> fourSum(vector<int>& nums, int target) {
    vector<vector<int>> result;
    int n = nums.size();

    if (n < 4) return result; // If fewer than 4 elements, no solution is possible.

    sort(nums.begin(), nums.end()); // Sort the array for easier two-pointer traversal.

    for (int i = 0; i < n - 3; ++i) {
        // Avoid duplicates for the first number
        if (i > 0 && nums[i] == nums[i - 1]) continue;

        for (int j = i + 1; j < n - 2; ++j) {
            // Avoid duplicates for the second number
            if (j > i + 1 && nums[j] == nums[j - 1]) continue;

            long long remainingTarget = (long long)target - nums[i] - nums[j];
            int left = j + 1, right = n - 1;

            while (left < right) {
                int twoSum = nums[left] + nums[right];
                if (twoSum == remainingTarget) {
                    result.push_back({nums[i], nums[j], nums[left], nums[right]});

                    // Move left and right pointers while avoiding duplicates
                    while (left < right && nums[left] == nums[left + 1]) ++left;
                    while (left < right && nums[right] == nums[right - 1]) --right;

                    ++left;
                    --right;
                } else if (twoSum < remainingTarget) {
                    ++left; // Increase the sum
                } else {
                    --right; // Decrease the sum
                }
            }
        }
    }

    return result;
}

int main() {
    vector<int> nums = {1, 0, -1, 0, -2, 2};
    int target = 0;

    vector<vector<int>> result = fourSum(nums, target);

    for (const auto& quad : result) {
        for (int num : quad) {
            cout << num << " ";
        }
        cout << endl;
    }

    return 0;
}
```

## Explanation

### Steps:
1. **Sorting the Array**: Sorting helps easily skip duplicates and use two-pointer techniques.
2. **Outer Loops**: Two nested loops pick the first two numbers. Skipping duplicates ensures unique combinations.
3. **Two-Pointer Technique**: The two-pointer approach finds pairs that sum to the remaining target efficiently.
4. **Avoiding Duplicates**: By incrementing or decrementing pointers appropriately, duplicates are avoided, ensuring unique results.
5. **Edge Cases**: The code handles cases like fewer than four elements, large numbers, or empty input.

### Complexity:
- **Time Complexity**: \(O(n^3)\), due to three nested loops and two-pointer traversal.
- **Space Complexity**: \(O(1)\), aside from the output storage.

### Example Input/Output
#### Input:
```text
nums = [1, 0, -1, 0, -2, 2], target = 0
```
#### Output:
```text
[-2, -1, 1, 2]
[-2, 0, 0, 2]
[-1, 0, 0, 1]
```

This solution is efficient for typical competitive programming constraints.
