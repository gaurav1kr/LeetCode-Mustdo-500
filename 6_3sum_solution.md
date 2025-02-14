
# Optimal C++ Solution for 3Sum Problem

This is an optimal solution for the [3Sum problem](https://leetcode.com/problems/3sum/) using sorting and the two-pointer approach. The time complexity of this solution is **O(n^2)**.

## Code

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {
        std::vector<std::vector<int>> result;
        int n = nums.size();

        // Sort the array
        std::sort(nums.begin(), nums.end());

        for (int i = 0; i < n - 2; ++i) {
            // Avoid duplicates for the first number
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];

                if (sum == 0) {
                    result.push_back({nums[i], nums[left], nums[right]});

                    // Avoid duplicates for the second and third numbers
                    while (left < right && nums[left] == nums[left + 1]) {
                        ++left;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        --right;
                    }

                    ++left;
                    --right;
                } else if (sum < 0) {
                    ++left;
                } else {
                    --right;
                }
            }
        }

        return result;
    }
};
```

## Explanation

### 1. Sorting
- The input array is sorted to simplify finding unique triplets and enable the two-pointer approach.

### 2. Outer Loop
- Iterate through the array to fix the first element of the triplet (`nums[i]`).
- Skip duplicate values for `nums[i]` to avoid duplicate triplets.

### 3. Two-Pointer Technique
- Use two pointers (`left` and `right`) to find pairs that sum to `-nums[i]`:
  - If the sum is 0, a valid triplet is found.
  - Move both pointers inward and skip duplicate values for `nums[left]` and `nums[right]` to avoid duplicate triplets.
  - Adjust pointers (`left` or `right`) based on whether the sum is less than or greater than 0.

### 4. Complexity
- **Sorting**: Sorting takes **O(n \* log(n))**.
- **Two-pointer search**: For each fixed element, the two-pointer approach runs in **O(n)**.
- Overall complexity: **O(n^2)**.

## Example

### Input
```plaintext
nums = [-1, 0, 1, 2, -1, -4]
```

### Output
```plaintext
[[-1, -1, 2], [-1, 0, 1]]
```

### Explanation
The triplets [-1, -1, 2] and [-1, 0, 1] are the two unique triplets that sum to zero.

## Edge Cases
1. If the input array has fewer than 3 elements, return an empty result.
2. Handle arrays with duplicate numbers gracefully by skipping duplicates.
3. Arrays with all positive or all negative numbers will return an empty result since no triplets can sum to zero.

---

Feel free to use this solution and let me know if you have further questions or need additional explanations!
