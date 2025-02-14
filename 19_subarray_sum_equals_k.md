# Subarray Sum Equals K (LeetCode 560)

## Optimal C++ Solution

```cpp
#include <unordered_map>
#include <vector>
using namespace std;

int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> prefixSum{{0, 1}};
    int sum = 0, count = 0;
    for (int num : nums) {
        sum += num;
        count += prefixSum[sum - k];
        prefixSum[sum]++;
    }
    return count;
}
```

### Explanation
- Uses **prefix sum** and an **unordered_map** to count occurrences of previous prefix sums.
- If `sum - k` exists in the map, it indicates a subarray ending at the current index sums to `k`.
- **Time Complexity:** \(O(n)\) (Single pass through `nums`).
- **Space Complexity:** \(O(n)\) in the worst case.

### LeetCode Link
[Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
