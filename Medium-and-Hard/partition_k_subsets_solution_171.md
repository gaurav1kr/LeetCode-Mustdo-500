# Partition to K Equal Sum Subsets

This is an optimized C++ solution to solve the problem of partitioning a set into `k` subsets with equal sums.

```cpp
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

class Solution {
public:
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % k != 0) return false;
        int target = sum / k;

        sort(nums.rbegin(), nums.rend()); // Sort in descending order to optimize
        vector<int> buckets(k, 0);

        return backtrack(nums, buckets, 0, target);
    }

private:
    bool backtrack(const vector<int>& nums, vector<int>& buckets, int index, int target) {
        if (index == nums.size())
            return all_of(buckets.begin(), buckets.end(), [&](int sum) { return sum == target; });

        for (int i = 0; i < buckets.size(); ++i) {
            if (buckets[i] + nums[index] > target) continue;

            buckets[i] += nums[index];
            if (backtrack(nums, buckets, index + 1, target)) return true;
            buckets[i] -= nums[index];

            if (buckets[i] == 0) break; // Prune duplicate states
        }

        return false;
    }
};
```
