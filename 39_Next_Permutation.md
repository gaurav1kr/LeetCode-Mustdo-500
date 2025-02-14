## Next Permutation - C++ Solution

### **Problem Statement**
Given an array of integers `nums`, rearrange them to get the **next lexicographically greater permutation** of numbers. If no greater permutation exists, rearrange it to the smallest possible order (i.e., sorted in ascending order).

### **Approach (Optimal - In-place)**
The algorithm follows these steps:

1. **Find the first decreasing element** (from right to left).
2. **Find the next greater element** (from right) and swap it with the first decreasing element.
3. **Reverse the right part** to get the next lexicographical permutation.

### **Code Implementation**
```cpp
#include <bits/stdc++.h>
using namespace std;

void nextPermutation(vector<int>& nums) {
    int n = nums.size(), i = n - 2;

    // Step 1: Find the first decreasing element from the right
    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }

    // Step 2: Find the next greater element and swap
    if (i >= 0) {
        int j = n - 1;
        while (nums[j] <= nums[i]) {
            j--;
        }
        swap(nums[i], nums[j]);
    }

    // Step 3: Reverse the right part
    reverse(nums.begin() + i + 1, nums.end());
}

int main() {
    vector<int> nums = {1, 2, 3};
    nextPermutation(nums);
    for (int num : nums) {
        cout << num << " ";
    }
    return 0;
}
```

### **Time & Space Complexity**
- **Time Complexity:** \(O(n)\) (single pass with swap & reverse)
- **Space Complexity:** \(O(1)\) (modifies input in-place)

### **Example**
#### **Input:**
```
nums = [1, 2, 3]
```
#### **Output:**
```
1 3 2
```

This solution efficiently finds the next lexicographical permutation while handling edge cases like already highest permutations. 🚀
