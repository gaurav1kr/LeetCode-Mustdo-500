## Optimal C++ Solution for "Product of Array Except Self"

### **Problem Statement**
Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all elements in `nums` except `nums[i]`. The solution should run in **O(N) time complexity** and should not use division.

---

### **C++ Solution**
```cpp
#include <vector>

using namespace std;

vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, 1);
    
    int left = 1, right = 1;
    for (int i = 0; i < n; ++i) {
        result[i] *= left;
        left *= nums[i];
        
        result[n - 1 - i] *= right;
        right *= nums[n - 1 - i];
    }
    
    return result;
}
```

---

### **Explanation:**
1. **Initialize a `result` array** with `1` as the base value.
2. **Compute left product** in the first loop:
   - `left` keeps track of the cumulative product from the left.
   - Multiply `result[i]` with `left` and update `left` with `nums[i]`.
3. **Compute right product** in the same loop (in reverse order):
   - `right` keeps track of the cumulative product from the right.
   - Multiply `result[n - 1 - i]` with `right` and update `right` with `nums[n - 1 - i]`.

---

### **Complexity Analysis:**
- **Time Complexity:** `O(N)`, since we traverse the array twice.
- **Space Complexity:** `O(1) extra space`, since the output array is not considered extra space.

This approach efficiently avoids using extra arrays for `left` and `right` products, making it an optimal solution. 🚀
