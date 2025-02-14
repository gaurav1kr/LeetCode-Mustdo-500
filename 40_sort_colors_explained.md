# Sort Colors (LeetCode Problem)

## **Problem Statement**
Given an array `nums` with `n` objects colored red (`0`), white (`1`), or blue (`2`), sort them **in-place** so that objects of the same color are adjacent, in the order red, white, and blue.

You must solve this problem without using the `sort()` function.

---

## **Optimal Solution (Dutch National Flag Algorithm)**

### **C++ Code:**
```cpp
#include <vector>
using namespace std;

class Solution {
public:
    void sortColors(vector<int>& nums) {
        int low = 0, mid = 0, high = nums.size() - 1;
        
        while (mid <= high) {
            if (nums[mid] == 0) {
                swap(nums[low++], nums[mid++]);
            } else if (nums[mid] == 1) {
                mid++;
            } else { // nums[mid] == 2
                swap(nums[mid], nums[high--]);
            }
        }
    }
};
```

---

## **Explanation:**
- **Three pointers (`low`, `mid`, `high`) are used:**
  - `low` points to the boundary where `0`s should be.
  - `mid` scans through the array.
  - `high` points to the boundary where `2`s should be.
- **Operations:**
  - If `nums[mid] == 0`: Swap `nums[mid]` and `nums[low]`, then increment `low` and `mid`.
  - If `nums[mid] == 1`: Simply increment `mid`.
  - If `nums[mid] == 2`: Swap `nums[mid]` and `nums[high]`, then decrement `high`.
- **Time Complexity:** `O(n)`, as each element is processed at most once.
- **Space Complexity:** `O(1)`, as we sort the array in-place.

---

## **Why is this Optimal?**
✅ **Single pass (O(n))** – No unnecessary comparisons.  
✅ **Constant space (O(1))** – No extra arrays or structures.  
✅ **Efficient for large inputs.**
