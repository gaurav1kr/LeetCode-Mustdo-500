## Kth Largest Element in an Array (LeetCode)

### **Problem Statement**
Find the Kth largest element in an unsorted array.

### **Approach**
- Use a **min-heap** (priority queue) of size **K**.
- Traverse the array:
  - Push elements into the heap.
  - If the heap size exceeds **K**, remove the smallest element.
- The top of the heap is the **Kth largest element**.

### **Optimal C++ Code**
```cpp
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, greater<int>> minHeap;

        for (int num : nums) {
            minHeap.push(num);
            if (minHeap.size() > k) {
                minHeap.pop();
            }
        }
        
        return minHeap.top();
    }
};

int main() {
    Solution sol;
    vector<int> nums = {3, 2, 3, 1, 2, 4, 5, 5, 6};
    int k = 4;
    cout << "The " << k << "th largest element is: " << sol.findKthLargest(nums, k) << endl;
    return 0;
}
```

### **Complexity Analysis**
- **Time Complexity**: \(O(N \log K)\) (Each insert operation in heap is \(O(\log K)\), done \(N\) times)
- **Space Complexity**: \(O(K)\) (Heap stores \(K\) elements)

### **Alternative Approaches**
1. **Quickselect (Hoare’s Selection Algorithm)**
   - Average time complexity: \(O(N)\)
   - Worst case: \(O(N^2)\)
   - In-place, no extra space.
   
2. **Sorting**
   - Time complexity: \(O(N \log N)\)
   - Space complexity: \(O(1)\) (if sorting in place)

Would you like a Quickselect implementation as well? 🚀
