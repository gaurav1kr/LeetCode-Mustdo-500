# Top K Frequent Elements - Optimal C++ Solution

This document provides an optimal C++ solution for the **Top K Frequent Elements** problem using a **heap** (priority queue) for efficiency.

## Problem Description
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.

## C++ Code Implementation

```cpp
#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // Step 1: Count the frequency of each element using a hash map
        unordered_map<int, int> freqMap;
        for (int num : nums) {
            freqMap[num]++;
        }
        
        // Step 2: Use a min-heap to store the top K elements
        auto compare = [](pair<int, int>& a, pair<int, int>& b) {
            return a.second > b.second; // Min-heap based on frequency
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(compare)> minHeap(compare);
        
        for (auto& [num, freq] : freqMap) {
            minHeap.push({num, freq});
            if (minHeap.size() > k) {
                minHeap.pop(); // Keep only the top K elements
            }
        }
        
        // Step 3: Extract the elements from the heap
        vector<int> result;
        while (!minHeap.empty()) {
            result.push_back(minHeap.top().first);
            minHeap.pop();
        }
        
        return result;
    }
};
```

## Explanation

### Step 1: Frequency Count
- Use a hash map (`unordered_map`) to count the frequency of each element in the input array.
- Complexity: **O(n)**, where `n` is the size of the input array.

### Step 2: Heap Usage
- A min-heap (priority queue) is used to maintain the top `k` elements.
- The comparison ensures that the heap keeps the smallest frequency at the top, and elements with larger frequencies are retained in the heap.
  - Adding an element to the heap is **O(log k)**, and the size of the heap is capped at `k`.

### Step 3: Result Extraction
- Extract elements from the heap and store them in the result vector.
- Complexity: **O(k log k)** in total.

## Complexity Analysis

### Time Complexity:
1. Building the frequency map: **O(n)**
2. Managing the heap: **O(n log k)**
3. Total: **O(n log k)**

### Space Complexity:
1. Hash map: **O(n)**
2. Heap: **O(k)**
3. Total: **O(n + k)**

## Example Usage

```cpp
int main() {
    Solution solution;
    vector<int> nums = {1, 1, 1, 2, 2, 3};
    int k = 2;
    vector<int> result = solution.topKFrequent(nums, k);

    // Print result
    for (int num : result) {
        cout << num << " ";
    }
    return 0;
}
```

## Output
For the input `nums = {1, 1, 1, 2, 2, 3}` and `k = 2`, the output will be:
```
1 2
```

This solution is efficient and works well for large inputs.
