# LeetCode Solutions

## 1. Last Stone Weight
**Problem:** Given an array of stones, repeatedly take the two heaviest stones and smash them. If they are of equal weight, both are destroyed; otherwise, the heavier stone is reduced by the weight of the lighter stone. Return the weight of the last remaining stone (or 0 if none remain).

### Approach
- Use a max heap (priority queue) to efficiently get the two heaviest stones.
- Pop the two heaviest stones, compute the difference, and push back the result if non-zero.
- Continue this process until one or zero stones remain.

### Time Complexity
- **O(N log N)** due to heap operations.

### C++ Code
```cpp
#include <queue>
#include <vector>
using namespace std;

int lastStoneWeight(vector<int>& stones) {
    priority_queue<int> maxHeap(stones.begin(), stones.end());
    while (maxHeap.size() > 1) {
        int stone1 = maxHeap.top(); maxHeap.pop();
        int stone2 = maxHeap.top(); maxHeap.pop();
        if (stone1 != stone2) {
            maxHeap.push(stone1 - stone2);
        }
    }
    return maxHeap.empty() ? 0 : maxHeap.top();
}
```

---

## 2. Number Complement
### Approach
- Find the highest bit position of the number.
- Create a bitmask of all 1s up to that position.
- XOR the number with the bitmask to flip its bits.

### Time Complexity
- **O(log N)** since we iterate over the bits of the number.

### C++ Code
```cpp
int findComplement(int num) {
    int mask = (1 << (int)log2(num) + 1) - 1;
    return num ^ mask;
}
```

---

## 3. Valid Perfect Square
### Approach
- Use binary search to check if there exists an integer whose square equals `num`.

### Time Complexity
- **O(log N)** due to binary search.

### C++ Code
```cpp
bool isPerfectSquare(int num) {
    long left = 1, right = num;
    while (left <= right) {
        long mid = left + (right - left) / 2;
        long square = mid * mid;
        if (square == num) return true;
        if (square < num) left = mid + 1;
        else right = mid - 1;
    }
    return false;
}
```

---

## 4. Arranging Coins
### Approach
- Use binary search to find the largest `k` such that `k(k+1)/2 ≤ n`.

### Time Complexity
- **O(log N)**.

### C++ Code
```cpp
int arrangeCoins(int n) {
    long left = 0, right = n;
    while (left <= right) {
        long mid = left + (right - left) / 2;
        long coins = mid * (mid + 1) / 2;
        if (coins == n) return mid;
        if (coins < n) left = mid + 1;
        else right = mid - 1;
    }
    return right;
}
```

---

## 5. Flipping an Image
### Approach
- Reverse each row and flip bits (0 to 1 and 1 to 0).

### Time Complexity
- **O(N^2)**.

### C++ Code
```cpp
vector<vector<int>> flipAndInvertImage(vector<vector<int>>& image) {
    for (auto &row : image) {
        reverse(row.begin(), row.end());
        for (int &pixel : row) pixel ^= 1;
    }
    return image;
}
```

---

## 6. Kth Largest Element in a Stream
### Approach
- Use a min-heap of size `k` to store the `k` largest elements.

### Time Complexity
- **O(log k)** per insert.

### C++ Code
```cpp
class KthLargest {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    int K;
public:
    KthLargest(int k, vector<int>& nums) {
        K = k;
        for (int num : nums) add(num);
    }
    int add(int val) {
        minHeap.push(val);
        if (minHeap.size() > K) minHeap.pop();
        return minHeap.top();
    }
};
```

---

## 7. Maximum Depth of N-ary Tree
### Approach
- Use BFS or DFS to compute the depth.

### Time Complexity
- **O(N)**.

### C++ Code
```cpp
int maxDepth(Node* root) {
    if (!root) return 0;
    int depth = 0;
    for (Node* child : root->children) {
        depth = max(depth, maxDepth(child));
    }
    return depth + 1;
}
```

---

## 8. Degree of an Array
### Approach
- Count frequencies and track first and last positions of elements.
- Compute the smallest subarray containing the most frequent number.

### Time Complexity
- **O(N)**.

### C++ Code
```cpp
int findShortestSubarray(vector<int>& nums) {
    unordered_map<int, int> freq, first, last;
    int degree = 0, minLen = nums.size();
    for (int i = 0; i < nums.size(); i++) {
        if (!first.count(nums[i])) first[nums[i]] = i;
        last[nums[i]] = i;
        degree = max(degree, ++freq[nums[i]]);
    }
    for (auto [num, count] : freq) {
        if (count == degree) minLen = min(minLen, last[num] - first[num] + 1);
    }
    return minLen;
}
```

---

## 9. Find the Difference
### Approach
- Use XOR to cancel out matching characters.

### Time Complexity
- **O(N)**.

### C++ Code
```cpp
char findTheDifference(string s, string t) {
    char result = 0;
    for (char c : s) result ^= c;
    for (char c : t) result ^= c;
    return result;
}
```

---

## 10. Minimum Absolute Difference in BST
### Approach
- Perform an in-order traversal and find the minimum difference between adjacent values.

### Time Complexity
- **O(N)**.

### C++ Code
```cpp
int getMinimumDifference(TreeNode* root) {
    int minDiff = INT_MAX, prev = -1;
    function<void(TreeNode*)> dfs = [&](TreeNode* node) {
        if (!node) return;
        dfs(node->left);
        if (prev != -1) minDiff = min(minDiff, node->val - prev);
        prev = node->val;
        dfs(node->right);
    };
    dfs(root);
    return minDiff;
}
```