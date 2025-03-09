# LeetCode Optimized C++ Solutions

## 1. Two Sum
**[Problem Link](https://leetcode.com/problems/two-sum/)**  
**Approach:** Use an unordered map (hash table) to store the indices of visited elements and find the complement in O(1) time.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> numIndex;
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (numIndex.find(complement) != numIndex.end()) {
            return {numIndex[complement], i};
        }
        numIndex[nums[i]] = i;
    }
    return {};
}
```

---

## 2. Maximum Subarray
**[Problem Link](https://leetcode.com/problems/maximum-subarray/)**  
**Approach:** Use **Kadane’s Algorithm** to find the maximum sum subarray.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0], currentSum = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        currentSum = max(nums[i], currentSum + nums[i]);
        maxSum = max(maxSum, currentSum);
    }
    return maxSum;
}
```

---

## 3. Best Time to Buy and Sell Stock
**[Problem Link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)**  
**Approach:** Track the minimum price so far and compute the max profit.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

int maxProfit(vector<int>& prices) {
    int minPrice = INT_MAX, maxProfit = 0;
    for (int price : prices) {
        minPrice = min(minPrice, price);
        maxProfit = max(maxProfit, price - minPrice);
    }
    return maxProfit;
}
```

---

## 4. Valid Parentheses
**[Problem Link](https://leetcode.com/problems/valid-parentheses/)**  
**Approach:** Use a stack to match opening and closing brackets.  
**Time Complexity:** O(n)

```cpp
#include <stack>
#include <unordered_map>
#include <string>

using namespace std;

bool isValid(string s) {
    stack<char> st;
    unordered_map<char, char> pairs = {{')', '('}, {']', '['}, {'}', '{'}};
    
    for (char c : s) {
        if (pairs.count(c)) {
            if (st.empty() || st.top() != pairs[c]) return false;
            st.pop();
        } else {
            st.push(c);
        }
    }
    return st.empty();
}
```

---

## 5. Reverse Linked List
**[Problem Link](https://leetcode.com/problems/reverse-linked-list/)**  
**Approach:** Use an iterative method with three pointers.  
**Time Complexity:** O(n)

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *curr = head;
    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
```

---

## 6. Merge Two Sorted Lists
**[Problem Link](https://leetcode.com/problems/merge-two-sorted-lists/)**  
**Approach:** Use a dummy node and merge iteratively.  
**Time Complexity:** O(n + m)

```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy(0), *tail = &dummy;
    while (l1 && l2) {
        if (l1->val < l2->val) {
            tail->next = l1;
            l1 = l1->next;
        } else {
            tail->next = l2;
            l2 = l2->next;
        }
        tail = tail->next;
    }
    tail->next = l1 ? l1 : l2;
    return dummy.next;
}
```

---

## 7. Climbing Stairs
**[Problem Link](https://leetcode.com/problems/climbing-stairs/)**  
**Approach:** Fibonacci approach using dynamic programming.  
**Time Complexity:** O(n)

```cpp
int climbStairs(int n) {
    if (n <= 2) return n;
    int first = 1, second = 2;
    for (int i = 3; i <= n; i++) {
        int third = first + second;
        first = second;
        second = third;
    }
    return second;
}
```

---

## 8. Symmetric Tree
**[Problem Link](https://leetcode.com/problems/symmetric-tree/)**  
**Approach:** Use DFS to compare left and right subtrees recursively.  
**Time Complexity:** O(n)

```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

bool isMirror(TreeNode* t1, TreeNode* t2) {
    if (!t1 || !t2) return t1 == t2;
    return (t1->val == t2->val) && isMirror(t1->left, t2->right) && isMirror(t1->right, t2->left);
}

bool isSymmetric(TreeNode* root) {
    return !root || isMirror(root->left, root->right);
}
```

---

## 9. Single Number
**[Problem Link](https://leetcode.com/problems/single-number/)**  
**Approach:** Use XOR to cancel out duplicates.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int num : nums) result ^= num;
    return result;
}
```

---

## 10. Move Zeroes
**[Problem Link](https://leetcode.com/problems/move-zeroes/)**  
**Approach:** Use two pointers to shift non-zero elements forward.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

void moveZeroes(vector<int>& nums) {
    int lastNonZero = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != 0) {
            swap(nums[i], nums[lastNonZero]);
            lastNonZero++;
        }
    }
}
```# LeetCode Solutions in C++

This document provides optimized C++ solutions for a set of LeetCode problems. Each problem includes an approach explanation, time complexity analysis, and the corresponding C++ code.

## 1. Intersection of Two Arrays

**Approach:** Use an unordered set for quick lookup and another set to store the intersection.

**Time Complexity:** O(m + n)

```cpp
#include <vector>
#include <unordered_set>
using namespace std;

vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
    unordered_set<int> set1(nums1.begin(), nums1.end()), result;
    for (int num : nums2) {
        if (set1.count(num)) result.insert(num);
    }
    return vector<int>(result.begin(), result.end());
}
```

---

## 2. Longest Palindrome

**Approach:** Use an unordered map to count character frequencies.

**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>
using namespace std;

int longestPalindrome(string s) {
    unordered_map<char, int> count;
    for (char c : s) count[c]++;
    int length = 0, odd = 0;
    for (auto& p : count) {
        length += p.second / 2 * 2;
        if (p.second % 2) odd = 1;
    }
    return length + odd;
}
```

---

## 3. Search in a Binary Search Tree

**Approach:** Recursively traverse the BST to find the target value.

**Time Complexity:** O(h) (where h is the height of the tree)

```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* searchBST(TreeNode* root, int val) {
    if (!root || root->val == val) return root;
    return val < root->val ? searchBST(root->left, val) : searchBST(root->right, val);
}
```

---

## 4. Reverse Words in a String III

**Approach:** Reverse each word in place.

**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

string reverseWords(string s) {
    int i = 0, n = s.size();
    for (int j = 0; j <= n; j++) {
        if (j == n || s[j] == ' ') {
            reverse(s.begin() + i, s.begin() + j);
            i = j + 1;
        }
    }
    return s;
}
```

---

## 5. Excel Sheet Column Number

**Approach:** Convert from base-26 notation to decimal.

**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

int titleToNumber(string columnTitle) {
    int result = 0;
    for (char c : columnTitle) {
        result = result * 26 + (c - 'A' + 1);
    }
    return result;
}
```

---

## 6. Design HashMap

**Approach:** Use an array of linked lists for collision handling.

**Time Complexity:** O(1) (average case)

```cpp
#include <vector>
using namespace std;

class MyHashMap {
private:
    vector<int> map;
public:
    MyHashMap() : map(1e6 + 1, -1) {}
    void put(int key, int value) { map[key] = value; }
    int get(int key) { return map[key]; }
    void remove(int key) { map[key] = -1; }
};
```

---

## 7. Convert Binary Number in a Linked List to Integer

**Approach:** Use bitwise operations to compute the integer value.

**Time Complexity:** O(n)

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

int getDecimalValue(ListNode* head) {
    int num = 0;
    while (head) {
        num = (num << 1) | head->val;
        head = head->next;
    }
    return num;
}
```

---

## 8. Max Consecutive Ones

**Approach:** Iterate through the array while keeping track of max consecutive ones.

**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

int findMaxConsecutiveOnes(vector<int>& nums) {
    int maxCount = 0, count = 0;
    for (int num : nums) {
        count = (num == 1) ? count + 1 : 0;
        maxCount = max(maxCount, count);
    }
    return maxCount;
}
```

---

## 9. Running Sum of 1D Array

**Approach:** Maintain a running sum while iterating through the array.

**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

vector<int> runningSum(vector<int>& nums) {
    for (int i = 1; i < nums.size(); i++) {
        nums[i] += nums[i - 1];
    }
    return nums;
}
```

---

## 10. Pascal's Triangle II

**Approach:** Use dynamic programming with a single vector.

**Time Complexity:** O(k^2)

```cpp
#include <vector>
using namespace std;

vector<int> getRow(int rowIndex) {
    vector<int> row(rowIndex + 1, 1);
    for (int i = 1; i < rowIndex; i++) {
        for (int j = i; j > 0; j--) {
            row[j] += row[j - 1];
        }
    }
    return row;
}
```
# Optimized C++ Solutions for LeetCode Problems

## 1. Increasing Order Search Tree
**Approach:** Perform an in-order traversal and store nodes in a new tree.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    TreeNode* increasingBST(TreeNode* root, TreeNode* tail = nullptr) {
        if (!root) return tail;
        TreeNode* res = increasingBST(root->left, root);
        root->left = nullptr;
        root->right = increasingBST(root->right, tail);
        return res;
    }
};
```

## 2. Find Common Characters
**Approach:** Use frequency arrays to track character counts.
**Time Complexity:** O(N*M)
```cpp
class Solution {
public:
    vector<string> commonChars(vector<string>& words) {
        vector<int> common(26, INT_MAX);
        for (const string& word : words) {
            vector<int> freq(26, 0);
            for (char c : word) freq[c - 'a']++;
            for (int i = 0; i < 26; ++i)
                common[i] = min(common[i], freq[i]);
        }
        vector<string> result;
        for (int i = 0; i < 26; ++i)
            while (common[i]-- > 0)
                result.push_back(string(1, 'a' + i));
        return result;
    }
};
```

## 3. Shortest Distance to a Character
**Approach:** Use two passes (left-to-right and right-to-left) to track distances.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    vector<int> shortestToChar(string s, char c) {
        int n = s.size();
        vector<int> res(n, n);
        int pos = -n;
        for (int i = 0; i < n; ++i) {
            if (s[i] == c) pos = i;
            res[i] = i - pos;
        }
        for (int i = n - 1; i >= 0; --i) {
            if (s[i] == c) pos = i;
            res[i] = min(res[i], abs(i - pos));
        }
        return res;
    }
};
```

## 4. Can Place Flowers
**Approach:** Iterate and check adjacent spots for availability.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int count = 0;
        for (int i = 0; i < flowerbed.size(); ++i) {
            if (flowerbed[i] == 0 && (i == 0 || flowerbed[i - 1] == 0) && (i == flowerbed.size() - 1 || flowerbed[i + 1] == 0)) {
                flowerbed[i] = 1;
                count++;
                if (count >= n) return true;
            }
        }
        return count >= n;
    }
};
```

## 5. Contains Duplicate II
**Approach:** Use an unordered_map to track indices.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        for (int i = 0; i < nums.size(); ++i) {
            if (mp.count(nums[i]) && i - mp[nums[i]] <= k)
                return true;
            mp[nums[i]] = i;
        }
        return false;
    }
};
```

## 6. Number of Good Pairs
**Approach:** Use a frequency map to count occurrences.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    int numIdenticalPairs(vector<int>& nums) {
        unordered_map<int, int> freq;
        int count = 0;
        for (int num : nums) {
            count += freq[num];
            freq[num]++;
        }
        return count;
    }
};
```

## 7. Shuffle the Array
**Approach:** Reconstruct array using interleaving.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    vector<int> shuffle(vector<int>& nums, int n) {
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            res.push_back(nums[i]);
            res.push_back(nums[i + n]);
        }
        return res;
    }
};
```

## 8. Kth Missing Positive Number
**Approach:** Use binary search for optimized lookup.
**Time Complexity:** O(log N)
```cpp
class Solution {
public:
    int findKthPositive(vector<int>& arr, int k) {
        int left = 0, right = arr.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] - (mid + 1) < k)
                left = mid + 1;
            else
                right = mid;
        }
        return left + k;
    }
};
```

## 9. Peak Index in a Mountain Array
**Approach:** Use binary search for efficient peak finding.
**Time Complexity:** O(log N)
```cpp
class Solution {
public:
    int peakIndexInMountainArray(vector<int>& arr) {
        int left = 0, right = arr.size() - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] < arr[mid + 1])
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }
};
```

## 10. Find Mode in Binary Search Tree
**Approach:** Perform in-order traversal and track frequency.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    vector<int> findMode(TreeNode* root) {
        unordered_map<int, int> freq;
        int maxFreq = 0;
        vector<int> result;
        function<void(TreeNode*)> inorder = [&](TreeNode* node) {
            if (!node) return;
            inorder(node->left);
            maxFreq = max(maxFreq, ++freq[node->val]);
            inorder(node->right);
        };
        inorder(root);
        for (const auto& p : freq)
            if (p.second == maxFreq)
                result.push_back(p.first);
        return result;
    }
};
```# LeetCode Solutions

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
```# Optimized C++ Solutions for LeetCode Problems

## 1. Count Negative Numbers in a Sorted Matrix

**Approach:**
- Use binary search in each row since they are sorted.
- Count the number of negatives efficiently.

**Time Complexity:** O(m log n)

```cpp
class Solution {
public:
    int countNegatives(vector<vector<int>>& grid) {
        int count = 0;
        for (auto &row : grid) {
            count += row.end() - upper_bound(row.begin(), row.end(), -1, greater<int>());
        }
        return count;
    }
};
```

---
## 2. Reshape the Matrix

**Approach:**
- Flatten the matrix and reshape it using indexing.

**Time Complexity:** O(m*n)

```cpp
class Solution {
public:
    vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
        int m = mat.size(), n = mat[0].size();
        if (m * n != r * c) return mat;
        vector<vector<int>> res(r, vector<int>(c));
        for (int i = 0; i < m * n; ++i)
            res[i / c][i % c] = mat[i / n][i % n];
        return res;
    }
};
```

---
## 3. Toeplitz Matrix

**Approach:**
- Check if each diagonal contains the same element.

**Time Complexity:** O(m*n)

```cpp
class Solution {
public:
    bool isToeplitzMatrix(vector<vector<int>>& matrix) {
        for (int i = 1; i < matrix.size(); i++) {
            for (int j = 1; j < matrix[0].size(); j++) {
                if (matrix[i][j] != matrix[i - 1][j - 1])
                    return false;
            }
        }
        return true;
    }
};
```

---
## 4. Range Sum Query - Immutable

**Approach:**
- Use prefix sum array.

**Time Complexity:** O(1) per query

```cpp
class NumArray {
    vector<int> prefixSum;
public:
    NumArray(vector<int>& nums) {
        prefixSum.push_back(0);
        for (int num : nums)
            prefixSum.push_back(prefixSum.back() + num);
    }
    int sumRange(int left, int right) {
        return prefixSum[right + 1] - prefixSum[left];
    }
};
```

---
## 5. Implement Stack using Queues

**Approach:**
- Use two queues to simulate stack behavior.

**Time Complexity:** O(1) for push, O(n) for pop

```cpp
class MyStack {
    queue<int> q;
public:
    void push(int x) {
        q.push(x);
        for (int i = 0; i < q.size() - 1; ++i) {
            q.push(q.front());
            q.pop();
        }
    }
    int pop() {
        int top = q.front();
        q.pop();
        return top;
    }
    int top() { return q.front(); }
    bool empty() { return q.empty(); }
};
```

---
## 6. Next Greater Element I

**Approach:**
- Use a stack to maintain the next greater element.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> nge;
        stack<int> s;
        for (int num : nums2) {
            while (!s.empty() && s.top() < num) {
                nge[s.top()] = num;
                s.pop();
            }
            s.push(num);
        }
        vector<int> res;
        for (int num : nums1)
            res.push_back(nge.count(num) ? nge[num] : -1);
        return res;
    }
};
```

---
## 7. Sum of Root to Leaf Binary Numbers

**Approach:**
- Use DFS to traverse the tree and calculate sum.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int sumRootToLeaf(TreeNode* root, int sum = 0) {
        if (!root) return 0;
        sum = sum * 2 + root->val;
        return root->left == root->right ? sum : sumRootToLeaf(root->left, sum) + sumRootToLeaf(root->right, sum);
    }
};
```

---
## 8. Set Mismatch

**Approach:**
- Use frequency count to find missing and duplicate numbers.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    vector<int> findErrorNums(vector<int>& nums) {
        vector<int> count(nums.size() + 1, 0);
        for (int num : nums) count[num]++;
        int dup = -1, missing = -1;
        for (int i = 1; i < count.size(); i++) {
            if (count[i] == 2) dup = i;
            if (count[i] == 0) missing = i;
        }
        return {dup, missing};
    }
};
```

---
## 9. Rotate String

**Approach:**
- Check if `goal` is a substring of `s + s`.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool rotateString(string s, string goal) {
        return s.size() == goal.size() && (s + s).find(goal) != string::npos;
    }
};
```

---
## 10. Unique Email Addresses

**Approach:**
- Normalize emails and use a set to count unique ones.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int numUniqueEmails(vector<string>& emails) {
        unordered_set<string> uniqueEmails;
        for (string email : emails) {
            string local, domain;
            int i = 0;
            while (i < email.size() && email[i] != '@' && email[i] != '+') {
                if (email[i] != '.') local += email[i];
                i++;
            }
            while (email[i] != '@') i++;
            domain = email.substr(i);
            uniqueEmails.insert(local + domain);
        }
        return uniqueEmails.size();
    }
};
```# Optimized C++ Solutions for LeetCode Problems

## 1. Majority Element
**Problem Link:** [Majority Element](https://leetcode.com/problems/majority-element)

### Approach:
- Use **Boyer-Moore Voting Algorithm** to find the majority element.
- Maintain a `candidate` element and a `count`. Iterate through the array adjusting the count.

### Code:
```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int candidate = 0, count = 0;
        for (int num : nums) {
            if (count == 0) candidate = num;
            count += (num == candidate) ? 1 : -1;
        }
        return candidate;
    }
};
```

### Time Complexity:
- **O(N)**, where N is the number of elements in the array.

---

## 2. Palindrome Linked List
**Problem Link:** [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list)

### Approach:
- Find the middle using slow and fast pointers.
- Reverse the second half and compare it with the first half.

### Code:
```cpp
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        ListNode *slow = head, *fast = head, *prev = nullptr;
        while (fast && fast->next) {
            fast = fast->next->next;
            ListNode* temp = slow;
            slow = slow->next;
            temp->next = prev;
            prev = temp;
        }
        if (fast) slow = slow->next;
        while (slow) {
            if (slow->val != prev->val) return false;
            slow = slow->next;
            prev = prev->next;
        }
        return true;
    }
};
```

### Time Complexity:
- **O(N)**, where N is the length of the linked list.

---

## 3. Invert Binary Tree
**Problem Link:** [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree)

### Approach:
- Recursively swap left and right children.

### Code:
```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;
        swap(root->left, root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
};
```

### Time Complexity:
- **O(N)**, where N is the number of nodes in the tree.

---

## 4. Min Stack
**Problem Link:** [Min Stack](https://leetcode.com/problems/min-stack)

### Approach:
- Use a stack to store values and another stack to track the minimum values.

### Code:
```cpp
class MinStack {
    stack<int> s, min_s;
public:
    void push(int val) {
        s.push(val);
        if (min_s.empty() || val <= min_s.top()) min_s.push(val);
    }
    void pop() {
        if (s.top() == min_s.top()) min_s.pop();
        s.pop();
    }
    int top() { return s.top(); }
    int getMin() { return min_s.top(); }
};
```

### Time Complexity:
- **O(1)** for all operations.

---

## 9. Search Insert Position
**Problem Link:** [Search Insert Position](https://leetcode.com/problems/search-insert-position)

### Approach:
- Use binary search to find the correct position.

### Code:
```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return left;
    }
};
```

### Time Complexity:
- **O(log N)**

---

## 10. Find All Numbers Disappeared in an Array
**Problem Link:** [Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array)

### Approach:
- Use marking technique to mark indices corresponding to seen numbers.

### Code:
```cpp
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        for (int num : nums) nums[abs(num) - 1] = -abs(nums[abs(num) - 1]);
        vector<int> result;
        for (int i = 0; i < nums.size(); i++)
            if (nums[i] > 0) result.push_back(i + 1);
        return result;
    }
};
```

### Time Complexity:
- **O(N)**
# Optimized C++ Solutions for LeetCode Problems

## 1. [Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees)
### Approach:
Use recursion to merge nodes from both trees. If a node exists in both trees, sum their values; otherwise, use the non-null node.
### Time Complexity: O(min(N, M))
### Code:
```cpp
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        if (!root1) return root2;
        if (!root2) return root1;
        root1->val += root2->val;
        root1->left = mergeTrees(root1->left, root2->left);
        root1->right = mergeTrees(root1->right, root2->right);
        return root1;
    }
};
```

## 2. [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree)
### Approach:
Use depth-first search (DFS) to traverse the tree and compute the depth recursively.
### Time Complexity: O(N)
### Code:
```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};
```

## 3. [Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree)
### Approach:
Use a divide-and-conquer approach to create a balanced BST by selecting the middle element as the root recursively.
### Time Complexity: O(N)
### Code:
```cpp
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return helper(nums, 0, nums.size() - 1);
    }
    TreeNode* helper(vector<int>& nums, int left, int right) {
        if (left > right) return nullptr;
        int mid = left + (right - left) / 2;
        TreeNode* node = new TreeNode(nums[mid]);
        node->left = helper(nums, left, mid - 1);
        node->right = helper(nums, mid + 1, right);
        return node;
    }
};
```

## 4. [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array)
### Approach:
Use a two-pointer technique to overwrite duplicate elements.
### Time Complexity: O(N)
### Code:
```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int i = 0;
        for (int j = 1; j < nums.size(); j++) {
            if (nums[j] != nums[i]) nums[++i] = nums[j];
        }
        return i + 1;
    }
};
```

## 5. [Counting Bits](https://leetcode.com/problems/counting-bits)
### Approach:
Use dynamic programming with the relation `dp[i] = dp[i / 2] + (i % 2)`.
### Time Complexity: O(N)
### Code:
```cpp
class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> dp(n + 1, 0);
        for (int i = 1; i <= n; i++) dp[i] = dp[i / 2] + (i % 2);
        return dp;
    }
};
```

## 6. [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs)
### Approach:
Use dynamic programming to find the minimum cost to reach the top.
### Time Complexity: O(N)
### Code:
```cpp
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> dp(n + 1);
        for (int i = 2; i <= n; i++) {
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
    }
};
```

## 7. [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree)
### Approach:
Use DFS to check the height of left and right subtrees.
### Time Complexity: O(N)
### Code:
```cpp
class Solution {
public:
    int height(TreeNode* root) {
        if (!root) return 0;
        int left = height(root->left);
        int right = height(root->right);
        if (left == -1 || right == -1 || abs(left - right) > 1) return -1;
        return max(left, right) + 1;
    }
    bool isBalanced(TreeNode* root) {
        return height(root) != -1;
    }
};
```

## 8. [Palindrome Number](https://leetcode.com/problems/palindrome-number)
### Approach:
Reverse half of the number and compare with the other half.
### Time Complexity: O(logN)
### Code:
```cpp
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) return false;
        int rev = 0;
        while (x > rev) {
            rev = rev * 10 + x % 10;
            x /= 10;
        }
        return x == rev || x == rev / 10;
    }
};
```

## 9. [Same Tree](https://leetcode.com/problems/same-tree)
### Approach:
Use recursion to check if both trees are identical.
### Time Complexity: O(N)
### Code:
```cpp
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (!p || !q) return p == q;
        return p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};
```

## 10. [Lowest Common Ancestor of a BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree)
### Approach:
Use the BST property to traverse to the LCA node.
### Time Complexity: O(H), where H is the height of the tree
### Code:
```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        while (root) {
            if (root->val > p->val && root->val > q->val) root = root->left;
            else if (root->val < p->val && root->val < q->val) root = root->right;
            else return root;
        }
        return nullptr;
    }
};
```# LeetCode Solutions with Optimized C++ Code

## 1. [Path Sum](https://leetcode.com/problems/path-sum)
### Approach:
- Use DFS to traverse the tree.
- Subtract the node value from the target sum at each step.
- If we reach a leaf node and the remaining sum equals the leaf's value, return true.

```cpp
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (!root) return false;
        if (!root->left && !root->right && targetSum == root->val) return true;
        return hasPathSum(root->left, targetSum - root->val) || hasPathSum(root->right, targetSum - root->val);
    }
};
```

## 2. [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree)
### Approach:
- Use DFS to check if two trees are identical.
- Traverse the main tree and compare each node's subtree with the given subtree.

```cpp
class Solution {
public:
    bool isSame(TreeNode* s, TreeNode* t) {
        if (!s || !t) return s == t;
        return (s->val == t->val) && isSame(s->left, t->left) && isSame(s->right, t->right);
    }
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        if (!root) return false;
        return isSame(root, subRoot) || isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot);
    }
};
```

## 3. [Missing Number](https://leetcode.com/problems/missing-number)
### Approach:
- Use the sum formula `(n * (n+1)) / 2` and subtract the sum of elements.

```cpp
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int sum = n * (n + 1) / 2;
        for (int num : nums) sum -= num;
        return sum;
    }
};
```

## 4. [Pascal's Triangle](https://leetcode.com/problems/pascals-triangle)
### Approach:
- Generate rows iteratively using previous row values.

```cpp
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> res(numRows);
        for (int i = 0; i < numRows; i++) {
            res[i] = vector<int>(i + 1, 1);
            for (int j = 1; j < i; j++) {
                res[i][j] = res[i - 1][j - 1] + res[i - 1][j];
            }
        }
        return res;
    }
};
```

## 5. [Happy Number](https://leetcode.com/problems/happy-number)
### Approach:
- Use Floyd's cycle detection to detect loops.

```cpp
class Solution {
public:
    int getNext(int n) {
        int sum = 0;
        while (n) {
            sum += (n % 10) * (n % 10);
            n /= 10;
        }
        return sum;
    }
    bool isHappy(int n) {
        int slow = n, fast = getNext(n);
        while (fast != 1 && slow != fast) {
            slow = getNext(slow);
            fast = getNext(getNext(fast));
        }
        return fast == 1;
    }
};
```

## 6. [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list)
### Approach:
- Use slow and fast pointers.

```cpp
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }
};
```

## 7. [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted)
### Approach:
- Use two pointers from left and right.

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int left = 0, right = numbers.size() - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) return {left + 1, right + 1};
            else if (sum < target) left++;
            else right--;
        }
        return {};
    }
};
```

## 8. [First Unique Character in a String](https://leetcode.com/problems/first-unique-character-in-a-string)
### Approach:
- Use a frequency map.

```cpp
class Solution {
public:
    int firstUniqChar(string s) {
        vector<int> count(26, 0);
        for (char c : s) count[c - 'a']++;
        for (int i = 0; i < s.size(); i++) if (count[s[i] - 'a'] == 1) return i;
        return -1;
    }
};
```

## 9. [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array)
### Approach:
- Use two pointers from both ends of the array.

```cpp
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int n = nums.size(), left = 0, right = n - 1;
        vector<int> res(n);
        for (int i = n - 1; i >= 0; i--) {
            if (abs(nums[left]) > abs(nums[right])) res[i] = nums[left] * nums[left++];
            else res[i] = nums[right] * nums[right--];
        }
        return res;
    }
};
```

## 10. [Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements)
### Approach:
- Use a dummy node and traverse the list.

```cpp
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode dummy(0), *prev = &dummy;
        dummy.next = head;
        while (head) {
            if (head->val == val) prev->next = head->next;
            else prev = head;
            head = head->next;
        }
        return dummy.next;
    }
};
```
# LeetCode Solutions in C++

## 1. Island Perimeter
**Approach:**  
Iterate through the grid and count the number of land cells. For each land cell, add 4 to the perimeter and subtract 2 for each adjacent land cell.

**Time Complexity:** O(m * n)

```cpp
class Solution {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int perimeter = 0;
        int rows = grid.size(), cols = grid[0].size();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    perimeter += 4;
                    if (i > 0 && grid[i-1][j] == 1) perimeter -= 2;
                    if (j > 0 && grid[i][j-1] == 1) perimeter -= 2;
                }
            }
        }
        return perimeter;
    }
};
```

## 2. Remove Duplicates from Sorted List
**Approach:**  
Use a single pass through the linked list to remove consecutive duplicate elements.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* current = head;
        while (current && current->next) {
            if (current->val == current->next->val) {
                current->next = current->next->next;
            } else {
                current = current->next;
            }
        }
        return head;
    }
};
```

## 3. Add Binary
**Approach:**  
Traverse both strings from end to start, add digits with carry and construct the result.

**Time Complexity:** O(max(n, m))

```cpp
class Solution {
public:
    string addBinary(string a, string b) {
        string result = "";
        int carry = 0, i = a.size() - 1, j = b.size() - 1;
        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) sum += a[i--] - '0';
            if (j >= 0) sum += b[j--] - '0';
            result += (sum % 2) + '0';
            carry = sum / 2;
        }
        reverse(result.begin(), result.end());
        return result;
    }
};
```

## 4. Valid Anagram
**Approach:**  
Count the frequency of characters in both strings and compare.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        if (s.size() != t.size()) return false;
        vector<int> count(26, 0);
        for (char c : s) count[c - 'a']++;
        for (char c : t) {
            if (--count[c - 'a'] < 0) return false;
        }
        return true;
    }
};
```

## 5. First Bad Version
**Approach:**  
Use binary search to minimize API calls.

**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int firstBadVersion(int n) {
        int left = 1, right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (isBadVersion(mid)) right = mid;
            else left = mid + 1;
        }
        return left;
    }
};
```

## 6. Valid Palindrome II
**Approach:**  
Use two pointers and allow at most one deletion.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isPalindrome(string s, int left, int right) {
        while (left < right) {
            if (s[left++] != s[right--]) return false;
        }
        return true;
    }
    
    bool validPalindrome(string s) {
        int left = 0, right = s.size() - 1;
        while (left < right) {
            if (s[left] != s[right]) {
                return isPalindrome(s, left + 1, right) || isPalindrome(s, left, right - 1);
            }
            left++; right--;
        }
        return true;
    }
};
```

## 7. Delete Node in a Linked List
**Approach:**  
Copy the next node's value and delete it.

**Time Complexity:** O(1)

```cpp
class Solution {
public:
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }
};
```

## 8. Intersection of Two Arrays II
**Approach:**  
Use a hashmap to count occurrences in one array and match with the other.

**Time Complexity:** O(n + m)

```cpp
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> count;
        vector<int> result;
        for (int num : nums1) count[num]++;
        for (int num : nums2) {
            if (count[num] > 0) {
                result.push_back(num);
                count[num]--;
            }
        }
        return result;
    }
};
```

## 9. Reverse String
**Approach:**  
Use two-pointer technique.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    void reverseString(vector<char>& s) {
        int left = 0, right = s.size() - 1;
        while (left < right) {
            swap(s[left++], s[right--]);
        }
    }
};
```

## 10. Range Sum of BST
**Approach:**  
Use DFS to sum nodes within range.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int rangeSumBST(TreeNode* root, int low, int high) {
        if (!root) return 0;
        int sum = 0;
        if (root->val >= low && root->val <= high) sum += root->val;
        if (root->val > low) sum += rangeSumBST(root->left, low, high);
        if (root->val < high) sum += rangeSumBST(root->right, low, high);
        return sum;
    }
};
```# Optimized C++ Solutions for LeetCode Problems

## 1. Binary Tree Postorder Traversal
### Approach
We use an iterative approach with a stack to efficiently perform postorder traversal (Left, Right, Root).

### Time Complexity
**O(n)** - We visit each node exactly once.

### C++ Code
```cpp
#include <vector>
#include <stack>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

vector<int> postorderTraversal(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    stack<TreeNode*> s;
    TreeNode* last = nullptr;
    while (!s.empty() || root) {
        if (root) {
            s.push(root);
            root = root->left;
        } else {
            TreeNode* node = s.top();
            if (node->right && last != node->right)
                root = node->right;
            else {
                result.push_back(node->val);
                last = node;
                s.pop();
            }
        }
    }
    return result;
}
```

---
## 2. Is Subsequence
### Approach
Use two pointers to check if `s` is a subsequence of `t`.

### Time Complexity
**O(n)** - We traverse `t` at most once.

### C++ Code
```cpp
#include <string>
using namespace std;

bool isSubsequence(string s, string t) {
    int i = 0, j = 0;
    while (i < s.size() && j < t.size()) {
        if (s[i] == t[j]) i++;
        j++;
    }
    return i == s.size();
}
```

---
## 3. Binary Tree Paths
### Approach
Use DFS with backtracking to generate all root-to-leaf paths.

### Time Complexity
**O(n)** - We visit each node once.

### C++ Code
```cpp
#include <vector>
#include <string>
using namespace std;

void dfs(TreeNode* root, string path, vector<string>& paths) {
    if (!root) return;
    path += to_string(root->val);
    if (!root->left && !root->right) {
        paths.push_back(path);
        return;
    }
    path += "->";
    dfs(root->left, path, paths);
    dfs(root->right, path, paths);
}

vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> paths;
    dfs(root, "", paths);
    return paths;
}
```

---
## 4. Minimum Depth of Binary Tree
### Approach
Use BFS to find the first leaf node.

### Time Complexity
**O(n)** - We visit each node at most once.

### C++ Code
```cpp
#include <queue>
using namespace std;

int minDepth(TreeNode* root) {
    if (!root) return 0;
    queue<pair<TreeNode*, int>> q;
    q.push({root, 1});
    while (!q.empty()) {
        auto [node, depth] = q.front(); q.pop();
        if (!node->left && !node->right) return depth;
        if (node->left) q.push({node->left, depth + 1});
        if (node->right) q.push({node->right, depth + 1});
    }
    return 0;
}
```

---
## 5. Binary Tree Preorder Traversal
### Approach
Use an iterative approach with a stack.

### Time Complexity
**O(n)**

### C++ Code
```cpp
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    stack<TreeNode*> s;
    s.push(root);
    while (!s.empty()) {
        TreeNode* node = s.top(); s.pop();
        result.push_back(node->val);
        if (node->right) s.push(node->right);
        if (node->left) s.push(node->left);
    }
    return result;
}
```

---
## 6. Plus One
### Approach
Simulate addition from the least significant digit.

### Time Complexity
**O(n)**

### C++ Code
```cpp
#include <vector>
using namespace std;

vector<int> plusOne(vector<int>& digits) {
    int n = digits.size();
    for (int i = n - 1; i >= 0; i--) {
        if (digits[i] < 9) {
            digits[i]++;
            return digits;
        }
        digits[i] = 0;
    }
    digits.insert(digits.begin(), 1);
    return digits;
}
```

---
## 7. Backspace String Compare
### Approach
Use a stack to simulate typing.

### Time Complexity
**O(n)**

### C++ Code
```cpp
#include <string>
using namespace std;

string buildString(string s) {
    string result;
    for (char c : s) {
        if (c == '#' && !result.empty()) result.pop_back();
        else if (c != '#') result.push_back(c);
    }
    return result;
}

bool backspaceCompare(string s, string t) {
    return buildString(s) == buildString(t);
}
```

---
## 8. Implement strStr()
### Approach
Use the KMP algorithm for efficient substring search.

### Time Complexity
**O(n + m)**

### C++ Code
```cpp
#include <vector>
using namespace std;

int strStr(string haystack, string needle) {
    return haystack.find(needle);
}
```

---
## 9. Contains Duplicate
### Approach
Use an unordered set to check for duplicates.

### Time Complexity
**O(n)**

### C++ Code
```cpp
#include <unordered_set>
using namespace std;

bool containsDuplicate(vector<int>& nums) {
    unordered_set<int> s;
    for (int num : nums) {
        if (s.count(num)) return true;
        s.insert(num);
    }
    return false;
}
```

---
## 10. Jewels and Stones
### Approach
Use an unordered set for quick lookups.

### Time Complexity
**O(n)**

### C++ Code
```cpp
#include <unordered_set>
using namespace std;

int numJewelsInStones(string jewels, string stones) {
    unordered_set<char> jewelSet(jewels.begin(), jewels.end());
    int count = 0;
    for (char stone : stones) {
        if (jewelSet.count(stone)) count++;
    }
    return count;
}
```
# Optimized C++ Solutions for LeetCode Problems

## 1. Flood Fill
**Approach:** Use DFS or BFS to traverse and fill connected pixels.
**Time Complexity:** O(m * n)

```cpp
class Solution {
public:
    void dfs(vector<vector<int>>& image, int sr, int sc, int color, int newColor) {
        if (sr < 0 || sc < 0 || sr >= image.size() || sc >= image[0].size() || image[sr][sc] != color) return;
        image[sr][sc] = newColor;
        dfs(image, sr + 1, sc, color, newColor);
        dfs(image, sr - 1, sc, color, newColor);
        dfs(image, sr, sc + 1, color, newColor);
        dfs(image, sr, sc - 1, color, newColor);
    }
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        if (image[sr][sc] != newColor) dfs(image, sr, sc, image[sr][sc], newColor);
        return image;
    }
};
```

## 2. Two Sum IV - Input is a BST
**Approach:** Use inorder traversal with a hash set to check for sum.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool findTarget(TreeNode* root, int k) {
        unordered_set<int> s;
        return inorder(root, k, s);
    }
    bool inorder(TreeNode* node, int k, unordered_set<int>& s) {
        if (!node) return false;
        if (s.count(k - node->val)) return true;
        s.insert(node->val);
        return inorder(node->left, k, s) || inorder(node->right, k, s);
    }
};
```

## 3. Sqrt(x)
**Approach:** Use binary search to find the square root.
**Time Complexity:** O(log x)

```cpp
class Solution {
public:
    int mySqrt(int x) {
        if (x == 0) return 0;
        int left = 1, right = x, ans;
        while (left <= right) {
            long mid = left + (right - left) / 2;
            if (mid * mid <= x) {
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;
    }
};
```

## 4. Isomorphic Strings
**Approach:** Use two maps to track character mappings.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        unordered_map<char, char> s_map, t_map;
        for (int i = 0; i < s.size(); i++) {
            if (s_map[s[i]] && s_map[s[i]] != t[i]) return false;
            if (t_map[t[i]] && t_map[t[i]] != s[i]) return false;
            s_map[s[i]] = t[i];
            t_map[t[i]] = s[i];
        }
        return true;
    }
};
```

## 5. Binary Search
**Approach:** Classic binary search.
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
};
```

## 6. Repeated Substring Pattern
**Approach:** Check string concatenation without the first and last character.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        string doubled = s + s;
        return doubled.substr(1, doubled.size() - 2).find(s) != string::npos;
    }
};
```

## 7. Remove Element
**Approach:** Use two pointers.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int j = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != val) nums[j++] = nums[i];
        }
        return j;
    }
};
```

## 8. Sum of Left Leaves
**Approach:** Use recursion to sum left leaves.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        if (!root) return 0;
        int sum = 0;
        if (root->left && !root->left->left && !root->left->right) sum += root->left->val;
        return sum + sumOfLeftLeaves(root->left) + sumOfLeftLeaves(root->right);
    }
};
```

## 9. Hamming Distance
**Approach:** Use XOR and count bits.
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int hammingDistance(int x, int y) {
        int xor_val = x ^ y, count = 0;
        while (xor_val) {
            count += xor_val & 1;
            xor_val >>= 1;
        }
        return count;
    }
};
```

## 10. Valid Palindrome
**Approach:** Two pointers, ignoring non-alphanumeric characters.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isPalindrome(string s) {
        int left = 0, right = s.size() - 1;
        while (left < right) {
            while (left < right && !isalnum(s[left])) left++;
            while (left < right && !isalnum(s[right])) right--;
            if (tolower(s[left]) != tolower(s[right])) return false;
            left++, right--;
        }
        return true;
    }
};
```

---

Each solution is optimized for time and space efficiency. Let me know if you need any modifications!
# LeetCode Solutions in C++

## 1. Add Strings ([Problem Link](https://leetcode.com/problems/add-strings))

### Approach:
- Use two pointers to traverse the strings from the end.
- Perform addition digit by digit, keeping track of carry.
- Append the result to a string and reverse it.

### Time Complexity: O(max(N, M))

```cpp
class Solution {
public:
    string addStrings(string num1, string num2) {
        int i = num1.size() - 1, j = num2.size() - 1, carry = 0;
        string result = "";
        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) sum += num1[i--] - '0';
            if (j >= 0) sum += num2[j--] - '0';
            result += (sum % 10) + '0';
            carry = sum / 10;
        }
        reverse(result.begin(), result.end());
        return result;
    }
};
```

## 2. Implement Queue using Stacks ([Problem Link](https://leetcode.com/problems/implement-queue-using-stacks))

### Approach:
- Use two stacks: one for enqueue and one for dequeue.
- Transfer elements from the input stack to the output stack when needed.

### Time Complexity: O(1) amortized per operation

```cpp
class MyQueue {
    stack<int> input, output;
public:
    void push(int x) {
        input.push(x);
    }
    int pop() {
        peek();
        int val = output.top(); output.pop();
        return val;
    }
    int peek() {
        if (output.empty()) {
            while (!input.empty()) {
                output.push(input.top());
                input.pop();
            }
        }
        return output.top();
    }
    bool empty() {
        return input.empty() && output.empty();
    }
};
```

## 3. Roman to Integer ([Problem Link](https://leetcode.com/problems/roman-to-integer))

### Approach:
- Use a hashmap to store values.
- Traverse from left to right, subtracting when a smaller value precedes a larger one.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    int romanToInt(string s) {
        unordered_map<char, int> values = {{'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}};
        int result = 0, prev = 0;
        for (char c : s) {
            int curr = values[c];
            result += (curr > prev) ? (curr - 2 * prev) : curr;
            prev = curr;
        }
        return result;
    }
};
```

## 4. Find the Town Judge ([Problem Link](https://leetcode.com/problems/find-the-town-judge))

### Approach:
- Use an array to track trust scores.
- Judge should have N-1 trust and trust no one.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    int findJudge(int n, vector<vector<int>>& trust) {
        vector<int> trustScore(n + 1, 0);
        for (auto& t : trust) {
            trustScore[t[0]]--;
            trustScore[t[1]]++;
        }
        for (int i = 1; i <= n; ++i) {
            if (trustScore[i] == n - 1) return i;
        }
        return -1;
    }
};
```

## 5. Merge Sorted Array ([Problem Link](https://leetcode.com/problems/merge-sorted-array))

### Approach:
- Merge from the end to avoid extra space.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while (j >= 0) {
            nums1[k--] = (i >= 0 && nums1[i] > nums2[j]) ? nums1[i--] : nums2[j--];
        }
    }
};
```

## 6. Remove All Adjacent Duplicates in String ([Problem Link](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string))

### Approach:
- Use a stack-based approach.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    string removeDuplicates(string s) {
        string result;
        for (char c : s) {
            if (!result.empty() && result.back() == c)
                result.pop_back();
            else
                result.push_back(c);
        }
        return result;
    }
};
```

## 7. Average of Levels in Binary Tree ([Problem Link](https://leetcode.com/problems/average-of-levels-in-binary-tree))

### Approach:
- Use BFS to compute level-wise sum.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    vector<double> averageOfLevels(TreeNode* root) {
        vector<double> result;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            long sum = 0, count = q.size();
            for (int i = 0; i < count; i++) {
                TreeNode* node = q.front(); q.pop();
                sum += node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            result.push_back((double)sum / count);
        }
        return result;
    }
};
```

## 8. Find Pivot Index ([Problem Link](https://leetcode.com/problems/find-pivot-index))

### Approach:
- Calculate prefix sums and compare.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        int totalSum = accumulate(nums.begin(), nums.end(), 0), leftSum = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (leftSum == totalSum - leftSum - nums[i]) return i;
            leftSum += nums[i];
        }
        return -1;
    }
};
```

## 9. How Many Numbers Are Smaller Than the Current Number ([Problem Link](https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number))

### Time Complexity: O(N)

```cpp
class Solution {
public:
    vector<int> smallerNumbersThanCurrent(vector<int>& nums) {
        vector<int> count(101, 0);
        for (int num : nums) count[num]++;
        for (int i = 1; i < 101; i++) count[i] += count[i - 1];
        vector<int> result;
        for (int num : nums) result.push_back(num ? count[num - 1] : 0);
        return result;
    }
};
```

## 10. Power of Two ([Problem Link](https://leetcode.com/problems/power-of-two))

### Time Complexity: O(1)

```cpp
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
};
```# Optimized C++ Solutions for LeetCode Problems

## 1. Fibonacci Number
**Approach:** Use dynamic programming with memoization to achieve O(n) time complexity.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int fib(int n) {
        if (n <= 1) return n;
        int a = 0, b = 1, c;
        for (int i = 2; i <= n; ++i) {
            c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
};
```

---

## 2. Reverse Bits
**Approach:** Iterate over 32 bits, shift and set them in reverse order.

**Time Complexity:** O(1)

```cpp
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t result = 0;
        for (int i = 0; i < 32; ++i) {
            result = (result << 1) | (n & 1);
            n >>= 1;
        }
        return result;
    }
};
```

---

## 3. Word Pattern
**Approach:** Use two hash maps to maintain mappings between words and characters.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool wordPattern(string pattern, string s) {
        unordered_map<char, string> pToS;
        unordered_map<string, char> sToP;
        istringstream in(s);
        string word;
        vector<string> words;
        while (in >> word) words.push_back(word);
        if (words.size() != pattern.size()) return false;
        
        for (int i = 0; i < pattern.size(); ++i) {
            if (pToS.count(pattern[i]) && pToS[pattern[i]] != words[i]) return false;
            if (sToP.count(words[i]) && sToP[words[i]] != pattern[i]) return false;
            pToS[pattern[i]] = words[i];
            sToP[words[i]] = pattern[i];
        }
        return true;
    }
};
```

---

## 4. Verifying an Alien Dictionary
**Approach:** Use a map to store the order of characters and compare words accordingly.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isAlienSorted(vector<string>& words, string order) {
        unordered_map<char, int> dict;
        for (int i = 0; i < order.size(); ++i) dict[order[i]] = i;
        for (int i = 0; i < words.size() - 1; ++i) {
            for (int j = 0; j < words[i].size(); ++j) {
                if (j >= words[i+1].size()) return false;
                if (dict[words[i][j]] < dict[words[i+1][j]]) break;
                if (dict[words[i][j]] > dict[words[i+1][j]]) return false;
            }
        }
        return true;
    }
};
```

---

## 5. Cousins in Binary Tree
**Approach:** Use BFS to find depth and parent of the given nodes.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isCousins(TreeNode* root, int x, int y) {
        queue<pair<TreeNode*, TreeNode*>> q;
        q.push({root, nullptr});
        while (!q.empty()) {
            int size = q.size();
            TreeNode *parentX = nullptr, *parentY = nullptr;
            for (int i = 0; i < size; ++i) {
                auto [node, parent] = q.front(); q.pop();
                if (node->val == x) parentX = parent;
                if (node->val == y) parentY = parent;
                if (node->left) q.push({node->left, node});
                if (node->right) q.push({node->right, node});
            }
            if (parentX && parentY) return parentX != parentY;
            if (parentX || parentY) return false;
        }
        return false;
    }
};
```

---

## 6. Count Binary Substrings
**Approach:** Use a single pass to count consecutive groups of 0s and 1s.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int countBinarySubstrings(string s) {
        int prev = 0, cur = 1, count = 0;
        for (int i = 1; i < s.size(); ++i) {
            if (s[i] == s[i - 1])
                cur++;
            else {
                count += min(prev, cur);
                prev = cur;
                cur = 1;
            }
        }
        return count + min(prev, cur);
    }
};
```

---

## 7. Sort Array by Parity
**Approach:** Use two-pointer technique.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    vector<int> sortArrayByParity(vector<int>& nums) {
        int i = 0, j = nums.size() - 1;
        while (i < j) {
            if (nums[i] % 2 > nums[j] % 2) swap(nums[i], nums[j]);
            if (nums[i] % 2 == 0) ++i;
            if (nums[j] % 2 == 1) --j;
        }
        return nums;
    }
};
```

---

## 8. Number of 1 Bits
**Approach:** Use bitwise AND and shift operations.

**Time Complexity:** O(1)

```cpp
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count = 0;
        while (n) {
            count += n & 1;
            n >>= 1;
        }
        return count;
    }
};
```

---

## 9. Maximum Product of Three Numbers
**Approach:** Find the top three largest and two smallest numbers.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int maximumProduct(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        return max(nums[0] * nums[1] * nums[n - 1],
                   nums[n - 1] * nums[n - 2] * nums[n - 3]);
    }
};
```

---

## 10. Excel Sheet Column Title
**Approach:** Convert number to base-26 format with character mapping.

**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    string convertToTitle(int columnNumber) {
        string result = "";
        while (columnNumber) {
            columnNumber--;
            result = char('A' + columnNumber % 26) + result;
            columnNumber /= 26;
        }
        return result;
    }
};
```
