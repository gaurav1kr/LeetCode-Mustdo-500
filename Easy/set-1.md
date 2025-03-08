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
```