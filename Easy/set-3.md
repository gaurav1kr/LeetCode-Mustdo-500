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
```