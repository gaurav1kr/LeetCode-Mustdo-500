# LeetCode Solutions with Optimized C++ Code

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
