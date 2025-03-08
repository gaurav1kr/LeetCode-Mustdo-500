# Optimized C++ Solutions for LeetCode Problems

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
