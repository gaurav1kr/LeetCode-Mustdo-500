# Convert Sorted List to Binary Search Tree

This repository contains a C++ solution for the LeetCode problem [Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/). The solution ensures an efficient \(O(n)\) time complexity by simulating in-order traversal.

## Problem Description

Given the head of a singly linked list where elements are sorted in ascending order, convert it to a height-balanced binary search tree.

### Example

Input: `[-10, -3, 0, 5, 9]`  
Output: A height-balanced binary search tree.

### Code

```cpp
#include <iostream>
#include <memory>
using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    TreeNode* sortedListToBST(ListNode* head) {
        if (!head) return nullptr;

        // Count the number of nodes in the linked list
        int size = getSize(head);

        // Use a pointer reference to simulate in-order traversal
        ListNode* current = head;
        return buildTree(current, 0, size - 1);
    }

private:
    // Helper function to count the size of the linked list
    int getSize(ListNode* head) {
        int count = 0;
        while (head) {
            ++count;
            head = head->next;
        }
        return count;
    }

    // Helper function to build the BST
    TreeNode* buildTree(ListNode*& current, int left, int right) {
        if (left > right) return nullptr;

        // Find the middle of the current range
        int mid = left + (right - left) / 2;

        // Recursively build the left subtree
        TreeNode* leftChild = buildTree(current, left, mid - 1);

        // Create the root node with the current value
        TreeNode* root = new TreeNode(current->val);
        root->left = leftChild;

        // Move to the next list node
        current = current->next;

        // Recursively build the right subtree
        root->right = buildTree(current, mid + 1, right);

        return root;
    }
};

// Helper function to print the tree in-order (for testing)
void printInOrder(TreeNode* root) {
    if (!root) return;
    printInOrder(root->left);
    cout << root->val << " ";
    printInOrder(root->right);
}

// Example usage
int main() {
    ListNode* head = new ListNode(-10, new ListNode(-3, new ListNode(0, new ListNode(5, new ListNode(9)))));
    Solution solution;
    TreeNode* root = solution.sortedListToBST(head);

    // Print the in-order traversal of the resulting BST
    printInOrder(root);

    return 0;
}
```

### Key Features
1. **Efficient Traversal**:
   - The `ListNode*& current` pointer simulates an in-order traversal. This avoids repeatedly finding the middle of the list, improving time complexity to \(O(n)\).

2. **Recursive Construction**:
   - The tree is built recursively by dividing the linked list into left and right parts using the `left` and `right` indices.

3. **Memory Efficiency**:
   - Only one pointer (`current`) moves through the list, so no extra space is required apart from the recursion stack.

## License

This code is provided under the MIT License.
