#include <bits/stdc++.h>
using namespace std;

struct TreeNode 
{
public:
    int data;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int x):data(x),left(nullptr),right(nullptr) {}
};

TreeNode* leftMost(TreeNode* node) 
{
    TreeNode* curr = node;
    while (curr->left != nullptr) 
        curr = curr->left;
    return curr;
}


TreeNode* getSuccessor(TreeNode* root, int target) 
{
    // Base Case 1: No Inorder Successoressor
    if (root == nullptr)
        return nullptr;
     
    // Base Case 2: root is same as target and 
    // right child is not empty
    if (root->data == target && root->right != nullptr)
        return leftMost(root->right);

    // Use BST properties to search for 
    // target and Successor
    TreeNode* succ = nullptr;
    TreeNode* curr = root; 
    while (curr != nullptr) 
    {
        // If curr node is greater, then it
        // is a potential Successor
        if (target < curr->data) 
        {
            succ = curr;
            curr = curr->left;
        }
      
        // If smaller, then Successor must
        // be in the right child
        else if (target >= curr->data) 
            curr = curr->right;
      
        // If same, the last visited
        // greater value is the Successor
        else 
            break;
    }

    return succ;
}

int main() {
  
    // Construct a BST
    //            20
    //          /    \
    //         8      22
    //       /   \
    //      4    12
    //          /  \
    //         10   14
    TreeNode *root = new TreeNode(20);
    root->left = new TreeNode(8);
    root->right = new TreeNode(22);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(12);
    root->left->right->left = new TreeNode(10);
    root->left->right->right = new TreeNode(14);
  
    int target = 14;
    TreeNode* succ = getSuccessor(root, target);
    if (succ != nullptr)
        cout << succ->data;
    else
        cout << "null";
    return 0;
}
