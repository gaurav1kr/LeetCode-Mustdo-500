class Solution 
{
public:
    bool isSameTree(TreeNode* p, TreeNode* q) 
    {
        if (!p && !q) return true;  // Both are null
        if (!p || !q) return false; // One is null, the other is not
        if (p->val != q->val) return false;  // Values are different
        
        // Recursively check left and right subtrees
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};
