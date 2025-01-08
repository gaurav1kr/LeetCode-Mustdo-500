class Solution 
{
public:
    
    void solve(TreeNode *root, int &num)
    {
		// Base Condition
        if(root == NULL)
            return;
		
        solve(root->right, num);
		
        root->val = num + root->val;
		
        num = root->val;
		
        solve(root->left, num);
    }
    
    TreeNode* convertBST(TreeNode* root) 
    {
		// keeps sum of RightSubTree of current Node.
        int num = 0;
        solve(root, num);
        return root;
    }
};