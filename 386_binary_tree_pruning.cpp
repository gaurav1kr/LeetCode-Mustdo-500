class Solution 
{
public:
    int solve(TreeNode* &root)
    {
        if(!root)return 0;
        int  sum = 0;
        sum = root->val + solve(root->left) + solve(root->right);
        if(sum == 0)
	{
            root = NULL;
        }
        return sum;
    }
    TreeNode* pruneTree(TreeNode* root) 
    {
       solve(root);
       return root;   
    }
};

//TC :- O(N)
