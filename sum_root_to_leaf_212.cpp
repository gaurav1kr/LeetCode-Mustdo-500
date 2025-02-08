/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution 
{
	public:
    	int helper(TreeNode *root, int sum)
	{
        	if (root==NULL)
		{
            		return 0;
        	}
                sum = sum*10+root->val;
                int l = helper(root->left,sum);
                int r = helper(root->right,sum);
                if (l==0 && r==0)
		{
            		return sum;
        	}
        	return l+r;
        }

    	int sumNumbers(TreeNode* root) 
	{
        	return helper(root,0);
    	}
};
