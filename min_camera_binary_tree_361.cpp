]/**
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

    int ans;
    int fun(TreeNode* root) 
    {
        if(!root) return 1;

        int k1 = fun(root->left);
        int k2 = fun(root->right);

        if(k1==0 || k2==0) 
        {
            ans++; 
            return 2;
        }
        if(k1==2 || k2==2)
        {
            return 1;
        }
        
        return 0;

    }
    int minCameraCover(TreeNode* root) 
	{
        ans=0;
        return fun(root)==0 ? ans+1 : ans;
    }
};