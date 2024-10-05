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
	void bstToGstUtil(TreeNode *root, int & temp)
	{
		if(!root)
		{
			return ;
		}
		bstToGstUtil(root->right , temp) ;
		root->val += temp ;
        	temp = root->val ;
		bstToGstUtil(root->left , temp) ;
	}
   
 	TreeNode* bstToGst(TreeNode* root) 
    	{
       		int temp = 0;
	   	bstToGstUtil(root, temp);
       		return root ;
    	}
};


// TC :- O(N)
// SC :- O(N)
