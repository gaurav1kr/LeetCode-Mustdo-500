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
    TreeNode* trimBST(TreeNode* root, int low, int high) 
	{
        if(!root) return root;
       
        // If the current node's value is less than low, trim the right subtree
        // must be in the left subtree
        if(root->val<low) 
        {
            return trimBST(root->right,low,high);
        }
         // If the current node's value is greater than high, trim the left subtree
        if(root->val>high) 
        {
            return trimBST(root->left,low,high);
        }
         // Otherwise, trim both subtrees
        root->left=trimBST(root->left,low,high);
        root->right=trimBST(root->right,low,high);
        return root;
    }
};