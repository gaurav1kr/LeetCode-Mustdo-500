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
    bool isBalanced(TreeNode* root) 
    {
        bool balanced = true;
        dfs(root, balanced);
        return balanced;
    }

    int dfs(TreeNode* root, bool& balanced) 
    {
        if (root == nullptr ) 
	{
            return 0;
        }

        int left = dfs(root->left, balanced);
        int right = dfs(root->right, balanced);

        if (abs(left - right) > 1) 
	{
            balanced = false;
        }

        return 1 + max(left, right);
    }
};
