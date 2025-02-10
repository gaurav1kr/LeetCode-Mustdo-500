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
    int sum ;
public:
    Solution()
    {
        sum = 0 ;
    }
    int countNodes(TreeNode* root) 
    {
        Traverse(root) ;
        return sum ;
    }
    void Traverse(TreeNode *root)
    {
        if(root)
        {
            Traverse(root->left) ;
            sum++;
            Traverse(root->right) ;
        }
    }
};
