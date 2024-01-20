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
    int sum;
public:
    Solution()
    {
        sum = 0;
    }
    int sumRootToLeaf(TreeNode* root)
    {
        sumPaths(root);
        return sum;
    }

    void sumPaths(TreeNode* node)
    {
        int path[1000];
        sumPathsRecur(node, path, 0);
    }

    void sumPathsRecur(TreeNode* node, int path[], int pathLen)
    {
        if (node == NULL)
            return;

        path[pathLen] = node->val;
        pathLen++;

        if (node->left == NULL && node->right == NULL)
        {
            sumArray(path, pathLen);
        }
        else
        {
            sumPathsRecur(node->left, path, pathLen);
            sumPathsRecur(node->right, path, pathLen);
        }
    }

    void sumArray(int ints[], int len)
    {
        int i;
        int power = len - 1;
        for (i = 0; i < len; i++)
        {
            if (ints[i])
            {
                sum = sum + pow(2, power);
            }
            power--;
        }
    }
};
