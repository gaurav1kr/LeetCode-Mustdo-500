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
/* Solution Logic
if root == null,
return pair(0, null)

if left depth == right depth,
deepest nodes both in the left and right subtree,
return pair (left.depth + 1, root)

if left depth > right depth,
deepest nodes only in the left subtree,
return pair (left.depth + 1, left subtree)

if left depth < right depth,
deepest nodes only in the right subtree,
return pair (right.depth + 1, right subtree)
*/

class Solution 
{
public:
    pair<int, TreeNode*> deep(TreeNode* root)
    {
        if (!root) return { 0, NULL };
        pair<int, TreeNode*> l = deep(root->left);
        pair<int, TreeNode*> r = deep(root->right);

        int d1 = l.first;
        int d2 = r.first;
        
        int maxDepth = max(d1, d2) + 1;
        TreeNode* secondNode;
        if (d1 == d2) 
        {
            secondNode = root;
        }
        else if (d1 > d2) 
        {
            secondNode = l.second;
        }
        else 
        {
            secondNode = r.second;
        }
        return { maxDepth, secondNode };
    }

    TreeNode* subtreeWithAllDeepest(TreeNode* root) 
    {
        return deep(root).second;
    }
};
