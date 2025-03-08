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
    vector<int> v1 ;
public:
    bool isValidBST(TreeNode* root) 
    {
        if(!root)
            return true ;
        if(root && root->left == NULL && root->right == NULL)
            return true ;
        v1.push_back(INT_MIN) ;
        stack<TreeNode *> stk1 ;
        TreeNode *Curr = root ;
        while(Curr || !stk1.empty())
        {
            while(Curr)
            {
                stk1.push(Curr) ;
                Curr = Curr->left ;
            }
            Curr = stk1.top() ;
            stk1.pop() ;
            
            if(Curr->val <= v1.back() && v1.size() >1)
            {
                return false ;
            }
            v1.push_back(Curr->val) ;
            
            
            Curr = Curr->right ;
        }
        v1.clear() ;
        return true ;
    }
};

// Time complexiety :- o(n) - n is no of nodes in the binary tree
