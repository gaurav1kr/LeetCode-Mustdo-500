class Solution 
{
public:
    bool isValidBST(TreeNode* root) 
    {
        if (!root) 
            return true;

        stack<TreeNode*> stk;
        TreeNode* curr = root;
        long prev = LONG_MIN; // Use LONG_MIN to handle edge cases with INT_MIN

        while (curr || !stk.empty()) 
        {
            while (curr) 
            {
                stk.push(curr);
                curr = curr->left;
            }

            curr = stk.top();
            stk.pop();

            if (curr->val <= prev) 
                return false;

            prev = curr->val;
            curr = curr->right;
        }

        return true;
    }
};
