class Solution 
{
public:
    int findBottomLeftValue(TreeNode* root) 
    {
        TreeNode* last=nullptr;
        queue<TreeNode*>q;
        if (root)
            q.push(root);
        while (!q.empty()) 
        {
            int size=q.size();
            for (int i=0; i<size;i++) 
	    {
                TreeNode* node=q.front();
                q.pop();
                if (i==0) 
                    last=node;
                if (node->left)
                    q.push(node->left);
                if (node->right)
                    q.push(node->right);
            }
        }
        return last->val;
    }
};
