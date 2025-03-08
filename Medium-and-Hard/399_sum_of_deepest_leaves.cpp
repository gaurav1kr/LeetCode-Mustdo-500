class Solution 
{
public:
    int deepestLeavesSum(TreeNode* root) 
	{
        int ans = 0;
        queue<TreeNode*> q;
        q.push(root);
        q.push(NULL);

        while(!q.empty())
		{
            
            int sum = 0;
            TreeNode* t = q.front();
            q.pop();
            while(t != NULL)
			{
                sum += t->val;
                if(t->left) q.push(t->left);
                if(t->right) q.push(t->right);
                t = q.front();
                q.pop();
            }
            if(!q.empty()) 
				q.push(NULL);
			
			ans = sum;
        }
        return ans;

    }
};

// TC :- O(N)
// SC :- O(N)
// Approach :- we will try to find out sum of nodes at every level and update the value to ans.
// Finally we will reach on the deepest level and get the sum.