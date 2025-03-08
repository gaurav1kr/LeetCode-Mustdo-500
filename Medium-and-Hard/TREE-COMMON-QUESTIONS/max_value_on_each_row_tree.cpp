class Solution 
{
	public:
    		vector<int> largestValues(TreeNode* root) 
		{
    			vector<int> result;
    			if (!root) 
			{
        			return result;
    			}
    			queue<TreeNode*> current;
    			current.push(root);
    			while (!current.empty()) 
			{
        			int levelMax = INT_MIN;
        			int levelSize = current.size();
        			for (int i = 0; i < levelSize; i++) 
				{
            				TreeNode* node = current.front();
            				current.pop();
            				levelMax = max(levelMax, node->val);
            				if (node->left) 
					{
                				current.push(node->left);
            				}
            				if (node->right) 
					{
                				current.push(node->right);
            				}
        			}
        			result.push_back(levelMax);
    			}
    			return result;
		}
};
