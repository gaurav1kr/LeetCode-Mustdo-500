class Solution 
{
public:
    TreeNode* buildTree(vector<int>&nums, int start, int end)
	{
        if(start == end)
		{
            return NULL;
        }
        int maxIndex = start;
        for(int i = start + 1; i < end; i++)
		{
            if(nums[i] > nums[maxIndex])
			{
                maxIndex = i;
            }
        }
        TreeNode* root = new TreeNode(nums[maxIndex]);
        root->left = buildTree(nums, start, maxIndex);
        root->right = buildTree(nums, maxIndex + 1, end);
        return root;
    }
    
	TreeNode* constructMaximumBinaryTree(vector<int>& nums)
	{
        return buildTree(nums, 0, nums.size());
    }
};