class Solution 
{
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) 
    {
        int n = nums.size();
        return solve(nums,0, n-1);
    }
    TreeNode* solve(vector<int>& nums, int l, int r)
    {
        if(l > r)
	{
            return NULL;
        }
        int mid  =  (l+r)/2;
        TreeNode* node = new TreeNode(nums[mid]);
        node->left = solve(nums, l, mid-1);
        node->right = solve(nums, mid+1, r);
        return node;
    }
};
