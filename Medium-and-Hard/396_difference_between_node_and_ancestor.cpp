class Solution 
{
		int fun(TreeNode* root, int a, int b)
		{
			if(root==NULL) return abs(a-b);

			a=max(a, root->val);
			b=min(b, root->val);

			int l=fun(root->left, a, b);
			int r=fun(root->right, a, b);

			return max(l,r);
		}
    public:
		int maxAncestorDiff(TreeNode* root) 
		{
			return fun(root, INT_MIN, INT_MAX);
		}
};