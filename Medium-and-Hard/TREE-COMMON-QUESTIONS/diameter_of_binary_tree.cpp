class Solution 
{
public:
    int res=0;
    
    int diameter_util(TreeNode* root)
	{
        if(!root)
            return 0;
        int l= diameter_util(root->left);
        int r= diameter_util(root->right);
        int temp=max(l,r)+1;
        res=max(res,(l+r));
        return temp;
    }
    
    
    int diameterOfBinaryTree(TreeNode* root) 
	{
       if(!root)
             return 0;
        int l= diameterOfBinaryTree(root->left);
        int d= diameterOfBinaryTree(root->right);
        int a=diameter_util(root);
        return res;
    }
};
