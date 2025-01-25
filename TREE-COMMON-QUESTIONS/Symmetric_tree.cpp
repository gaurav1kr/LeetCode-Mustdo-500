class Solution 
{
public:
    bool isSymmetricUtil(TreeNode *p , TreeNode *q)
    {
        if(!p && !q)
            return true ;

        if(!p || !q)
            return false;

        return( (p->val == q->val) && isSymmetricUtil(p->left , q->right) && isSymmetricUtil(p->right , q->left) ) ;
    }

    bool isSymmetric(TreeNode* root) 
    {
        return isSymmetricUtil(root->left , root->right) ;
    }
};
