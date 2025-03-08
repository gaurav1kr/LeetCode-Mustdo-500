class Solution 
{
public:
    void swap(TreeNode *root)
    {
        TreeNode *temp_node = root->left ;
        root->left = root->right ;
        root->right = temp_node ;
    }
    TreeNode* invertTree(TreeNode* root) 
    {
       if(!root)
           return NULL ;
       queue<TreeNode *> tree_queue ;
       tree_queue.push(root) ;
       while(!tree_queue.empty())
       {
            TreeNode *front_node = tree_queue.front() ;
            if(front_node)
            {
                tree_queue.push(front_node->left) ;
                tree_queue.push(front_node->right) ;
                swap(front_node) ;
            }
           tree_queue.pop() ;
       }
       return root ;
    }
};

