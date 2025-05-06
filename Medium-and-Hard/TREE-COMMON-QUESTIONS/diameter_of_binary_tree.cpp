class Solution 
{
public:
    int res = 0;  // To store the maximum diameter

    int diameter_util(TreeNode* root) 
    {
        if (!root)
            return 0;

        int leftHeight = diameter_util(root->left);   // height of left subtree
        int rightHeight = diameter_util(root->right); // height of right subtree

        // Diameter passing through current node is sum of left and right subtree heights
        res = max(res, leftHeight + rightHeight);

        // Return height of current subtree
        return max(leftHeight, rightHeight) + 1;
    }

    int diameterOfBinaryTree(TreeNode* root) 
    {
        res = 0;                     // Initialize result
        diameter_util(root);        // Compute diameter
        return res;                 // Return the result
    }
};

