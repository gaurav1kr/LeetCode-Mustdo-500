string inorder(TreeNode* root, unordered_map<string, int>& mp, vector<TreeNode*>& res)                
{
        if(!root)
         return "";

        string str = "(";
        str += inorder(root -> left, mp, res);
        str += to_string(root -> val);
        str += inorder(root -> right, mp, res);
        str += ")";

        if(mp[str] == 1)
            res.push_back(root);
    
        mp[str]++;

        return str;
}
    
vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) 
{
	unordered_map<string, int> mp;
	vector<TreeNode*> res;
	inorder(root, mp, res);
	return res;
}