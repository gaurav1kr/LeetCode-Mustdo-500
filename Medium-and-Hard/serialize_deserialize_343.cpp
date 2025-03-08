/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec 
{
	int i ;
	
public:

    string serialize(TreeNode* root) 
	{
        if(!root) return "null,";
        return to_string(root->val)+ "," +serialize(root->left)  +serialize(root->right);        
    }

    TreeNode* deserialize(string data) 
	{
        if(i>=data.size())return nullptr;

        string t="";
        while(data[i]!=',') t+=data[i++];
        i++;

        if(t=="null") return nullptr;
        TreeNode* root=new TreeNode(stoi(t));
        root->left=deserialize(data);
        root->right=deserialize(data);
        return root;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec* ser = new Codec();
// Codec* deser = new Codec();
// string tree = ser->serialize(root);
// TreeNode* ans = deser->deserialize(tree);
// return ans;