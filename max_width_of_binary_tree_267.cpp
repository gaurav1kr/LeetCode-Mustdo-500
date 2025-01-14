/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
#define it unsigned long long 
class Solution 
{
public:
    int widthOfBinaryTree(TreeNode* root) 
	{
        if(root == NULL)
			return 0;
        int ans=INT_MIN; 
        queue<pair<TreeNode* ,it>>q;
        q.push({root,1}); 
        while(!q.empty())
		{ 
            int size=q.size();
            it s,e;
            for(int i=0;i<size;i++)
			{ 
                auto [node,index] = q.front();
                q.pop();
                if(i==0)
				{
					s=index;
				}
                if(i==size-1)
				{
					e=index;
				}
                if(node->left)
				{
                    q.push({node->left,2*index});
                }
                if(node->right)
				{
                    q.push({node->right,2*index+1});
                }
            }
            int width=e-s+1;
            ans=max(ans,width); 
        }
        return ans;  
    }
};