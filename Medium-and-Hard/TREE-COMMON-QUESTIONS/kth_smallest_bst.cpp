#include<iostream>
using namespace std ;

struct TreeNode
{
public:
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode():val(0),left(nullptr),right(nullptr) {}
    TreeNode(int x):val(x),left(nullptr),right(nullptr) {}
    TreeNode(int x, TreeNode *l , TreeNode *r):val(x),left(l),right(r) {}
};

class Solution 
{
public:
        void solve(TreeNode* root, int &cnt, int &ans, int k)
	{
        	if(root == NULL)    
			return;
        //left, root, right 
        	solve(root->left, cnt, ans, k);
        	cnt++;
        	if(cnt == k)
		{
            		ans = root->val;
            		return;
        	}
        	solve(root->right, cnt, ans, k);
    	}

       int kthSmallest(TreeNode* root, int k) 
       {
        
        int cnt = 0;        
        int ans;
        solve(root, cnt, ans, k);
        return ans;
       }
};

int main() {

    // Construct a BST
    //            20
    //          /    \
    //         8      22
    //       /   \
    //      4    12
    //          /  \
    //         10   14
    TreeNode *root = new TreeNode(20);
    root->left = new TreeNode(8);
    root->right = new TreeNode(22);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(12);
    root->left->right->left = new TreeNode(10);
    root->left->right->right = new TreeNode(14);

    int k =3 ;
    Solution sol ;
    cout<<sol.kthSmallest(root,k) ;
    return 0 ;
}
