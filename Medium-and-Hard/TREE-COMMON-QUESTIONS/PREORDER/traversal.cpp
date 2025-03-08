#include<iostream>
#include<stack>
using namespace std ;
struct TreeNode
{
	int data;
	TreeNode *left ;
	TreeNode *right ;
	TreeNode():data(0),left(nullptr),right(nullptr){}
	TreeNode(int val):data(val),left(nullptr),right(nullptr){}
	TreeNode(int val, TreeNode *left , TreeNode *right):data(val),left(left),right(right){}
};

void PreOrderTraversalRecursive(TreeNode *root)
{
	if(root)	
	{
		cout<<root->data<<"\t" ;
		PreOrderTraversalRecursive(root->left) ;
		PreOrderTraversalRecursive(root->right) ;
	}
}
void PreOrderTraversalIterative(TreeNode *root)
{
	stack<TreeNode*> stk;
	if(!root) return ;

	stk.push(root);
	while(!stk.empty())
	{
		TreeNode *n = stk.top() ;
		stk.pop() ;

		cout<<n->data<<"\t" ;

		if(n->right) stk.push(n->right) ;
		if(n->left) stk.push(n->left) ;
	}
}
int main()
{
	TreeNode *root = new TreeNode(3) ;
	root->left = new TreeNode(9) ;
	root->right = new TreeNode(20) ;
	root->right->left = new TreeNode(15) ;
	root->right->right = new TreeNode(7) ;
	PreOrderTraversalRecursive(root) ;	
	cout<<"\n" ;
	PreOrderTraversalIterative(root) ;	
	return 0 ;
}
