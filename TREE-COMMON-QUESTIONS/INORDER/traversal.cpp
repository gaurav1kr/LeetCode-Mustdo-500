#include<iostream>
#include<stack>
using namespace std ;
struct TreeNode
{
	int data;
	TreeNode *left ;
	TreeNode *right ;
	TreeNode():data(0),left(nullptr),right(nullptr){}
	TreeNode(int data):data(data),left(nullptr),right(nullptr){}
	TreeNode(int data, TreeNode *left , TreeNode *right):data(data),left(left),right(right){}
};

void InOrderTraversalRecursive(TreeNode *root)
{
	if(root)	
	{
		InOrderTraversalRecursive(root->left) ;
		cout<<root->data<<"\t" ;
		InOrderTraversalRecursive(root->right) ;
	}
}
void InOrderTraversalIterative(TreeNode *root)
{
        stack<TreeNode*> todo;
        while (root || !todo.empty()) 
	{
            while (root) 
	    {
                todo.push(root);
                root = root -> left;
            }
            root = todo.top();
            todo.pop();
            cout<<root->data<<"\t";
            root = root -> right;
    	}
}

int main()
{
	TreeNode *root = new TreeNode(3) ;
	root->left = new TreeNode(9) ;
	root->right = new TreeNode(20) ;
	root->right->left = new TreeNode(15) ;
	root->right->right = new TreeNode(7) ;
	InOrderTraversalRecursive(root) ;	
	cout<<"\n" ;
	InOrderTraversalIterative(root) ;	
	return 0 ;
}
