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

void PostOrderTraversalRecursive(TreeNode *root)
{
	if(root)	
	{
		PostOrderTraversalRecursive(root->left) ;
		PostOrderTraversalRecursive(root->right) ;
		cout<<root->data<<"\t" ;
	}
}
void PostOrderTraversalIterative(TreeNode *root)
{
	stack<TreeNode *> todo ;
        TreeNode* last = NULL;
        while (root || !todo.empty()) 
	{
            if (root) 
	    {
                todo.push(root);
                root = root -> left;
            } 
	    else 
	    {
                TreeNode* node = todo.top();
                if (node -> right && last != node -> right) 
		{
                    root = node -> right;
                } 
		else 
		{
                    cout<<node->data<<"\t";
                    last = node;
                    todo.pop();
                }
            }
        }	
}

int main()
{
	TreeNode *root = new TreeNode(3) ;
	root->left = new TreeNode(9) ;
	root->right = new TreeNode(20) ;
	root->right->left = new TreeNode(15) ;
	root->right->right = new TreeNode(7) ;
	PostOrderTraversalRecursive(root) ;	
	cout<<"\n" ;
	PostOrderTraversalIterative(root) ;	
	return 0 ;
}
