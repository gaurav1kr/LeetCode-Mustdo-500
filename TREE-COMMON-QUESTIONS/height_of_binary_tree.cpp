#include<iostream>
using namespace std ;

struct Tree
{
	int data ;
	Tree *left ;
	Tree *right;
	
	Tree():left(nullptr),right(nullptr),data(0){}
	
	Tree(int d):left(nullptr),right(nullptr),data(d){}
	
	Tree(int d , Tree *l , Tree *r):left(l),right(r),data(d){}
	
};

int height(Tree *root)
{
	if(!root)
		return 0 ;
	int lheight = height(root->left) ;
	int rheight = height (root->right) ;
	if(lheight > rheight) return (lheight+1) ;
	
	return (rheight+1) ;
}

int main()
{
	Tree *root = new Tree(3) ;
	root->left = new Tree(9) ;
	root->right = new Tree(20) ;
	root->right->left = new Tree(15) ;
	root->right->right = new Tree(7) ;
	cout<<height(root) ;
	return 0 ;
}
