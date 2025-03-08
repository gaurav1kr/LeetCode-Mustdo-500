#include<iostream>

struct Node
{
	int data ;
	Node *left ;
	Node *right ;

	Node(int value):data(value) , left(nullptr) , right(nullptr){}
};

class BST
{
	public:
	Node *root;
	BST():root(nullptr){}

	void insert(int value){
		insertRec(root , int);
	}

	private:
	Noide *insertRec(Node *node , int value)
	{
		if(node == nullptr)
		{
			return new Node(value) ;
		}
		if(value > node->data)
		{
			node->left = insertRec(node->left , value); 
		}
	}
}


