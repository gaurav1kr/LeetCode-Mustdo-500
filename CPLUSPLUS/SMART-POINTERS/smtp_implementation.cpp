//smart pointer wraps a normal pointer in such a way that we don't need to take care for the deletion of the memory
//which has been allocated dynamically to it.
//To implement smart pointers, we need an overloaded member access operator and an explicit constructor.
//A normal implementation is below :- 
#include<iostream>
using namespace std ;
class MyInt
{
	int *data ;
	public:
	explicit MyInt(int *p = nullptr):data(p){}
	int & operator * () 
	{
		return *data;
	}
	~MyInt()
	{
		delete data ;
		cout<<"Element deleted" ;
	}
};
int main()
{
	int *p = new int(10) ;
	MyInt mt(p) ;
	cout<<*mt ;
	return 0 ;
}
