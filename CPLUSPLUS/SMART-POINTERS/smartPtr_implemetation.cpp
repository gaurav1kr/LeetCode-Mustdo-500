//smart poTer wraps a normal poTer in such a way that we don't need to take care for the deletion of the memory
//which has been allocated dynamically to it.
//To implement smart poTers, we need an overloaded member access operator and an explicit constructor.
//A normal implementation is below :- 
#include<iostream>
using namespace std ;
template<class T>
class MySmart
{
	T *data ;
	public:
	explicit MySmart(T *p = nullptr):data(p){}
	T & operator * () 
	{
		return *data;
	}
	~MySmart()
	{
		cout<<*data<<" deleted" ;
		delete data ;
	}
};
int main()
{
	MySmart<string> mt(new string()) ;
	*mt = "gaurav" ;
	cout<<*mt<<" added\n";
	return 0 ;
}
