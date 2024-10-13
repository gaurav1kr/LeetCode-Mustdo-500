#include<iostream>
#include<memory>
using namespace std ;
class Foo
{
	private:
		int x ;
	public:
		explicit Foo(int x):x(x){}
		int getX() { return x ;}
};

int main()
{
	unique_ptr<Foo> p1(new Foo(10)) ; 
	unique_ptr<Foo> p6(new Foo(100)) ; 
	unique_ptr<Foo> p2 = make_unique<Foo>(20) ; //This will be compiled with C++14 only

	cout<<p1->getX()<<"\n" ;
	cout<<p2->getX()<<"\n" ;
	cout<<(*p2).getX()<<"\n" ;

	unique_ptr<Foo> p3 = move(p1) ;
	if(p1)
	{
		cout<<p1->getX() ;	
	}
	else
	{
		cout<<"Ownership of p1 has been transferred to p3" ;
	}

	Foo *p = p3.get() ;	
	Foo *p4 = p3.release() ;	
	return 0 ;
}
