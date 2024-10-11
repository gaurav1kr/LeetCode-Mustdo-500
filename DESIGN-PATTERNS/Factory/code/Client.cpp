//Implementation of FDP Demo
#include<iostream>
#include "ToyFactory.cpp"
using namespace std ;
int main()
{
	int type ;
	while(1)	
	{
		cout<<"Enter the type" ;
		cin>> type ;
		if(!type)
		{
			break ;
		}
		Toy *v = ToyFactory::createToy(type) ;
		if(v)
		{
			v->showProduct() ;
			delete v ;
		}
	}
	cout<<"Exit.."<<endl ;
	return 0 ;
}
