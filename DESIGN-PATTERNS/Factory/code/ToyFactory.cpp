//ToyFactory class implementation
#include "Object.cpp"
#include<iostream>
using namespace std ;

class ToyFactory
{
	public:
		static Toy *createToy(int type)
		{
			Toy *toy = NULL ;
			switch(type)
			{
				case 1:
					toy = new Car ;
					break;
				case 2:
					toy = new Bike ;
					break;
				case 3:
					toy = new Plane ;
					break;
				default:
					cout<<"Invalid toy. Please re-enter type"<<endl ;
					return NULL ;
			}
			toy->prepareParts() ;
			toy->combineParts() ;
			toy->assembleParts() ;
			return toy ;
		}
};
