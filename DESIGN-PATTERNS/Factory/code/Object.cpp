#include<iostream>
using namespace std ;
class Toy
{
	protected:
		string name ;
		float price ;
	public:
		virtual void prepareParts() = 0 ;
		virtual void combineParts() = 0 ;
		virtual void assembleParts() = 0 ;
		virtual void showProduct() = 0 ;
};

class Car : public Toy
{
	public:
		void prepareParts()
		{
			cout<<"Preparing Car parts\n" ;
		}
		void combineParts()
		{
			cout<<"Combining Car parts\n" ;
		}
		void assembleParts()
		{
			cout<<"Assembling Car parts\n" ;
		}
		void showProduct()
		{
			cout<<"Showing Car parts\n" ;
		}
};

class Bike : public Toy
{
	public:
		void prepareParts()
		{
			cout<<"Preparing Bike parts\n" ;
		}
		void combineParts()
		{
			cout<<"Combining Bike parts\n" ;
		}
		void assembleParts()
		{
			cout<<"Assembling Bike parts\n" ;
		}
		void showProduct()
		{
			cout<<"Showing Bike parts\n" ;
		}
};

class Plane : public Toy
{
	public:
		void prepareParts()
		{
			cout<<"Preparing Plane parts\n" ;
		}
		void combineParts()
		{
			cout<<"Combining Plane parts\n" ;
		}
		void assembleParts()
		{
			cout<<"Assembling Plane parts\n" ;
		}
		void showProduct()
		{
			cout<<"Showing Plane parts\n" ;
		}
};
