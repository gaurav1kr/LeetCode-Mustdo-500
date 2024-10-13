#include<iostream>
using namespace std;

#include "CarFactory.cpp"

int main()
{
#ifdef SIMPLE_CAR
	CarFactory* factory = new SimpleCarFactory;
#elif LUXURY_CAR
	CarFactory* factory = new LuxuryCarFactory;
#endif
	Car* car = factory->buildWholeCar();
	car->printDetails();
	return 0;
}
