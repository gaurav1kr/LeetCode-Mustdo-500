//thread creation ways - 
// Function pointer way
// Lambda expression way
// functor way
// Non static function way
// static function way
#include<thread>
#include<iostream>
using namespace std;

//This will be used for thread creation via functor way
class Functor
{
public:
	void operator ()(int x)
	{
		while (x-- > 0)
		{
			cout << x << "\n";
		}
	}
};

//This will be used for thread creation via non-static function way
class BaseNonstatic
{
public:
	void fun(int x)
	{
		while (x-- > 0)
		{
			cout << x << "\n";
		}
	}
};

//This will be used for thread creation via static way
class Basestatic
{
public:
	static void fun(int x)
	{
		while (x-- > 0)
		{
			cout << x << "\n";
		}
	}
};

void fun(int x)
{
	while (x-- > 0)
	{
		cout << x << "\n";
	}
}

/*Function pointer way
int main()
{
	thread t1(fun, 20); //function pointer way
	t1.join();
	return 0; 
}*/

/*Lambda expression way
int main()
{
	thread t1([](int x) {
				while (x-- > 0)
				{
					cout << x << "\n";
				}
			}, 20); //function pointer way
	t1.join();
	return 0;
}*/

/*Functor way
int main()
{
	thread t1(Functor(), 20);
	t1.join();
	return 0;
}*/

/*Non static way
int main()
{
	BaseNonstatic b;
	thread t1(&Base::fun, &b, 20);
	t1.join();
	return 0;
}*/

//static way
int main()
{
	thread t1(&Basestatic::fun, 20);
	t1.join();
	return 0;
	return 0;
}

