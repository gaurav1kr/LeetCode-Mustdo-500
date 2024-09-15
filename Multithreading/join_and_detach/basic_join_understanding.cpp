#include<iostream>
#include<chrono>
#include<thread>
using namespace std;

void run(int count)
{
	while (count-- > 0)
	{
		cout << "gaurav"<<"\n";
	}
}

int main()
{
	thread t1(run, 10);
	//If we don'join the thread --> when t1's destructor will be called and it will check whether t1 is joinable. If yes, then it will abort.
	//If we will add join , t1's destructor will be called and since already it is joined , so it will pass
	t1.join(); 
	return 0;
}
