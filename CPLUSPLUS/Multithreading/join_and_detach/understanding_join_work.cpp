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
	this_thread::sleep_for(chrono::seconds(3));
}

int main()
{
	thread t1(run, 10);
	cout << "main before thread join\n"; // thread will start and control will come here.
	t1.join(); //here we are waiting for the thread t1 to be completed which will take 3 seconds as we mentioned in the underlying function
	cout << "main after thread join\n";
	return 0;
}
