#include<iostream>
#include<thread>
#include<mutex>

#define MAX 100000
using namespace std;
using namespace std::chrono;

int amount = 0;
mutex m;
void increment_amount()
{
	for (int i = 0; i < MAX; i++)
	{
		if (m.try_lock())
		{
			++amount;
			m.unlock();
		}
	}
}
int main()
{
	thread t1(increment_amount);
	thread t2(increment_amount);
	

	t1.join();
	t2.join();

	cout << amount;
	return 0;
}

// try_lock tries to lock the mutex. It will return immediately. On successful lock aquisition , it returnns true. Otherwise
// it will return false and control can be taken by another thread which is trying to access the same critical section.
// In the above example , both t1 and t2 are trying to accesss 'amount'. As soon as t1 will fail to acquire lock, t2 can take
//contol and will try to increment it on the same value of 'i' where t1 left. so complete count '100000' will not be completed and 
//thus valuf of amount at line no. '31' will always be less than '200000'

// If we want to get the correct amount , we can apply simple m.lock() and m.unlock().

// we generally use try_lock instead of lock in C++ when we want to attempt locking a mutex without blocking the current thread
// if the mutex is already locked. This will be useful in situations where we don't want the program to wait indefinitely for the mutex
// to become available, allowing the thread to either perform alternative task of skip certain operations. 

// A practical scenario will be implementation of a logging system where multiple threads can write to a shared log file.
// To ensure thread safety, you use a mutex to control access to the log. However,
//  you don't want the logging operation to block critical tasks if the log file is currently being written to by another thread. 
// In such a case, you can use try_lock to attempt to lock the mutex, and if it's already locked, 
// the thread can proceed with other tasks instead of waiting.
