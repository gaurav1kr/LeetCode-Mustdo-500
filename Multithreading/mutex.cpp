#include<iostream>
#include<thread>
#include<mutex>
#include<chrono>
using namespace std;
using namespace std::chrono;

int amount = 0;
mutex m;
void increment_amount()
{
	m.lock();
	amount++;
	this_thread::sleep_for(chrono::seconds(2));
	amount++;
	m.unlock();
}
int main()
{
	auto start_time = high_resolution_clock::now();
	thread t1(increment_amount);
	thread t2(increment_amount);
	thread t3(increment_amount);
	

	t1.join();
	t2.join();
	t3.join();
	 
	auto end_time = high_resolution_clock::now();
	auto time_taken = duration_cast<microseconds>(end_time - start_time);
	cout << "Execution time = " << time_taken.count() / 1000000 <<"\n";

	cout << amount;
	return 0;
}

//Mutex we use to avoid race condition (a condittion where 2 or more threads are trying to access the same piece of section at the same time. This section is referred to be 'Critical section'.
// If we remove lock and unlock from line no 12 and 16 , the result can be sometime 5 and sometime 6, althought the correct value is 6.
// Mutex locking and unlocking ensures that each thread will complete the execution and then will give control to next thread and the result will always be '6'.
// The above code will execute in 6 seconds as each thread is locking the thread for 2 seconds and releasing it. If we will remove the locking and unlocking , It till take 2 seconds as each theread will parallely run the function.
//line no 13 to 15 is our 'critical section' which threads are trying to access. So both threads will try to access it at the same time but one of threads will access it and lock it. Othher threads will be be
// in blocked state till the first thread will unlock it.
