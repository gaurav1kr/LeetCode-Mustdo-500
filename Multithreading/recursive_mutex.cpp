#include<iostream>
#include<thread>
#include<mutex>
using namespace std ;

recursive_mutex m1 ;

int buffer = 0 ;

void recursion(char c , int loopFor)
{
	if(loopFor<0)
		return ;
	
	m1.lock() ;
	cout<<"ThreadID:" <<c<<" "<<buffer++ << "\n" ;
	recursion(c , --loopFor) ;
	m1.unlock() ;
	cout<<"Unlock by thread "<< c << endl ;
}

int main()
{
	thread t1(recursion, '0',10) ;
	thread t2(recursion , '1' , 10) ;
	t1.join() ;
	t2.join() ;
	return 0 ;
}


// Synopsys :- 

// A std::mutex is a non-recursive mutex, meaning a thread can only lock it once. 
// If a thread tries to lock the same std::mutex again without unlocking it first, 
// it will result in a deadlock.

// Usage:
// std::mutex is suitable for simple locking scenarios where a thread acquires a lock and releases it
// before trying to lock it again.

// A std::recursive_mutex allows a thread to lock the same mutex multiple times without deadlocking itself, 
// as long as it unlocks it the same number of times.
// Each lock operation by the same thread increases the lock count, and each unlock operation decreases the
// lock count. The mutex is only released when the lock count reaches zero.

// Consider one more scenario where a class has 2 functions where one function is calling other
// and both the functions are locking same mutex then we can't do it with normal mutex.

// Consider below program :- 

#include <iostream>
#include <thread>
#include <mutex>

class MyClass 
{
	public:
		void methodA() 
		{
			mtx.lock();
			std::cout << "methodA acquired lock" << std::endl;
			methodB();  // Calls methodB, which also needs to acquire the lock
			mtx.unlock();
		}

		void methodB()
		{
			mtx.lock();
			std::cout << "methodB acquired lock" << std::endl;
			// Do some work
			mtx.unlock();
		}

	private:
		std::mutex mtx;
};

int main() 
{
    MyClass obj;
    std::thread t1(&MyClass::methodA, &obj);
    t1.join();
    return 0;
}



// This will stuck at below point as a deadlock scenario has been created here :-
// gakumar@DESKTOP-CTCEABV:~/LeetCode-Mustdo-500/Multithreading$ ./a.out
// methodA acquired lock


// Now we can change this program and add recursive_mutex rather than mutex and can see the change :- 

#include <iostream>
#include <thread>
#include <mutex>

class MyClass 
{
	public:
		void methodA() 
		{
			recursive_mtx.lock();
			std::cout << "methodA acquired lock" << std::endl;
			methodB();  // Calls methodB, which also needs to acquire the lock
			recursive_mtx.unlock();
		}

		void methodB()
		{
			recursive_mtx.lock();
			std::cout << "methodB acquired lock" << std::endl;
			// Do some work
			recursive_mtx.unlock();
		}

	private:
		std::recursive_mutex recursive_mtx ;
};

int main() 
{
    MyClass obj;
    std::thread t1(&MyClass::methodA, &obj);
    t1.join();
    return 0;
}

// gakumar@DESKTOP-CTCEABV:~/LeetCode-Mustdo-500/Multithreading$ ./a.out
// methodA acquired lock
// methodB acquired lock
