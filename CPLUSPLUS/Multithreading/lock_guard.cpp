//lock_guard usage
#include<iostream>
#include<mutex>
#include<thread>

using namespace std ;

mutex m ;
int buffer = 0 ;

void fun(const char *thread_num , int loop)
{
	lock_guard<mutex> lock(m) ;
	for(int i=0;i<loop;i++)
	{
		buffer++;
		cout<<thread_num<<" "<<buffer<<"\n" ;
	}
}

int main()
{
	thread t1(fun , "t0" , 10) ;
	thread t2(fun , "t1" , 10) ;
	t1.join() ;
	t2.join() ;
	return 0 ;
}

// Facts about lock_guard :- 
// 1. It is very light weight wrapper for owning mutex on scoped basis
// 2. It acquires mutex lock the moment we create the object of lock_guard
// 3. It automatically removes the lock while goes out of scope
// 4. we cannot explicitly unlock the lock_guard
// 5. we can't copy lock_guard.
