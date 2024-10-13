#include<iostream>
#include<chrono>
#include<thread>

using namespace std;
using namespace std::chrono;
typedef unsigned long long ull;
#define MAX 1900000000

ull getOddSum(ull start, ull end)
{
	ull sum = 0;
	for (ull i = start; i <= end; i++)
	{
		if (i & 1)
		{
			sum += i;
		}
	}
	return sum;
}

ull getEvenSum(ull start, ull end)
{
	ull sum = 0;
	for (ull i = start; i <= end; i++)
	{
		if (!(i & 1))
		{
			sum += i;
		}
	}
	return sum;
}
/* using single thread - it takes 8 seconds
int main()
{
	auto start_time = high_resolution_clock::now();

	ull oddsum = getOddSum(1, MAX);
	ull evensum = getEvenSum(1, MAX);

	auto stop_time = high_resolution_clock::now();

	cout << "Evensum= " << evensum << "\t" << "Oddsum= " << oddsum << "\n";

	auto time_taken = duration_cast<microseconds>(stop_time - start_time);
	cout << "Time taken = " << time_taken.count() / 1000000 << " seconds\n";
	return 0;
}*/

// using multi-thr4eading - it takes 3 seconds as we are opening 2 separate threads which will run in parallel
int main()
{
	auto start_time = high_resolution_clock::now();
	thread t1(getOddSum, 1, MAX);
	thread t2(getEvenSum, 1, MAX);
	t1.join();
	t2.join();
	auto end_time = high_resolution_clock::now();

	auto time_taken = duration_cast<microseconds>(end_time - start_time);

	cout << "Time taken = " << time_taken.count() / 1000000 << " Seconds\n";
	return 0;
}


