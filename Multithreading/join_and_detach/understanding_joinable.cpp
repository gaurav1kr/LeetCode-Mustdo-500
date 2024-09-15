#include<chrono>
#include<thread>
#include<iostream>
using namespace std;

void run(int count)
{
    while (count-- > 0)
    {
        cout << "gaurav" << "\n";
    }
    this_thread::sleep_for(chrono::seconds(3));
}

int main()
{
    thread t1(run, 10);
    cout << "main before thread join\n";
    if (t1.joinable()) // This will be true since t1 is joinable as first time we are going to use it.
    {
        cout << "Thread is joinable\n";
        t1.join();
    }
   // t1.join(); //It will abort the program as when we call join. It checks whether the thread is joinable or not. If it is not joinable and we are calling join, the program will abort.

    //To avoid the situation which occurred in line no 24, we can check whether it is joinable
    if (t1.joinable())
    {
        t1.join();
    }
    else
    {
        cout << "Thread is not joinable now\n";
    }

    //The best practice is to check t1.joinable() before using t1.join()
    cout << "main after thread join\n";
    return 0;
}
