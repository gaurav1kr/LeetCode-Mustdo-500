#include<chrono>
#include<thread>
#include<iostream>
using namespace std;

void run(int count)
{
    while (count-- > 0)
    {
        cout << "count= " << count << "\n";
        cout << "gaurav" << "\n";
    }
   
    cout << "thread finished";
}

int main()
{
    thread t1(run, 10);
    cout << "Before main\n";
    t1.detach(); // It will immediately detach t1 but function run would be executed as we are adding some wait time here

    //If we will detach it again , it will abort the program. So like before join, we can add 'joinable' check before detaching the thread.
    if(t1.joinable())
        t1.detach();
    this_thread::sleep_for(chrono::seconds(5));
    cout << "After main\n";

}

//Final output -
//Before main
//After main
