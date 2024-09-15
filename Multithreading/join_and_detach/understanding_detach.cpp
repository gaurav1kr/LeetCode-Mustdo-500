nclude<chrono>
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
    this_thread::sleep_for(chrono::seconds(5));
    cout << "thread finished";
}

int main()
{
    thread t1(run, 10);
    cout << "Before main\n";
    t1.detach(); // It will immediately detach t1 and there is no wait also so 'run' won't be executed.
    cout << "After main\n";

}

//Final output -
//Before main
//After main
