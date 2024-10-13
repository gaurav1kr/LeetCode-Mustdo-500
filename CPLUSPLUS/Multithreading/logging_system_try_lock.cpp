#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <string>

std::mutex logMutex;

void logMessage(const std::string& message)
{
    // Try to acquire the mutex without blocking
    if (logMutex.try_lock())
    {
        // If successful, log the message
        std::cout << message << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate time spent logging
        logMutex.unlock();
    }
    else 
    {
        // If the mutex is already locked, skip logging or handle differently
        std::cout << "Log is busy, skipping message: " << message << std::endl;
    }
}

void worker(int id) 
{
    for (int i = 0; i < 5; ++i) 
    {
        logMessage("Thread " + std::to_string(id) + " message " + std::to_string(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
    }
}

int main() {
    std::thread t1(worker, 1);
    std::thread t2(worker, 2);

    t1.join();
    t2.join();

    return 0;
}

//Output - 
//Log is busy, skipping message : Thread 2 message 0Thread 1 message 0
//
//Thread 1 message 1
//Thread 2 message 1
//Thread 1 message 2
//Thread 2 message 2
//Thread 1 message 3
//Thread 2 message 3
//Thread 1 message 4
//Thread 2 message 4
