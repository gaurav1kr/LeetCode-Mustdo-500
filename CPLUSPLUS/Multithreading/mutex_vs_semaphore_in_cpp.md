
# Mutex vs Semaphore in C++

In concurrent programming, **mutexes** and **semaphores** are synchronization primitives used to control access to shared resources by multiple threads or processes. Although they serve similar purposes, they differ significantly in terms of their behavior, use cases, and underlying mechanisms. Here's a detailed distinction with examples in C++:

## Mutex (Mutual Exclusion Object)
- A **mutex** is a locking mechanism used to synchronize access to a resource. It is primarily used when only one thread needs access to a shared resource at any given time.
- Mutexes are typically **binary**, meaning they can either be locked or unlocked (like a binary semaphore).
- Once a thread locks a mutex, no other thread can access the shared resource until the mutex is unlocked by the thread that locked it. This ensures **exclusive access** to the resource.

### Key Points:
- A mutex is **locked** by a thread and **unlocked** by the same thread.
- Only one thread can hold the mutex at a time.
- If another thread attempts to lock the mutex while it is already locked, that thread will be blocked (i.e., it will wait until the mutex becomes available).

### C++ Example of Mutex:
In C++, mutexes are provided in the `<mutex>` header, and their usage typically looks like this:

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx; // global mutex

void print_function(int thread_id) {
    // Locking the mutex before accessing the shared resource
    mtx.lock();
    std::cout << "Thread " << thread_id << " is accessing the shared resource\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Simulate work
    // Unlocking the mutex after accessing the shared resource
    mtx.unlock();
}

int main() {
    std::thread t1(print_function, 1);
    std::thread t2(print_function, 2);

    t1.join();
    t2.join();

    return 0;
}
```

In this example, the mutex (`mtx`) ensures that only one thread at a time can access the shared resource (in this case, printing to the console). Without the mutex, both threads might interleave their output, leading to mixed or unpredictable results.

---

## Semaphore
A **semaphore** is more flexible than a mutex. It can be thought of as a counter that regulates access to a shared resource. Semaphores are of two types:
1. **Binary Semaphore**: It behaves similarly to a mutex, where the semaphore can only take the value `0` or `1` (locked or unlocked).
2. **Counting Semaphore**: It allows more than one thread to access the shared resource simultaneously. The semaphore has a count that indicates how many threads can access the resource at the same time.

### Key Points:
- A semaphore **allows multiple threads** to access the shared resource, depending on its count.
- A thread **waits** on the semaphore (i.e., decrements the semaphore count) before entering the critical section, and **signals** the semaphore (i.e., increments the count) after exiting.
- A **counting semaphore** is typically used when a resource has multiple units that can be used by more than one thread simultaneously.
- Unlike a mutex, **any thread can signal the semaphore**, not necessarily the one that waited on it.

### C++ Example of Semaphore:
Semaphores are not directly provided in the C++ standard library, but we can implement them using the `<condition_variable>` and `<mutex>` headers. Here is an example of a counting semaphore:

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

class Semaphore {
public:
    Semaphore(int count = 1) : count(count) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return count > 0; });
        --count;
    }

    void signal() {
        std::unique_lock<std::mutex> lock(mtx);
        ++count;
        cv.notify_one();
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    int count;
};

Semaphore semaphore(3); // Allow up to 3 threads to access the resource at the same time

void access_shared_resource(int thread_id) {
    semaphore.wait(); // Wait for the semaphore to become available
    std::cout << "Thread " << thread_id << " is accessing the shared resource\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Simulate work
    semaphore.signal(); // Signal that the thread is done
}

int main() {
    std::thread t1(access_shared_resource, 1);
    std::thread t2(access_shared_resource, 2);
    std::thread t3(access_shared_resource, 3);
    std::thread t4(access_shared_resource, 4);
    std::thread t5(access_shared_resource, 5);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();

    return 0;
}
```

In this example, the `Semaphore` class regulates access to a shared resource with a count of 3, meaning up to three threads can access the resource simultaneously. Additional threads are blocked until the count increases (i.e., when another thread finishes and signals the semaphore).

---

## Key Differences Between Mutex and Semaphore:

| Aspect         | **Mutex**                                           | **Semaphore**                                       |
|----------------|-----------------------------------------------------|-----------------------------------------------------|
| Basic Concept  | Mutual exclusion – only one thread can access at a time. | Synchronization – multiple threads can access based on count. |
| Ownership      | The thread that locks the mutex must unlock it.      | Any thread can signal a semaphore, not necessarily the one that waited on it. |
| Count          | Binary (locked or unlocked).                        | Can be binary (0 or 1) or counting (greater than 1). |
| Blocking       | A thread trying to lock an already locked mutex will be blocked. | A thread waits if the semaphore count is 0; otherwise, it proceeds. |
| Use Case       | Use when exclusive access is required.              | Use when multiple accesses are allowed (e.g., resource pooling). |

## Summary:
- Use a **mutex** when you need **exclusive access** to a shared resource (only one thread can access it at a time).
- Use a **semaphore** when you need to allow **limited concurrent access** to a shared resource (multiple threads can access it, but the access is limited).
