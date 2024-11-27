
# Thread-Safe Data Structures

## **What is a Thread-Safe Data Structure?**
A **thread-safe data structure** is designed to work correctly in multi-threaded environments where multiple threads can access or modify the data concurrently. Thread-safety ensures that:
1. **Integrity is maintained**: The data structure behaves predictably without corruption.
2. **Race conditions are avoided**: Concurrent modifications do not lead to undefined or incorrect results.
3. **Deadlocks and livelocks are prevented**: Mechanisms ensure threads don't block indefinitely.

---

## **How Thread-Safety is Achieved**
Thread-safe data structures use synchronization mechanisms to manage concurrent access. Common approaches include:

### 1. **Locks (Mutexes)**
   - Protects critical sections by ensuring only one thread can modify the data at a time.
   - Examples:
     - `std::mutex` in C++ or `synchronized` keyword in Java.

#### **Implementation Example (C++ with `std::mutex`):**
```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

class ThreadSafeVector {
private:
    std::vector<int> data;
    std::mutex mtx;
public:
    void add(int value) {
        std::lock_guard<std::mutex> lock(mtx); // Automatically unlocks after scope
        data.push_back(value);
    }

    void print() {
        std::lock_guard<std::mutex> lock(mtx);
        for (int v : data)
            std::cout << v << " ";
        std::cout << "\n";
    }
};

int main() {
    ThreadSafeVector tsVector;

    std::thread t1([&]() { tsVector.add(1); });
    std::thread t2([&]() { tsVector.add(2); });

    t1.join();
    t2.join();

    tsVector.print(); // Prints: 1 2
    return 0;
}
```

---

### 2. **Atomic Operations**
   - Use low-level atomic primitives to ensure atomicity without locks.
   - Examples:
     - `std::atomic` in C++ or `AtomicInteger` in Java.

#### **Implementation Example (C++ with `std::atomic`):**
```cpp
#include <iostream>
#include <atomic>
#include <thread>

class Counter {
private:
    std::atomic<int> count;
public:
    Counter() : count(0) {}
    void increment() {
        count.fetch_add(1, std::memory_order_relaxed);
    }

    int get() const {
        return count.load(std::memory_order_relaxed);
    }
};

int main() {
    Counter counter;

    std::thread t1([&]() { for (int i = 0; i < 1000; ++i) counter.increment(); });
    std::thread t2([&]() { for (int i = 0; i < 1000; ++i) counter.increment(); });

    t1.join();
    t2.join();

    std::cout << "Counter: " << counter.get() << "\n"; // Prints: Counter: 2000
    return 0;
}
```

---

### 3. **Lock-Free Algorithms**
   - Avoid locks altogether by leveraging atomic primitives like `compare_and_swap` (CAS).
   - Examples:
     - Concurrent data structures like lock-free queues and stacks.

#### **Implementation Example (Lock-Free Stack):**
```cpp
#include <atomic>
#include <iostream>

struct Node {
    int value;
    Node* next;
    Node(int val) : value(val), next(nullptr) {}
};

class LockFreeStack {
private:
    std::atomic<Node*> head;
public:
    LockFreeStack() : head(nullptr) {}

    void push(int value) {
        Node* newNode = new Node(value);
        newNode->next = head.load();
        while (!head.compare_exchange_weak(newNode->next, newNode)) {
            // Retry until the head is updated atomically
        }
    }

    bool pop(int& result) {
        Node* topNode = head.load();
        while (topNode && !head.compare_exchange_weak(topNode, topNode->next)) {
            // Retry until the head is updated atomically
        }
        if (topNode) {
            result = topNode->value;
            delete topNode;
            return true;
        }
        return false;
    }
};

int main() {
    LockFreeStack stack;
    stack.push(1);
    stack.push(2);

    int value;
    while (stack.pop(value)) {
        std::cout << value << " ";
    } // Prints: 2 1
    return 0;
}
```

---

### 4. **Readers-Writers Locks**
   - Allow multiple readers or one writer at a time.
   - Useful for scenarios where reads are more frequent than writes.

---

### 5. **Thread-Safe Libraries**
   - Some programming languages provide thread-safe data structures out of the box.
     - **Java**: `ConcurrentHashMap`, `CopyOnWriteArrayList`.
     - **C++**: `std::shared_mutex` for shared ownership, `std::lock_guard`.

---

## **Performance Trade-Offs**
Thread-safety often comes with performance costs:
1. **Locks** introduce contention and can degrade performance under heavy load.
2. **Atomic operations** can be faster but harder to design for complex data structures.
3. **Lock-free algorithms** offer high performance but are complex to implement and debug.

---

## **When to Use Thread-Safe Data Structures**
- Use thread-safe data structures when multiple threads access shared data.
- Consider non-thread-safe alternatives if your code ensures no concurrent access. For example:
  - Using thread-local storage.
  - Copying data for each thread.
