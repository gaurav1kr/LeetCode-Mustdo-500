
# Understanding `std::weak_ptr` in C++

In C++, `std::weak_ptr` is a smart pointer that provides a non-owning ("weak") reference to an object managed by `std::shared_ptr`. It is part of the C++ Standard Library (`<memory>`).

## Why `std::weak_ptr`?

When using `std::shared_ptr`, objects are reference-counted. This reference counting creates a problem when two objects refer to each other using `std::shared_ptr`: they form a cyclic dependency. This prevents their memory from being released, even when there are no external references.

To solve this, `std::weak_ptr` allows you to refer to a `std::shared_ptr` without affecting its reference count. If the `std::shared_ptr` managing the object is destroyed, the `std::weak_ptr` becomes invalid.

## Key Features of `std::weak_ptr`

1. **Non-owning:** It doesn't affect the reference count of the shared object.
2. **Use with `std::shared_ptr`:** `std::weak_ptr` works only with objects managed by `std::shared_ptr`.
3. **Checking validity:** You can check if the managed object still exists using `expired()` or lock the `weak_ptr` to a `shared_ptr`.
4. **Thread-safety:** It is thread-safe to use `std::weak_ptr` with `std::shared_ptr`.

## How to Use `std::weak_ptr`

```cpp
#include <iostream>
#include <memory>

class Node {
public:
    std::weak_ptr<Node> next;  // Weak reference to avoid cyclic dependency
    ~Node() {
        std::cout << "Node destroyed\n";
    }
};

int main() {
    auto node1 = std::make_shared<Node>();
    auto node2 = std::make_shared<Node>();

    node1->next = node2;  // `node1` points to `node2`
    node2->next = node1;  // `node2` points back to `node1` (cyclic reference)

    // Break the cycle by resetting one shared pointer
    node1.reset();
    node2.reset();  // Both nodes are destroyed

    return 0;
}
```

## How to Implement Your Own `weak_ptr`

Here is a basic idea of how you could write your own `weak_ptr` and `shared_ptr`. For simplicity, this implementation won't include all the features of the standard library versions.

### Step 1: Define a `ControlBlock`

The `ControlBlock` maintains both a reference count for `shared_ptr` and a weak reference count.

```cpp
struct ControlBlock {
    int shared_count = 0;
    int weak_count = 0;
    ControlBlock() = default;
};
```

### Step 2: Implement `SharedPtr`

A simple `SharedPtr` manages the object and increments/decrements the `shared_count`.

```cpp
template <typename T>
class SharedPtr {
public:
    T* ptr = nullptr;
    ControlBlock* ctrl = nullptr;

    SharedPtr(T* obj) : ptr(obj), ctrl(new ControlBlock) {
        ctrl->shared_count = 1;
    }

    SharedPtr(const SharedPtr& other) : ptr(other.ptr), ctrl(other.ctrl) {
        if (ctrl) ++ctrl->shared_count;
    }

    ~SharedPtr() {
        release();
    }

    void release() {
        if (ctrl && --ctrl->shared_count == 0) {
            delete ptr;
            if (ctrl->weak_count == 0) {
                delete ctrl;
            }
        }
    }
};
```

### Step 3: Implement `WeakPtr`

A `WeakPtr` works with the `ControlBlock` but doesn't affect `shared_count`.

```cpp
template <typename T>
class WeakPtr {
public:
    T* ptr = nullptr;
    ControlBlock* ctrl = nullptr;

    WeakPtr() = default;

    WeakPtr(const SharedPtr<T>& shared) : ptr(shared.ptr), ctrl(shared.ctrl) {
        if (ctrl) ++ctrl->weak_count;
    }

    ~WeakPtr() {
        if (ctrl && --ctrl->weak_count == 0 && ctrl->shared_count == 0) {
            delete ctrl;
        }
    }

    SharedPtr<T> lock() const {
        if (ctrl && ctrl->shared_count > 0) {
            return SharedPtr<T>(*this);
        }
        return SharedPtr<T>(nullptr);
    }

    bool expired() const {
        return !ctrl || ctrl->shared_count == 0;
    }
};
```

### Step 4: Usage Example

```cpp
int main() {
    SharedPtr<int> sp = SharedPtr<int>(new int(10));
    WeakPtr<int> wp = sp;

    if (auto locked = wp.lock()) {
        std::cout << "Value: " << *locked.ptr << "\n";
    } else {
        std::cout << "Object is expired\n";
    }

    sp.release();

    if (wp.expired()) {
        std::cout << "Weak pointer expired\n";
    }

    return 0;
}
```

## Key Points

- Writing your own `weak_ptr` requires managing a control block for reference counting.
- The `ControlBlock` is shared between `SharedPtr` and `WeakPtr`.
- A `WeakPtr` becomes invalid when the shared count in the control block drops to 0.

This implementation is educational and skips some complexities (like thread-safety, atomic operations, etc.) that are present in the standard library implementation.
