
# Custom Implementation of `unique_ptr` in C++

## Overview

A `unique_ptr` in C++ is a smart pointer that owns and manages the lifetime of a dynamically allocated object. It provides sole ownership of the object and ensures that the object is deleted automatically when the `unique_ptr` goes out of scope.

## Features of `unique_ptr`

1. **Sole Ownership**:
   - Only one `unique_ptr` can own a particular object.

2. **Automatic Deletion**:
   - The managed object is automatically deleted when the `unique_ptr` is destroyed.

3. **Move Semantics**:
   - Ownership can be transferred using move semantics (e.g., with `std::move`).

4. **Custom Deleters**:
   - Supports custom deleters for specialized cleanup operations.

5. **Non-Copyable**:
   - `unique_ptr` cannot be copied, ensuring unique ownership.

---

## Implementation of `unique_ptr`

Here is an example of how you can implement a basic `unique_ptr`:

```cpp
#include <iostream>
#include <utility> // for std::move

template <typename T>
class UniquePtr 
{
private:
    T* ptr; // Pointer to the managed object

public:
    // Constructor
    explicit UniquePtr(T* p = nullptr) : ptr(p) 
	{
        std::cout << "UniquePtr created.\n";
    }

    // Destructor
    ~UniquePtr() 
	{
        if (ptr) 
		{
            delete ptr;
            std::cout << "UniquePtr deleted the managed object.\n";
        }
    }

    // Delete Copy Constructor and Copy Assignment Operator
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;

    // Move Constructor
    UniquePtr(UniquePtr&& other) noexcept : ptr(other.ptr) 
	{
        other.ptr = nullptr; // Transfer ownership
        std::cout << "UniquePtr moved.\n";
    }

    // Move Assignment Operator
    UniquePtr& operator=(UniquePtr&& other) noexcept 
	{
        if (this != &other) 
		{
            // Release current resource
            if (ptr) 
			{
                delete ptr;
            }
            ptr = other.ptr; // Transfer ownership
            other.ptr = nullptr;
            std::cout << "UniquePtr move-assigned.\n";
        }
        return *this;
    }

    // Overloaded * and -> operators
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }

    // Release ownership of the managed object
    T* release() 
	{
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }

    // Replace the managed object
    void reset(T* p = nullptr) 
	{
        if (ptr) 
		{
            delete ptr;
        }
        ptr = p;
    }

    // Get the raw pointer
    T* get() const { return ptr; }
};

int main() {
    UniquePtr<int> up1(new int(10));
    std::cout << "Value: " << *up1 << "\n";

    UniquePtr<int> up2 = std::move(up1); // Transfer ownership
    if (!up1.get()) {
        std::cout << "up1 no longer owns the resource.\n";
    }

    up2.reset(new int(20)); // Replace the managed object
    std::cout << "Value after reset: " << *up2 << "\n";

    up2.release(); // Release ownership without deleting
    std::cout << "up2 released ownership.\n";

    return 0;
}
```

---

## Explanation

1. **Ownership Management**:
   - The `UniquePtr` ensures that only one `UniquePtr` instance owns the resource at any time.

2. **Move Semantics**:
   - Move constructor and move assignment operator transfer ownership of the resource.
   - The source pointer is set to `nullptr` after the transfer.

3. **Resource Release**:
   - The `release` function relinquishes ownership of the resource without deleting it.

4. **Resetting and Replacement**:
   - The `reset` function replaces the managed object with a new one, deleting the old one.

5. **Raw Pointer Access**:
   - The `get` function provides access to the raw pointer without transferring ownership.

6. **Non-Copyable**:
   - The copy constructor and copy assignment operator are deleted to prevent copying.

---

## Example Output

```
UniquePtr created.
Value: 10
UniquePtr moved.
up1 no longer owns the resource.
UniquePtr deleted the managed object.
Value after reset: 20
UniquePtr deleted the managed object.
up2 released ownership.
```

---

## Advantages of `unique_ptr`

- Eliminates manual `delete` calls, reducing the risk of memory leaks.
- Provides strong ownership semantics, ensuring no dangling references to the object.
- Lightweight compared to `shared_ptr` since it does not use reference counting.

---

## Improvements for Real-World Use

- Add support for custom deleters.
- Thread safety is not required as `unique_ptr` is designed for single-threaded ownership.

---
