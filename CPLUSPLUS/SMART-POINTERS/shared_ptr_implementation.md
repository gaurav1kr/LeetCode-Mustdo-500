
# Custom Implementation of `shared_ptr` in C++

## Overview

A `shared_ptr` in C++ is a smart pointer used to manage the lifetime of a dynamically allocated object through reference counting. 
When the last `shared_ptr` owning an object is destroyed or reset, the object is automatically deleted. This helps avoid memory leaks 
and simplifies memory management.

## Features of `shared_ptr`

1. **Reference Counting**:
   - Keeps track of how many `shared_ptr` instances share ownership of the same object.
   - Reference count is incremented for new owners and decremented for owners being destroyed or reset.

2. **Automatic Deletion**:
   - When reference count reaches zero, the object is automatically deleted.

3. **Custom Deleters**:
   - Supports custom deleters to define how the managed object should be destroyed.

4. **Thread-Safe Reference Counting**:
   - Incrementing and decrementing the reference count is thread-safe.

5. **Circular References Issue**:
   - If `shared_ptr`s reference each other in a cycle, memory leaks can occur. Use `weak_ptr` to break such cycles.

---

## Implementation of `shared_ptr`

Here is an example of how you can implement a basic `shared_ptr`:

```cpp
#include <iostream>

template <typename T>
class SharedPtr 
{
	private:
		T* ptr;                       // Pointer to the managed object
		size_t* ref_count;            // Pointer to the reference count

    void release() 
	{
        if (ref_count) 
		{
            (*ref_count)--;       // Decrement reference count
            if (*ref_count == 0) {
                delete ptr;       // Delete managed object
                delete ref_count; // Delete reference count
                std::cout << "Object and ref_count deleted.\n";
            }
        }
    }

	public:
    // Constructor
    explicit SharedPtr(T* p = nullptr): ptr(p), ref_count(new size_t(1)) 
	{
        std::cout << "SharedPtr created.\n";
    }

    // Copy Constructor
    SharedPtr(const SharedPtr& other): ptr(other.ptr), ref_count(other.ref_count) 
	{
        (*ref_count)++;
        std::cout << "SharedPtr copied. Ref count: " << *ref_count << "\n";
    }

    // Move Constructor
    SharedPtr(SharedPtr&& other) noexcept: ptr(other.ptr), ref_count(other.ref_count) 
	{
        other.ptr = nullptr;
        other.ref_count = nullptr;
        std::cout << "SharedPtr moved.\n";
    }

    // Copy Assignment Operator
    SharedPtr& operator=(const SharedPtr& other) 
	{
        if (this != &other) 
		{
            release(); // Release current object
            ptr = other.ptr;
            ref_count = other.ref_count;
            (*ref_count)++;
            std::cout << "SharedPtr assigned. Ref count: " << *ref_count << "\n";
        }
        return *this;
    }

    // Move Assignment Operator
    SharedPtr& operator=(SharedPtr&& other) noexcept 
	{
        if (this != &other) 
		{
            release(); // Release current object
            ptr = other.ptr;
            ref_count = other.ref_count;
            other.ptr = nullptr;
            other.ref_count = nullptr;
            std::cout << "SharedPtr move-assigned.\n";
        }
        return *this;
    }

    // Destructor
    ~SharedPtr() 
	{
        release();
    }

    // Overloaded * and -> operators
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }

    // Utility function to get reference count
    size_t use_count() const 
	{ 
		return (ref_count) ? *ref_count : 0; 
	}
};

int main() {
    {
        SharedPtr<int> sp1(new int(10));
        std::cout << "sp1 use_count: " << sp1.use_count() << "\n";

        SharedPtr<int> sp2 = sp1;
        std::cout << "sp2 use_count: " << sp2.use_count() << "\n";

        {
            SharedPtr<int> sp3 = sp2;
            std::cout << "sp3 use_count: " << sp3.use_count() << "\n";
        } // sp3 goes out of scope

        std::cout << "sp2 use_count after sp3 is destroyed: " << sp2.use_count() << "\n";
    } // sp1 and sp2 go out of scope

    return 0;
}
```

---

## Explanation

1. **Control Block**:
   - Reference count is managed using `size_t*`. In a complete implementation, it could also hold a custom deleter.

2. **Reference Count Management**:
   - Incremented in copy constructor and copy assignment operator.
   - Decremented in destructor and `release` method.

3. **Resource Management**:
   - The `release` function ensures proper cleanup when reference count drops to zero.

4. **Copy and Move Semantics**:
   - Supports shared ownership through the copy constructor.
   - Efficiently transfers ownership through the move constructor.

5. **Thread Safety**:
   - This implementation is **not thread-safe**. Use `std::atomic<size_t>` for thread safety.

---

## Improvements for Real-World Use

- Use a control block structure to hold both reference count and custom deleter.
- Add thread safety with `std::atomic`.
- Handle weak references (`weak_ptr`) to avoid circular reference issues.

---

## Example Output

```
SharedPtr created.
sp1 use_count: 1
SharedPtr copied. Ref count: 2
sp2 use_count: 2
SharedPtr copied. Ref count: 3
sp3 use_count: 3
Object and ref_count deleted.
sp2 use_count after sp3 is destroyed: 2
Object and ref_count deleted.
```

---

This implementation provides a strong understanding of how `shared_ptr` works internally and how you can manage resource lifetimes effectively in C++.
