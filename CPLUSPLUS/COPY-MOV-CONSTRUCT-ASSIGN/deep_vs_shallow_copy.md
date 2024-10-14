
# Deep Copy vs Shallow Copy in C++

In C++, when dealing with dynamic memory or resource ownership, understanding the difference between **deep copy** and **shallow copy** is crucial. These terms refer to how the resources (typically heap-allocated memory) are handled when objects are copied.

## 1. Shallow Copy

A **shallow copy** simply copies the values of the member variables from one object to another. In the case of pointer members, this means copying the address stored in the pointer, rather than allocating new memory and copying the contents. As a result, both objects end up pointing to the same memory location.

### Characteristics:
- Only the pointers' addresses are copied.
- Both the original and copied objects point to the same resource (memory).
- This can lead to **double deletion** (when both destructors attempt to free the same memory) and undefined behavior.

### Example of Shallow Copy:
```cpp
#include <iostream>
using namespace std;

class MyClass {
    int *data;

public:
    // Constructor
    MyClass(int value) {
        data = new int(value);  // Allocate memory dynamically
        cout << "Constructor called" << endl;
    }

    // Default copy constructor (shallow copy)
    MyClass(const MyClass &other) = default;

    // Destructor
    ~MyClass() {
        delete data;  // Release the memory
        cout << "Destructor called" << endl;
    }

    int getData() const {
        return *data;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2 = obj1;  // Shallow copy (default copy constructor)

    cout << "obj1 data: " << obj1.getData() << endl;
    cout << "obj2 data: " << obj2.getData() << endl;

    return 0;
}
```

### Explanation:
- In this example, both `obj1` and `obj2` point to the same memory location because the default copy constructor performs a shallow copy.
- This would cause a problem when both objects are destroyed because both will attempt to free the same memory, leading to a **double free** error.

### Output:
```
Constructor called
obj1 data: 10
obj2 data: 10
Destructor called
Destructor called
```

The program crashes due to a double deletion of the same memory.

## 2. Deep Copy

A **deep copy** creates a new copy of the dynamically allocated memory or resource, meaning that each object has its own copy of the resource. This avoids the pitfalls of shallow copying, such as double deletion and unintended modifications.

### Characteristics:
- Dynamically allocated memory is duplicated.
- Both the original and the copied objects own independent memory blocks.
- There’s no risk of double deletion or unexpected side effects from modifying one object.

### Example of Deep Copy:
```cpp
#include <iostream>
using namespace std;

class MyClass {
    int *data;

public:
    // Constructor
    MyClass(int value) {
        data = new int(value);  // Allocate memory dynamically
        cout << "Constructor called" << endl;
    }

    // Deep copy constructor
    MyClass(const MyClass &other) {
        data = new int(*(other.data));  // Allocate new memory and copy value
        cout << "Copy constructor (deep copy) called" << endl;
    }

    // Destructor
    ~MyClass() {
        delete data;  // Release the memory
        cout << "Destructor called" << endl;
    }

    int getData() const {
        return *data;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2 = obj1;  // Deep copy

    cout << "obj1 data: " << obj1.getData() << endl;
    cout << "obj2 data: " << obj2.getData() << endl;

    return 0;
}
```

### Explanation:
- The deep copy constructor allocates new memory for `data` and copies the contents from `other.data`. This ensures `obj1` and `obj2` have independent copies of the data.
- Now, when the destructors are called, each object frees its own memory, avoiding double deletion.

### Output:
```
Constructor called
Copy constructor (deep copy) called
obj1 data: 10
obj2 data: 10
Destructor called
Destructor called
```

There’s no crash, and both destructors clean up their respective memory properly.

## 3. Summary Table

| Feature           | Shallow Copy                                         | Deep Copy                                   |
|-------------------|-----------------------------------------------------|---------------------------------------------|
| **Memory Handling**| Copies pointer addresses, shares the same resource  | Allocates new memory and copies contents    |
| **Pointers**       | Both objects point to the same memory               | Each object points to its own memory        |
| **Risk**           | Double deletion, unintended side effects            | No risk of double deletion or side effects  |
| **Use Case**       | Faster, but risky when managing resources           | Safer for objects managing dynamic memory   |

### When to Use Each:
- **Shallow copy**: Appropriate when your class doesn't own dynamic resources (like heap-allocated memory). It's faster because it doesn't need to duplicate the data.
- **Deep copy**: Necessary when your class manages dynamic resources. It ensures that each object has its own independent copy of the data, preventing issues like double deletion and unintended sharing.

