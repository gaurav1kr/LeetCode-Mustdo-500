
# Copy Constructor, Copy Assignment Operator, Move Constructor, and Move Assignment Operator in C++

In C++, copy and move constructors and assignment operators are used to define how objects of a class are copied or moved. They help manage resource ownership, such as dynamic memory or file handles. Let's explore each in detail:

## 1. Copy Constructor
The **copy constructor** creates a new object by copying an existing object. It’s called when:
- You pass an object by value to a function.
- You return an object by value from a function.
- You explicitly create a copy of an object.

### Syntax:
```cpp
ClassName(const ClassName &other);
```

### Example:
```cpp
#include<iostream>
using namespace std;

class MyClass {
    int *data;

public:
    // Constructor
    MyClass(int value) {
        data = new int(value);
    }

    // Copy constructor
    MyClass(const MyClass &other) {
        data = new int(*(other.data)); // Deep copy
        cout << "Copy constructor called" << endl;
    }

    // Destructor
    ~MyClass() {
        delete data;
        cout << "Destructor called" << endl;
    }

    int getData() const {
        return *data;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2 = obj1; // Calls copy constructor

    cout << "obj2 data: " << obj2.getData() << endl;
    return 0;
}
```

### Explanation:
- The copy constructor takes a constant reference to another object of the same class.
- **Deep Copy**: In the example, the copy constructor allocates new memory for the `data` pointer and copies the value stored in `other.data`.
- **Shallow Copy**: If you just copy the pointer without allocating new memory, both objects would point to the same memory location, which could lead to a double deletion when destructors are called.

### Output:
```
Copy constructor called
obj2 data: 10
Destructor called
Destructor called
```

## 2. Copy Assignment Operator
The **copy assignment operator** is used to assign the contents of one object to another existing object. It’s called when you use the `=` operator to copy objects.

### Syntax:
```cpp
ClassName& operator=(const ClassName &other);
```

### Example:
```cpp
class MyClass {
    int *data;

public:
    // Constructor
    MyClass(int value) {
        data = new int(value);
    }

    // Copy assignment operator
    MyClass& operator=(const MyClass &other) {
        if (this == &other) // Self-assignment check
            return *this;

        delete data;  // Clean up existing resource
        data = new int(*(other.data)); // Allocate new resource and copy
        cout << "Copy assignment operator called" << endl;
        return *this;
    }

    ~MyClass() {
        delete data;
    }

    int getData() const {
        return *data;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2(20);
    obj2 = obj1; // Calls copy assignment operator

    cout << "obj2 data: " << obj2.getData() << endl;
    return 0;
}
```

### Explanation:
- The **copy assignment operator** must first clean up the existing resources (delete the old `data`), then allocate new memory and copy the contents from the `other` object.
- It also includes a **self-assignment check** to avoid issues when assigning an object to itself (`obj = obj`).

### Output:
```
Copy assignment operator called
obj2 data: 10
```

## 3. Move Constructor
The **move constructor** transfers ownership of resources from one object to another, leaving the original object in a valid but unspecified state. It’s called when:
- An object is initialized with an r-value (temporary object).
- `std::move()` is called on an object.

### Syntax:
```cpp
ClassName(ClassName &&other) noexcept;
```

### Example:
```cpp
class MyClass {
    int *data;

public:
    // Constructor
    MyClass(int value) {
        data = new int(value);
    }

    // Move constructor
    MyClass(MyClass &&other) noexcept {
        data = other.data;  // Transfer ownership
        other.data = nullptr;  // Leave the other object in a valid state
        cout << "Move constructor called" << endl;
    }

    ~MyClass() {
        delete data;
    }

    int getData() const {
        return data ? *data : 0;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2 = std::move(obj1); // Calls move constructor

    cout << "obj2 data: " << obj2.getData() << endl;

    return 0;
}
```

### Explanation:
- The **move constructor** transfers ownership of the `data` pointer from `other` to the new object and then nullifies `other.data` to ensure it doesn't point to the same memory.
- `std::move()` forces the object to be treated as an r-value, triggering the move constructor.

### Output:
```
Move constructor called
obj2 data: 10
```

## 4. Move Assignment Operator
The **move assignment operator** transfers ownership of resources from one object to another existing object. It’s called when you use the `=` operator with an r-value.

### Syntax:
```cpp
ClassName& operator=(ClassName &&other) noexcept;
```

### Example:
```cpp
class MyClass {
    int *data;

public:
    // Constructor
    MyClass(int value) {
        data = new int(value);
    }

    // Move assignment operator
    MyClass& operator=(MyClass &&other) noexcept {
        if (this == &other) // Self-assignment check
            return *this;

        delete data;   // Clean up existing resource
        data = other.data;  // Transfer ownership
        other.data = nullptr;  // Leave the other object in a valid state
        cout << "Move assignment operator called" << endl;
        return *this;
    }

    ~MyClass() {
        delete data;
    }

    int getData() const {
        return data ? *data : 0;
    }
};

int main() {
    MyClass obj1(10);
    MyClass obj2(20);
    obj2 = std::move(obj1); // Calls move assignment operator

    cout << "obj2 data: " << obj2.getData() << endl;
    return 0;
}
```

### Explanation:
- The **move assignment operator** first releases the resources of the current object, then transfers ownership of the `data` pointer from `other` to the current object and nullifies `other.data`.
- The object is left in a valid but empty state, ready for cleanup when destructed.

### Output:
```
Move assignment operator called
obj2 data: 10
```

## Summary Table

| Feature                | When Used                                          | Key Behavior                              |
|------------------------|---------------------------------------------------|-------------------------------------------|
| **Copy Constructor**    | Called when creating a new object as a copy       | Allocates new memory and copies data      |
| **Copy Assignment**     | Called when assigning one object to another       | Cleans up old data, allocates new, copies |
| **Move Constructor**    | Called when transferring resources from a temporary object | Transfers ownership without copying     |
| **Move Assignment**     | Called when assigning resources from a temporary object | Releases old resources, transfers ownership |
