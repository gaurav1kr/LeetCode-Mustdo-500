
# Features Introduced in C++11 and C++14

## C++11 Features:
C++11 (often referred to as "C++0x" before its official release) brought major changes and additions to the C++ language. Key features include:

### 1. Auto Keyword
- **Type Inference:** The `auto` keyword allows the compiler to automatically deduce the type of a variable from its initializer. This reduces verbosity and helps make code more flexible.
```cpp
auto x = 10;    // x is deduced as int
auto y = 10.5;  // y is deduced as double
```

### 2. Range-Based For Loop
- Simplified syntax for iterating over containers like arrays, vectors, etc.
```cpp
for (int x : myVector) {
    // Do something with x
}
```

### 3. Lambda Expressions
- Anonymous functions that can be defined in place, primarily for use in algorithms or as callbacks.
```cpp
auto add = [](int a, int b) { return a + b; };
std::cout << add(2, 3);  // Output: 5
```

### 4. Move Semantics and rvalue References
- Introduced the concept of move semantics through rvalue references (`&&`), enabling the transfer of resources from temporary objects, which can significantly improve performance, especially in cases involving dynamic memory management.
```cpp
std::vector<int> v1 = {1, 2, 3};
std::vector<int> v2 = std::move(v1);  // Moves v1 to v2
```

### 5. Smart Pointers
- Added `std::shared_ptr` and `std::unique_ptr` to manage dynamic memory safely, reducing the chances of memory leaks.
```cpp
std::unique_ptr<int> ptr(new int(10));
```

### 6. nullptr
- A dedicated keyword (`nullptr`) was introduced for representing null pointers, replacing the old `NULL` or `0`.
```cpp
int* p = nullptr;
```

### 7. Threading Support
- C++11 introduced a thread library (`<thread>`) to allow native multithreading support.
```cpp
std::thread t([]() { std::cout << "Thread\n"; });
t.join();  // Wait for the thread to finish
```

### 8. Strongly-Typed Enums
- Scoped enums using `enum class` were added to avoid polluting the global namespace and providing better type safety.
```cpp
enum class Color { Red, Green, Blue };
Color c = Color::Red;
```

### 9. constexpr
- Functions and variables can be evaluated at compile time, making code more efficient by avoiding runtime computations.
```cpp
constexpr int square(int x) {
    return x * x;
}
```

### 10. Variadic Templates
- Template parameter packs enable functions and classes to take a variable number of template arguments.
```cpp
template<typename... Args>
void print(Args... args) {
    (std::cout << ... << args);  // Fold expression in C++17
}
```

### 11. Type Aliases (using)
- A more readable and flexible way to create type aliases, replacing `typedef`.
```cpp
using String = std::string;
```

### 12. std::tuple and std::array
- C++11 added `std::tuple`, which can hold a fixed number of heterogeneous elements, and `std::array`, which is a static-size array with the same functionality as `std::vector`.
```cpp
std::tuple<int, double, std::string> t(1, 3.14, "hello");
std::array<int, 3> arr = {1, 2, 3};
```

## C++14 Features:
C++14 was a smaller release compared to C++11, focused on refinements and minor feature additions. Key features include:

### 1. Relaxed `constexpr`
- C++14 relaxed the rules on `constexpr` functions, allowing them to contain more complex logic like loops and conditionals, as long as they still result in constant expressions.
```cpp
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}
```

### 2. Return Type Deduction for Functions
- Functions can now automatically deduce their return type using the `auto` keyword, simplifying the declaration of template functions.
```cpp
auto add(int a, int b) {
    return a + b;  // Return type is automatically deduced as int
}
```

### 3. Generic Lambda Expressions
- Lambdas were enhanced to support template-like generic parameters, making them more powerful.
```cpp
auto lambda = [](auto x, auto y) { return x + y; };
std::cout << lambda(2, 3);   // Output: 5
```

### 4. `std::make_unique`
- Added `std::make_unique` for creating `std::unique_ptr` objects more safely and efficiently.
```cpp
auto ptr = std::make_unique<int>(10);
```

### 5. Binary Literals
- C++14 introduced binary literals, allowing you to specify numbers in binary format by prefixing them with `0b` or `0B`.
```cpp
int binaryNum = 0b1010;  // Decimal 10
```

### 6. Digit Separators
- You can now use single quotes (`'`) as digit separators to improve the readability of large numbers.
```cpp
int largeNumber = 1'000'000;
```

### 7. std::exchange
- A utility function that exchanges the value of an object with a new one and returns the old value.
```cpp
int oldValue = std::exchange(x, newValue);
```

### 8. decltype(auto)
- Allows `decltype(auto)` to be used to deduce the return type in a more flexible manner, preserving reference or const qualifiers.
```cpp
decltype(auto) func() {
    int x = 5;
    return (x);  // Deduces int&
}
```

These updates in C++11 and C++14 laid the groundwork for modern C++ development, introducing powerful features that promote safer, faster, and more readable code.
