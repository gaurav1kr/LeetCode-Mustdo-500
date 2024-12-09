
# C++17 Features and Examples

## 1. `std::optional`
Represents optional (nullable) values, useful for functions that may not return a value.

**Example**:
```cpp
#include <iostream>
#include <optional>

std::optional<int> findValue(bool condition) {
    if (condition) return 42;
    return std::nullopt;
}

int main() {
    auto result = findValue(true);
    if (result) {
        std::cout << "Value: " << *result << '\n';
    } else {
        std::cout << "No value found\n";
    }
}
```

---

## 2. `std::variant`
A type-safe union that holds one of several types.

**Example**:
```cpp
#include <iostream>
#include <variant>

int main() {
    std::variant<int, std::string> value;
    value = 42;
    std::cout << "Integer: " << std::get<int>(value) << '\n';

    value = "Hello, C++17!";
    std::cout << "String: " << std::get<std::string>(value) << '\n';
}
```

---

## 3. `std::any`
A type-safe container for single values of any type.

**Example**:
```cpp
#include <iostream>
#include <any>

int main() {
    std::any data = 10;
    std::cout << "Integer: " << std::any_cast<int>(data) << '\n';

    data = std::string("Hello");
    std::cout << "String: " << std::any_cast<std::string>(data) << '\n';
}
```

---

## 4. Structured Bindings
Allows decomposition of objects into individual variables.

**Example**:
```cpp
#include <iostream>
#include <tuple>

int main() {
    auto [x, y, z] = std::make_tuple(1, 2.5, "hello");
    std::cout << x << ", " << y << ", " << z << '\n';
}
```

---

## 5. `if constexpr`
Compile-time conditional evaluation.

**Example**:
```cpp
#include <iostream>
#include <type_traits>

template <typename T>
void printType(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integer: " << value << '\n';
    } else {
        std::cout << "Non-integer: " << value << '\n';
    }
}

int main() {
    printType(42);
    printType(3.14);
}
```

---

## 6. Inline Variables
Allows variables to be defined in header files without violating the one-definition rule.

**Example**:
```cpp
// header.h
inline int myVar = 10;

// main.cpp
#include <iostream>
#include "header.h"

int main() {
    std::cout << "Value: " << myVar << '\n';
}
```

---

## 7. `std::filesystem`
Provides a way to work with files and directories.

**Example**:
```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    fs::path p = "example.txt";
    std::cout << "File exists: " << fs::exists(p) << '\n';
}
```

---

## 8. `std::string_view`
A non-owning reference to a string.

**Example**:
```cpp
#include <iostream>
#include <string_view>

void print(std::string_view str) {
    std::cout << "String: " << str << '\n';
}

int main() {
    print("Hello, World!");
}
```

---

## 9. Fold Expressions
Simplifies variadic template programming.

**Example**:
```cpp
#include <iostream>

template <typename... Args>
auto sum(Args... args) {
    return (args + ...);
}

int main() {
    std::cout << "Sum: " << sum(1, 2, 3, 4) << '\n';
}
```

---

## 10. `constexpr` Enhancements
Expanded to allow more complex computations at compile time.

**Example**:
```cpp
#include <iostream>

constexpr int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    constexpr int result = factorial(5);
    std::cout << "Factorial: " << result << '\n';
}
```

---

## 11. `std::clamp`
Clamps a value within a given range.

**Example**:
```cpp
#include <iostream>
#include <algorithm>

int main() {
    int value = 15;
    std::cout << "Clamped: " << std::clamp(value, 0, 10) << '\n';
}
```

---

## 12. Parallel Algorithms
Adds support for parallelism in the Standard Library.

**Example**:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::for_each(std::execution::par, vec.begin(), vec.end(), [](int& n) { n *= 2; });
    for (int n : vec) {
        std::cout << n << ' ';
    }
}
```

---

## 13. `std::uncaught_exceptions`
Counts the number of uncaught exceptions.

**Example**:
```cpp
#include <iostream>
#include <exception>

int main() {
    try {
        std::cout << "Uncaught exceptions: " << std::uncaught_exceptions() << '\n';
        throw std::runtime_error("Error!");
    } catch (...) {
        std::cout << "Caught an exception!\n";
    }
}
```
