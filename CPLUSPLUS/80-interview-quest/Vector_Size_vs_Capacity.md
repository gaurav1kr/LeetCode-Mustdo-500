
# Difference Between Size and Capacity of a Vector in C++

`std::vector` in C++ provides two important properties: **size** and **capacity**. While both relate to the storage of elements, they have distinct meanings and use cases.

---

## **1. Size**

- **Definition**: The number of elements currently stored in the vector.
- **Type**: `size_t`
- **Accessed via**: `vector.size()`
- **Dynamic Behavior**: 
  - Increases when elements are added using methods like `push_back()`.
  - Decreases when elements are removed using methods like `pop_back()` or `erase()`.

### **Example**:
```cpp
std::vector<int> vec = {1, 2, 3};
std::cout << "Size: " << vec.size();  // Output: 3
```

---

## **2. Capacity**

- **Definition**: The total number of elements that the vector can hold without requiring reallocation of memory.
- **Type**: `size_t`
- **Accessed via**: `vector.capacity()`
- **Dynamic Behavior**:
  - Starts with a default capacity (implementation-dependent).
  - Automatically increases when more elements are added beyond its current capacity.
  - Initially size and capacity are same. As soon as size gets increased and it becomes greater than
    current capacity , capacity gets doubled from it's previous value.
  - Can be manually set using `reserve()`.


---

## **Key Differences**

| **Aspect**         | **Size**                                        | **Capacity**                                      |
|---------------------|------------------------------------------------|--------------------------------------------------|
| **Meaning**         | Number of elements currently in the vector.    | Maximum elements the vector can hold before reallocation. |
| **Dynamic Behavior**| Changes as elements are added or removed.      | Grows automatically when needed but doesn't shrink automatically. |
| **Direct Control**  | Cannot be controlled directly.                 | Can be manually controlled using `reserve()` or `shrink_to_fit()`. |
| **Function**        | Measures usage of the vector.                  | Indicates memory reserved for potential growth.  |

---

## **Illustrative Example**

```cpp
#include <iostream>
#include <vector>

int main() 
{
    std::vector<int> vec;
    std::cout<<"size :"<<vec.size()<<"\n"<<"capacity :"<<vec.capacity()<<"\n" ;

    vec.push_back(1);
    std::cout<<"size :"<<vec.size()<<"\n"<<"capacity :"<<vec.capacity()<<"\n" ;

    vec.push_back(2);
    std::cout<<"size :"<<vec.size()<<"\n"<<"capacity :"<<vec.capacity()<<"\n" ;

    vec.push_back(3);
    std::cout<<"size :"<<vec.size()<<"\n"<<"capacity :"<<vec.capacity()<<"\n" ;

    vec.push_back(4);
    std::cout<<"size :"<<vec.size()<<"\n"<<"capacity :"<<vec.capacity()<<"\n" ;

    vec.push_back(5);
    std::cout<<"size :"<<vec.size()<<"\n"<<"capacity :"<<vec.capacity()<<"\n" ;
    
    vec.push_back(6);
    std::cout<<"size :"<<vec.size()<<"\n"<<"capacity :"<<vec.capacity()<<"\n" ;
    return 0 ;
}

#output - we can see that the size is getting doubled as soon as size > capacity
size :0
capacity :0
size :1
capacity :1
size :2
capacity :2
size :3
capacity :4
size :4
capacity :4
size :5
capacity :8
size :6
capacity :8
```

---

## **Key Observations**

1. **Size changes immediately** with additions or deletions of elements.
2. **Capacity changes dynamically**, but only when the current capacity is exceeded during an addition.
3. You can **manually reserve memory** to optimize performance if you know the required size in advance.
4. Use `shrink_to_fit()` to reduce the capacity to match the size if memory usage is a concern.

---
