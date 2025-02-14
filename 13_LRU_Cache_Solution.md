# LRU Cache Solution in C++

This document provides an optimal solution for the **LRU Cache** problem on LeetCode. The solution is implemented using C++ and leverages `unordered_map` and `list` for efficient operations.

## Code
```cpp
#include <unordered_map>
#include <list>

class LRUCache {
private:
    int capacity;
    std::list<std::pair<int, int>> cacheList; // Stores key-value pairs
    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> cacheMap; // Maps key to list iterator

public:
    LRUCache(int capacity) {
        this->capacity = capacity;
    }

    int get(int key) {
        if (cacheMap.find(key) == cacheMap.end()) {
            return -1; // Key not found
        }

        // Move the accessed node to the front of the list (most recently used)
        auto it = cacheMap[key];
        cacheList.splice(cacheList.begin(), cacheList, it);
        return it->second;
    }

    void put(int key, int value) {
        if (cacheMap.find(key) != cacheMap.end()) {
            // Key already exists, update the value and move to the front
            auto it = cacheMap[key];
            it->second = value;
            cacheList.splice(cacheList.begin(), cacheList, it);
        } else {
            if (cacheList.size() == capacity) {
                // Cache is full, remove the least recently used (LRU) item
                auto lru = cacheList.back();
                cacheMap.erase(lru.first);
                cacheList.pop_back();
            }

            // Insert the new key-value pair at the front of the list
            cacheList.push_front({key, value});
            cacheMap[key] = cacheList.begin();
        }
    }
};
```

## Explanation

### Data Structures
1. **`std::list`**: Used to store key-value pairs in the order of access. The front represents the most recently used item.
2. **`std::unordered_map`**: Maps a key to an iterator pointing to its position in the `list`. This provides O(1) lookup and removal.

### Operations

#### `get(key)`
- Check if the key exists in `cacheMap`.
- If it exists, move the corresponding list node to the front (most recently used) and return its value.
- If it doesn't exist, return `-1`.

#### `put(key, value)`
- If the key already exists, update its value and move it to the front.
- If the key doesn't exist:
  - Check if the cache is full. If it is, remove the least recently used item from the back of the `list` and erase its entry from the `unordered_map`.
  - Add the new key-value pair to the front of the `list` and update the `unordered_map`.

### Time Complexity
- `get`: O(1) due to `unordered_map` and `list` operations.
- `put`: O(1) for insertion and eviction.

### Space Complexity
- O(C), where C is the capacity of the cache.

## Advantages
- Efficiently handles LRU Cache operations with constant time complexity.
- Clean and easy-to-understand implementation using STL containers.

## Example Usage
```cpp
int main() {
    LRUCache lruCache(2); // Initialize cache with capacity 2

    lruCache.put(1, 1); // Cache: {1=1}
    lruCache.put(2, 2); // Cache: {2=2, 1=1}

    std::cout << lruCache.get(1) << std::endl; // Returns 1 (Cache: {1=1, 2=2})

    lruCache.put(3, 3); // Evicts key 2 (Cache: {3=3, 1=1})
    std::cout << lruCache.get(2) << std::endl; // Returns -1 (not found)

    lruCache.put(4, 4); // Evicts key 1 (Cache: {4=4, 3=3})
    std::cout << lruCache.get(1) << std::endl; // Returns -1 (not found)
    std::cout << lruCache.get(3) << std::endl; // Returns 3
    std::cout << lruCache.get(4) << std::endl; // Returns 4

    return 0;
}
```
