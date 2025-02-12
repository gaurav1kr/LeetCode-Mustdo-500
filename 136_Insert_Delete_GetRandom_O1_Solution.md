
# Insert Delete GetRandom O(1) - Optimized C++ Solution

This is a concise and optimized C++ solution for the LeetCode problem ["Insert Delete GetRandom O(1)"](https://leetcode.com/problems/insert-delete-getrandom-o1/).

## Solution

```cpp
#include <unordered_map>
#include <vector>
#include <cstdlib>

class RandomizedSet {
    std::unordered_map<int, int> valToIndex;
    std::vector<int> values;

public:
    RandomizedSet() {}

    bool insert(int val) {
        if (valToIndex.count(val)) return false;
        valToIndex[val] = values.size();
        values.push_back(val);
        return true;
    }

    bool remove(int val) {
        if (!valToIndex.count(val)) return false;
        int idx = valToIndex[val];
        int lastVal = values.back();
        values[idx] = lastVal; // Replace with last element
        valToIndex[lastVal] = idx; // Update index of last element
        values.pop_back(); // Remove last element
        valToIndex.erase(val); // Erase removed element
        return true;
    }

    int getRandom() {
        return values[rand() % values.size()];
    }
};
```

## Key Points
1. **Insert Operation**: Uses an `unordered_map` for O(1) lookup and insertion of indices.
2. **Remove Operation**:
   - Replace the element to remove with the last element in the `vector`.
   - Update the `unordered_map` to reflect the new index of the moved element.
   - Pop the last element and erase the removed element from the map.
3. **GetRandom Operation**: Access a random index in the `vector` using `rand()` in O(1).

This implementation ensures O(1) complexity for all operations as required by the problem.
