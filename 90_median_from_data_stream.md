# Find Median from Data Stream

## Problem Description

Design a data structure that supports adding numbers from a data stream and finding the median of all added numbers efficiently. Implement the following operations:

- `addNum(int num)`: Add an integer `num` from the data stream to the data structure.
- `findMedian()`: Return the median of all elements so far.

---

## Optimal Solution in C++

To solve this problem, we use the **two heaps approach**.

### Algorithm:

1. **Use Two Heaps**:
   - A max-heap (`lower`) to store the smaller half of the numbers.
   - A min-heap (`upper`) to store the larger half of the numbers.

2. **Balancing the Heaps**:
   - Ensure the size difference between the two heaps is at most 1.
   - The max-heap contains elements less than or equal to the median, and the min-heap contains elements greater than or equal to the median.

3. **Finding the Median**:
   - If the heaps are of the same size, the median is the average of the two middle elements.
   - If one heap is larger, the median is the top element of that heap.

### C++ Code:

```cpp
#include <queue>
using namespace std;

class MedianFinder {
private:
    priority_queue<int> lower; // Max-heap
    priority_queue<int, vector<int>, greater<int>> upper; // Min-heap

public:
    MedianFinder() {}

    void addNum(int num) {
        // Step 1: Add to max-heap first
        lower.push(num);

        // Step 2: Balance the heaps
        // Move the largest element from lower to upper
        upper.push(lower.top());
        lower.pop();

        // Step 3: Ensure size property (lower >= upper)
        if (lower.size() < upper.size()) {
            lower.push(upper.top());
            upper.pop();
        }
    }

    double findMedian() {
        if (lower.size() > upper.size()) {
            return lower.top(); // Odd total, median is the top of lower
        } else {
            return (lower.top() + upper.top()) / 2.0; // Even total, average of tops
        }
    }
};

/**
 * Usage:
 * MedianFinder mf;
 * mf.addNum(1);
 * mf.addNum(2);
 * double median = mf.findMedian(); // 1.5
 * mf.addNum(3);
 * median = mf.findMedian(); // 2
 */
```

---

## Explanation:

### Adding Numbers:
- Add the number to the max-heap (`lower`).
- Transfer the largest element from `lower` to `upper` to maintain order.
- If `upper` becomes larger, move the smallest element from `upper` back to `lower`.

### Finding the Median:
- If `lower` has more elements, the median is the top of `lower`.
- If both heaps are of equal size, the median is the average of the tops of `lower` and `upper`.

---

## Complexity:

- **Time Complexity**:
  - `addNum`: \(O(\log n)\), since insertion and balancing involve heap operations.
  - `findMedian`: \(O(1)\), since it just involves accessing the tops of the heaps.

- **Space Complexity**:
  - \(O(n)\), where \(n\) is the number of elements stored.

---

## Example Usage:

```cpp
MedianFinder mf;
mf.addNum(1);
mf.addNum(2);
cout << mf.findMedian() << endl; // Output: 1.5
mf.addNum(3);
cout << mf.findMedian() << endl; // Output: 2
```
