
# Maximal Rectangle - Optimized C++ Solution

## Problem Description
The problem "Maximal Rectangle" on LeetCode requires finding the largest rectangle containing only `1`s in a binary matrix.

Here is an optimized **C++** solution using a **histogram-based approach**, which reduces the problem to finding the largest rectangle in a histogram for each row of the matrix.

---

## Optimized C++ Solution
```cpp
#include <vector>
#include <stack>
#include <algorithm>
#include <iostream>

using namespace std;

class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if (matrix.empty()) return 0;

        int rows = matrix.size();
        int cols = matrix[0].size();
        vector<int> heights(cols, 0); // Histogram heights for each column
        int maxArea = 0;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // Update histogram heights: if '1', increment; if '0', reset to 0
                heights[j] = (matrix[i][j] == '1') ? heights[j] + 1 : 0;
            }
            // Calculate max rectangle in this row's histogram
            maxArea = max(maxArea, largestRectangleArea(heights));
        }

        return maxArea;
    }

private:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        heights.push_back(0); // Add a sentinel value for easier calculation
        int maxArea = 0;

        for (int i = 0; i < heights.size(); ++i) {
            while (!st.empty() && heights[st.top()] > heights[i]) {
                int height = heights[st.top()];
                st.pop();
                int width = st.empty() ? i : i - st.top() - 1;
                maxArea = max(maxArea, height * width);
            }
            st.push(i);
        }

        heights.pop_back(); // Restore the heights vector
        return maxArea;
    }
};
```

---

## Explanation of the Solution

### 1. **Histogram Representation**
- For each row in the matrix, we treat it as the base of a histogram. The height of each column is updated based on consecutive `1`s above the current row.

### 2. **Calculate Largest Rectangle**
- For each histogram (row), the largest rectangle is calculated using the **monotonic stack approach**, which efficiently finds the maximum rectangle area in \(O(n)\) time.

### 3. **Logic for `largestRectangleArea`**
- A monotonic stack is used to keep track of indices of histogram bars in non-decreasing order.
- When a bar of smaller height is encountered, rectangles are calculated using the heights in the stack as potential rectangle heights, and their areas are updated.

### 4. **Time Complexity**
- \(O(m \times n)\), where \(m\) is the number of rows and \(n\) is the number of columns:
  - \(O(n)\) for updating the histogram for each row.
  - \(O(n)\) for finding the largest rectangle area in the histogram.

### 5. **Space Complexity**
- \(O(n)\) for storing the histogram heights and the stack.

---

## Example Usage
```cpp
int main() {
    vector<vector<char>> matrix = {
        {'1', '0', '1', '0', '0'},
        {'1', '0', '1', '1', '1'},
        {'1', '1', '1', '1', '1'},
        {'1', '0', '0', '1', '0'}
    };

    Solution sol;
    cout << "Maximal Rectangle Area: " << sol.maximalRectangle(matrix) << endl;

    return 0;
}
```

### Output:
```
Maximal Rectangle Area: 6
```
