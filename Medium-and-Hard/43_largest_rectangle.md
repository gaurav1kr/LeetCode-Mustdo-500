## Largest Rectangle in Histogram - Optimal C++ Solution

### Approach
The optimal solution for the **"Largest Rectangle in Histogram"** problem on LeetCode uses a **monotonic stack** approach, which runs in **O(n) time complexity**.

#### Steps:
1. **Use a stack** to store indices of the histogram bars.
2. **Iterate through the histogram heights**, ensuring the stack maintains a **non-decreasing order**.
3. **If a smaller height is encountered**, process the stored heights by calculating the area using the popped height as the shortest bar.
4. **Continue until all bars are processed** and the stack is empty.
5. **Return the maximum area found**.

### Optimal C++ Solution
```cpp
#include <iostream>
#include <vector>
#include <stack>

using namespace std;

class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        int maxArea = 0;
        int n = heights.size();

        for (int i = 0; i <= n; i++) {
            while (!st.empty() && (i == n || heights[i] < heights[st.top()])) {
                int h = heights[st.top()];
                st.pop();
                int width = st.empty() ? i : (i - st.top() - 1);
                maxArea = max(maxArea, h * width);
            }
            st.push(i);
        }

        return maxArea;
    }
};

// Example usage
int main() {
    Solution sol;
    vector<int> heights = {2, 1, 5, 6, 2, 3};
    cout << "Largest Rectangle Area: " << sol.largestRectangleArea(heights) << endl;
    return 0;
}
```

### Explanation
1. **Iterate through each bar**:
   - Push index into the stack if the stack is empty or the current height is greater than or equal to the height at the top of the stack.
   - Otherwise, pop from the stack and calculate the **area** considering the popped height as the shortest in the rectangle.
   
2. **Handling Width**:
   - If the stack is empty after popping, the width extends from index `0` to `i`.
   - Otherwise, the width is `(i - st.top() - 1)`, as `st.top()` represents the next smaller height index.

3. **Edge Case**:
   - The loop runs until `i == n` to ensure all bars are processed.

### Complexity Analysis
- **Time Complexity:** **O(n)** (Each bar is pushed and popped once).
- **Space Complexity:** **O(n)** (Stack stores at most `n` indices in the worst case).

This solution efficiently finds the **largest rectangle in a histogram** with an **optimal time complexity**. 🚀