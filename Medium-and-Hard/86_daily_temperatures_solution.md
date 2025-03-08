
```cpp
#include <vector>
#include <stack>
#include <iostream>

using namespace std;

vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> result(n, 0);
    stack<int> st; // Stack to keep track of indices

    for (int i = 0; i < n; ++i) {
        while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
            int idx = st.top();
            st.pop();
            result[idx] = i - idx;
        }
        st.push(i);
    }

    return result;
}

// Test the function
int main() {
    vector<int> temperatures = {73, 74, 75, 71, 69, 72, 76, 73};
    vector<int> result = dailyTemperatures(temperatures);

    for (int days : result) {
        cout << days << " ";
    }
    return 0;
}
```

### Explanation

1. **Core Idea**:
   - Use a **monotonic decreasing stack** to keep track of indices where we need to find the next warmer day.
   - If the current day's temperature is higher than the temperature at the top of the stack, we calculate the difference in indices and update the result for that day.

2. **Steps**:
   - Initialize a `stack<int>` to store indices of temperatures.
   - Iterate through the `temperatures` array.
   - For each temperature, check if it's greater than the temperature at the top of the stack:
     - If true, calculate the number of days until the warmer temperature and update the result.
     - Pop the index from the stack.
   - Push the current day's index onto the stack.

3. **Edge Cases**:
   - For days without a warmer temperature in the future, the result remains 0 (default).

4. **Complexity**:
   - **Time Complexity**: \(O(n)\), as each index is pushed and popped from the stack at most once.
   - **Space Complexity**: \(O(n)\) for the stack.
