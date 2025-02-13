# Task Scheduler - Optimized C++ Solution

## Problem Statement
Given a list of tasks represented by characters and a positive integer `n` representing the cooldown period, you need to find the least amount of time required to finish all the tasks, ensuring that the same task cannot be executed again within the cooldown period.

[LeetCode Problem Link](https://leetcode.com/problems/task-scheduler/description/)

## Optimized C++ Solution
```cpp
#include <vector>
#include <algorithm>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        // Step 1: Count the frequency of each task
        vector<int> freq(26, 0);
        for (char task : tasks) {
            freq[task - 'A']++;
        }

        // Step 2: Find the task with the maximum frequency
        int maxFreq = *max_element(freq.begin(), freq.end());
        int maxCount = count(freq.begin(), freq.end(), maxFreq);

        // Step 3: Calculate the minimum time using the formula
        int partCount = maxFreq - 1;
        int partLength = n - (maxCount - 1);
        int emptySlots = partCount * partLength;
        int availableTasks = tasks.size() - maxFreq * maxCount;
        int idleTime = max(0, emptySlots - availableTasks);

        return tasks.size() + idleTime;
    }
};
```

## Explanation
### 1. Frequency Count
- Count the occurrences of each task using a frequency array (`freq`).

### 2. Maximum Frequency
- Identify the task with the maximum frequency (`maxFreq`) and count how many tasks have this frequency (`maxCount`).

### 3. Greedy Calculation
- Use the formula to calculate the idle slots:
  ```
  emptySlots = (maxFreq - 1) * (n - (maxCount - 1))
  ```
- Calculate how many tasks are available to fill those empty slots and adjust the idle time accordingly.

### 4. Final Result
- The total time is the sum of the tasks' size and the idle time:
  ```
  tasks.size() + idleTime
  ```

## Complexity
- **Time Complexity**: \(O(N + 26) \rightarrow O(N)\), where \(N\) is the number of tasks. Counting and finding the max in a fixed-size array are constant operations.
- **Space Complexity**: \(O(26) \rightarrow O(1)\), as the frequency array is always of size 26.

## Notes
This solution uses a greedy approach to minimize idle time while respecting the task cooldown constraint. The algorithm is efficient and concise, leveraging the fixed alphabet size (26) for optimization.
