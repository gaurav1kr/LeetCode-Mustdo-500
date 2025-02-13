
# Solution for Course Schedule II

This document contains an optimized and concise C++ solution for the **Course Schedule II** problem from LeetCode.

---

## Problem Description

The problem can be found at: [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/description/)

### Objective
Given the number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses. If it is impossible to finish all courses, return an empty array.

---

## Optimized C++ Solution
The solution uses **Kahn's Algorithm** for topological sorting.

```cpp
#include <vector>
#include <queue>

class Solution {
public:
    std::vector<int> findOrder(int numCourses, std::vector<std::vector<int>>& prerequisites) {
        std::vector<int> inDegree(numCourses, 0), result;
        std::vector<std::vector<int>> adjList(numCourses);

        // Build adjacency list and compute in-degrees
        for (const auto& prereq : prerequisites) {
            adjList[prereq[1]].push_back(prereq[0]);
            ++inDegree[prereq[0]];
        }

        // Push courses with no prerequisites into the queue
        std::queue<int> q;
        for (int i = 0; i < numCourses; ++i)
            if (inDegree[i] == 0) q.push(i);

        // Process the courses
        while (!q.empty()) {
            int course = q.front();
            q.pop();
            result.push_back(course);

            for (int next : adjList[course]) {
                if (--inDegree[next] == 0) q.push(next);
            }
        }

        // Check if all courses can be completed
        return result.size() == numCourses ? result : std::vector<int>();
    }
};
```

---

## Explanation

1. **Graph Representation**:
   - The courses and their dependencies are represented as a graph using an adjacency list.

2. **In-Degree Array**:
   - Maintain an array `inDegree` to track the number of prerequisites for each course.

3. **Queue Initialization**:
   - Start with all courses that have zero prerequisites (in-degree = 0).

4. **Topological Sorting**:
   - Iteratively process courses from the queue:
     - Add the current course to the result list.
     - Reduce the in-degree of its dependent courses.
     - If a dependent course's in-degree becomes zero, add it to the queue.

5. **Cycle Detection**:
   - After processing, if the result does not contain all courses, it means there's a cycle, and completing all courses is impossible.

---

## Complexity Analysis

- **Time Complexity**: **O(V + E)**
  - `V`: Number of courses (nodes in the graph)
  - `E`: Number of prerequisites (edges in the graph)

- **Space Complexity**: **O(V + E)**
  - Space is used for the adjacency list and in-degree array.

---

## Example

### Input:
```cpp
numCourses = 4;
prerequisites = {{1, 0}, {2, 0}, {3, 1}, {3, 2}};
```

### Output:
```cpp
[0, 1, 2, 3]  // Or [0, 2, 1, 3]
```

---

## Notes
- The solution efficiently handles cycles and ensures only valid orderings are returned.
- Kahn's Algorithm ensures a reliable topological sort when one exists.
- In cases where it's impossible to finish all courses, the solution gracefully returns an empty array.
