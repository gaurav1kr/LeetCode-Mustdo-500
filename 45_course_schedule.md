## Course Schedule Problem (LeetCode)

### **Problem Statement:**
Given `numCourses` and a list of `prerequisites`, determine if it is possible to finish all courses.

---

### **Algorithm Explanation:**
1. **Build an adjacency list** representation of the course prerequisites.
2. **Compute in-degrees** for all nodes (i.e., number of prerequisites each course has).
3. **Use a queue** (BFS) to process nodes with zero in-degree (no prerequisites).
4. **Process the queue**:
   - Remove a course from the queue.
   - Reduce the in-degree of its dependent courses.
   - If any course reaches zero in-degree, add it to the queue.
5. **Check if all courses are processed**. If yes, return `true`; otherwise, return `false` (cycle detected).

---

### **C++ Code:**
```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> adj(numCourses);  // Adjacency list
        vector<int> inDegree(numCourses, 0); // In-degree array
        
        // Build the graph
        for (const auto& pre : prerequisites) {
            adj[pre[1]].push_back(pre[0]);  // pre[1] -> pre[0]
            inDegree[pre[0]]++;
        }
        
        queue<int> q;
        int count = 0;  // To track number of processed courses
        
        // Add all courses with zero in-degree to the queue
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                q.push(i);
            }
        }
        
        // Process the courses in topological order
        while (!q.empty()) {
            int course = q.front();
            q.pop();
            count++;

            // Reduce in-degree of dependent courses
            for (int next : adj[course]) {
                if (--inDegree[next] == 0) {
                    q.push(next);
                }
            }
        }

        // If we processed all courses, return true
        return count == numCourses;
    }
};
```

---

### **Time & Space Complexity:**
- **Time Complexity**: `O(V + E)`, where **V** is the number of courses and **E** is the number of prerequisites.
- **Space Complexity**: `O(V + E)` due to the adjacency list and in-degree array.

---

### **Alternative Approach (DFS - Cycle Detection)**
Another approach is using **DFS with cycle detection**. Let me know if you’d like that version as well! 🚀
