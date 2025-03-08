
## Rotting Oranges Problem - Optimized C++ Solution

### Problem Description
You are given an `m x n` grid where each cell can have one of three values:
- `0` representing an empty cell,
- `1` representing a fresh orange,
- `2` representing a rotten orange.

Every minute, any fresh orange that is adjacent (4-directionally) to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return `-1`.

---

### Solution
The following is a concise and optimized C++ solution using **Breadth-First Search (BFS)**:

```cpp
#include <vector>
#include <queue>
using namespace std;

int orangesRotting(vector<vector<int>>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    queue<pair<int, int>> q;
    int freshOranges = 0, minutes = 0;

    // Initialize the queue with all rotten oranges and count fresh oranges
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] == 2) {
                q.push({r, c});
            } else if (grid[r][c] == 1) {
                ++freshOranges;
            }
        }
    }

    // Directions for 4 adjacent cells (right, down, left, up)
    vector<pair<int, int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    // Perform BFS to spread the rot
    while (!q.empty() && freshOranges > 0) {
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            auto [r, c] = q.front();
            q.pop();

            for (auto [dr, dc] : directions) {
                int nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                    grid[nr][nc] = 2; // Mark as rotten
                    --freshOranges;
                    q.push({nr, nc});
                }
            }
        }
        ++minutes;
    }

    // If there are still fresh oranges, return -1
    return freshOranges > 0 ? -1 : minutes;
}
```

---

### Explanation

#### 1. **Initialization**:
- Use a `queue` to store the coordinates of all rotten oranges (`2`).
- Count the total number of fresh oranges (`1`) in the grid using the variable `freshOranges`.

#### 2. **Breadth-First Search (BFS)**:
- Process all rotten oranges in the queue:
  - For each orange, check its four adjacent cells (right, down, left, up).
  - If an adjacent cell contains a fresh orange (`1`):
    - Mark it as rotten (`2`).
    - Decrease the count of `freshOranges`.
    - Add it to the queue for processing in the next minute.
- After processing all oranges in the current layer, increment the `minutes` counter.

#### 3. **Termination**:
- If all fresh oranges are rotted, return the `minutes` elapsed.
- If fresh oranges remain after processing all rotten ones, return `-1`.

---

### Complexity Analysis

- **Time Complexity**: 
  - \(O(n \times m)\), where \(n\) is the number of rows and \(m\) is the number of columns. Each cell is processed at most once.

- **Space Complexity**: 
  - \(O(n \times m)\), due to the queue used to store the coordinates of rotten oranges.

---

### Example Usage
```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<vector<int>> grid = {
        {2, 1, 1},
        {1, 1, 0},
        {0, 1, 1}
    };

    int result = orangesRotting(grid);
    cout << "Minimum minutes to rot all oranges: " << result << endl;
    return 0;
}
```

#### Input:
```
[[2, 1, 1],
 [1, 1, 0],
 [0, 1, 1]]
```

#### Output:
```
Minimum minutes to rot all oranges: 4
```
