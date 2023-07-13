#include<iostream>
#include<vector>
using namespace std;
#define AXIS 4
class Solution
{
public:
    int dx[AXIS];
    int dy[AXIS];
    int n;
    Solution() : dx{ 0,-1,0,1 }, dy{ 1,0,-1,0 }
    {
    }
    bool dfs(int x, int y, int t, vector<vector<int>>& grid, vector<vector<int>>& visit) {
        visit[x][y] = 1;

        if (t < grid[x][y])
        {
            return false;
        }

        if (x == n - 1 && y == n - 1) {
            return true;
        }

        bool an = false;

        for (int i = 0; i < 4; i += 1)
        {
            int nx = x + dx[i];
            int ny = y + dy[i];

            if (nx >= 0 && ny >= 0 && nx < n && ny < n && grid[nx][ny] <= t && !visit[nx][ny]) {
                if (nx == n - 1 && ny == n - 1)
                {
                    an = true;
                }
                else
                {
                    an = an | dfs(nx, ny, t, grid, visit);
                }
            }
        }
        return an;
    }

    int swimInWater(vector<vector<int>>& grid) {
        n = grid.size();
        int lo = 0, hi = 1e5;
        while (lo < hi)
        {
            int mid = lo + (hi - lo) / 2;
            vector<vector<int>> visit(n + 1, vector<int>(n + 1, 0));
            if (dfs(0, 0, mid, grid, visit))
            {
                hi = mid;
            }
            else
            {
                lo = mid + 1;
            }
        }
        return hi;
    }
};