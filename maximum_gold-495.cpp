#include<iostream>
#include<vector>
#define AXIS 4
using namespace std;
int max_gold = 0;
class Solution
{
    int x_axis[AXIS];
    int y_axis[AXIS];
public:
    Solution() : x_axis{ 0,-1,0,1 }, y_axis{ 1,0,-1,0 }
    {
    }
    void check_max(int i, int j, vector<vector<bool>> visited, vector<vector<int>> grid, int row_grid, int col_grid, int  sum)
    {
        if (i < 0 || i >= row_grid || j < 0 || j >= col_grid || !grid[i][j] || visited[i][j])
            return;
        visited[i][j] = true;
        sum += grid[i][j];
        if (sum > max_gold)
        {
            max_gold = sum;
        }

        for (int k = 0; k < AXIS; k++)
        {
            check_max(i + x_axis[k], j + y_axis[k], visited, grid, row_grid, col_grid, sum);
        }
        visited[i][j] = false;
    }
    int getMaximumGold(vector<vector<int>>& grid)
    {
        int row_grid = grid.size();
        int col_grid = grid[0].size();
        max_gold = 0;
        vector<vector<bool>> visited(row_grid, vector<bool>(col_grid, false));
        for (int i = 0; i < row_grid; i++)
        {
            for (int j = 0; j < col_grid; j++)
            {
                if (grid[i][j])
                {
                    int sum = 0;
                    check_max(i, j, visited, grid, row_grid, col_grid, sum);
                }
            }
        }
        return max_gold;
    }
};