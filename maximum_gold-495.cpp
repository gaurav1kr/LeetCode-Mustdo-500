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
    bool check_equal_elem(vector<vector<int>> grid , int row_grid , int col_grid)
    {
        int source = grid[0][0];
        for(int i=0;i<row_grid;i++)
        for (int j = 1; j < col_grid; j++)
        {
            if (grid[i][j] != source) return false;
        }
        return true;
    }
    void check_max(int i, int j, vector<vector<int>> grid, int row_grid, int col_grid, int  sum)
    {
        sum += grid[i][j];
        grid[i][j] = -grid[i][j];
        if (sum > max_gold)
        {
            max_gold = sum;
        }

        for (int k = 0; k < AXIS; k++)
        {
            int new_x = i + x_axis[k];
            int new_y = j + y_axis[k];
            if (new_x >= 0 && new_y >= 0 && new_x < row_grid && new_y < col_grid && grid[new_x][new_y] > 0)
                check_max(new_x, new_y, grid, row_grid, col_grid, sum);
        }
        grid[i][j] = -grid[i][j];
    }
    int getMaximumGold(vector<vector<int>>& grid)
    {
        int row_grid = grid.size();
        int col_grid = grid[0].size();
        max_gold = 0;

        bool equal_elem = check_equal_elem(grid, row_grid, col_grid);
        if (equal_elem)
        {
            return row_grid * col_grid * grid[0][0];
        }

        for (int i = 0; i < row_grid; i++)
        {
            for (int j = 0; j < col_grid; j++)
            {
                if (grid[i][j])
                {
                    int sum = 0;
                    check_max(i, j, grid, row_grid, col_grid, sum);
                }
            }
        }
        return max_gold;
    }
};

int main()
{
    vector<vector<int>> grid = { {1,1,1,1,1},
                                 {1,1,1,1,1},
                                 {1,1,1,1,1},
                                 {1,1,1,1,1},
                                 {1,1,1,1,1} 
                               };
    Solution sol;
    cout << sol.getMaximumGold(grid);
    return 0;
}