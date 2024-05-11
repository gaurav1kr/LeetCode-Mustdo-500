int x_axis[] = { 0,-1,0,1 };
int y_axis[] = { 1,0,-1,0 };
class Solution
{
public:
    int numIslands(vector<vector<char>>& grid)
    {
        int row = grid.size();
        int col = grid[0].size();
        int no_of_islands = 0;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                if (grid[i][j] == '1')
                {
                    no_of_islands++;
                    dfs(i, j, grid, row, col);
                }
            }
        }
        return no_of_islands;
    }
    void dfs(int i, int j, vector<vector<char>> & grid, int row, int col)
    {
        if (( (i < 0) || (i > row - 1) || (j < 0) || (j > col - 1) || grid[i][j] != '1') )
            return;
        
        grid[i][j] = 'x';

        for (int k = 0; k < 4; k++)
        {
            dfs(i + x_axis[k], j + y_axis[k], grid, row, col);
        }
    }
};
