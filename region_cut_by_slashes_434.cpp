class Solution 
{
public:
    const vector<vector<vector<int>>> aux = {
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, // " "
        {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}, // "/"
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}  // "\\"
    };

    const vector<int> dx = {-1, 1, 0, 0};
    const vector<int> dy = {0, 0, -1, 1};

    void dfs(vector<vector<int>>& graph, int i, int j, int rows, int cols) 
    {
        if (i<0 || i>=rows || j<0 || j>=cols || graph[i][j]==1) 
        {
            return;
        }
        graph[i][j] = 1;
        for (int k=0; k<4; k++) 
        {
            dfs(graph, i+dx[k], j+dy[k], rows, cols);
        }
    }

    vector<vector<int>> generateGraph(const vector<string>& grid) 
    {
        int n = grid.size(), m = grid[0].size();
        vector<vector<int>> graph(3*n, vector<int>(3*m, 0));

        for (int i = 0; i < n; ++i) 
	{
            for (int j = 0; j < m; ++j) 
	    {
                int index = grid[i][j] == '/' ? 1 : (grid[i][j] == '\\' ? 2 : 0);
                const vector<vector<int>>& replacement = aux[index];
                for (int x=0; x<3; x++) 
		{
                    for (int y=0; y<3; y++) 
		    {
                        graph[3*i + x][3*j + y] = replacement[x][y];
                    }
                }
            }
        }

        return graph;
    }

    int regionsBySlashes(vector<string>& grid) 
    {
        int n = grid.size(), m = grid[0].size();
        vector<vector<int>> graph = generateGraph(grid);

        int noOfRegions = 0;

        for (int i = 0; i < 3 * n; ++i) 
	{
            for (int j = 0; j < 3 * m; ++j) 
	    {
                if (graph[i][j] == 0) 
		{
                    dfs(graph, i, j, 3 * n, 3 * m);
                    noOfRegions++;
                }
            }
        }

        return noOfRegions;
    }
};
