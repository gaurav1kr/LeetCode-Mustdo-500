class Solution 
{
public:
   vector<vector<int>> updateMatrix(vector<vector<int>>& mat) 
   {
    int rows = mat.size(), cols = mat[0].size();
    vector<vector<int>> dist(rows, vector<int>(cols, INT_MAX));
    queue<pair<int, int>> q;

    // Step 1: Initialize the queue with all 0 cells
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
	{
            if (mat[i][j] == 0) 
	    {
                dist[i][j] = 0;
                q.push({i, j});
            }
        }
    }

    // Step 2: Directions array for exploring neighbors
    vector<pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    // Step 3: Perform BFS
    while (!q.empty()) 
    {
        auto [curRow, curCol] = q.front();
        q.pop();

        for (auto [dr, dc] : directions) 
	{
            int newRow = curRow + dr;
            int newCol = curCol + dc;

            // Check if the new cell is within bounds and can be updated
            if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols) 
	    {
                if (dist[newRow][newCol] > dist[curRow][curCol] + 1) 
		{
                    dist[newRow][newCol] = dist[curRow][curCol] + 1;
                    q.push({newRow, newCol}); // Add the updated cell to the queue
                }
            }
        }
    }

    return dist;
}
};
