ass Solution 
{
    using p = tuple<int, int, int>;

public:
    int minimumEffortPath(vector<vector<int>>& A) 
    {
        int m = A.size(), n = A[0].size();
        priority_queue<p, vector<p>, greater<p>> q;

        vector<vector<int>> Eff(m, vector<int>(n, INT_MAX));
        q.push({0, 0, 0}); Eff[0][0]=0;

        vector<pair<int, int>> dirn = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        while (!q.empty()) 
        {
            auto [eff, i, j] = q.top();
            q.pop();

            if (eff > Eff[i][j])
                continue;

            for (auto& dir : dirn) 
            {
                int x = i + dir.first, y = j + dir.second;
                if (x >= 0 && x < m && y >= 0 && y < n) 
                {
                    int diff = abs(A[x][y] - A[i][j]), effort = max(eff, diff);
                    if(effort<Eff[x][y])
                    {
                        Eff[x][y] = effort, q.push({effort, x, y});
                    }
                }
            }
        }
        return Eff[m - 1][n - 1];
    }
};
