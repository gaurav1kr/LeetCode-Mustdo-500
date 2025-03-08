class Solution
{
public:
    double f(int n, int k, int nr, int nc, vector<vector<vector<double>>>& dp)
    {
        if (nc < 0 || nr < 0 || nc >= n || nr >= n) return 0;
        if (k == 0) return 1;
        if (dp[nr][nc][k] != -1) return dp[nr][nc][k];
        int r[] = { -2,-2,-1,1,2,2,1,-1 };
        int c[] = { -1,1,2,2,1,-1,-2,-2 };
        double prob = 0;
        for (int i = 0; i < 8; i++)
        {
            prob += f(n, k - 1, nr + r[i], nc + c[i], dp) / 8.0;
        }
        return dp[nr][nc][k] = prob;
    }

    double knightProbability(int n, int k, int row, int col) 
    {
        if (k == 0) return 1;
        vector<vector<vector<double>>> dp(n, vector<vector<double>>(n, vector<double>(k + 1, -1)));
        return f(n, k, row, col, dp);
    }
};
