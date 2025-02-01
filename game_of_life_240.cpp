class Solution 
{
public:
    int cntNeighbours(int i, int j, int m, int n, vector<vector<int>> board)
    {
        int cnt = 0;
        vector<int> dx = {-1,-1,-1,0,0,1,1,1};
        vector<int> dy = {-1,0,1,-1,1,-1,0,1};
        for(int k = 0; k < 8;k++)
        {
            int r, c;
            r = i + dx[k]; c = j + dy[k];
            if(r >= 0 && r < m && c >=0 && c < n)
            {
                if(board[r][c]==1||board[r][c] == 2)
                    cnt++;
            }
        }
        return cnt;
    }

    void gameOfLife(vector<vector<int>>& board) 
    {
        int m = board.size();
        int n = board[0].size();
        //add temp values
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                int cnt = cntNeighbours(i,j,m,n,board);
                if((board[i][j]== 1 ) && cnt < 2)
                    board[i][j] = 2;
                else if(board[i][j] == 0 && cnt == 3) //dead cell with 3 live neighbours
                {
                    board[i][j] = 3;
                }
                else if((board[i][j] == 1) &&cnt > 3)
                    board[i][j] = 2;
            }
        }
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                if(board[i][j] == 2)
                    board[i][j] = 0;
                if(board[i][j] == 3)
                    board[i][j] = 1;
            }
        }
        
    }
};
