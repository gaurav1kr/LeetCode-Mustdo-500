class Solution 
{
public:
    int min_three(int a , int b , int c)
    {
        if(a<b)
        {
            if(a<c)
            {
                return a ;
            }
            else
            {
                return c ;
            }
        }
        else if(b<c)
        {
            return b ;
        }
        else
        {
            return c ;
        }
    }
    int countSquares(vector<vector<int>>& matrix) 
    {
        int row = matrix.size() ;
        int col = matrix[0].size() ;
        int sum = 0;
        int i = 0 , j = 0 ;
        vector<vector<int>> dp_matrix(row, vector<int>(col,0)) ;
        for(i=0;i<row;i++)
        {
            dp_matrix[i][0] = matrix[i][0] ;
        }
        
        for(i=0;i<col;i++)
        {
            dp_matrix[0][i] = matrix[0][i] ;
        }
        for(i=1;i<row;i++)
        {
            for(j=1;j<col;j++)
            {
                if(matrix[i][j])
                dp_matrix[i][j] = min_three(dp_matrix[i-1][j-1] , dp_matrix[i][j-1] , dp_matrix[i-1][j]) + matrix[i][j] ;
            }
        }
        for(i=0;i<row;i++)
        {
            for(j=0;j<col;j++)
            {
                sum += dp_matrix[i][j] ;   
            }
        }
        return sum ;
    }
};