class NumMatrix 
{
	private:
		int row,col;
		vector<vector<int>> prefix;
	public:
		NumMatrix(vector<vector<int>>& matrix) 
		{
			row = matrix.size(), col = matrix[0].size();
			prefix = vector<vector<int>>(row+1, vector<int>(col+1, 0));
			for(int i=0;i<matrix.size();i++)
			{
				for(int j=0;j<matrix[0].size();j++)
				{
					prefix[i+1][j+1]=matrix[i][j]+prefix[i][j+1]+prefix[i+1][j]-prefix[i][j];
				}
			}
		}
    
		int sumRegion(int row1, int col1, int row2, int col2) 
		{
			return prefix[row2+1][col2+1]-prefix[row2+1][col1]-prefix[row1][col2+1]+prefix[row1][col1];
		}
};

// This formula works by:

// Adding the current element matrix[i][j].
// Adding the sum of all elements in the previous row up to column j: prefix[i+1][j].
// Adding the sum of all elements in the previous column up to row i: prefix[i][j+1].
// Subtracting the overlapping region, which was added twice: prefix[i][j].

// matrix = [
  // [3, 0, 1],
  // [5, 6, 3],
  // [1, 2, 0]
// ]

// prefix = [
  // [0, 0, 0, 0],
  // [0, 3, 3, 4],
  // [0, 8, 14, 18],
  // [0, 9, 17, 21]
// ]

// For a query like sumRegion(1, 1, 2, 2), which asks for the sum of the submatrix:
// The result would be calculated as:-
// prefix[3][3] - prefix[1][3] - prefix[3][1] + prefix[1][1] = 21 - 4 - 9 + 3 = 11

// Time Complexity:
// Preprocessing (constructor): O(m * n) where m is the number of rows and n is the number of columns.
// Query (sumRegion): Each query is computed in constant time, i.e., O(1).


