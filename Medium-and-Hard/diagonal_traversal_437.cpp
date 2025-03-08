class Solution 
{
public:
        vector<int> findDiagonalOrder(vector<vector<int>>& matrix) 
        {
            vector<int>ans;
            map<int,vector<int>>m;
            for(int i=0;i<matrix.size();i++)
            {
                for(int j=0;j<matrix[0].size();j++)
                {
                    m[i+j].push_back(matrix[i][j]);
                }
            }
            
           for(auto i:m) 
           {
            if((i.first)%2 == 0) 
                reverse(i.second.begin(), i.second.end()); 
            
            for(auto j: i.second) 
            ans.push_back(j);
           }

            return ans;
        }
  
};
