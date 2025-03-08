class Solution 
{
public:
    bool canJump(vector<int>& A) 
    {
       int n = A.size() ;
       if(n <= 1)
            return true;
 
       int max = A[0]; 
 
        for(int i=0; i<n; i++)
        {
            if(max <= i && A[i] == 0) 
                return false;
            if(i + A[i] > max)
            {
                max = i + A[i];
            }
            if(max >= n-1) 
                return true;
        }
 
        return false;    
    }
};
