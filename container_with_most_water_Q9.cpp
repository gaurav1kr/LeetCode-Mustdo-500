class Solution {
public:
    int min1(int a , int b)
    {
        return ((a<b)?a:b) ;
    }
    int max1(int a , int b)
    {
        return ((a>b)?a:b) ;
    }
    int maxArea(vector<int>& A) 
    {
        int l = 0;
        int len = A.size() ;
        int r = len - 1;
        int area = 0;

        while (l < r)
        {
            area = max1(area, min1(A[l], A[r]) * (r - l));

            if (A[l] < A[r])
                l += 1;
            else
                r -= 1;
        }
        return area;   
    }
};
/*
The overall idea of the maxArea method is to start with the widest possible container and gradually narrow it down while maximizing the area enclosed. This is achieved by moving the pointers l and r towards each other, always choosing the pointer corresponding to the shorter line to ensure that there is a possibility of increasing the area.
*/
