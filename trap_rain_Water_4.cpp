class Solution 
{
public:
    int trap(vector<int>& arr) 
    {
        int n= arr.size() ;
        int result = 0; 
        int left_max = 0, right_max = 0; 
        int lo = 0, hi = n - 1; 

        while (lo <= hi) 
        { 
            if (arr[lo] < arr[hi]) 
            { 
                if (arr[lo] > left_max) 
                {
                    left_max = arr[lo]; 
                }
                else
                {
                    result += left_max - arr[lo]; 
                }
                lo++; 
            } 
            else 
            { 
                if (arr[hi] > right_max) 
                    right_max = arr[hi]; 
                else
                    result += right_max - arr[hi]; 
                hi--; 
            } 
        } 
        return result;     
    }
};
/*
The idea is to maintain two pointers, one starting from the beginning and one from the end, and keep track of the maximum height seen so far from both ends. Then, you can calculate the trapped water at each index based on the minimum of the maximum heights from both ends.
*/
