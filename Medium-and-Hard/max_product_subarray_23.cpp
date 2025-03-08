class Solution 
{
public:
    int maxProduct(vector<int>& arr)
    {
        int n = arr.size() ;
        if(n==1)
            return arr[0] ;
        if( (n==2) && arr[0]<0 && arr[1]<0 )
            return (arr[0]*arr[1]) ;
        int max_ending_here = 1; 
        int min_ending_here = 1; 
        int max_so_far = 1; 
        int flag = 0; 
    
        for (int i = 0; i < n; i++) 
        {    
            if (arr[i] > 0) 
            { 
                max_ending_here = max_ending_here * arr[i]; 
                min_ending_here = min(min_ending_here * arr[i], 1); 
                flag = 1; 
            } 
            else if (arr[i] == 0) 
            { 
                max_ending_here = 1; 
                min_ending_here = 1; 
            } 
            else 
            { 
                int temp = max_ending_here; 
                max_ending_here = max(min_ending_here * arr[i], 1); 
                min_ending_here = temp * arr[i]; 
            } 
            if (max_so_far < max_ending_here) 
                max_so_far = max_ending_here; 
     } 
        
    if (flag == 0 && max_so_far == 1) 
        return 0; 
    return max_so_far; 
    }
};

//Approach
The code solve the "Maximum Product Subarray" problem using dynamic programming. The problem requires finding the contiguous subarray within an array of integers that has the largest product.

Here's how the code works:

It initializes several variables to keep track of the maximum product subarray: max_ending_here, min_ending_here, and max_so_far. It also initializes a flag variable to check if any positive element exists in the array.

It iterates through the elements of the input array arr.

For each element arr[i], it updates max_ending_here and min_ending_here based on whether arr[i] is positive, negative, or zero. If arr[i] is positive, max_ending_here and min_ending_here are updated accordingly. If arr[i] is zero, both max_ending_here and min_ending_here are reset to 1. If arr[i] is negative, max_ending_here and min_ending_here are swapped and updated.

It updates max_so_far to the maximum of max_so_far and max_ending_here at each step.

Finally, it checks if no positive element exists in the array and if max_so_far is still 1. If both conditions are true, it returns 0; otherwise, it returns max_so_far.

This solution has a time complexity of O(n) since it iterates through the array only once. The space complexity is O(1) because it uses only a constant amount of additional space regardless of the input size.
