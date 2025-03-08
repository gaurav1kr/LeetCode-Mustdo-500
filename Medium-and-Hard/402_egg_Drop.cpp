//Memoization + Binary Search
class Solution 
{
public:
    //Time: O(n*k*logn), Space: O(n*k)
    int helper(int k, int n, vector<vector<int>>& memo)
	{
        if(n == 0 || n == 1) return n;
		
        if(k == 1) return n;
        
        if(memo[k][n] != -1) return memo[k][n];
        
        int mn = INT_MAX, low = 0, high = n, temp = 0;
        
        while(low<=high){
            
            int mid = (low + high)/2;
            
            int left = helper(k-1, mid-1, memo);
            int right = helper(k, n-mid, memo);
            
            temp = 1 + max(left, right);
            
            if(left < right) 
                low = mid+1;
            else 
                high = mid-1;     
    
            mn = min(mn, temp); 
        }
        return memo[k][n] = mn;
    }
    
    int superEggDrop(int k, int n) 
	{
        vector<vector<int>> memo(k+1, vector<int>(n+1, -1));
        return helper(k, n, memo);
    }
};

// The approach to solving the Egg Drop Problem in this C++ code can be summarized as follows:

// Dynamic Programming with Memorization:

// A memo table is used to store the results of previously computed subproblems, avoiding redundant calculations and speeding up the solution.
// Recursive Problem Breakdown:

// For a given number of eggs k and floors n, the problem is recursively broken down into two cases:
// Egg breaks: Check the floors below (mid - 1) with k - 1 eggs.
// Egg doesn't break: Check the floors above (n - mid) with the same k eggs.
// The worst-case attempts are calculated for each floor and minimized using binary search.

// Binary Search to Optimize:
// Binary search is used to find the optimal floor (mid) to drop the egg, reducing the range of floors and minimizing the maximum number of attempts needed in the worst case.

// Base Cases:
// If there is only one egg, the solution requires linear attempts (n).
// If there are no or only one floor, the solution requires n attempts (0 or 1).
// This method reduces the overall time complexity by narrowing the search space through binary search while storing intermediate results for faster retrieval.