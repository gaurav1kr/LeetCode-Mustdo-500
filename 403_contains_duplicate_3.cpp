class Solution 
{
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int z) 
	{
        set<int> window;
        for (int i = 0; i < nums.size(); i++) 
		{
            if (i > k) 
			{
				window.erase(nums[i-k-1]);
			}
            auto it = window.lower_bound( nums[i] - z);
			
            if (it != window.end() && *it <= ( nums[i] + z))
			{
                return true;
			}
            window.insert(nums[i]);
        }
        return false;
    }
};
// Summary of the Code Flow
// The code maintains a sliding window of size k by using a set called window.

// As the loop iterates over the array, the oldest element in the window (if necessary) is removed 
// to ensure the window size doesn't exceed k.

// For each element, the code uses the lower_bound function to find if there exists any element in the
// window such that the absolute difference with the current element is less than or equal to z.

// If such an element is found, the function returns true. Otherwise, the loop continues.

// If no such pair is found by the end of the loop, the function returns false.

// This approach ensures that the solution runs efficiently, with a time complexity of approximately 
// O(nlogk), where n is the size of the nums array, and k is the size of the sliding window.