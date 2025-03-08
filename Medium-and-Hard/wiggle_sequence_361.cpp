class Solution 
{
public:
    int wiggleMaxLength(vector<int>& nums) 
    {
        int size=nums.size(), peak=1, valley=1;
        for(int i=1; i<size; ++i)
        {
                 if(nums[i]>nums[i-1])
                 {
                    peak = valley + 1;
                 }
                else if(nums[i]<nums[i-1]) 
                {
                    valley = peak + 1;
                }
        }
        return max(peak , valley );
        
    }
};
/*
The demand of the question is to get INCREASING and DECREASING sequences alternating.
Now re-arranging of array is not possible, Hence NUMBER OF TIMES ARRAY CHANGES from increasing to decreasing & vice versa REMAINS CONSTANT.

The question can now reduced to find number of times array changes pattern.
Further , it can be deduced that we have to COUNT THE NUMBER OF PEAKS AND VALLEY POINTS.
*/
