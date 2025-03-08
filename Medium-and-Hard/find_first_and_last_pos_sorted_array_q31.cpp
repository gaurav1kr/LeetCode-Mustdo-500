class Solution 
{
public:
    vector<int> searchRange(vector<int>& nums, int target) 
    {
        vector<int> result ;
        int startindex = GetPos(nums , target , true) ;    
        int endindex = GetPos(nums , target , false) ;
        result.push_back(startindex) ;
        result.push_back(endindex) ;
        return result ;
    }
    
    int GetPos(vector<int> nums , int target , bool isstartindex)
    {
        int size_of_nums = nums.size() ;
        int low = 0 ;
        int ans = -1 ;
        int high = size_of_nums - 1 ;
        
        while(low <= high)
        {
            int mid = low + (high-low)/2 ;
            if(nums[mid] > target)
            {
                high = mid -1 ;
            }
            else if(nums[mid] < target)
            {
                low = mid+1 ;
            }
            else
            {
                ans = mid ;
                if(isstartindex)
                {
                    high = mid-1 ;
                }
                else
                {
                    low = mid+1 ;
                }
            }
        }
        return ans ;
    }
};
