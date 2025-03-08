class Solution 
{
public:
    int GetPivotElem(vector<int> nums)
    {
        int start = 0 ;
        int end = nums.size() - 1 ;
        while(start < end)
        {
            int mid = start + (end-start)/2 ;
            
            if( (mid<end) && (nums[mid]> nums[mid+1]) )
            {
                return mid ;
            }
            else if( (mid > start) && (nums[mid]<nums[mid-1]) )
            {
                return mid-1 ;
            }
            else if(nums[mid] <= nums[start])
            {
                end = mid-1 ;
            }
            else
            {
                start = mid+1 ;
            }
        }
        return -1 ;
    }
    int BinarySearchUtil(vector<int> nums , int start, int end , int target)
    {
        while(start<=end)
        {
            int mid = start + (end-start)/2 ;
            if(nums[mid] == target)
            {
                return mid ;
            }
            if(nums[mid] > target)
            {
                end = mid-1 ;
            }
           else
            {
                start = mid+1 ;
            }
        }
        return -1 ;
    }
    int search(vector<int>& nums, int target) 
    {
        int size_nums = nums.size() ;
        int elem_search = -1 ;
        if(size_nums == 0)
        {
            return -1;
        }
        if(size_nums == 1)
        {
            if(nums[0] == target)
            {
                return 0 ;
            }
            else
            {
                return -1 ;
            }
        }
        
        int pivot = GetPivotElem(nums) ;
       /* if(pivot == -1)
        {
            return elem_search ;
        }*/
        elem_search = BinarySearchUtil(nums,0,pivot,target) ;
        
        if(elem_search == -1)
        {
            elem_search = BinarySearchUtil(nums,pivot+1,size_nums-1,target) ;
        }
        return elem_search ;
    }
};
