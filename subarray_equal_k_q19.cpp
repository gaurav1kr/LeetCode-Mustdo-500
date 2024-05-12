class Solution 
{
public:
    int subarraySum(vector<int>& nums, int k) 
    {
        size_t nums_size = nums.size() ;
        unordered_map<int,int> prev ;
        int s = 0 ;
        int res = 0 ;
        for(auto & e: nums)
        {
            s += e ;
            if(s == k) res++ ;
            int rem = s-k ;
            if(prev.find(rem) != prev.end())
            {
                res += prev[rem] ;
            }
            prev[s]++ ;
        }
        return res ;
    }
};
