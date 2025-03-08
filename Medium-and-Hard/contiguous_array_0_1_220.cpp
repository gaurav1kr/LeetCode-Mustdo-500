class Solution 
{
public:
    int findMaxLength(vector<int>& nums) 
    {
        for(auto &it : nums) it = (it == 1) ? 1 : -1;
        map<int, int> lastOccurance;

        lastOccurance[0] = -1;
        int ans = 0, sum = 0;

        for(int i=0; i<nums.size(); i++)
        {
            sum += nums[i];
            if(lastOccurance.find(sum) != lastOccurance.end())
                ans = max(ans, i - lastOccurance[sum] );
            else
                lastOccurance[sum] = i;
        }
        return ans;
    }
};

//According to question we have to find the length of the largest subarray having equal no of zeros and ones, but if we replace zeros with -1, question will become find the length of largest of subarray having sum zero, because equal no 1 and -1 will give total sum equal to 0.

