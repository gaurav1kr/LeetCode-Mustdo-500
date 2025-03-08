class Solution 
{
public:
    void solve(int ind,vector<int>&nums,vector<int>&ds,vector<vector<int>>&res,int k)
    {
        if(ind==nums.size())
        {
            if(ds.size()==k)
            {
                res.push_back(ds);
            }
            return;
        }
        ds.push_back(nums[ind]);
        solve(ind+1,nums,ds,res,k);
        ds.pop_back();
        solve(ind+1,nums,ds,res,k);
    }

    vector<vector<int>> combine(int n, int k) 
    {
        vector<vector<int>>res;
        vector<int>ds;
        vector<int>nums;
        int i;
        for(i=1;i<=n;i++)
        {
            nums.push_back(i);
        }
        solve(0,nums,ds,res,k);
        return res;
        
    }
};
