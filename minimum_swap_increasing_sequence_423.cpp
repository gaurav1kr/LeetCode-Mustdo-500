class Solution {
public:
    int minSwap(vector<int>& nums1, vector<int>& nums2) 
	{
        int ans0 = 0;
        int ans1 = 0;
        int ans00 = 0;
        for(int i = nums1.size()-1;i>=0;i--)
		{
            for(int j = 0;j<2;j++){
                int ans = INT_MAX;
                if(j)
				{
                    if(i==0||nums1[i]>nums2[i-1]&&nums2[i]>nums1[i-1])
					{
                        ans = min(ans,ans00);
                    }
                    if(i==0||nums1[i]>nums1[i-1]&&nums2[i]>nums2[i-1])
					{
                        ans = min(ans,1+ans1);
                    }
                }
				else
				{
                    if(i==0||nums1[i]>nums1[i-1]&&nums2[i]>nums2[i-1])
					{
                        ans = min(ans,ans0);
                    }
                    if(i==0||nums1[i]>nums2[i-1]&&nums2[i]>nums1[i-1])
					{
                        ans = min(ans,1+ans1);
                    }
                }
                if(j)
				{
                    ans1 = ans;
                }
				else
				{
                    ans00 = ans0;
                    ans0 = ans;
                }
            }
        }
        return min(ans1,ans0);
        
    }
};