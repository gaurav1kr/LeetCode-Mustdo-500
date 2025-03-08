//Iterative 
class Solution {
public:
    int arrayNesting(vector<int>& nums) {
        int max_len = 0;
        int n = nums.size();
        vector<bool> visited(n, false);

        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                int start = i, count = 0;
                while (!visited[start]) {
                    visited[start] = true;
                    start = nums[start];
                    ++count;
                }
                max_len = max(max_len, count);
            }
        }
        return max_len;
    }
};
//Recursive
class Solution {
    int max ;
public:
    Solution()
    {
        max = INT_MIN ;
    }
    void dfs(int elem , vector<int> nums , vector<bool> visited , int & c)
    {
        if (visited[elem])
            return ;
 
        if (c > max)
        {
            max = c ;
        }
 
        visited[elem] = true ;
        c++ ;
        dfs(nums[elem], nums , visited , c) ;
        c-- ;
        visited[elem] = false ;
    }
    int arrayNesting(vector<int>& nums) 
    {
        int size_num = sizeof(nums) ;
        vector<bool> visited(size_num,false) ;
        int count = 1 ;
        for(auto & elem:nums)
        {
            dfs(elem , nums , visited , count);
        }    
        return max ;
    }
};
