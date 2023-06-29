#define MAX 10001
class Solution {
    int hash[MAX];
public:
    Solution()
    {
        for (int i = 0; i < MAX; i++)
        {
            hash[i] = 0;
        }
    }
    vector<int> findErrorNums(vector<int>& nums)
    {
        vector<int> res;
        int size_nums = nums.size();
        for (auto& i : nums)
        {
            hash[i]++;
            if (hash[i] == 2) res.push_back(i);
        }
        for (int i = 1; i < MAX; i++)
        {
            if (hash[i] == 0)
            {
                res.push_back(i);
                break;
            }
        }

        return res;
    }
};