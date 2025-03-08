class Solution
{
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals)
    {
        vector<vector<int>> result;
        size_t interval_size = intervals.size();
        sort(intervals.begin(), intervals.end());
        for (int i = 0; i < interval_size; i++)
        {
            if (result.empty() || (result.back()[1] < intervals[i][0]))
            {
                vector<int> v = { intervals[i][0] , intervals[i][1] };
                result.push_back(v);
            }
            else
            {
                result.back()[1] = max(result.back()[1], intervals[i][1]);
            }
        }
        return result;
    }
};
