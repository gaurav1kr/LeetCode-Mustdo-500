class Solution 
{
public:
    int leastBricks(vector<vector<int>>& wall) 
    {
        int wall_height = wall.size();
        int ans = 0;
        unordered_map<int,int> s;
        for(int i=0; i<wall_height; i++)
        {
            int total_width = 0;
            for(int k=0; k<wall[i].size()-1; k++)
            {
                total_width = total_width + wall[i][k];
                s[total_width]++;
                ans = max(ans, s[total_width]);
            } 
        }
        return wall_height - ans;
    }
};
