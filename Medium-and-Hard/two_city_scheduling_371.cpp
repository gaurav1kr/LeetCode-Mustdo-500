class Solution 
{
public:
    static bool compare(pair<int, int>& ele1,pair<int, int>& ele2) 
	{
        return ele1.second-ele1.first> ele2.second-ele2.first;
    }

    int twoCitySchedCost(std::vector<std::vector<int>>& costs) 
	{
        int cost = 0;
        int a = 0;
        int b = 0;
        vector<pair<int, int>> a1;

        for (auto& c : costs) 
		{
            a1.push_back({c[0], c[1]});
        }

        // Sort the vector of pairs using the custom compare function
        sort(a1.begin(), a1.end(), compare);

        for (const auto& c : a1) 
		{
            if (a != (costs.size() / 2)) 
			{
                cost += c.first;
                a++;
            } 
			else 
			{
                cost += c.second;
                b++;
            }
        }
        return cost;
    }
};
