class Solution 
{
public:
    int maxProfit(int k, vector<int>& prices) 
    {
        vector<int> states(k*2, INT_MIN);
        states[0] = -prices[0];

        for (auto price : prices) 
	{
            states[0] = max(states[0], -price);
            int sign = 1;
            // for each k, we have 2 new states
            for (int i=1; i<k*2; ++i) 
	    {
                states[i] = max(states[i], states[i-1] + sign * price);
                sign *= -1;
            } 
        }

        return states[k*2-1];
    }
};
