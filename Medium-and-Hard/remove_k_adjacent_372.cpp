class Solution 
{
public:
    string removeDuplicates(string s, int k) 
	{
        stack<pair<char, int>> st;
        string ans;
        for (int i = s.size() - 1; i >= 0; i--) 
		{
            char curr = s[i];
            if (!st.empty() && st.top().first == curr) 
			{
                st.top().second++;
                if (st.top().second == k) 
				{
                    st.pop();
                }
            } 
			else 
			{
                st.push({curr, 1});
            }
        }
        while (!st.empty()) 
		{
            char curr = st.top().first;
            while(st.top().second--)  
			{
				ans.push_back(curr);
			}
            st.pop();
        }
        return ans;
    }
};