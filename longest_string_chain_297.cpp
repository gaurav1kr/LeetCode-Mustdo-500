class Solution 
{
public:
    int longestStrChain(vector<string>& words) 
	{
        sort(words.begin(), words.end(), [&](const string &a, const string &b) 
		{
            return a.length() < b.length();
        });
        int N = words.size();
        unordered_set<string> s(words.begin(), words.end());
        unordered_set<string> visited;
        int ans = 0;
        for(int i = N-1;i>=0;i--)
        {
            if(visited.find(words[i]) == visited.end()) 
			{
                queue<string> Q;
                Q.push(words[i]);
                visited.insert(words[i]);
                int count = 0;
                while(!Q.empty())
                {
                    int sz = Q.size();
                    while(sz--)
                    {
                        string str = Q.front();
                        //cout<<str<<" ";
                        Q.pop();
                        for(int j=0;j<str.length();j++)
                        {
                            string temp = str;
                            temp.erase(j, 1);
                            if(s.find(temp) != s.end() && visited.find(temp) == visited.end())
                            {
                                Q.push(temp);
                                visited.insert(temp);
                            }
                        }
                    }
                    count++;
                    ans = max(ans, count);
                }
            }
        }
        return ans;
    }
};