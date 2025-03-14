class Solution 
{
public:
    bool check(string s1,string s2)
    {
        if(s1.length()<s2.length())
            return false;

        int i = 0, j = 0;
        while(i<s1.length() && j<s2.length())
        {
            if(s1[i]==s2[j])
            {
                i++;
                j++;
            }
            else
            {
                i++;
            }
        }
        return (j==s2.length());
    }
    int numMatchingSubseq(string s, vector<string>& words)
    {
        int ans = 0;
        unordered_map<string,int> mp;
        for(auto &i: words)
        {
            if(mp.count(i))
            {
                ans += mp[i];
            }
            else
            {
                ans += mp[i] = check(s,i);
            }
        }
        return ans;
    }
};