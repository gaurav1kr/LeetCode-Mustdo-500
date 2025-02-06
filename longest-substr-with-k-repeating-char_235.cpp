class Solution 
{
public:
    int longestSubstring(string s, int k) 
    {
        return helper(s, 0, s.length(), k);
    }
    int helper(string s, int start, int end, int k) 
    {
        if (start >= end) return 0;

        vector<int> c (26, 0);
        for (int i = start; i < end; i++) c[s[i]-'a']++;

        for (int i = start; i < end; i++) 
	{
            if (c[s[i]-'a'] < k) 
	    {
                int j = i + 1;
                while (j < end && c[s[j]-'a'] < k) j++;
                return max(helper(s, start, i, k), helper(s, j, end, k));
            }
        }
        return end - start;
    }
};
