#define MAX 26
#define INDEX(c) (int)c - (int)'a'
class Solution
{
public:  
    int dictionaryContains(string word , vector<string> dictionary)
    {
        int size = dictionary.size();
        for (int i = 0; i < size; i++)
            if (dictionary[i].compare(word) == 0)
                return true;
        return false;
    }

    bool wordBreak(string s, vector<string>& wordDict)
    {
        int n = s.size();
        if (n == 0)
            return true;
        vector<bool> dp(n + 1, 0);
        vector<int> matched_index;
        matched_index.push_back(-1);

        for (int i = 0; i < n; i++) 
        {
            int msize = matched_index.size();
            int f = 0;
            for (int j = msize - 1; j >= 0; j--) 
            {
                string sb = s.substr(matched_index[j] + 1, i - matched_index[j]);

                if (dictionaryContains(sb,wordDict))
                {
                    f = 1;
                    break;
                }
            }
            if (f == 1) 
            {
                dp[i] = 1;
                matched_index.push_back(i);
            }
        }
        return dp[n - 1];
    }
};
// Code flow explanation
Initialization:

The method wordBreak is called with a string s and a vector of strings wordDict.
It initializes n to the length of the string s.
If the string is empty (n == 0), it returns true (an empty string can be segmented trivially).
A dp vector of size n+1 is initialized to false (0). This vector is used to track if the substring up to index i can be segmented.
A matched_index vector is initialized with -1, representing the position before the start of the string.
Main Loop:

Iterate through each character in the string s using the index i.
For each i, check substrings ending at i using previously matched indices stored in matched_index.
Substring Matching:

For each index j in matched_index, extract the substring sb from s[matched_index[j] + 1] to s[i].
Check if sb exists in the wordDict using dictionaryContains.
If a match is found (f = 1), mark dp[i] as true and add i to matched_index.
Result:

After processing, return the value of dp[n-1]. If dp[n-1] is true, it means the string s can be segmented into words from wordDict.
Example Walkthrough
Consider s = "leetcode" and wordDict = ["leet", "code"]:

Initialize:

n = 8, dp = [0, 0, 0, 0, 0, 0, 0, 0, 0], matched_index = [-1].
Process each character in "leetcode":

For i = 3 (end of "leet"):
Check substrings ("l", "le", "lee", "leet").
"leet" is in wordDict, so dp[3] = 1, matched_index = [-1, 3].
For i = 7 (end of "code"):
Check substrings ("c", "co", "cod", "code").
"code" is in wordDict, so dp[7] = 1, matched_index = [-1, 3, 7].
Final state: dp[7] = 1, so return true.

//Time and time complexiety
Time Complexity: O(n^2)
Space Complexity: O(n + m)

