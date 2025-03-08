#include<iostream>
#include<vector>
#include<unordered_map>
using namespace std;
class Solution {
public:
    void normalize(string& s)
    {
        unordered_map<char, char> mp;
        char c = 'a';
        for (auto& i : s)
        {
            if (mp.find(i) == mp.end())
            {
                mp[i] = c++;
            }
        }

        for (int i = 0; i < s.length(); i++)
        {
            s[i] = mp[s[i]];
        }

    }
    vector<string> findAndReplacePattern(vector<string>& w, string p) {
        normalize(p);
        vector<string> res;
        for (auto& s : w)
        {
            string t = s;
            normalize(t);
            if (p == t)
                res.push_back(s);
        }
        return res;
    }
};