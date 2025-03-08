//https://leetcode.com/problems/unique-email-addresses/
#include<iostream>
#include<vector>
#include<unordered_map>
#include<string>
using namespace std;

class Solution {
public:
    int numUniqueEmails(vector<string>& emails)
    {
        unordered_map<string, int> freqMap;
        for (string str : emails)
        {
            RemovePlus(str);
            ++freqMap[str];
        }
        return freqMap.size();
    }

    void RemovePlus(string& str)
    {
        size_t pos_plus = str.find("+");
        size_t pos_atrate = str.find("@");
        if ((pos_plus != string::npos) && (pos_atrate != string::npos) && (pos_atrate > pos_plus))
        {
            str.erase(str.begin() + pos_plus, str.begin() + pos_atrate);
            pos_atrate = str.find('@');
        }

        for (size_t i = 0; i < pos_atrate; i++)
        {
            if (str[i] == '.')
            {
                str.erase(str.begin() + i);
            }
        }
    }
};
