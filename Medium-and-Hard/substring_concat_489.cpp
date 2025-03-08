#include<vector>
#include<string>
#include<unordered_map>
#include<vector>
using namespace std;
class Solution 
{
public:
    vector<int> findSubstring(string s, vector<string>& words) 
    {
        unordered_map<string, int> cnt, record;
        for (auto w : words) cnt[w]++;
        vector <int> ans;
        int n = s.size(), m = words.size(), k = words[0].size();

        for (int P = 0; P < k; P++)
        {
            int left = P; int sum = 0; record.clear();
            for (int j = P; j <= n - k; j += k)
            {
                string tmp = s.substr(j, k);
                if (cnt.count(tmp))
                {
                    record[tmp]++; sum++;
                    while (record[tmp] > cnt[tmp])
                    {
                        string rem = s.substr(left, k);
                        record[rem]--;
                        if (record[rem] == 0) record.erase(rem);
                        left += k;
                        sum--;
                    }
                    if (sum == m)
                    {
                        ans.push_back(left);
                    }
                }
                else
                {
                    left = j + k;
                    record.clear();
                    sum = 0;
                }
            }
        }
        return ans;
    }
};