#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        unordered_map<string, string> emailToName;
        unordered_map<string, string> parent;

        // Find function for Union-Find
        function<string(string)> find = [&](string s) {
            return parent[s] == s ? s : parent[s] = find(parent[s]);
        };

        // Build Union-Find for emails
        for (auto& acc : accounts) {
            for (int i = 1; i < acc.size(); ++i) {
                emailToName[acc[i]] = acc[0];
                if (!parent.count(acc[i])) parent[acc[i]] = acc[i];
                parent[find(acc[i])] = find(acc[1]);
            }
        }

        // Group emails by root
        unordered_map<string, set<string>> groups;
        for (auto& [email, _] : emailToName) {
            groups[find(email)].insert(email);
        }

        // Prepare the result
        vector<vector<string>> res;
        for (auto& [root, emails] : groups) {
            vector<string> merged(emails.begin(), emails.end());
            merged.insert(merged.begin(), emailToName[root]);
            res.push_back(merged);
        }

        return res;
    }
};

Key Points:
Union-Find:

Used to group emails by finding common roots.
Efficient with path compression for find operation.
Email Grouping:

Emails with the same root are grouped together.
Sorting:

Emails are automatically sorted using set.
Time Complexity:
O(N * α(N) + N * logK), where:
N is the total number of emails.
α(N) is the inverse Ackermann function from Union-Find.
K is the average size of email groups.
