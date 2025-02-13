```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;

vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> mp;
    
    for (string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());  // Sort to create the key
        mp[key].push_back(s);
    }
    
    vector<vector<string>> result;
    for (auto& [_, group] : mp) {
        result.push_back(move(group));  // Move for efficiency
    }
    
    return result;
}
```