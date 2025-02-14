```cpp
class Solution {
public:
    void dfs(int i, int target, vector<int>& c, vector<int>& path, vector<vector<int>>& res) {
        if (target == 0) { res.push_back(path); return; }
        if (i == c.size() || target < 0) return;
        
        path.push_back(c[i]);
        dfs(i, target - c[i], c, path, res);
        path.pop_back();
        
        dfs(i + 1, target, c, path, res);
    }
    
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> path;
        dfs(0, target, candidates, path, res);
        return res;
    }
};
```