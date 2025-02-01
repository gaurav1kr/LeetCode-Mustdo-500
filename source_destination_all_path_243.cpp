class Solution 
{
public:
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) 
    {
        int target = graph.size() - 1;
        vector<vector<int>> res;
        if (graph.size() == 0) 
		return res;

        vector<int> path;
        queue<vector<int>> q;
        path.push_back(0);
        q.push(path);

        while (!q.empty()) 
	{
            auto temp = q.front();
            q.pop();
            int node = temp.back();

            for (auto next : graph[node]) 
	    {
                auto tempPath = temp;
                tempPath.push_back(next);
                if (next == target) 
                    res.push_back(tempPath);
                else
                    q.push(tempPath);
            }
        }

        return res;
    }
};
