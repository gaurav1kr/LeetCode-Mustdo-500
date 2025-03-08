class Solution
{
    unordered_map<int, int> umap;
    stack<int> stk;
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2)
    {
        vector<int> result;
        for (auto & i : nums2)
        {
            if (stk.empty() ||i < stk.top())
            {
                stk.push(i);
            }
            else 
            {
                while (!stk.empty() && i > stk.top())
                {
                    umap[stk.top()] = i;
                    stk.pop();
                }
                stk.push(i);
            }
        }
        while (!stk.empty())
        {
            umap[stk.top()] = -1;
            stk.pop();
        }
        
        
        for (auto& i : nums1)
        {
            
            result.push_back(umap[i]);
        }
        return result; 
    }
};
