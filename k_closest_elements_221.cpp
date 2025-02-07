class Solution 
{
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x) 
    {
        priority_queue<pair<int,int>> p;
        vector<int>v;
        for(auto i:arr)
        {
            int n=abs(x-i);
            p.push({n,i});

            if(p.size()>k)
            {
                 p.pop();
            }
        }
        while(p.size()>0)
        {
            v.push_back(p.top().second);
            p.pop();
        }
        sort(v.begin(),v.end());
        return v;
    }
};
