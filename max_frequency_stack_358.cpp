class FreqStack 
{
    unordered_map<int, int> mp;
    priority_queue<vector<int> > pq;
    int index = 0;

public:
    FreqStack() {}

    void push(int val) 
    {
        mp[val]++;
        pq.push({mp[val], index, val});
        index++;
    }

    int pop() 
    {
        vector<int> ans = pq.top();
        pq.pop();
        mp[ans[2]]--;
        return ans[2];
    }
};
