class Solution 
{
public:
    bool f(vector<int>& v, int ind, vector<int> &vis) 
	{
        if(ind < 0 || ind >= v.size() || vis[ind] == true) 
			return false;
		
        if(v[ind] == 0) 
			return true;
		
        vis[ind] = true;
        return f(v, ind - v[ind], vis) || f(v, ind + v[ind], vis);
    }
    
    bool canReach(vector<int>& v, int start) 
	{
        int n = v.size();
        vector<int> vis(n, false);
        return f(v, start, vis);
    }
};