class Solution 
{
public:
void findNumbers(vector<int>& ar, int sum,vector<vector<int> >& res, vector<int>& r, int i) 
{ 
    if (sum < 0) 
        return; 
    if (sum == 0) 
    { 
        res.push_back(r); 
        return; 
    } 
    while (i < ar.size() && sum - ar[i] >= 0) 
    { 
        r.push_back(ar[i]);  
        findNumbers(ar, sum - ar[i], res, r, i); 
        i++; 
        r.pop_back(); 
    } 
}
    
vector<vector<int>> combinationSum(vector<int>& ar, int sum) 
    {
        sort(ar.begin(), ar.end()); 
        ar.erase(unique(ar.begin(), ar.end()), ar.end()); 
        vector<int> r; 
        vector<vector<int> > res; 
        findNumbers(ar, sum, res, r, 0); 
        return res;     
    }
};
// Time complexiety :- o(2^n * n)
