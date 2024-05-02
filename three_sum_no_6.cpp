class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& arr) 
    {
        int n = arr.size();
    vector<vector<int>> res;
    vector<int> single_vec;
    int cnt1 = 0;
    bool found = false;
    // sort array elements 
    sort(arr.begin(), arr.end());
    for (int i = 0; i < n - 1; i++)
    {
        // initialize left and right 
        int l = i + 1;
        int r = n - 1;
        int x = arr[i];
        while (l < r)
        {
            if (x + arr[l] + arr[r] == 0)
            {
                // print elements if it's sum is zero
                single_vec.push_back(x);
                single_vec.push_back(arr[l]);
                single_vec.push_back(arr[r]);
                res.push_back(single_vec);
                cnt1++;
                single_vec.clear();
                l++;
                r--;
                found = true;
            }
            else if (x + arr[l] + arr[r] < 0)
                l++;
            else
                r--;
        }
    }
    set<vector<int>> s(res.cbegin(), res.cend());
    res = vector<vector<int>>(s.cbegin(), s.cend());
    return res;
    }
};
