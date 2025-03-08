class Solution 
{
public:
    vector<int> nextGreaterElements(vector<int>& nums) 
    {
        int n = nums.size();
        vector<int> res(n, -1); // Initialize result array with -1
        stack<int> st;         // Stack to store indices

    // Iterate over the array twice for circular traversal
        for (int i = 0; i < 2 * n; ++i) 
        {
            while (!st.empty() && nums[st.top()] < nums[i % n])
            {
                res[st.top()] = nums[i % n];
                st.pop();
            }
            if (i < n) 
            { // Push indices only in the first pass
                st.push(i);
            }
        }

        return res;
    }
};
