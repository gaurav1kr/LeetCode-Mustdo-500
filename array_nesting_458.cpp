//Iterative 
class Solution {
public:
    int arrayNesting(vector<int>& nums) {
        int max_len = 0;
        int n = nums.size();
        vector<bool> visited(n, false);

        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                int start = i, count = 0;
                while (!visited[start]) {
                    visited[start] = true;
                    start = nums[start];
                    ++count;
                }
                max_len = max(max_len, count);
            }
        }
        return max_len;
    }
};
