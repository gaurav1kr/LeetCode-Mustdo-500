class Solution {
public:
    bool stoneGame(vector<int>& p) 
    {
         vector<int> dp = p;
         for (int d = 1; d < p.size(); d++)
         {
            for (int i = 0; i < p.size() - d; i++)
            {
                dp[i] = max(p[i] - dp[i + 1], p[i + d] - dp[i]);
            }
         }
        return dp[0] > 0;
    }
};
/*
Explanation :- 
This C++ function stoneGame appears to implement a dynamic programming solution to solve a variation of the stone game problem. Let's break it down step by step:

Function Signature:

bool stoneGame(vector<int>& p): This function takes a reference to a vector of integers p as input and returns a boolean value.
Variable Initialization:

vector<int> dp = p;: A new vector dp is created and initialized with the same values as vector p. This vector will be used to store the optimal scores.
Dynamic Programming Loop:

for (int d = 1; d < p.size(); d++): This outer loop iterates over the possible lengths of subarrays, from 1 to the size of the input vector p.
for (int i = 0; i < p.size() - d; i++): This inner loop iterates over the starting indices of the subarrays of length d.
dp[i] = max(p[i] - dp[i + 1], p[i + d] - dp[i]);: For each subarray, it calculates the maximum score that the current player can achieve. The score is determined by choosing either the value at the current index i or at index i+d, subtracting the opponent's score, which is the previously calculated optimal score in dp.
Return Statement:

return dp[0] > 0;: After calculating the optimal scores for all possible subarrays, the function returns true if the score of the first player (starting player) is greater than 0, indicating that the starting player can win, and false otherwise.
*/
