class Solution {
public:
    int coinChange(vector<int>& coins, int amount) \
    {
        vector<int> dp(amount+1, 1e8);
        dp[0] = 0;
        for (int i=1; i<=amount; ++i)
        for (int n : coins)
            if (i-n >= 0)
                dp[i] = min(dp[i], dp[i-n]+1);
        return (dp[amount] == 1e8) ? -1 : dp[amount];
    }
};

/*
1. It initializes a vector dp of size amount + 1 and fills it with a large value (1e8). This vector is used to store the minimum number of coins required to make up each amount from 0 to amount.

2. It sets dp[0] to 0, as no coins are needed to make up the amount 0.

3. It iterates from 1 to amount, and for each i, it iterates through each coin denomination in coins.

4. For each coin denomination n, it checks if i - n is non-negative (i.e., if using coin n doesn't result in a negative index). If it is non-negative, it updates dp[i] to be the minimum of its current value and dp[i - n] + 1, indicating the minimum number of coins required to make up the amount i using coin n plus one additional coin (as n itself is used).

5. Finally, it returns dp[amount], which represents the minimum number of coins required to make up the target amount. If dp[amount] is still equal to the large initial value (1e8), it means it was not possible to make up the amount using the given coins, so it returns -1.
*/ 
TC :- O(Amount * no_of_coins)
