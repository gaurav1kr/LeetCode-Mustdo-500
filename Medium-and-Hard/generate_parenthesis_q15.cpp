class Solution
{
    vector<string> str;
public:
    void brackets(string output, int open, int close, int pairs, int n)
    {
        if (open == pairs && close == pairs && (output.length() == 2 * n))
            str.push_back(output);
        else
        {
            if (open < pairs)
                brackets(output + "(", open + 1, close, pairs, n);
            if (close < open)
                brackets(output + ")", open, close + 1, pairs, n);
        }
    }
    vector<string> generateParenthesis(int n)
    {
        for (int i = 1; i <= n; i++)
        {
            brackets("", 0, 0, i, n);
        }
        return str;
    }
};
//Approach
The brackets function recursively generates valid combinations of parentheses. It takes several parameters:

output: The current combination of parentheses being constructed.
open: The count of open parentheses in the current combination.
close: The count of closed parentheses in the current combination.
pairs: The number of pairs of parentheses to be formed.
n: The total number of pairs of parentheses to be generated.
In the brackets function, if the counts of open and close parentheses reach the desired number of pairs (pairs), and the length of the output string is equal to 2 * n, it means a valid combination has been formed, so it is added to the str vector.

Otherwise, the function recursively calls itself with either an open parenthesis added (output+"("), or a closed parenthesis added (output+")"), depending on the current counts and constraints.

The generateParenthesis function initializes the process by calling brackets for each number of pairs of parentheses from 1 to n.

Finally, it returns the vector str containing all the valid combinations of parentheses.


//TC
 The time complexity of your solution is O(4^n / sqrt(n)), where n is the number of pairs of parentheses. This complexity arises because for each position in the resulting string, there are 4 choices: open left parenthesis, close left parenthesis, open right parenthesis, and close right parenthesis, and the recursion depth is limited by 2 * n. So, the total number of combinations is bounded by the Catalan number, which is approximately 4^n / sqrt(n)
