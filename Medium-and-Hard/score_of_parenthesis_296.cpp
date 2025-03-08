class Solution 
{
public:
    int scoreOfParentheses(string s)
	{
        stack<int> stack;
        int currentScore = 0;
        for(auto&c : s)
		{
            if(c == '(')
			{
                stack.push(currentScore);
                currentScore = 0;
            }
			else 
			{
                int lastScore = stack.top();
                stack.pop();
                currentScore = lastScore + std::max(1, 2 * currentScore);
            }
        }
        return currentScore;
    }
};