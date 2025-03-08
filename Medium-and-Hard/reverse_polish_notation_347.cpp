class Solution 
{
public:
    int evalRPN(vector<string>& tokens)
    {
		std::stack<int> stk;
		for (const auto& token : tokens) 
		{
			if (token == "+" || token == "-" || token == "*" || token == "/")
			{
				int op2 = stk.top(); stk.pop();
				int op1 = stk.top(); stk.pop();

				if (token == "+") 
				{
					stk.push(op1 + op2);
				}
				else if (token == "-") 
				{
					stk.push(op1 - op2);
				}
				else if (token == "*") 
				{
					stk.push(op1 * op2);
				}
				else if (token == "/") 
				{
					stk.push(op1 / op2);
				}
			}
			else 
			{
				stk.push(std::stoi(token));
			}
		}

		return stk.top();
    }
};