class Solution
{
	stack<char> stk;
public:
	string reverseWords(string s)
	{
		string result;
		string final_result;
		for (auto& i : s)
		{
			stk.push(i);
		}

		while (!stk.empty())
		{
			while (!stk.empty() && stk.top() == ' ')
			{
				stk.pop();
			}
			if (stk.empty())
				break;
			while (!stk.empty() && stk.top() != ' ')
			{
				result.push_back(stk.top());
				stk.pop();
			}

			reverse(result.begin(), result.end());
			
			for (auto & c : result)
			final_result.push_back(c);

			result.clear();
			final_result.push_back(' ');
		}
		final_result.pop_back();
		return final_result;
	}
};