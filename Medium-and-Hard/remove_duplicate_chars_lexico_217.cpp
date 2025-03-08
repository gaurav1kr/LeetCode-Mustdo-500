class Solution 
{  
public:  
    string removeDuplicateLetters(string s) 
    {  
        unordered_map<char, int> charCount;  
        vector<bool> inResult(26, false);  
        stack<char> stack;  

        for (char c : s) charCount[c]++;  

        for (char c : s) 
	{  
            if (--charCount[c] < 0) continue;  
            if (inResult[c - 'a']) continue;  
            while (!stack.empty() && stack.top() > c && charCount[stack.top()] > 0) 
	    {  
                inResult[stack.top() - 'a'] = false;  
                stack.pop();  
            }  
            stack.push(c);  
            inResult[c - 'a'] = true;  
        }  

        string result;  
        while (!stack.empty()) 
	{  
            result += stack.top();  
            stack.pop();  
        }  
        reverse(result.begin(), result.end());  
        return result;  
    }  
};
