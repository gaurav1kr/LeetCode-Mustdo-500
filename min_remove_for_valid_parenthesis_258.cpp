class Solution 
{
public:
    string minRemoveToMakeValid(string s) 
    {
        stack<pair<char,int>> stk;
        vector<int> vec(s.size(),0);
        for(int i=0;i<s.size();i++){
            if(!stk.empty() && stk.top().first == '(' && s[i] == ')')
            {
                int num = stk.top().second; 
                vec[num] = 1;
                stk.pop();
                vec[i] = 1;
            }
            else if(s[i] == ')' || s[i] == '(')
            {
                stk.push(make_pair(s[i],i));
            }
            else{
                vec[i] = 1;
            }
        }
        string res = "";
        for(int i=0;i<s.size();i++)
        {
            if(vec[i])
            {
                res += s[i];
            }
        }
        return res;
    }
};