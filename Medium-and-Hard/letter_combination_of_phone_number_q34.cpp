class Solution 
{
    string  table[10] ;
public:
    Solution():table{ "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" }
    {
        
    }
    vector<string> letterCombinations(string digits)
    {
        int n = digits.length() ;
        int *number = new int[n] ;
        for(int i=0;i<n;i++)
        {
            number[i] = digits[i]-48 ;
        }
        vector<string> list;
        queue<string> q;
        q.push("");
        
        if(n==0)
            return list ;
        while (!q.empty()) 
        {
            string s = q.front();
            q.pop();
            if (s.length() == n)
            {
                list.push_back(s);
            }
            else
            {
                for (auto letter : table[number[s.length()]])
                    q.push(s + letter);
            }
    }
    return list;
    }
};
//Time complexiety :- o(n*4^n)
