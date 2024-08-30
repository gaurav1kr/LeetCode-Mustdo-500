class Solution 
{
    stack<char> stk1 ;
public:
    int minAddToMakeValid(string s) 
    {
        int count = 0 ;
        for(auto & e:s)
        {
            if(e==')')
            {
                if(stk1.empty())
                {
                    count++ ;
                }
                else
                {
                    stk1.pop() ;
                }
            }
            else
            {
                stk1.push(e) ;
            }
        }
		count += stk1.size() ;
		return count ;
    }
};