class Solution 
{
    public:
    int compress(vector<char>& s) 
	{
        int c=1;
        for(int i=0;i<s.size()-1;i++)
	{
            if(s[i]==s[i+1])
	    {
                c++, i--;
                s.erase(s.begin()+i+1);
            }
	    else
	    {
                if(c>1)
		{
                        for(char c:to_string(c))
			{
                        	s.insert(s.begin()+i+1,c),i++;
			}
		}
                c=1;
            }
        }
        if(c>1)
	{
            for(char c:to_string(c))
	    {
                s.insert(s.end(),c);
	    }
        }
        return s.size();
    }
};
