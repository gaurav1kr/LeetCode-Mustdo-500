class Solution 
{
public:
    int rangeBitwiseAnd(int n, int m) 
	{
		int ans=0;
		for(int i=0;i<32;i++)
		{
			if(((n/(1<<i))==(m/(1<<i))) && (n & (1 << i)))
			{
				ans=(ans|(1<<i));
			}
		}
		return ans;
    }
};

// The condition if (((n / (1 << i)) == (m / (1 << i))) && (n & (1 << i))) 
// in the given C++ code is a crucial check for determining whether the i-th bit of
// the result of the bitwise AND operation can be set to 1. If 'yes', we are setting the ith 
// bit of answer to 1.