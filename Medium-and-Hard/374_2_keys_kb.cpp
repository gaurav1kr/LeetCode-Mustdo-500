class Solution 
{
public:
    int minSteps(int n) 
	{
        int current = 1 ;
		int steps = 0;
		int copy = 0 ;
		while(curr < n)
		{
			if( (n-curr) & 2  )
			{
				steps += 2 ;
				copy = curr ;
			}
			else
			{
				steps += 1 ;
			}
			curr += copy ;
		}
		return steps ;
    }
};

// Intuition
// A copy operation can only be performed when the remaining characters can be evenly divided by the
// current number of characters on the screen; otherwise, it will be impossible to reach the desired 
// total of n characters.

// Approach
// Declare curr = 1 (given in question), copy = 0 and steps = 0
// Now iterate till curr < n
// At each step check if (n-curr) % curr == 0
// If yes, copy the characters and increase steps by 2 (since copy function adds an additional operation)
// If no, then simply add 1 in steps, since only paste operation is being performed
// And keep on adding copy in the curr