class Solution 
{
public:
    vector<int> asteroidCollision(vector<int>& astr) 
	{
        int prev = -1;
        for (int i = 0; i < astr.size(); i++) 
		{
            bool notSame = true;
            if (astr[i] > 0) 
			{
                astr[++prev] = astr[i];
            } 
			else 
			{
                while (prev >= 0 && astr[prev] > 0) 
				{
                    if (astr[prev] < -astr[i]) 
					{
                        prev--;
                    } 
					else if (astr[prev] == -astr[i]) 
					{
                        prev--;
                        notSame = false;
                        break;
                    } 
					else 
					{
                        break;
                    }
                }

                if (notSame && (prev < 0 || astr[prev] < 0)) 
				{
                    astr[++prev] = astr[i];
                }
            }
        }

        astr.resize(prev + 1);
        return astr;
    }
};