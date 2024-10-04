class Solution 
{
public:
    int maximumSwap(int num) 
	{
        auto s = std::to_string(num);
        int maxPos = s.size() - 1, bestSrc = -1, bestDest = -1;
		
        for (int i = s.size() - 1; i >= 0; --i) 
		{
            if (s[i] > s[maxPos]) 
                maxPos = i;
            else if (s[i] < s[maxPos]) 
			{
                bestSrc = maxPos;
                bestDest = i;
            }
        }
        if (bestSrc != -1) 
		{
            std::swap(s[bestSrc], s[bestDest]);
		}
            
        return std::stoi(s);
    }
};

//TC :- o(n)
//SC :- o(n)