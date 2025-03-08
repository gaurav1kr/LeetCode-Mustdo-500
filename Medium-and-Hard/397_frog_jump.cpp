class Solution 
{
public:
    bool canCross(vector<int>& stones) 
	{
        unordered_map<int, unordered_set<int>> mp;

        mp[stones[0] + 1] = {1};

        for(int i = 1; i < stones.size(); ++i)
		{
            int stone = stones[i];
            for(auto it : mp[stone])
			{
                mp[stone + it].insert(it);
                mp[stone + it + 1].insert(it + 1);
                mp[stone + it - 1].insert(it - 1);
            }
        }
        return mp[stones.back()].size() != 0;
    }
};
// For each stone, the frog checks all possible jump lengths (it) stored in mp[stone].
// For each jump it, the frog can try three different jumps:
// Same Jump Length (it): Frog jumps exactly the same length as before and lands on stone + it.
// Jump + 1: Frog jumps one unit more than before and lands on stone + it + 1.
// Jump - 1: Frog jumps one unit less than before and lands on stone + it - 1.
// These new jump lengths are inserted into the map for the corresponding landing stone.
//return mp[stones.back()].size() != 0; --> Finally, the function checks if there are any valid jump 
// lengths stored for the last stone (stones.back()). If the set for the last stone is non-empty, 
// it means the frog can reach that stone, so the function returns true. 
// Otherwise, it returns false.