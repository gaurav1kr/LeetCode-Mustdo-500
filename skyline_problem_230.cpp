class Solution 
{
public:
    vector<vector<int>> getSkyline(vector<vector<int>>& buildings) 
    {
        vector<pair<int, int>> h;

        for (auto b : buildings) 
        {
            h.push_back({b[0], -b[2]}); // Start of a building with negative height
            h.push_back({b[1], b[2]});  // End of a building with positive height
        }

        sort(h.begin(), h.end());

        multiset<int> heights; // Multiset to maintain the current active heights
        heights.insert(0);     // Initialize with ground level (height = 0)

        int prev = 0, cur = 0; // Track the previous and current maximum heights
        vector<vector<int>> res; // Result to store the key points

        for (auto i : h) 
	{
            if (i.second < 0) 
	    {
                // Left boundary of a building, insert height
                heights.insert(-i.second);
            } 
	    else 
	    {
                // Right boundary of a building, remove height
                heights.erase(heights.find(i.second));
            }

            // Current maximum height from the active heights
            cur = *heights.rbegin();

            // If the maximum height changes, it's a key point
            if (cur != prev) 
	    {
                res.push_back({i.first, cur});
                prev = cur;
            }
        }
        return res;
    }
};

//Approach - 
Approach

Convert each building into two segments.
Sort the segments.
If second value of segment is less than zero, it is left boundary, else it is right boundary.
If current maximum is not equal to maximum previous height, it is a key point.
