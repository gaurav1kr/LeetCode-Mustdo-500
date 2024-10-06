class Solution 
{
public:
    int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations) 
    {
        priority_queue<int>q;
        long long int i=startFuel,ans=0,j=0; 
        while(target-i>0)
        {
            while(j<stations.size() && i>=stations[j][0])
            {
                q.push(stations[j][1]);
                j++;
            }
            
            if(i<target)
            {
                if(q.empty())return -1;
                ans++;
                i+=q.top();
                q.pop();
            }
        }
        return ans;
    }
};

// TC :- o(nlogn)
// SC :- o(n)

// Example Walkthrough:
// Input:

// target = 100
// startFuel = 10
// stations = [[10, 60], [20, 30], [30, 30], [60, 40]]
// Initial State:

// Fuel available: 10
// Target: 100
// Stations: [[10, 60], [20, 30], [30, 30], [60, 40]]
// First Loop:

// Current fuel i = 10, reach station 10, add fuel 60 to the heap.
// Still short of the target, fuel up with 60, new fuel level 70, stop count 1.
// Second Loop:

// Reach station 20, add fuel 30 to the heap.
// Reach station 30, add fuel 30 to the heap.
// Reach station 60, add fuel 40 to the heap.
// Still short of target, fuel up with 40, new fuel level 110, stop count 2.
// Reached Target:

// Reached the target with 2 stops.
// Output: 2