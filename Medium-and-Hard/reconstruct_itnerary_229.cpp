class Solution 
{
public:
    void dfs(string& airport, unordered_map<string, vector<string>>& graph,vector<string>& itinerary)
    {
        while(!graph[airport].empty())
	{
            string next = graph[airport].back();
            graph[airport].pop_back();
            dfs(next, graph, itinerary);
        }
        itinerary.push_back(airport);
    }

    vector<string> findItinerary(vector<vector<string>>& tickets) 
    {
        unordered_map<string, vector<string>> graph;
        for(auto& ticket : tickets)
	{
            graph[ticket[0]].push_back(ticket[1]);
        }
        for(auto& temp : graph)
	{
            sort(temp.second.rbegin(), temp.second.rend());
        }

        vector<string> itinerary;
        string jfk = "JFK";
        dfs(jfk, graph, itinerary);

        reverse(itinerary.begin(), itinerary.end());

        return itinerary;
        
    }
};

//Approach
We need to find the itirenary such that it has a smallest lexical order.

So we start by making a graph with the help of a map. We do this by pushing destination ticket[1] city to every source Aicket[0] city.

After this we need to sort our destination cities in reverse order so that we visit lexically smaller destinations.

Create an array of strings itinerary to store our answer.

Calldfsfunction on our source, i.e JFK airport.

The dfs function works like this -

It check if there is a destination city to current airport, if there is, it stores the city in next variable. We do graph[airport].back() so that we get the lexically smallest destination, this is why we sorted in reverse order since we can't pop lexically smallest string from front.
After choosing next, pop_back() the lexically smallest destination.
Call dfs on this next city.
After all calls are made,push_back()our original city Airport.
At last, we reverse our itinerary and return it.
