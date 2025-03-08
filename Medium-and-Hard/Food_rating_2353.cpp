class FoodRatings 
{
private:
    struct compare
	{
        bool operator() (const pair<int, string>& a, const pair<int, string>& b) const 
		{
            if (a.first == b.first) return (a.second < b.second);
            return (a.first > b.first);
        }
    };
    
    unordered_map<string, set<pair<int, string>, compare>> rank;
    unordered_map<string, string> food2cuisine;
    unordered_map<string, int> food2rating;
	
public:

    FoodRatings(vector<string>& foods, vector<string>& cuisines, vector<int>& ratings) 
	{
        int n = foods.size();
        for (int i = 0; i < n; i++) 
		{
            rank[cuisines[i]].insert({ratings[i], foods[i]});
            food2cuisine[foods[i]] = cuisines[i];
            food2rating[foods[i]] = ratings[i];
        }
    }
    
    void changeRating(string food, int newRating) 
	{
        auto& cuisine = food2cuisine[food];
        rank[cuisine].erase({food2rating[food], food});
        rank[cuisine].insert({newRating, food});
        food2rating[food] = newRating;
    }
    
  
    string highestRated(string cuisine) 
	{
        return (rank[cuisine].begin())->second;
    }

};