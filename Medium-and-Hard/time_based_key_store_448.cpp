class TimeMap 
{
public:
    unordered_map<string, vector<pair<int, string>>> hash;
    TimeMap() 
    {
 
    }
 
    void set(string key, string value, int timestamp) 
    {
        hash[key].push_back({ timestamp, value});
    }
 
    string get(string key, int timestamp) 
    {
        if(hash.find(key) == hash.end()) return "";
        if(timestamp < hash[key][0].first) return "";
        string ans = "";
        int low = 0, hi = hash[key].size()-1;
        while(low <= hi)
        {
            int mid = low + (hi-low)/2;
            if(hash[key][mid].first <= timestamp) 
                ans = hash[key][mid].second, low = mid+1;
            else 
                hi = mid-1;
        }
        return ans;
    }
}
