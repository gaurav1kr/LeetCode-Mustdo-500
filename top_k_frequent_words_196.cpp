class Solution 
{
public:
    vector<string> topKFrequent(vector<string>& words, int k) 
    {
        // Step 1: Count the frequency of each word
        unordered_map<string, int> frequency;
        for (const string& word : words) 
	{
            frequency[word]++;
        }

        // Step 2: Use a min-heap to store the top k frequent words
        auto cmp = [](const pair<string, int>& a, const pair<string, int>& b) 
	{
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        };

        priority_queue<pair<string, int>, vector<pair<string, int>>, decltype(cmp)> minHeap(cmp);

        for (const auto& [word, count] : frequency) 
	{
            minHeap.push({word, count});
            if (minHeap.size() > k) 
	    {
                minHeap.pop();
            }
        }

        // Step 3: Extract the words from the heap into a vector
        vector<string> result;
        while (!minHeap.empty()) 
	{
            result.push_back(minHeap.top().first);
            minHeap.pop();
        }

        // Step 4: Reverse the vector since the heap gives the least frequent words first
        reverse(result.begin(), result.end());

        return result;
    }
};
