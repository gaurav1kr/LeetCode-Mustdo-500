class Solution
 {
public:
    vector<int> deckRevealedIncreasing(vector<int>& deck) 
    {
        sort(deck.begin() , deck.end()) ;
        int n = deck.size();
        int i=0;
        queue<int> deck_queue;
        vector<int> ans(n);
        while(i<n)
        {
            deck_queue.push(i++);
        }

        for(i=0;i<n;i++)
        {
            ans[deck_queue.front()] = deck[i];
            deck_queue.pop() ;
            deck_queue.push(deck_queue.front()) ;
            deck_queue.pop() ;
        }
        return ans ;
    }
};
