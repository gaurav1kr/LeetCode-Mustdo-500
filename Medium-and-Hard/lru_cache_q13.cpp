class LRUCache 
{
        int cacheSize;
        unordered_map<int, int> m;
        deque<int> dq;
    
    public:
        LRUCache(int capacity) 
        {
            m.clear();
            dq.clear();
            cacheSize = capacity;
        }

        int get(int key) 
        {
            if (m.find(key) == m.end()) 
            {
                return -1;
            } 
            else 
            {
                deque<int>::iterator it = dq.begin();
                while (*it != key) 
                {
                    it++;
                }
                dq.erase(it);
                dq.push_front(key);
                return m[key];
            }
        }

        void put(int key, int value) 
        {
            if (m.find(key) == m.end()) 
            {
                if (cacheSize == dq.size()) 
                {
                    int last = dq.back();
                    dq.pop_back();
                    m.erase(last);
                }
            } 
            else 
            {
                deque<int>::iterator it = dq.begin();
                while (*it != key) 
                {
                    it++;
                }
                dq.erase(it);
                m.erase(key);
            }
            dq.push_front(key);
            m[key] = value;    
        }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */

/*
Explanations :- 
The LRUCache class maintains a deque (dq) to keep track of the order in which keys are accessed, with the most recently used key at the front and the least recently used key at the back.

It also maintains an unordered map (m) to store key-value pairs for quick lookups.

The constructor initializes the cache size and clears both the deque and the unordered map.

The get function checks if the key exists in the cache. If it does, it updates the deque to reflect that the key was recently used and returns the corresponding value. If not, it returns -1.

The put function adds a new key-value pair to the cache. If the key already exists, it updates its value and moves it to the front of the deque. If the cache is full, it removes the least recently used key-value pair from both the deque and the unordered map before inserting the new key-value pair at the front.

*/
