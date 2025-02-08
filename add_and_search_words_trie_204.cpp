#include <string>
#include <vector>
using namespace std;

class WordDictionary 
{
    private:
    struct TrieNode 
    {
        TrieNode* children[26] = {nullptr}; // Pointers to child nodes
        bool isEnd = false; // Flag for end of word
    };
    
    TrieNode* root;
    
    bool dfsSearch(const string& word, int index, TrieNode* node) 
    {
        if (!node) return false; 
        if (index == word.size()) return node->isEnd; 
        
        char c = word[index];
        if (c == '.') 
	{
            // Try all possible children
            for (int i = 0; i < 26; ++i) 
	    {
                if (dfsSearch(word, index + 1, node->children[i])) 
		{
                    return true;
                }
            }
            return false;
        } 
        else 
	{
            // Normal character, move to the next node
            return dfsSearch(word, index + 1, node->children[c - 'a']);
        }
    }
    
public:
    WordDictionary() 
    {
        root = new TrieNode(); // Initialize root
    }
    
    void addWord(string word) 
    {
        TrieNode* node = root;
        for (char c : word) 
	{
            int idx = c - 'a';
            if (!node->children[idx]) 
	    {
                node->children[idx] = new TrieNode();
            }
            node = node->children[idx];
        }
        node->isEnd = true; // Mark the end of the word
    }
    
    bool search(string word) 
    {
        return dfsSearch(word, 0, root); // Start DFS search from root
    }
};

TC :- addWord :- o(L) , where L is the length of the word.
search :- worst case O(26^L) (if all characters are .) , average case O(L)

SC :- O(N*26) , where N is the total number of characters inserted in the Trie.
