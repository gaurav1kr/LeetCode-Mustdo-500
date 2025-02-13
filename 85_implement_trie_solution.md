
```cpp
#include <unordered_map>
#include <string>
#include <iostream>

using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord;

    TrieNode() : isEndOfWord(false) {}
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(string word) {
        TrieNode* current = root;
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = new TrieNode();
            }
            current = current->children[ch];
        }
        current->isEndOfWord = true;
    }

    bool search(string word) {
        TrieNode* current = root;
        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }
            current = current->children[ch];
        }
        return current->isEndOfWord;
    }

    bool startsWith(string prefix) {
        TrieNode* current = root;
        for (char ch : prefix) {
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }
            current = current->children[ch];
        }
        return true;
    }
};

// Test the Trie
int main() {
    Trie* trie = new Trie();
    trie->insert("apple");
    cout << trie->search("apple") << endl;  // true
    cout << trie->search("app") << endl;    // false
    cout << trie->startsWith("app") << endl; // true
    trie->insert("app");
    cout << trie->search("app") << endl;    // true
    delete trie;
    return 0;
}
```

### Explanation

1. **Structure**:
   - A TrieNode contains:
     - A map `children` to store child nodes for each character.
     - A boolean `isEndOfWord` to indicate the end of a word.

2. **Trie Operations**:
   - **Insert**:
     - Traverse the Trie, creating new nodes for characters not already in the tree.
     - Mark the last node as the end of the word.
   - **Search**:
     - Traverse the Trie to check if the word exists.
     - Return true only if the traversal ends at a node marked as the end of a word.
   - **Prefix Check**:
     - Traverse the Trie to ensure all characters of the prefix exist.
     - Return true if traversal is successful.

3. **Complexity**:
   - **Insert/Search/Prefix**:
     - Time Complexity: \(O(m)\), where \(m\) is the length of the word or prefix.
     - Space Complexity: Depends on the total number of characters in all inserted words.
