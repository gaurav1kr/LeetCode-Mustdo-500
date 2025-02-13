
# Word Ladder Problem - Optimal C++ Solution

The **Word Ladder** problem on LeetCode is about finding the shortest transformation sequence length from a `beginWord` to an `endWord`, with all intermediate words existing in a given word list, and where only one letter can change per step.

## Optimal C++ Solution
```cpp
#include <iostream>
#include <unordered_set>
#include <queue>
#include <string>
#include <vector>

using namespace std;

int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (wordSet.find(endWord) == wordSet.end()) {
        return 0; // endWord is not in the word list
    }

    queue<pair<string, int>> q; // Pair of current word and transformation steps
    q.push({beginWord, 1}); // Start from the beginWord with step 1

    while (!q.empty()) {
        auto [currentWord, steps] = q.front();
        q.pop();

        // Check all possible single-letter transformations
        for (int i = 0; i < currentWord.length(); ++i) {
            string temp = currentWord;
            for (char c = 'a'; c <= 'z'; ++c) {
                temp[i] = c;

                if (temp == endWord) {
                    return steps + 1; // Found the endWord
                }

                // If the transformed word is in the word list, add it to the queue
                if (wordSet.find(temp) != wordSet.end()) {
                    q.push({temp, steps + 1});
                    wordSet.erase(temp); // Remove to avoid revisiting
                }
            }
        }
    }

    return 0; // No transformation sequence exists
}
```

## Explanation

### Data Structures Used
- **`unordered_set<string>`**: Efficient lookup for words in the dictionary.
- **Queue**: For BFS traversal, storing pairs of the current word and the number of steps taken to reach it.

### Algorithm
1. Use BFS to explore all possible transformations layer by layer.
2. For each word, try changing each character (`'a'` to `'z'`) and check if it exists in the word list.
3. If the transformed word is the `endWord`, return the number of steps.
4. Remove words from the word list (`wordSet.erase(temp)`) once visited to prevent cycles.

### Complexity

#### Time Complexity:
- Generating all transformations for a word: \( O(L \times 26) = O(L) \), where \( L \) is the word length.
- BFS traversal over all words: \( O(N \times L) \), where \( N \) is the size of the word list.
- **Total**: \( O(N \times L) \).

#### Space Complexity:
- \( O(N) \): For the word list (`wordSet`) and the BFS queue.

## Example

### Input:
```cpp
string beginWord = "hit";
string endWord = "cog";
vector<string> wordList = {"hot", "dot", "dog", "lot", "log", "cog"};
```

### Output:
```cpp
int result = ladderLength(beginWord, endWord, wordList);
cout << result; // Output: 5
```

### Transformation:
`hit -> hot -> dot -> dog -> cog` (5 steps)
