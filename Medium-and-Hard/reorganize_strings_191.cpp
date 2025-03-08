# Reorganize String - C++ Solution

## Problem Description
You are given a string `s`. You need to reorganize it so that no two adjacent characters are the same. If it's not possible to achieve this, return an empty string.

Problem link: [Reorganize String](https://leetcode.com/problems/reorganize-string/)

## Optimized C++ Solution
Below is a concise and optimized solution using a **priority queue**:

```cpp
#include <string>
#include <unordered_map>
#include <queue>
using namespace std;

class Solution 
{
public:
    string reorganizeString(string s) 
    {
        unordered_map<char, int> freq;
        for (char c : s) freq[c]++;
        
        priority_queue<pair<int, char>> pq;
        for (auto& [ch, count] : freq) 
	{
            if (count > (s.size() + 1) / 2) return ""; // Impossible to reorganize
            pq.push({count, ch});
        }

        string result;
        while (pq.size() > 1) 
	{
            auto [freq1, ch1] = pq.top(); pq.pop();
            auto [freq2, ch2] = pq.top(); pq.pop();

            result += ch1;
            result += ch2;

            if (--freq1 > 0) pq.push({freq1, ch1});
            if (--freq2 > 0) pq.push({freq2, ch2});
        }

        if (!pq.empty()) result += pq.top().second; // Add the last remaining character
        return result;
    }
};
```

## Explanation
1. **Count Frequencies**:
   - Use a hash map to count the frequency of each character in the input string.
   
2. **Priority Queue**:
   - Push characters and their frequencies into a max-heap (priority queue), so the character with the highest frequency is at the top.
   - If the frequency of any character exceeds `(s.size() + 1) / 2`, it is impossible to reorganize, so return an empty string.

3. **Build Result String**:
   - Repeatedly take the two most frequent characters from the heap and append them to the result string.
   - Decrement their frequencies and push them back into the heap if their frequency is still greater than zero.

4. **Handle Remaining Character**:
   - If there's one character left in the heap, append it to the result string.

## Complexity
- **Time Complexity**: 
  - Counting frequencies takes \(O(n)\).
  - Each operation on the priority queue takes \(O(\log k)\), where \(k\) is the number of distinct characters.
  - Total complexity: \(O(n \log k)\), where \(n\) is the length of the string.

- **Space Complexity**: \(O(k)\), where \(k\) is the number of distinct characters (space for the hash map and priority queue).

## Example
### Input:
```plaintext
s = "aab"
```

### Output:
```plaintext
"aba"
```

### Input:
```plaintext
s = "aaab"
```

### Output:
```plaintext
""

