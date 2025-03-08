
```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

using namespace std;

vector<int> partitionLabels(string s) {
    // Step 1: Store the last occurrence of each character
    vector<int> lastIndex(26, 0);
    for (int i = 0; i < s.length(); i++) {
        lastIndex[s[i] - 'a'] = i;
    }

    // Step 2: Partition the string
    vector<int> result;
    int start = 0, end = 0;
    for (int i = 0; i < s.length(); i++) {
        end = max(end, lastIndex[s[i] - 'a']);
        if (i == end) {
            result.push_back(end - start + 1);
            start = i + 1;
        }
    }

    return result;
}

// Test the function
int main() {
    string s = "ababcbacadefegdehijhklij";
    vector<int> result = partitionLabels(s);

    for (int size : result) {
        cout << size << " ";
    }
    return 0;
}
```

### Explanation:

1. **Step 1: Last Occurrence**  
   - Compute the last occurrence of each character in the string.  
   - This is done using a vector where the index corresponds to the character (e.g., `a` maps to index 0).

2. **Step 2: Partition**  
   - Traverse the string and maintain a variable `end` that tracks the furthest index any character in the current partition can reach.  
   - If the current index equals `end`, it means the partition ends here. Add the partition size to the result.

3. **Complexity**  
   - **Time Complexity**: \(O(n)\) where \(n\) is the length of the string (due to single traversal).  
   - **Space Complexity**: \(O(1)\) because the `lastIndex` array has a fixed size of 26.
