
##101 ****[Problem Link]https://leetcode.com/problems/buddy-strings****  
**Approach:** Check for two mismatched characters. If there are exactly two, swap them to see if the strings match. Special case for duplicates.  
**Time Complexity:** O(n)  

```cpp
#include <string>
#include <unordered_set>
using namespace std;

class Solution {
public:
    bool buddyStrings(string s, string goal) {
        if (s.size() != goal.size()) return false;
        if (s == goal) {
            unordered_set<char> seen(s.begin(), s.end());
            return seen.size() < s.size();
        }
        vector<int> diff;
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] != goal[i]) diff.push_back(i);
        }
        return diff.size() == 2 && s[diff[0]] == goal[diff[1]] && s[diff[1]] == goal[diff[0]];
    }
};
```

---

##102 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden****  
**Approach:** Use a greedy algorithm to maximize coverage using the fewest taps possible.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minTaps(int n, vector<int>& ranges) {
        vector<int> maxReach(n + 1, 0);
        for (int i = 0; i <= n; ++i) {
            int left = max(0, i - ranges[i]);
            int right = min(n, i + ranges[i]);
            maxReach[left] = max(maxReach[left], right);
        }
        int taps = 0, end = 0, farthest = 0;
        for (int i = 0; i <= n; ++i) {
            if (i > farthest) return -1;
            if (i > end) {
                taps++;
                end = farthest;
            }
            farthest = max(farthest, maxReach[i]);
        }
        return taps;
    }
};
```

---

##103 ****[Problem Link]https://leetcode.com/problems/kids-with-the-greatest-number-of-candies****  
**Approach:** Find the maximum number of candies and compare each kid's candies with it.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies) {
        int maxCandies = *max_element(candies.begin(), candies.end());
        vector<bool> result;
        for (int candy : candies) {
            result.push_back(candy + extraCandies >= maxCandies);
        }
        return result;
    }
};
```

---

##104 ****[Problem Link]https://leetcode.com/problems/repeated-string-match****  
**Approach:** Keep appending the string until it is longer than the target or contains the target using KMP algorithm.  
**Time Complexity:** O(n * m)  

```cpp
#include <string>
using namespace std;

class Solution {
public:
    int repeatedStringMatch(string a, string b) {
        int count = 1;
        string repeated = a;
        while (repeated.size() < b.size()) {
            repeated += a;
            count++;
        }
        if (repeated.find(b) != string::npos) return count;
        repeated += a;
        return (repeated.find(b) != string::npos) ? count + 1 : -1;
    }
};
```

---

##105 ****[Problem Link]https://leetcode.com/problems/satisfiability-of-equality-equations****  
**Approach:** Use Union-Find to check if contradictory equations exist.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    vector<int> parent;

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unionNodes(int x, int y) {
        parent[find(x)] = find(y);
    }

    bool equationsPossible(vector<string>& equations) {
        parent.resize(26);
        for (int i = 0; i < 26; ++i) parent[i] = i;
        for (const string& eq : equations) {
            if (eq[1] == '=') {
                unionNodes(eq[0] - 'a', eq[3] - 'a');
            }
        }
        for (const string& eq : equations) {
            if (eq[1] == '!' && find(eq[0] - 'a') == find(eq[3] - 'a')) {
                return false;
            }
        }
        return true;
    }
};
```
##106 ****[Problem Link]https://leetcode.com/problems/smallest-string-with-swaps****  
**Approach:** Use Union-Find to identify connected components and sort them to generate the lexicographically smallest string.  
**Time Complexity:** O(n log n)  

```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
using namespace std;

class Solution {
public:
    vector<int> parent;
    int find(int x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    }

    void unionNodes(int x, int y) {
        parent[find(x)] = find(y);
    }

    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
        int n = s.size();
        parent.resize(n);
        iota(parent.begin(), parent.end(), 0);

        for (auto& p : pairs) unionNodes(p[0], p[1]);

        unordered_map<int, priority_queue<char, vector<char>, greater<char>>> map;
        for (int i = 0; i < n; i++) {
            map[find(i)].push(s[i]);
        }

        for (int i = 0; i < n; i++) {
            s[i] = map[find(i)].top();
            map[find(i)].pop();
        }
        return s;
    }
};
```

---

##107 ****[Problem Link]https://leetcode.com/problems/shortest-path-visiting-all-nodes****  
**Approach:** Use BFS with bit masking to track visited nodes and find the shortest path.  
**Time Complexity:** O(n * 2^n)  

```cpp
#include <vector>
#include <queue>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int shortestPathLength(vector<vector<int>>& graph) {
        int n = graph.size();
        if (n == 1) return 0;

        int allVisited = (1 << n) - 1;
        queue<pair<int, int>> q;
        vector<vector<bool>> visited(n, vector<bool>(1 << n, false));

        for (int i = 0; i < n; i++) {
            q.push({i, 1 << i});
            visited[i][1 << i] = true;
        }

        int steps = 0;
        while (!q.empty()) {
            int size = q.size();
            while (size--) {
                auto [node, mask] = q.front(); q.pop();
                if (mask == allVisited) return steps;
                for (int neighbor : graph[node]) {
                    int nextMask = mask | (1 << neighbor);
                    if (!visited[neighbor][nextMask]) {
                        q.push({neighbor, nextMask});
                        visited[neighbor][nextMask] = true;
                    }
                }
            }
            steps++;
        }
        return -1;
    }
};
```

---

##108 ****[Problem Link]https://leetcode.com/problems/minimum-moves-to-equal-array-elements****  
**Approach:** The optimal solution is to increment all elements except one, which is equivalent to decrementing one element. Min moves = sum(nums) - min(nums) * n.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <numeric>
using namespace std;

class Solution {
public:
    int minMoves(vector<int>& nums) {
        int minNum = *min_element(nums.begin(), nums.end());
        int sum = accumulate(nums.begin(), nums.end(), 0);
        return sum - minNum * nums.size();
    }
};
```

---

##109 ****[Problem Link]https://leetcode.com/problems/knight-dialer****  
**Approach:** Use dynamic programming to track moves from each digit.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int knightDialer(int n) {
        const int MOD = 1e9 + 7;
        vector<vector<int>> moves = {{4, 6}, {6, 8}, {7, 9}, {4, 8}, {3, 9, 0}, {}, {1, 7, 0}, {2, 6}, {1, 3}, {2, 4}};
        vector<int> dp(10, 1);

        for (int i = 1; i < n; i++) {
            vector<int> temp(10);
            for (int j = 0; j < 10; j++) {
                for (int move : moves[j]) {
                    temp[move] = (temp[move] + dp[j]) % MOD;
                }
            }
            dp = temp;
        }
        return accumulate(dp.begin(), dp.end(), 0L) % MOD;
    }
};
```

---

##110 ****[Problem Link]https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list****  
**Approach:** Use a prefix sum and hashmap to detect and remove zero-sum sublists.  
**Time Complexity:** O(n)  

```cpp
#include <unordered_map>
using namespace std;


class Solution {
public:
    ListNode* removeZeroSumSublists(ListNode* head) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        unordered_map<int, ListNode*> sumMap;
        int sum = 0;
        for (ListNode* node = dummy; node; node = node->next) {
            sum += node->val;
            sumMap[sum] = node;
        }
        sum = 0;
        for (ListNode* node = dummy; node; node = node->next) {
            sum += node->val;
            node->next = sumMap[sum]->next;
        }
        return dummy->next;
    }
};
```
##111 ****[Problem Link]https://leetcode.com/problems/best-sightseeing-pair****  
**Approach:** Use a single pass with dynamic tracking of the maximum value to compute the best sightseeing pair.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int maxScoreSightseeingPair(vector<int>& values) {
        int maxScore = 0;
        int maxVal = values[0];
        for (int i = 1; i < values.size(); i++) {
            maxScore = max(maxScore, maxVal + values[i] - i);
            maxVal = max(maxVal, values[i] + i);
        }
        return maxScore;
    }
};
```

---

##112 ****[Problem Link]https://leetcode.com/problems/power-of-four****  
**Approach:** Check if a number is a power of four using bit manipulation. Ensure only one bit is set at an even position.  
**Time Complexity:** O(1)  

```cpp
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0x55555555);
    }
};
```

---

##113 ****[Problem Link]https://leetcode.com/problems/longest-word-in-dictionary-through-deleting****  
**Approach:** Sort the dictionary and use a two-pointer technique to check if a word can be formed by deleting characters.  
**Time Complexity:** O(n * m) where n is the dictionary size and m is the word length.  

```cpp
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class Solution {
public:
    string findLongestWord(string s, vector<string>& dictionary) {
        sort(dictionary.begin(), dictionary.end(), [](string &a, string &b) {
            return a.size() == b.size() ? a < b : a.size() > b.size();
        });

        for (string& word : dictionary) {
            int i = 0, j = 0;
            while (i < s.size() && j < word.size()) {
                if (s[i] == word[j]) j++;
                i++;
            }
            if (j == word.size()) return word;
        }
        return "";
    }
};
```

---

##114 ****[Problem Link]https://leetcode.com/problems/maximum-units-on-a-truck****  
**Approach:** Greedily choose the box types with the most units first using sorting.  
**Time Complexity:** O(n log n)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maximumUnits(vector<vector<int>>& boxTypes, int truckSize) {
        sort(boxTypes.begin(), boxTypes.end(), [](vector<int>& a, vector<int>& b) {
            return a[1] > b[1];
        });

        int units = 0;
        for (auto& box : boxTypes) {
            int count = min(truckSize, box[0]);
            units += count * box[1];
            truckSize -= count;
            if (truckSize == 0) break;
        }
        return units;
    }
};
```

---

##115 ****[Problem Link]https://leetcode.com/problems/construct-string-from-binary-tree****  
**Approach:** Perform a preorder traversal to generate the string representation of the binary tree.  
**Time Complexity:** O(n)  

```cpp
#include <string>
using namespace std;


class Solution {
public:
    string tree2str(TreeNode* t) {
        if (!t) return "";
        string result = to_string(t->val);
        if (t->left || t->right) {
            result += "(" + tree2str(t->left) + ")";
        }
        if (t->right) {
            result += "(" + tree2str(t->right) + ")";
        }
        return result;
    }
};
```
##116 ****[Problem Link]https://leetcode.com/problems/assign-cookies****  
**Approach:** Greedily assign cookies to children by sorting both arrays and using a two-pointer technique.  
**Time Complexity:** O(n log n + m log m)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int i = 0, j = 0;
        while (i < g.size() && j < s.size()) {
            if (s[j] >= g[i]) i++;
            j++;
        }
        return i;
    }
};
```

---

##117 ****[Problem Link]https://leetcode.com/problems/scramble-string****  
**Approach:** Use dynamic programming with memoization to check if one string is a scramble of another.  
**Time Complexity:** O(n^4)  

```cpp
#include <string>
#include <unordered_map>
using namespace std;

class Solution {
public:
    unordered_map<string, bool> memo;
    
    bool isScramble(string s1, string s2) {
        if (s1 == s2) return true;
        if (s1.size() != s2.size()) return false;
        string key = s1 + "#" + s2;
        if (memo.find(key) != memo.end()) return memo[key];

        int n = s1.size();
        vector<int> count(26, 0);
        for (int i = 0; i < n; i++) {
            count[s1[i] - 'a']++;
            count[s2[i] - 'a']--;
        }
        for (int c : count) {
            if (c != 0) return memo[key] = false;
        }

        for (int i = 1; i < n; i++) {
            if ((isScramble(s1.substr(0, i), s2.substr(0, i)) &&
                 isScramble(s1.substr(i), s2.substr(i))) ||
                (isScramble(s1.substr(0, i), s2.substr(n - i)) &&
                 isScramble(s1.substr(i), s2.substr(0, n - i)))) {
                return memo[key] = true;
            }
        }
        return memo[key] = false;
    }
};
```

---

##118 ****[Problem Link]https://leetcode.com/problems/continuous-subarray-sum****  
**Approach:** Use a hashmap to store the prefix sum modulo k and check for repeated remainders.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> remainderMap;
        remainderMap[0] = -1;
        int sum = 0;
        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
            int remainder = (k == 0) ? sum : sum % k;
            if (remainderMap.count(remainder)) {
                if (i - remainderMap[remainder] > 1) return true;
            } else {
                remainderMap[remainder] = i;
            }
        }
        return false;
    }
};
```

---

##119 ****[Problem Link]https://leetcode.com/problems/minesweeper****  
**Approach:** Perform a BFS or DFS from the clicked cell to reveal empty cells recursively.  
**Time Complexity:** O(m * n)  

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    vector<vector<int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, 
                                      {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
    
    vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
        int x = click[0], y = click[1];
        if (board[x][y] == 'M') {
            board[x][y] = 'X';
            return board;
        }
        dfs(board, x, y);
        return board;
    }
    
    void dfs(vector<vector<char>>& board, int x, int y) {
        int m = board.size(), n = board[0].size();
        if (x < 0 || x >= m || y < 0 || y >= n || board[x][y] != 'E') return;

        int mines = 0;
        for (auto& dir : directions) {
            int nx = x + dir[0], ny = y + dir[1];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && board[nx][ny] == 'M') {
                mines++;
            }
        }

        if (mines > 0) {
            board[x][y] = '0' + mines;
        } else {
            board[x][y] = 'B';
            for (auto& dir : directions) {
                dfs(board, x + dir[0], y + dir[1]);
            }
        }
    }
};
```

---

##120 ****[Problem Link]https://leetcode.com/problems/bulls-and-cows****  
**Approach:** Count bulls directly, and use a frequency array to calculate cows.  
**Time Complexity:** O(n)  

```cpp
#include <string>
#include <vector>
using namespace std;

class Solution {
public:
    string getHint(string secret, string guess) {
        int bulls = 0, cows = 0;
        vector<int> sCount(10, 0), gCount(10, 0);
        
        for (int i = 0; i < secret.size(); i++) {
            if (secret[i] == guess[i]) {
                bulls++;
            } else {
                sCount[secret[i] - '0']++;
                gCount[guess[i] - '0']++;
            }
        }
        
        for (int i = 0; i < 10; i++) {
            cows += min(sCount[i], gCount[i]);
        }
        
        return to_string(bulls) + "A" + to_string(cows) + "B";
    }
};
```

##121 ****[Problem Link]https://leetcode.com/problems/design-linked-list****  
**Approach:** Implement a doubly linked list with functions for adding, deleting, and getting nodes.  
**Time Complexity:** O(1) for add and delete, O(n) for get.  

```cpp
class Node {
public:
    int val;
    Node* prev;
    Node* next;
    Node(int x) : val(x), prev(nullptr), next(nullptr) {}
};

class MyLinkedList {
public:
    Node* head;
    Node* tail;
    int size;

    MyLinkedList() {
        head = new Node(0);
        tail = new Node(0);
        head->next = tail;
        tail->prev = head;
        size = 0;
    }

    int get(int index) {
        if (index < 0 || index >= size) return -1;
        Node* curr = head->next;
        for (int i = 0; i < index; i++) {
            curr = curr->next;
        }
        return curr->val;
    }

    void addAtHead(int val) {
        addAtIndex(0, val);
    }

    void addAtTail(int val) {
        addAtIndex(size, val);
    }

    void addAtIndex(int index, int val) {
        if (index < 0 || index > size) return;
        Node* prevNode = head;
        for (int i = 0; i < index; i++) {
            prevNode = prevNode->next;
        }
        Node* newNode = new Node(val);
        newNode->next = prevNode->next;
        newNode->prev = prevNode;
        prevNode->next->prev = newNode;
        prevNode->next = newNode;
        size++;
    }

    void deleteAtIndex(int index) {
        if (index < 0 || index >= size) return;
        Node* curr = head;
        for (int i = 0; i <= index; i++) {
            curr = curr->next;
        }
        curr->prev->next = curr->next;
        curr->next->prev = curr->prev;
        delete curr;
        size--;
    }
};
```

---

##122 ****[Problem Link]https://leetcode.com/problems/hand-of-straights****  
**Approach:** Use a hashmap to count card frequencies and a min-heap to form hands.  
**Time Complexity:** O(n log n)  

```cpp
class Solution {
public:
    bool isNStraightHand(vector<int>& hand, int groupSize) {
        if (hand.size() % groupSize != 0) return false; // Quick check

        map<int, int> count;
        for (int card : hand) count[card]++; // Count frequencies

        for (auto it = count.begin(); it != count.end(); ++it) {
            int start = it->first;  
            int freq = it->second;
            if (freq > 0) { // We must form a group starting from 'start'
                for (int i = 0; i < groupSize; i++) {
                    if (count[start + i] < freq) return false;
                    count[start + i] -= freq; // Reduce usage
                }
            }
        }
        return true;
    }
};
```

---

##123 ****[Problem Link]https://leetcode.com/problems/odd-even-jump****  
**Approach:** Use dynamic programming with sorted maps to track odd and even jumps.  
**Time Complexity:** O(n log n)  

```cpp
class Solution {
public:
    int oddEvenJumps(vector<int>& arr) {
        int n = arr.size();
        vector<bool> odd(n, false), even(n, false);
        odd[n - 1] = even[n - 1] = true;
        
        map<int, int> indexMap; // Ordered map to store indices
        indexMap[arr[n - 1]] = n - 1;
        int res = 1;

        for (int i = n - 2; i >= 0; --i) {
            auto hi = indexMap.lower_bound(arr[i]);  // Find smallest >= arr[i]
            auto lo = indexMap.upper_bound(arr[i]);  // Find largest < arr[i]
            if (hi != indexMap.end()) odd[i] = even[hi->second]; 
            if (lo != indexMap.begin()) even[i] = odd[prev(lo)->second]; 
            if (odd[i]) res++; 
            
            indexMap[arr[i]] = i;  // Store index for future jumps
        }
        return res;
    }
};
```

---

##124 ****[Problem Link]https://leetcode.com/problems/arithmetic-slices-ii-subsequence****  
**Approach:** Use dynamic programming with a hashmap to store the difference and count.  
**Time Complexity:** O(n^2)  

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int numberOfArithmeticSlices(vector<int>& nums) {
        int n = nums.size(), res = 0;
        vector<unordered_map<long long, int>> dp(n);

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                long long diff = (long long)nums[i] - nums[j];
                int count = dp[j].count(diff) ? dp[j][diff] : 0;
                dp[i][diff] += count + 1;
                res += count;
            }
        }
        return res;
    }
};
```

---

##125 ****[Problem Link]https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii****  
**Approach:** Calculate the median and find the sum of absolute differences.  
**Time Complexity:** O(n log n)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minMoves2(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int median = nums[nums.size() / 2];
        int moves = 0;
        for (int num : nums) {
            moves += abs(num - median);
        }
        return moves;
    }
};
```
##126 ****[Problem Link]https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree****  
**Approach:** Perform level-order traversal using BFS and calculate the sum of nodes at each level. Track the maximum sum.  
**Time Complexity:** O(n)  

```cpp
#include <queue>
#include <vector>
using namespace std;


class Solution {
public:
    int maxLevelSum(TreeNode* root) {
        if (!root) return 0;
        queue<TreeNode*> q;
        q.push(root);
        int level = 1, maxLevel = 1, maxSum = root->val;

        while (!q.empty()) {
            int size = q.size(), levelSum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front(); q.pop();
                levelSum += node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            if (levelSum > maxSum) {
                maxSum = levelSum;
                maxLevel = level;
            }
            level++;
        }
        return maxLevel;
    }
};
```

---

##127 ****[Problem Link]https://leetcode.com/problems/create-maximum-number****  
**Approach:** Use a monotonic stack to get the maximum number from two arrays using a merge technique.  
**Time Complexity:** O(n^3)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        vector<int> result;
        int m = nums1.size(), n = nums2.size();

        for (int i = max(0, k - n); i <= min(k, m); i++) {
            auto candidate = merge(getMax(nums1, i), getMax(nums2, k - i));
            result = max(result, candidate);
        }
        return result;
    }

    vector<int> getMax(vector<int>& nums, int k) {
        vector<int> res;
        int toPop = nums.size() - k;
        for (int num : nums) {
            while (!res.empty() && toPop > 0 && res.back() < num) {
                res.pop_back();
                toPop--;
            }
            res.push_back(num);
        }
        res.resize(k);
        return res;
    }

    vector<int> merge(vector<int> nums1, vector<int> nums2) {
        vector<int> res;
        while (!nums1.empty() || !nums2.empty()) {
            if (nums1 > nums2) {
                res.push_back(nums1.front());
                nums1.erase(nums1.begin());
            } else {
                res.push_back(nums2.front());
                nums2.erase(nums2.begin());
            }
        }
        return res;
    }
};
```

---

##128 ****[Problem Link]https://leetcode.com/problems/delete-leaves-with-a-given-value****  
**Approach:** Perform a post-order traversal and delete nodes if they match the target value.  
**Time Complexity:** O(n)  

```cpp

class Solution {
public:
    TreeNode* removeLeafNodes(TreeNode* root, int target) {
        if (!root) return nullptr;
        root->left = removeLeafNodes(root->left, target);
        root->right = removeLeafNodes(root->right, target);
        if (!root->left && !root->right && root->val == target) {
            return nullptr;
        }
        return root;
    }
};
```

---

##129 ****[Problem Link]https://leetcode.com/problems/find-smallest-letter-greater-than-target****  
**Approach:** Perform binary search to find the smallest letter greater than the target.  
**Time Complexity:** O(log n)  

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    char nextGreatestLetter(vector<char>& letters, char target) {
        int low = 0, high = letters.size() - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (letters[mid] <= target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return letters[low % letters.size()];
    }
};
```

---

##130 ****[Problem Link]https://leetcode.com/problems/longest-word-in-dictionary****  
**Approach:** Use a trie with BFS to find the longest word with all prefixes present.  
**Time Complexity:** O(n * k)  

```cpp
#include <string>
#include <unordered_set>
#include <queue>
using namespace std;

class TrieNode {
public:
    TrieNode* children[26] = {};
    bool isEnd = false;
};

class Solution {
public:
    string longestWord(vector<string>& words) {
        TrieNode* root = new TrieNode();
        for (const string& word : words) {
            TrieNode* node = root;
            for (char ch : word) {
                if (!node->children[ch - 'a']) {
                    node->children[ch - 'a'] = new TrieNode();
                }
                node = node->children[ch - 'a'];
            }
            node->isEnd = true;
        }

        queue<pair<TrieNode*, string>> q;
        q.push({root, ""});
        string result = "";

        while (!q.empty()) {
            auto [node, word] = q.front();
            q.pop();
            for (int i = 0; i < 26; i++) {
                if (node->children[i] && node->children[i]->isEnd) {
                    string nextWord = word + char(i + 'a');
                    q.push({node->children[i], nextWord});
                    if (nextWord.size() > result.size() || (nextWord.size() == result.size() && nextWord < result)) {
                        result = nextWord;
                    }
                }
            }
        }
        return result;
    }
};
```
##131 ****[Problem Link]https://leetcode.com/problems/heaters****  
**Approach:** Perform binary search to find the nearest heater for each house.  
**Time Complexity:** O(n log n)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int findRadius(vector<int>& houses, vector<int>& heaters) {
        sort(houses.begin(), houses.end());
        sort(heaters.begin(), heaters.end());
        int radius = 0;

        for (int house : houses) {
            auto it = lower_bound(heaters.begin(), heaters.end(), house);
            int rightDist = (it == heaters.end()) ? INT_MAX : *it - house;
            int leftDist = (it == heaters.begin()) ? INT_MAX : house - *(it - 1);
            radius = max(radius, min(leftDist, rightDist));
        }
        return radius;
    }
};
```

---

##132 ****[Problem Link]https://leetcode.com/problems/largest-component-size-by-common-factor****  
**Approach:** Use Union-Find to group numbers with common factors.  
**Time Complexity:** O(n * log n)  

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    vector<int> parent;

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unionNodes(int x, int y) {
        parent[find(x)] = find(y);
    }

    int largestComponentSize(vector<int>& nums) {
        int n = nums.size();
        int maxNum = *max_element(nums.begin(), nums.end());
        parent.resize(maxNum + 1);
        for (int i = 0; i <= maxNum; i++) parent[i] = i;

        for (int num : nums) {
            for (int i = 2; i * i <= num; i++) {
                if (num % i == 0) {
                    unionNodes(num, i);
                    unionNodes(num, num / i);
                }
            }
        }

        unordered_map<int, int> componentCount;
        int maxComponentSize = 0;
        for (int num : nums) {
            int root = find(num);
            maxComponentSize = max(maxComponentSize, ++componentCount[root]);
        }
        return maxComponentSize;
    }
};
```

---

##133 ****[Problem Link]https://leetcode.com/problems/linked-list-in-binary-tree****  
**Approach:** Perform a DFS to check if the linked list is present as a path in the binary tree.  
**Time Complexity:** O(n * m)  

```cpp

class Solution {
public:
    bool isSubPath(ListNode* head, TreeNode* root) {
        if (!root) return false;
        return dfs(head, root) || isSubPath(head, root->left) || isSubPath(head, root->right);
    }

    bool dfs(ListNode* head, TreeNode* node) {
        if (!head) return true;
        if (!node || head->val != node->val) return false;
        return dfs(head->next, node->left) || dfs(head->next, node->right);
    }
};
```

---

##134 ****[Problem Link]https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero****  
**Approach:** Perform BFS or DFS to count the edges that need to be reversed.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int minReorder(int n, vector<vector<int>>& connections) {
        vector<vector<int>> adj(n);
        unordered_set<string> directedEdges;

        for (auto& conn : connections) {
            adj[conn[0]].push_back(conn[1]);
            adj[conn[1]].push_back(conn[0]);
            directedEdges.insert(to_string(conn[0]) + "," + to_string(conn[1]));
        }

        int changes = 0;
        vector<bool> visited(n, false);
        dfs(0, adj, visited, directedEdges, changes);
        return changes;
    }

    void dfs(int node, vector<vector<int>>& adj, vector<bool>& visited, unordered_set<string>& directedEdges, int& changes) {
        visited[node] = true;
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                if (directedEdges.count(to_string(node) + "," + to_string(neighbor))) {
                    changes++;
                }
                dfs(neighbor, adj, visited, directedEdges, changes);
            }
        }
    }
};
```

---

##135 ****[Problem Link]https://leetcode.com/problems/sum-of-square-numbers****  
**Approach:** Use a two-pointer technique to check if any two numbers squared add up to the given number.  
**Time Complexity:** O(sqrt(n))  

```cpp
#include <cmath>
using namespace std;

class Solution {
public:
    bool judgeSquareSum(int c) {
        long long left = 0, right = sqrt(c);
        while (left <= right) {
            long long sum = left * left + right * right;
            if (sum == c) return true;
            else if (sum < c) left++;
            else right--;
        }
        return false;
    }
};
```
##136 ****[Problem Link]https://leetcode.com/problems/number-of-substrings-containing-all-three-characters****  
**Approach:** Use a sliding window to keep track of character frequencies and find all valid substrings.  
**Time Complexity:** O(n)  

```cpp
#include <string>
#include <vector>
using namespace std;

class Solution {
public:
    int numberOfSubstrings(string s) {
        vector<int> count(3, 0);
        int left = 0, res = 0;

        for (int right = 0; right < s.size(); right++) {
            count[s[right] - 'a']++;
            while (count[0] > 0 && count[1] > 0 && count[2] > 0) {
                count[s[left++] - 'a']--;
            }
            res += left;
        }
        return res;
    }
};
```

---

##137 ****[Problem Link]https://leetcode.com/problems/most-common-word****  
**Approach:** Use a hashmap to count word frequencies, ignoring banned words.  
**Time Complexity:** O(n)  

```cpp
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <cctype>
using namespace std;

class Solution {
public:
    string mostCommonWord(string paragraph, vector<string>& banned) {
        unordered_set<string> bannedSet(banned.begin(), banned.end());
        unordered_map<string, int> wordCount;
        string word, result;
        int maxCount = 0;

        for (char& c : paragraph) {
            c = isalpha(c) ? tolower(c) : ' ';
        }

        stringstream ss(paragraph);
        while (ss >> word) {
            if (!bannedSet.count(word)) {
                maxCount = max(maxCount, ++wordCount[word]);
                if (wordCount[word] == maxCount) result = word;
            }
        }
        return result;
    }
};
```

---

##138 ****[Problem Link]https://leetcode.com/problems/advantage-shuffle****  
**Approach:** Sort both arrays and use a two-pointer technique to maximize advantage.  
**Time Complexity:** O(n log n)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> advantageCount(vector<int>& nums1, vector<int>& nums2) {
        vector<int> sortedNums1(nums1.begin(), nums1.end());
        sort(sortedNums1.begin(), sortedNums1.end());
        vector<pair<int, int>> sortedNums2;

        for (int i = 0; i < nums2.size(); i++) {
            sortedNums2.push_back({nums2[i], i});
        }
        sort(sortedNums2.begin(), sortedNums2.end());

        int left = 0, right = nums2.size() - 1;
        vector<int> result(nums2.size());
        for (int num : sortedNums1) {
            if (num > sortedNums2[left].first) {
                result[sortedNums2[left].second] = num;
                left++;
            } else {
                result[sortedNums2[right].second] = num;
                right--;
            }
        }
        return result;
    }
};
```

---

##139 ****[Problem Link]https://leetcode.com/problems/prison-cells-after-n-days****  
**Approach:** Use bit manipulation to simulate the state of the cells and detect cycles.  
**Time Complexity:** O(1) due to cycle detection.  

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    vector<int> prisonAfterNDays(vector<int>& cells, int n) {
        unordered_map<int, int> seen;
        int state = 0;

        for (int cell : cells) {
            state <<= 1;
            state |= cell;
        }

        while (n > 0) {
            if (seen.count(state)) {
                n %= seen[state] - n;
            }
            seen[state] = n;
            if (n == 0) break;
            n--;

            int nextState = 0;
            for (int i = 1; i < 7; i++) {
                nextState |= ((state >> (i - 1) & 1) == (state >> (i + 1) & 1)) << i;
            }
            state = nextState;
        }

        vector<int> result(8);
        for (int i = 7; i >= 0; i--) {
            result[i] = state & 1;
            state >>= 1;
        }
        return result;
    }
};
```

---

##140 ****[Problem Link]https://leetcode.com/problems/longest-turbulent-subarray****  
**Approach:** Track the increasing and decreasing states using dynamic programming.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maxTurbulenceSize(vector<int>& arr) {
        int inc = 1, dec = 1, maxLen = 1;

        for (int i = 1; i < arr.size(); i++) {
            if (arr[i] > arr[i-1]) {
                inc = dec + 1;
                dec = 1;
            } else if (arr[i] < arr[i-1]) {
                dec = inc + 1;
                inc = 1;
            } else {
                inc = dec = 1;
            }
            maxLen = max(maxLen, max(inc, dec));
        }
        return maxLen;
    }
};
```
##141 ****[Problem Link]https://leetcode.com/problems/add-one-row-to-tree****  
**Approach:** Perform a level-order traversal using a queue. Insert the row at the specified depth.  
**Time Complexity:** O(n)  

```cpp
#include <queue>
using namespace std;


class Solution {
public:
    TreeNode* addOneRow(TreeNode* root, int val, int depth) {
        if (depth == 1) {
            TreeNode* newRoot = new TreeNode(val);
            newRoot->left = root;
            return newRoot;
        }
        
        queue<TreeNode*> q;
        q.push(root);
        int currentDepth = 1;
        
        while (!q.empty()) {
            int size = q.size();
            if (currentDepth == depth - 1) {
                for (int i = 0; i < size; i++) {
                    TreeNode* node = q.front(); q.pop();
                    TreeNode* leftChild = new TreeNode(val);
                    TreeNode* rightChild = new TreeNode(val);
                    leftChild->left = node->left;
                    rightChild->right = node->right;
                    node->left = leftChild;
                    node->right = rightChild;
                }
                return root;
            }
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front(); q.pop();
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            currentDepth++;
        }
        return root;
    }
};
```

---

##142 ****[Problem Link]https://leetcode.com/problems/count-different-palindromic-subsequences****  
**Approach:** Use dynamic programming to count unique palindromes.  
**Time Complexity:** O(n^2)  

```cpp
class Solution {
public:
    
    int dp[1001][1001];
    int mod = 1000*1000*1000+7;
    int dfs(int i, int j, string &s) {
        if(i > j) return 0;
        if(i==j) return 1;
        if(dp[i][j]) return dp[i][j];
        int ans=0;
        if(s[i]==s[j]) {
            ans = 2*dfs(i+1,j-1,s);
            int l=i+1, r=j-1;
            while(l<=r and s[l]!=s[i]) l++;
            while(l<=r and s[r]!=s[j]) r--;
            if(l < r) ans-=dfs(l+1,r-1,s);
            else if(l==r) ans++;
            else ans+=2;
        }
        else ans = dfs(i+1,j,s)+dfs(i,j-1,s)-dfs(i+1,j-1,s);
        return dp[i][j] = ans < 0 ? (ans+mod) % mod : ans % mod;
    }
    
    int countPalindromicSubsequences(string s) {
        memset(dp,0,sizeof(dp));
        return dfs(0,s.size()-1,s);
    }
};
```

---

##144 ****[Problem Link]https://leetcode.com/problems/ugly-number****  
**Approach:** Check if the number can be divided by 2, 3, or 5 until it becomes 1.  
**Time Complexity:** O(log n)  

```cpp
class Solution {
public:
    bool isUgly(int n) {
        if (n <= 0) return false;
        while (n % 2 == 0) n /= 2;
        while (n % 3 == 0) n /= 3;
        while (n % 5 == 0) n /= 5;
        return n == 1;
    }
};
```

---

##145 ****[Problem Link]https://leetcode.com/problems/stone-game-ii****  
**Approach:** Use dynamic programming with memoization to calculate the maximum stones a player can collect.  
**Time Complexity:** O(n^2)  

```cpp
class Solution {
public:
    int n;
    int dp[101][101];// stones diff
    int alice(int i, int m, vector<int>& piles){
        if (i == n) return 0;
        if (dp[i][m]!=-1) return dp[i][m];
        int diff=INT_MIN;
        int sum = 0, xN= min(2*m, n-i);
        for (int x = 1; x <= xN; x++) {
            sum += piles[i+x-1];
            diff=max(diff, sum-alice(i+x, max(m, x), piles));                      
        }
        return dp[i][m]=diff;
    }

    int stoneGameII(vector<int>& piles) {
        n = piles.size();
        memset(dp, -1, sizeof(dp));
        int sum=accumulate(piles.begin(), piles.end(), 0);
        return (sum+alice(0, 1,  piles))/2;// A=((A+B)+(A-B))/2
    }
};


auto init = []() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    return 'c';
}();
```
##146 ****[Problem Link]https://leetcode.com/problems/car-fleet****  
**Approach:** Sort cars by position and calculate the time required to reach the target. Count fleets using a stack.  
**Time Complexity:** O(n log n)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int carFleet(int target, vector<int>& position, vector<int>& speed) {
        int n = position.size();
        vector<pair<int, double>> cars;
        for (int i = 0; i < n; i++) {
            cars.emplace_back(position[i], (double)(target - position[i]) / speed[i]);
        }
        sort(cars.rbegin(), cars.rend());
        int fleets = 0;
        double curTime = 0;
        for (auto& car : cars) {
            if (car.second > curTime) {
                fleets++;
                curTime = car.second;
            }
        }
        return fleets;
    }
};
```

---

##147 ****[Problem Link]https://leetcode.com/problems/binary-subarrays-with-sum****  
**Approach:** Use a sliding window to count subarrays with sum equal to the target.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int numSubarraysWithSum(vector<int>& nums, int goal) {
        return atMost(nums, goal) - atMost(nums, goal - 1);
    }
    
    int atMost(vector<int>& nums, int goal) {
        if (goal < 0) return 0;
        int left = 0, sum = 0, count = 0;
        for (int right = 0; right < nums.size(); right++) {
            sum += nums[right];
            while (sum > goal) {
                sum -= nums[left++];
            }
            count += right - left + 1;
        }
        return count;
    }
};
```

---

##148 ****[Problem Link]https://leetcode.com/problems/consecutive-characters****  
**Approach:** Iterate through the string to find the maximum consecutive character length.  
**Time Complexity:** O(n)  

```cpp
#include <string>
using namespace std;

class Solution {
public:
    int maxPower(string s) {
        int maxLen = 1, curLen = 1;
        for (int i = 1; i < s.size(); i++) {
            if (s[i] == s[i - 1]) {
                curLen++;
                maxLen = max(maxLen, curLen);
            } else {
                curLen = 1;
            }
        }
        return maxLen;
    }
};
```

---

##149 ****[Problem Link]https://leetcode.com/problems/simplify-path****  
**Approach:** Use a stack to simulate the directory traversal using the given path.  
**Time Complexity:** O(n)  

```cpp
#include <string>
#include <sstream>
#include <vector>
using namespace std;

class Solution {
public:
    string simplifyPath(string path) {
        vector<string> stack;
        stringstream ss(path);
        string dir;
        
        while (getline(ss, dir, '/')) {
            if (dir == "..") {
                if (!stack.empty()) stack.pop_back();
            } else if (!dir.empty() && dir != ".") {
                stack.push_back(dir);
            }
        }
        
        string result = "/";
        for (int i = 0; i < stack.size(); i++) {
            result += stack[i];
            if (i < stack.size() - 1) result += "/";
        }
        return result;
    }
};
```

---

##150 ****[Problem Link]https://leetcode.com/problems/filling-bookcase-shelves****  
**Approach:** Use dynamic programming to find the minimum height required to fill the shelves.  
**Time Complexity:** O(n^2)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minHeightShelves(vector<vector<int>>& books, int shelfWidth) {
        int n = books.size();
        vector<int> dp(n + 1, 1e9);
        dp[0] = 0;

        for (int i = 1; i <= n; i++) {
            int width = 0, height = 0;
            for (int j = i; j > 0; j--) {
                width += books[j - 1][0];
                if (width > shelfWidth) break;
                height = max(height, books[j - 1][1]);
                dp[i] = min(dp[i], dp[j - 1] + height);
            }
        }
        return dp[n];
    }
};
```
##151 ****[Problem Link]https://leetcode.com/problems/univalued-binary-tree****  
**Approach:** Perform a DFS or BFS traversal to check if all nodes have the same value.  
**Time Complexity:** O(n)  

```cpp
#include <queue>
using namespace std;

class Solution {
public:
    bool isUnivalTree(TreeNode* root) {
        int val = root->val;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            if (node->val != val) return false;
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        return true;
    }
};
```

---

##152 ****[Problem Link]https://leetcode.com/problems/divisor-game****  
**Approach:** Alice wins if the number is even, otherwise Bob wins.  
**Time Complexity:** O(1)  

```cpp
class Solution {
public:
    bool divisorGame(int n) {
        return n % 2 == 0;
    }
};
```

---

##153 ****[Problem Link]https://leetcode.com/problems/guess-the-word****  
**Approach:** Use a strategy of minimizing the possible words based on matches from each guess.  
**Time Complexity:** O(n^2)  

```cpp
class Solution {
public:
    int solve(string s1, string s2) {
        int ans = 0;
        for (int i = 0; i < s1.size(); i++) {
            if (s1[i] == s2[i])
                ans++;
        }
        return ans;
    }

    void findSecretWord(vector<string>& words, Master& master) {
        unordered_set<string> st(words.begin(), words.end());

        while (!st.empty()) {
            string first = *st.begin();
            int guessed = master.guess(first);

            for (auto it = st.begin(); it!= st.end();) {
                if(solve(*it,first) != guessed) it = st.erase(it);
                else it++;
            }
            st.erase(first);
        }
    }
};
```

---

##154 ****[Problem Link]https://leetcode.com/problems/unique-morse-code-words****  
**Approach:** Convert words to their Morse code representations and use a set to find unique codes.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <string>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int uniqueMorseRepresentations(vector<string>& words) {
        vector<string> morse = {".-","-...","-.-.","-..",".","..-.","--.","....","..",
                                  ".---","-.-",".-..","--","-.","---",".--.","--.-",
                                  ".-.","...","-","..-","...-",".--","-..-","-.--",
                                  "--.."};
        unordered_set<string> transformations;
        for (const string& word : words) {
            string code = "";
            for (char c : word) {
                code += morse[c - 'a'];
            }
            transformations.insert(code);
        }
        return transformations.size();
    }
};
```

---

##155 ****[Problem Link]https://leetcode.com/problems/number-of-good-ways-to-split-a-string****  
**Approach:** Use prefix and suffix arrays to track unique character counts.  
**Time Complexity:** O(n)  

```cpp
#include <string>
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int numSplits(string s) {
        int n = s.size();
        vector<int> left(n), right(n);
        unordered_set<char> uniqueLeft, uniqueRight;

        for (int i = 0; i < n; i++) {
            uniqueLeft.insert(s[i]);
            left[i] = uniqueLeft.size();
        }

        for (int i = n - 1; i >= 0; i--) {
            uniqueRight.insert(s[i]);
            right[i] = uniqueRight.size();
        }

        int count = 0;
        for (int i = 0; i < n - 1; i++) {
            if (left[i] == right[i + 1]) count++;
        }
        return count;
    }
};
```
##156 ****[Problem Link]https://leetcode.com/problems/remove-outermost-parentheses****  
**Approach:** Use a counter to track open and close parentheses and remove outermost ones.  
**Time Complexity:** O(n)  

```cpp
#include <string>
using namespace std;

class Solution {
public:
    string removeOuterParentheses(string s) {
        string result;
        int count = 0;
        for (char c : s) {
            if (c == '(') {
                if (count > 0) result += c;
                count++;
            } else {
                count--;
                if (count > 0) result += c;
            }
        }
        return result;
    }
};
```

---

##157 ****[Problem Link]https://leetcode.com/problems/24-game****  
**Approach:** Use backtracking to try all possible operations and permutations of the numbers.  
**Time Complexity:** O(1) (Constant operations due to fixed input size)  

```cpp
#include <vector>
#include <cmath>
using namespace std;

class Solution {
public:
    bool judgePoint24(vector<int>& cards) {
        vector<double> nums(cards.begin(), cards.end());
        return solve(nums);
    }

    bool solve(vector<double>& nums) {
        if (nums.size() == 1) return abs(nums[0] - 24.0) < 1e-6;
        for (int i = 0; i < nums.size(); ++i) {
            for (int j = 0; j < nums.size(); ++j) {
                if (i == j) continue;
                vector<double> next;
                for (int k = 0; k < nums.size(); ++k) {
                    if (k != i && k != j) next.push_back(nums[k]);
                }
                for (double d : {nums[i] + nums[j], nums[i] - nums[j], nums[j] - nums[i],
                                  nums[i] * nums[j], nums[i] / nums[j], nums[j] / nums[i]}) {
                    if (isfinite(d)) {
                        next.push_back(d);
                        if (solve(next)) return true;
                        next.pop_back();
                    }
                }
            }
        }
        return false;
    }
};
```

---

##158 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes****  
**Approach:** Find nodes with zero in-degree, as they can't be reached from any other node.  
**Time Complexity:** O(n + m)  

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
        vector<int> inDegree(n, 0);
        for (auto& edge : edges) {
            inDegree[edge[1]]++;
        }
        vector<int> result;
        for (int i = 0; i < n; i++) {
            if (inDegree[i] == 0) result.push_back(i);
        }
        return result;
    }
};
```

---

##159 ****[Problem Link]https://leetcode.com/problems/jump-game-vi****  
**Approach:** Use a deque to maintain the maximum score within a sliding window.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <deque>
using namespace std;

class Solution {
public:
    int maxResult(vector<int>& nums, int k) {
        deque<int> dq;
        vector<int> dp(nums.size());
        dp[0] = nums[0];
        dq.push_back(0);
        for (int i = 1; i < nums.size(); ++i) {
            if (dq.front() < i - k) dq.pop_front();
            dp[i] = nums[i] + dp[dq.front()];
            while (!dq.empty() && dp[i] >= dp[dq.back()]) dq.pop_back();
            dq.push_back(i);
        }
        return dp.back();
    }
};
```

---

##160 ****[Problem Link]https://leetcode.com/problems/greatest-common-divisor-of-strings****  
**Approach:** Use GCD to find the largest common string divisor.  
**Time Complexity:** O(n)  

```cpp
#include <string>
using namespace std;

class Solution {
public:
    string gcdOfStrings(string str1, string str2) {
        if (str1 + str2 != str2 + str1) return "";
        return str1.substr(0, gcd(str1.size(), str2.size()));
    }
    
    int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }
};
```
##161 ****[Problem Link]https://leetcode.com/problems/maximum-subarray-sum-with-one-deletion****  
**Approach:** Use dynamic programming to track the max sum with and without deletion.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maximumSum(vector<int>& arr) {
        int n = arr.size();
        int noDelete = arr[0], oneDelete = 0, maxSum = arr[0];
        for (int i = 1; i < n; i++) {
            oneDelete = max(noDelete, oneDelete + arr[i]);
            noDelete = max(arr[i], noDelete + arr[i]);
            maxSum = max({maxSum, noDelete, oneDelete});
        }
        return maxSum;
    }
};
```

---

##162 ****[Problem Link]https://leetcode.com/problems/maximum-product-of-splitted-binary-tree****  
**Approach:** Perform a post-order traversal to compute subtree sums and maximize the product.  
**Time Complexity:** O(n)  

```cpp
#include <iostream>
using namespace std;

class Solution {
public:
    long totalSum = 0, maxProductResult = 0;
    int MOD = 1e9 + 7;

    int calculateSum(TreeNode* root) {
        if (!root) return 0;
        int sum = root->val + calculateSum(root->left) + calculateSum(root->right);
        totalSum += root->val;
        return sum;
    }

    int findMaxProduct(TreeNode* root) {
        if (!root) return 0;
        int subSum = root->val + findMaxProduct(root->left) + findMaxProduct(root->right);
        maxProductResult = max(maxProductResult, subSum * (totalSum - subSum));
        return subSum;
    }

    int maxProduct(TreeNode* root) {
        calculateSum(root);
        findMaxProduct(root);
        return maxProductResult % MOD;
    }
};
```

---

##163 ****[Problem Link]https://leetcode.com/problems/linked-list-random-node****  
**Approach:** Use Reservoir Sampling to select a random node.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <cstdlib>
using namespace std;


class Solution {
public:
    ListNode* head;

    Solution(ListNode* head) {
        this->head = head;
    }

    int getRandom() {
        ListNode* curr = head;
        int result = curr->val;
        int i = 1;
        while (curr->next) {
            curr = curr->next;
            if (rand() % (++i) == 0) {
                result = curr->val;
            }
        }
        return result;
    }
};
```

---

##164 ****[Problem Link]https://leetcode.com/problems/map-sum-pairs****  
**Approach:** Use a Trie with prefix sums for efficient queries.  
**Time Complexity:** O(n)  

```cpp
#include <unordered_map>
#include <string>
using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    int value = 0;
};

class MapSum {
private:
    TrieNode* root;
    unordered_map<string, int> keyMap;

public:
    MapSum() {
        root = new TrieNode();
    }

    void insert(string key, int val) {
        int delta = val - keyMap[key];
        keyMap[key] = val;
        TrieNode* node = root;
        for (char ch : key) {
            if (!node->children.count(ch)) {
                node->children[ch] = new TrieNode();
            }
            node = node->children[ch];
            node->value += delta;
        }
    }

    int sum(string prefix) {
        TrieNode* node = root;
        for (char ch : prefix) {
            if (!node->children.count(ch)) {
                return 0;
            }
            node = node->children[ch];
        }
        return node->value;
    }
};
```

---

##165 ****[Problem Link]https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards****  
**Approach:** Use GCD to find the common divisor for all card counts.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <unordered_map>
#include <numeric>
using namespace std;

class Solution {
public:
    bool hasGroupsSizeX(vector<int>& deck) {
        unordered_map<int, int> count;
        for (int card : deck) count[card]++;
        int gcdVal = 0;
        for (auto& pair : count) gcdVal = gcd(gcdVal, pair.second);
        return gcdVal >= 2;
    }
};
```
