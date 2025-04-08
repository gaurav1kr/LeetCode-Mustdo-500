
##1 ****[Problem Link]https://leetcode.com/problems/minimize-malware-spread****  
**Approach:** Use DFS/Union-Find to identify connected components and count infections per component.  
**Time Complexity:** O(N^2)

```cpp
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial) {
        int n = graph.size();
        vector<int> parent(n);
        iota(parent.begin(), parent.end(), 0);

        function<int(int)> find = [&](int u) {
            if (parent[u] != u) parent[u] = find(parent[u]);
            return parent[u];
        };

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (graph[i][j] && i != j)
                    parent[find(i)] = find(j);

        unordered_map<int, int> size;
        for (int i = 0; i < n; ++i)
            size[find(i)]++;

        unordered_map<int, int> infectedCount;
        for (int node : initial)
            infectedCount[find(node)]++;

        sort(initial.begin(), initial.end());
        int res = initial[0], maxSaved = -1;
        for (int node : initial) {
            int root = find(node);
            if (infectedCount[root] == 1 && size[root] > maxSaved) {
                maxSaved = size[root];
                res = node;
            }
        }
        return res;
    }
};
```

---

##2 ****[Problem Link]https://leetcode.com/problems/rank-transform-of-a-matrix****  
**Approach:** Use Union-Find to group equal values and assign ranks row and column wise.  
**Time Complexity:** O(mn log(mn))

```cpp
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<int>> matrixRankTransform(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> res(m, vector<int>(n));
        map<int, vector<pair<int, int>>> valueCells;

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                valueCells[matrix[i][j]].emplace_back(i, j);

        vector<int> row(m), col(n);
        for (auto& [val, cells] : valueCells) {
            unordered_map<int, int> parent;
            function<int(int)> find = [&](int u) {
                if (!parent.count(u)) parent[u] = u;
                if (u != parent[u]) parent[u] = find(parent[u]);
                return parent[u];
            };
            for (auto& [i, j] : cells)
                parent[find(i)] = find(~j);

            unordered_map<int, int> maxRank;
            for (auto& [i, j] : cells)
                maxRank[find(i)] = max(maxRank[find(i)], max(row[i], col[j]));

            for (auto& [i, j] : cells) {
                res[i][j] = maxRank[find(i)] + 1;
                row[i] = col[j] = res[i][j];
            }
        }
        return res;
    }
};
```

---

##3 ****[Problem Link]https://leetcode.com/problems/minimum-deletions-to-make-string-balanced****  
**Approach:** Count right-side 'a's and track left-side 'b's to compute min deletions.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minimumDeletions(string s) {
        int rightA = count(s.begin(), s.end(), 'a');
        int res = rightA, leftB = 0;
        for (char c : s) {
            if (c == 'a') rightA--;
            else leftB++;
            res = min(res, leftB + rightA);
        }
        return res;
    }
};
```

---

##4 ****[Problem Link]https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies****  
**Approach:** Build topological sort for items and groups separately.  
**Time Complexity:** O(n + e)

```cpp
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Solution {
public:
    vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) {
        for (int i = 0; i < n; ++i)
            if (group[i] == -1) group[i] = m++;

        vector<vector<int>> groupItems(m), itemGraph(n), groupGraph(m);
        vector<int> itemIndegree(n, 0), groupIndegree(m, 0);

        for (int i = 0; i < n; ++i) {
            groupItems[group[i]].push_back(i);
            for (int pre : beforeItems[i]) {
                itemGraph[pre].push_back(i);
                itemIndegree[i]++;
                if (group[i] != group[pre]) {
                    groupGraph[group[pre]].push_back(group[i]);
                    groupIndegree[group[i]]++;
                }
            }
        }

        auto topSort = [](vector<vector<int>>& graph, vector<int>& indegree, vector<int>& nodes) {
            queue<int> q;
            for (int i : nodes)
                if (indegree[i] == 0) q.push(i);
            vector<int> res;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                res.push_back(u);
                for (int v : graph[u])
                    if (--indegree[v] == 0) q.push(v);
            }
            return res.size() == nodes.size() ? res : vector<int>();
        };

        vector<int> groupOrder, itemOrder;
        for (int i = 0; i < m; ++i) groupOrder.push_back(i);
        groupOrder = topSort(groupGraph, groupIndegree, groupOrder);
        if (groupOrder.empty()) return {};

        vector<int> res;
        for (int g : groupOrder) {
            vector<int> items = topSort(itemGraph, itemIndegree, groupItems[g]);
            if (items.empty()) return {};
            res.insert(res.end(), items.begin(), items.end());
        }
        return res;
    }
};
```

---

##5 ****[Problem Link]https://leetcode.com/problems/special-array-with-x-elements-greater-than-or-equal-x****  
**Approach:** Sort array and use binary search to find valid x.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int specialArray(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        for (int x = 0; x <= n; ++x) {
            int cnt = lower_bound(nums.begin(), nums.end(), x) - nums.begin();
            if (n - cnt == x) return x;
        }
        return -1;
    }
};
```

---

##6 ****[Problem Link]https://leetcode.com/problems/the-kth-factor-of-n****  
**Approach:** Iterate through numbers from 1 to n and collect divisors.  
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int kthFactor(int n, int k) {
        for (int i = 1; i <= n; ++i) {
            if (n % i == 0 && --k == 0)
                return i;
        }
        return -1;
    }
};
```

---

##7 ****[Problem Link]https://leetcode.com/problems/rotated-digits****  
**Approach:** Count numbers that differ from original after rotation and remain valid.  
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int rotatedDigits(int n) {
        int count = 0;
        for (int i = 1; i <= n; ++i) {
            string s = to_string(i);
            bool valid = true, diff = false;
            for (char c : s) {
                if (c == '3' || c == '4' || c == '7') {
                    valid = false;
                    break;
                }
                if (c == '2' || c == '5' || c == '6' || c == '9')
                    diff = true;
            }
            if (valid && diff) count++;
        }
        return count;
    }
};
```

---

##8 ****[Problem Link]https://leetcode.com/problems/minimum-insertions-to-balance-a-parentheses-string****  
**Approach:** Track unmatched ')' and needed insertions for '(' as per the rule.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int minInsertions(string s) {
        int res = 0, need = 0;
        for (char c : s) {
            if (c == '(') {
                need += 2;
                if (need % 2) {
                    res++;
                    need--;
                }
            } else {
                need--;
                if (need < 0) {
                    res++;
                    need = 1;
                }
            }
        }
        return res + need;
    }
};
```

---

##9 ****[Problem Link]https://leetcode.com/problems/longest-uncommon-subsequence-i****  
**Approach:** If strings are equal, return -1; else return length of longer.  
**Time Complexity:** O(1)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int findLUSlength(string a, string b) {
        return a == b ? -1 : max(a.length(), b.length());
    }
};
```

---

##10 ****[Problem Link]https://leetcode.com/problems/number-of-boomerangs****  
**Approach:** For each point, count distance frequencies to find permutations.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int numberOfBoomerangs(vector<vector<int>>& points) {
        int res = 0;
        for (auto& p : points) {
            unordered_map<int, int> count;
            for (auto& q : points) {
                int dx = p[0] - q[0], dy = p[1] - q[1];
                count[dx * dx + dy * dy]++;
            }
            for (auto& [_, v] : count)
                res += v * (v - 1);
        }
        return res;
    }
};
```

---

##11 ****[Problem Link]https://leetcode.com/problems/reordered-power-of-2****  
**Approach:** Count digit frequencies and compare with all powers of 2.  
**Time Complexity:** O(log n)

```cpp
#include <string>
#include <algorithm>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool reorderedPowerOf2(int n) {
        string s = to_string(n);
        sort(s.begin(), s.end());
        for (int i = 0; i < 31; ++i) {
            string t = to_string(1 << i);
            sort(t.begin(), t.end());
            if (s == t) return true;
        }
        return false;
    }
};
```

---

##12 ****[Problem Link]https://leetcode.com/problems/find-lucky-integer-in-an-array****  
**Approach:** Count frequency and return max number where count equals the number.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int findLucky(vector<int>& arr) {
        unordered_map<int, int> count;
        for (int n : arr) count[n]++;
        int res = -1;
        for (auto& [k, v] : count)
            if (k == v) res = max(res, k);
        return res;
    }
};
```

---

##13 ****[Problem Link]https://leetcode.com/problems/maximal-network-rank****  
**Approach:** Count degrees and check all city pairs with/without shared edge.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int maximalNetworkRank(int n, vector<vector<int>>& roads) {
        vector<int> deg(n, 0);
        unordered_set<int> connected[n];
        for (auto& r : roads) {
            deg[r[0]]++;
            deg[r[1]]++;
            connected[r[0]].insert(r[1]);
            connected[r[1]].insert(r[0]);
        }
        int res = 0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                int rank = deg[i] + deg[j] - (connected[i].count(j) ? 1 : 0);
                res = max(res, rank);
            }
        return res;
    }
};
```

---

##14 ****[Problem Link]https://leetcode.com/problems/egg-drop-with-2-eggs-and-n-floors****  
**Approach:** Use DP or find minimal x where x(x+1)/2 ≥ n.  
**Time Complexity:** O(sqrt(n))

```cpp
class Solution {
public:
    int twoEggDrop(int n) {
        int x = 1;
        while (x * (x + 1) / 2 < n) ++x;
        return x;
    }
};
```

---

##15 ****[Problem Link]https://leetcode.com/problems/number-of-wonderful-substrings****  
**Approach:** Use prefix XOR and count even parity combinations.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    long long wonderfulSubstrings(string word) {
        unordered_map<int, int> count = {{0, 1}};
        int mask = 0;
        long long res = 0;
        for (char c : word) {
            mask ^= 1 << (c - 'a');
            res += count[mask];
            for (int i = 0; i < 10; ++i)
                res += count[mask ^ (1 << i)];
            count[mask]++;
        }
        return res;
    }
};
```

---

##16 ****[Problem Link]https://leetcode.com/problems/parsing-a-boolean-expression****  
**Approach:** Use stack to evaluate expression from inside out.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <stack>

using namespace std;

class Solution {
public:
    bool parseBoolExpr(string expression) {
        stack<char> ops;
        stack<vector<char>> operands;
        for (char c : expression) {
            if (c == '!' || c == '&' || c == '|') {
                ops.push(c);
                operands.push({});
            } else if (c == 't' || c == 'f') {
                operands.top().push_back(c);
            } else if (c == ')') {
                char op = ops.top(); ops.pop();
                vector<char> values = operands.top(); operands.pop();
                char result = values[0];
                for (int i = 1; i < values.size(); ++i) {
                    if (op == '&') result = (result == 't' && values[i] == 't') ? 't' : 'f';
                    else if (op == '|') result = (result == 't' || values[i] == 't') ? 't' : 'f';
                }
                if (op == '!') result = (result == 't') ? 'f' : 't';
                if (op != '!') operands.top().push_back(result);
                else operands.push({result});
            }
        }
        return operands.top()[0] == 't';
    }
};
```

---

##17 ****[Problem Link]https://leetcode.com/problems/even-odd-tree****  
**Approach:** Level-order traversal with value checks per level index.  
**Time Complexity:** O(n)

```cpp
#include <queue>


class Solution {
public:
    bool isEvenOddTree(TreeNode* root) {
        queue<TreeNode*> q;
        q.push(root);
        bool even = true;
        while (!q.empty()) {
            int size = q.size();
            int prev = even ? INT_MIN : INT_MAX;
            for (int i = 0; i < size; ++i) {
                TreeNode* node = q.front(); q.pop();
                if ((even && (node->val % 2 == 0 || node->val <= prev)) ||
                    (!even && (node->val % 2 == 1 || node->val >= prev)))
                    return false;
                prev = node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            even = !even;
        }
        return true;
    }
};
```

---

##18 ****[Problem Link]https://leetcode.com/problems/number-of-visible-people-in-a-queue****  
**Approach:** Monotonic stack from right to count visible people.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <stack>

using namespace std;

class Solution {
public:
    vector<int> canSeePersonsCount(vector<int>& heights) {
        vector<int> res(heights.size());
        stack<int> st;
        for (int i = heights.size() - 1; i >= 0; --i) {
            while (!st.empty() && heights[i] > st.top()) {
                st.pop();
                res[i]++;
            }
            if (!st.empty()) res[i]++;
            st.push(heights[i]);
        }
        return res;
    }
};
```

---

##19 ****[Problem Link]https://leetcode.com/problems/remove-comments****  
**Approach:** Process each line and track block comments using flags.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<string> removeComments(vector<string>& source) {
        vector<string> res;
        string line;
        bool inBlock = false;
        for (auto& src : source) {
            for (int i = 0; i < src.size(); ++i) {
                if (!inBlock && i + 1 < src.size() && src[i] == '/' && src[i + 1] == '*') {
                    inBlock = true;
                    ++i;
                } else if (inBlock && i + 1 < src.size() && src[i] == '*' && src[i + 1] == '/') {
                    inBlock = false;
                    ++i;
                } else if (!inBlock && i + 1 < src.size() && src[i] == '/' && src[i + 1] == '/') {
                    break;
                } else if (!inBlock) {
                    line += src[i];
                }
            }
            if (!inBlock && !line.empty()) {
                res.push_back(line);
                line.clear();
            }
        }
        return res;
    }
};
```

---

##20 ****[Problem Link]https://leetcode.com/problems/maximum-score-words-formed-by-letters****  
**Approach:** Backtrack with word scoring and letter count tracking.  
**Time Complexity:** O(2^n * k)

```cpp
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int maxScoreWords(vector<string>& words, vector<char>& letters, vector<int>& score) {
        vector<int> count(26);
        for (char c : letters) count[c - 'a']++;
        return dfs(words, score, count, 0);
    }

    int dfs(vector<string>& words, vector<int>& score, vector<int> count, int idx) {
        if (idx == words.size()) return 0;
        int skip = dfs(words, score, count, idx + 1), take = 0;
        vector<int> curr = count;
        bool valid = true;
        int wordScore = 0;
        for (char c : words[idx]) {
            if (--curr[c - 'a'] < 0) {
                valid = false;
                break;
            }
            wordScore += score[c - 'a'];
        }
        if (valid)
            take = wordScore + dfs(words, score, curr, idx + 1);
        return max(skip, take);
    }
};
```

---

##21 ****[Problem Link]https://leetcode.com/problems/stone-game-iv****  
**Approach:** DP with each state checking if current player can win by subtracting a square number.  
**Time Complexity:** O(n * sqrt(n))

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    bool winnerSquareGame(int n) {
        vector<bool> dp(n + 1, false);
        for (int i = 1; i <= n; ++i) {
            for (int k = 1; k * k <= i; ++k) {
                if (!dp[i - k * k]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
};
```

---

##22 ****[Problem Link]https://leetcode.com/problems/number-of-ways-to-wear-different-hats-to-each-other****  
**Approach:** DP + bitmask to track assignment of hats to persons.  
**Time Complexity:** O(2^n * 40)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int numberWays(vector<vector<int>>& hats) {
        const int MOD = 1e9 + 7;
        vector<vector<int>> hatToPerson(41);
        int n = hats.size();
        for (int i = 0; i < n; ++i)
            for (int h : hats[i])
                hatToPerson[h].push_back(i);

        vector<vector<int>> dp(41, vector<int>(1 << n, 0));
        dp[0][0] = 1;

        for (int h = 1; h <= 40; ++h) {
            for (int mask = 0; mask < (1 << n); ++mask) {
                dp[h][mask] = dp[h - 1][mask];
                for (int p : hatToPerson[h]) {
                    if (mask & (1 << p))
                        dp[h][mask] = (dp[h][mask] + dp[h - 1][mask ^ (1 << p)]) % MOD;
                }
            }
        }
        return dp[40][(1 << n) - 1];
    }
};
```

---

##23 ****[Problem Link]https://leetcode.com/problems/restore-the-array-from-adjacent-pairs****  
**Approach:** Build adjacency list and reconstruct path starting from degree 1 node.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Solution {
public:
    vector<int> restoreArray(vector<vector<int>>& adjacentPairs) {
        unordered_map<int, vector<int>> graph;
        for (auto& p : adjacentPairs) {
            graph[p[0]].push_back(p[1]);
            graph[p[1]].push_back(p[0]);
        }

        int start = 0;
        for (auto& [k, v] : graph)
            if (v.size() == 1) {
                start = k;
                break;
            }

        vector<int> res;
        unordered_set<int> visited;
        function<void(int)> dfs = [&](int node) {
            res.push_back(node);
            visited.insert(node);
            for (int nei : graph[node])
                if (!visited.count(nei))
                    dfs(nei);
        };

        dfs(start);
        return res;
    }
};
```

---

##24 ****[Problem Link]https://leetcode.com/problems/check-if-the-sentence-is-pangram****  
**Approach:** Use a set or boolean array to track presence of each character.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool checkIfPangram(string sentence) {
        unordered_set<char> seen(sentence.begin(), sentence.end());
        return seen.size() == 26;
    }
};
```

---

##25 ****[Problem Link]https://leetcode.com/problems/erect-the-fence****  
**Approach:** Monotone chain algorithm (Convex Hull).  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<int>> outerTrees(vector<vector<int>>& trees) {
        sort(trees.begin(), trees.end());
        vector<vector<int>> hull;

        auto cross = [](vector<int>& o, vector<int>& a, vector<int>& b) {
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
        };

        for (int i = 0; i < trees.size(); ++i) {
            while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), trees[i]) < 0)
                hull.pop_back();
            hull.push_back(trees[i]);
        }

        int lowerSize = hull.size();
        for (int i = trees.size() - 2; i >= 0; --i) {
            while (hull.size() > lowerSize && cross(hull[hull.size() - 2], hull.back(), trees[i]) < 0)
                hull.pop_back();
            hull.push_back(trees[i]);
        }

        sort(hull.begin(), hull.end());
        hull.erase(unique(hull.begin(), hull.end()), hull.end());
        return hull;
    }
};
```

---

##26 ****[Problem Link]https://leetcode.com/problems/loud-and-rich****  
**Approach:** Topological sort with DFS to propagate quietest person.  
**Time Complexity:** O(n + e)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    vector<int> loudAndRich(vector<vector<int>>& richer, vector<int>& quiet) {
        int n = quiet.size();
        vector<vector<int>> graph(n);
        vector<int> indegree(n, 0), ans(n);
        for (int i = 0; i < n; ++i) ans[i] = i;

        for (auto& r : richer) {
            graph[r[0]].push_back(r[1]);
            indegree[r[1]]++;
        }

        queue<int> q;
        for (int i = 0; i < n; ++i)
            if (indegree[i] == 0) q.push(i);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : graph[u]) {
                if (quiet[ans[u]] < quiet[ans[v]])
                    ans[v] = ans[u];
                if (--indegree[v] == 0)
                    q.push(v);
            }
        }

        return ans;
    }
};
```

---

##27 ****[Problem Link]https://leetcode.com/problems/max-dot-product-of-two-subsequences****  
**Approach:** DP with all combinations of matches and skips.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
#include <climits>

using namespace std;

class Solution {
public:
    int maxDotProduct(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, INT_MIN));

        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                int prod = nums1[i - 1] * nums2[j - 1];
                dp[i][j] = max({prod, dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + prod});
            }
        }
        return dp[m][n];
    }
};
```

---

##28 ****[Problem Link]https://leetcode.com/problems/orderly-queue****  
**Approach:** If k > 1 sort string, else find lexicographically smallest rotation.  
**Time Complexity:** O(n log n)

```cpp
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    string orderlyQueue(string s, int k) {
        if (k > 1) {
            sort(s.begin(), s.end());
            return s;
        }

        string res = s;
        for (int i = 1; i < s.size(); ++i) {
            string rotated = s.substr(i) + s.substr(0, i);
            res = min(res, rotated);
        }
        return res;
    }
};
```

---

##29 ****[Problem Link]https://leetcode.com/problems/remove-sub-folders-from-the-filesystem****  
**Approach:** Sort paths and skip subfolders of current prefix.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<string> removeSubfolders(vector<string>& folder) {
        sort(folder.begin(), folder.end());
        vector<string> res;
        string prev = "";
        for (string& f : folder) {
            if (prev.empty() || f.compare(0, prev.length(), prev) != 0 || f[prev.length()] != '/')
                res.push_back(prev = f);
        }
        return res;
    }
};
```

---

##30 ****[Problem Link]https://leetcode.com/problems/parallel-courses-ii****  
**Approach:** DP with bitmask to track state and schedule up to k per step.  
**Time Complexity:** O(n * 2^n)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int minNumberOfSemesters(int n, vector<vector<int>>& dependencies, int k) {
        vector<int> prereq(n, 0);
        for (auto& d : dependencies)
            prereq[d[1] - 1] |= (1 << (d[0] - 1));

        vector<int> dp(1 << n, n + 1);
        dp[0] = 0;

        for (int mask = 0; mask < (1 << n); ++mask) {
            int canTake = 0;
            for (int i = 0; i < n; ++i)
                if (!(mask & (1 << i)) && (mask & prereq[i]) == prereq[i])
                    canTake |= (1 << i);

            for (int sub = canTake; sub; sub = (sub - 1) & canTake)
                if (__builtin_popcount(sub) <= k)
                    dp[mask | sub] = min(dp[mask | sub], dp[mask] + 1);
        }

        return dp[(1 << n) - 1];
    }
};
```

---

##31 ****[Problem Link]https://leetcode.com/problems/goal-parser-interpretation****  
**Approach:** Iterate and match patterns for 'G', '()', and '(al)'.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string interpret(string command) {
        string res;
        for (int i = 0; i < command.size(); ++i) {
            if (command[i] == 'G') res += 'G';
            else if (command[i] == '(' && command[i + 1] == ')') {
                res += 'o';
                ++i;
            } else {
                res += "al";
                i += 3;
            }
        }
        return res;
    }
};
```

---

##32 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating****  
**Approach:** Sliding window to simulate cyclic rotations and match both alternations.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minFlips(string s) {
        int n = s.size();
        string s2 = s + s;
        string alt1, alt2;
        for (int i = 0; i < s2.size(); ++i) {
            alt1 += i % 2 ? '1' : '0';
            alt2 += i % 2 ? '0' : '1';
        }

        int res = n, diff1 = 0, diff2 = 0;
        for (int i = 0; i < s2.size(); ++i) {
            if (s2[i] != alt1[i]) diff1++;
            if (s2[i] != alt2[i]) diff2++;

            if (i >= n) {
                if (s2[i - n] != alt1[i - n]) diff1--;
                if (s2[i - n] != alt2[i - n]) diff2--;
            }
            if (i >= n - 1)
                res = min({res, diff1, diff2});
        }
        return res;
    }
};
```

---

##33 ****[Problem Link]https://leetcode.com/problems/smallest-string-with-a-given-numeric-value****  
**Approach:** Greedy fill from back with max 'z' until sum is met.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string getSmallestString(int n, int k) {
        string res(n, 'a');
        k -= n;
        for (int i = n - 1; i >= 0 && k > 0; --i) {
            int add = min(25, k);
            res[i] += add;
            k -= add;
        }
        return res;
    }
};
```

---

##34 ****[Problem Link]https://leetcode.com/problems/count-substrings-that-differ-by-one-character****  
**Approach:** Compare all substrings and count differing by one character.  
**Time Complexity:** O(m * n * min(m,n))

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int countSubstrings(string s, string t) {
        int res = 0;
        for (int i = 0; i < s.size(); ++i) {
            for (int j = 0; j < t.size(); ++j) {
                int diff = 0;
                for (int k = 0; i + k < s.size() && j + k < t.size(); ++k) {
                    if (s[i + k] != t[j + k]) diff++;
                    if (diff > 1) break;
                    if (diff == 1) res++;
                }
            }
        }
        return res;
    }
};
```

---

##35 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring****  
**Approach:** Use sliding window and hashmap to track counts of substrings.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int maxFreq(string s, int maxLetters, int minSize, int maxSize) {
        unordered_map<string, int> freq;
        unordered_map<char, int> window;
        int res = 0;

        for (int i = 0; i <= (int)s.size() - minSize; ++i) {
            window.clear();
            for (int j = 0; j < minSize; ++j)
                window[s[i + j]]++;
            if (window.size() <= maxLetters) {
                string sub = s.substr(i, minSize);
                res = max(res, ++freq[sub]);
            }
        }
        return res;
    }
};
```

---

##36 ****[Problem Link]https://leetcode.com/problems/maximum-students-taking-exam****  
**Approach:** DP with bitmask to validate seat arrangements per row.  
**Time Complexity:** O(m * 2^n * 2^n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int maxStudents(vector<string>& seats) {
        int m = seats.size(), n = seats[0].size();
        vector<vector<int>> dp(m + 1, vector<int>(1 << n, 0));

        vector<int> valid;
        for (int mask = 0; mask < (1 << n); ++mask) {
            if ((mask & (mask >> 1)) == 0)
                valid.push_back(mask);
        }

        for (int i = 0; i < m; ++i) {
            for (int cur : valid) {
                bool ok = true;
                int cnt = 0;
                for (int j = 0; j < n; ++j) {
                    if (((cur >> j) & 1) && seats[i][j] == '#') {
                        ok = false;
                        break;
                    }
                    cnt += (cur >> j) & 1;
                }
                if (!ok) continue;

                for (int pre : valid) {
                    if ((cur & (pre >> 1)) || (cur & (pre << 1))) continue;
                    dp[i + 1][cur] = max(dp[i + 1][cur], dp[i][pre] + cnt);
                }
            }
        }

        return *max_element(dp[m].begin(), dp[m].end());
    }
};
```

---

##37 ****[Problem Link]https://leetcode.com/problems/largest-time-for-given-digits****  
**Approach:** Try all permutations and check valid max time.  
**Time Complexity:** O(1)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    string largestTimeFromDigits(vector<int>& arr) {
        string res = "";
        sort(arr.begin(), arr.end());
        do {
            int h = arr[0] * 10 + arr[1], m = arr[2] * 10 + arr[3];
            if (h < 24 && m < 60) {
                char buf[6];
                sprintf(buf, "%02d:%02d", h, m);
                res = max(res, string(buf));
            }
        } while (next_permutation(arr.begin(), arr.end()));
        return res;
    }
};
```

---

##38 ****[Problem Link]https://leetcode.com/problems/design-parking-system****  
**Approach:** Use a counter array for available slots by type.  
**Time Complexity:** O(1)

```cpp
class ParkingSystem {
    int count[4];

public:
    ParkingSystem(int big, int medium, int small) {
        count[1] = big;
        count[2] = medium;
        count[3] = small;
    }

    bool addCar(int carType) {
        return count[carType]-- > 0;
    }
};
```

---

##39 ****[Problem Link]https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations****  
**Approach:** DP with memoization: try taking from start or end.  
**Time Complexity:** O(m^2)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int maximumScore(vector<int>& nums, vector<int>& multipliers) {
        int n = nums.size(), m = multipliers.size();
        vector<vector<int>> dp(m + 1, vector<int>(m + 1, 0));

        for (int i = m - 1; i >= 0; --i) {
            for (int l = i; l >= 0; --l) {
                int r = n - 1 - (i - l);
                dp[i][l] = max(
                    nums[l] * multipliers[i] + dp[i + 1][l + 1],
                    nums[r] * multipliers[i] + dp[i + 1][l]
                );
            }
        }
        return dp[0][0];
    }
};
```

---

##40 ****[Problem Link]https://leetcode.com/problems/car-fleet-ii****  
**Approach:** Monotonic stack with time calculation for fleet merging.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <stack>

using namespace std;

class Solution {
public:
    vector<double> getCollisionTimes(vector<vector<int>>& cars) {
        int n = cars.size();
        vector<double> res(n, -1);
        stack<int> st;

        for (int i = n - 1; i >= 0; --i) {
            int p = cars[i][0], s = cars[i][1];
            while (!st.empty()) {
                int j = st.top();
                int pj = cars[j][0], sj = cars[j][1];
                if (s <= sj || (res[j] > 0 && (pj - p) / double(s - sj) > res[j])) {
                    st.pop();
                } else {
                    break;
                }
            }

            if (!st.empty()) {
                int j = st.top();
                res[i] = (cars[j][0] - p) / double(s - cars[j][1]);
            }
            st.push(i);
        }

        return res;
    }
};
```

---

##41 ****[Problem Link]https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable****  
**Approach:** Union-Find for type 3 edges first, then type 1 and 2.  
**Time Complexity:** O(n + e)

```cpp
#include <vector>

using namespace std;

class UnionFind {
public:
    vector<int> parent;
    int count;

    UnionFind(int n) : parent(n + 1), count(n) {
        for (int i = 1; i <= n; ++i) parent[i] = i;
    }

    int find(int u) {
        if (u != parent[u]) parent[u] = find(parent[u]);
        return parent[u];
    }

    bool unite(int u, int v) {
        int pu = find(u), pv = find(v);
        if (pu == pv) return false;
        parent[pu] = pv;
        count--;
        return true;
    }
};

class Solution {
public:
    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        UnionFind alice(n), bob(n);
        int used = 0;

        for (auto& e : edges) {
            if (e[0] == 3) {
                bool a = alice.unite(e[1], e[2]);
                bool b = bob.unite(e[1], e[2]);
                if (a || b) used++;
            }
        }

        for (auto& e : edges) {
            if (e[0] == 1) {
                if (alice.unite(e[1], e[2])) used++;
            } else if (e[0] == 2) {
                if (bob.unite(e[1], e[2])) used++;
            }
        }

        return (alice.count == 1 && bob.count == 1) ? edges.size() - used : -1;
    }
};
```

---

##42 ****[Problem Link]https://leetcode.com/problems/k-th-smallest-in-lexicographical-order****  
**Approach:** Use prefix tree logic to count nodes under current prefix.  
**Time Complexity:** O(log n)^2

```cpp
class Solution {
public:
    int findKthNumber(int n, int k) {
        int curr = 1;
        k -= 1;

        while (k > 0) {
            long steps = calcSteps(n, curr, curr + 1);
            if (steps <= k) {
                curr += 1;
                k -= steps;
            } else {
                curr *= 10;
                k -= 1;
            }
        }

        return curr;
    }

    long calcSteps(long n, long curr, long next) {
        long steps = 0;
        while (curr <= n) {
            steps += min(n + 1, next) - curr;
            curr *= 10;
            next *= 10;
        }
        return steps;
    }
};
```

---

##43 ****[Problem Link]https://leetcode.com/problems/find-if-path-exists-in-graph****  
**Approach:** DFS or BFS traversal from start to end.  
**Time Complexity:** O(n + e)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
        vector<vector<int>> graph(n);
        for (auto& e : edges) {
            graph[e[0]].push_back(e[1]);
            graph[e[1]].push_back(e[0]);
        }

        vector<bool> visited(n, false);
        queue<int> q;
        q.push(source);
        visited[source] = true;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u == destination) return true;
            for (int v : graph[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }

        return false;
    }
};
```

---

##44 ****[Problem Link]https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree****  
**Approach:** Reconstruct tree using DFS and store all values in a hashset.  
**Time Complexity:** O(n)

```cpp
#include <unordered_set>


class FindElements {
    unordered_set<int> vals;

public:
    FindElements(TreeNode* root) {
        dfs(root, 0);
    }

    void dfs(TreeNode* node, int val) {
        if (!node) return;
        node->val = val;
        vals.insert(val);
        dfs(node->left, 2 * val + 1);
        dfs(node->right, 2 * val + 2);
    }

    bool find(int target) {
        return vals.count(target);
    }
};
```

---

##45 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-coins-you-can-get****  
**Approach:** Sort array and pick every third from end of sorted triplets.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxCoins(vector<int>& piles) {
        sort(piles.begin(), piles.end());
        int res = 0, n = piles.size() / 3;
        for (int i = piles.size() - 2; i >= n; i -= 2)
            res += piles[i];
        return res;
    }
};
```

---

##46 ****[Problem Link]https://leetcode.com/problems/stamping-the-sequence****  
**Approach:** Reverse stamping simulation with greedy matching.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<int> movesToStamp(string stamp, string target) {
        int m = stamp.size(), n = target.size();
        vector<int> res;
        bool changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i <= n - m; ++i) {
                bool canStamp = false, stamped = false;
                for (int j = 0; j < m; ++j) {
                    if (target[i + j] == '*') continue;
                    if (target[i + j] != stamp[j]) break;
                    canStamp = true;
                }
                if (canStamp) {
                    for (int j = 0; j < m; ++j) {
                        if (target[i + j] != '*') {
                            target[i + j] = '*';
                            stamped = true;
                        }
                    }
                    if (stamped) {
                        changed = true;
                        res.push_back(i);
                    }
                }
            }
        }

        for (char c : target)
            if (c != '*') return {};

        reverse(res.begin(), res.end());
        return res;
    }
};
```

---

##47 ****[Problem Link]https://leetcode.com/problems/reconstruct-original-digits-from-english****  
**Approach:** Count characters and use unique letters for digit identification.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string originalDigits(string s) {
        vector<int> count(26, 0), digit(10, 0);
        for (char c : s) count[c - 'a']++;

        digit[0] = count['z' - 'a'];
        digit[2] = count['w' - 'a'];
        digit[4] = count['u' - 'a'];
        digit[6] = count['x' - 'a'];
        digit[8] = count['g' - 'a'];

        digit[1] = count['o' - 'a'] - digit[0] - digit[2] - digit[4];
        digit[3] = count['h' - 'a'] - digit[8];
        digit[5] = count['f' - 'a'] - digit[4];
        digit[7] = count['s' - 'a'] - digit[6];
        digit[9] = count['i' - 'a'] - digit[5] - digit[6] - digit[8];

        string res;
        for (int i = 0; i < 10; ++i)
            res.append(digit[i], '0' + i);
        return res;
    }
};
```

---

##48 ****[Problem Link]https://leetcode.com/problems/number-of-students-doing-homework-at-a-given-time****  
**Approach:** Count how many intervals contain the query time.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int busyStudent(vector<int>& startTime, vector<int>& endTime, int queryTime) {
        int count = 0;
        for (int i = 0; i < startTime.size(); ++i) {
            if (startTime[i] <= queryTime && queryTime <= endTime[i])
                count++;
        }
        return count;
    }
};
```

---

##49 ****[Problem Link]https://leetcode.com/problems/minimize-deviation-in-array****  
**Approach:** Use max heap and reduce even numbers greedily.  
**Time Complexity:** O(n log maxVal)

```cpp
#include <vector>
#include <queue>
#include <climits>

using namespace std;

class Solution {
public:
    int minimumDeviation(vector<int>& nums) {
        priority_queue<int> pq;
        int minVal = INT_MAX;
        for (int& n : nums) {
            if (n % 2 == 1) n *= 2;
            pq.push(n);
            minVal = min(minVal, n);
        }

        int res = INT_MAX;
        while (!pq.empty()) {
            int maxVal = pq.top(); pq.pop();
            res = min(res, maxVal - minVal);
            if (maxVal % 2 == 0) {
                maxVal /= 2;
                minVal = min(minVal, maxVal);
                pq.push(maxVal);
            } else break;
        }
        return res;
    }
};
```

---

##50 ****[Problem Link]https://leetcode.com/problems/get-the-maximum-score****  
**Approach:** Two pointers with prefix sums and modulo.  
**Time Complexity:** O(m + n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int maxSum(vector<int>& nums1, vector<int>& nums2) {
        const int MOD = 1e9 + 7;
        int i = 0, j = 0;
        long sum1 = 0, sum2 = 0, total = 0;

        while (i < nums1.size() && j < nums2.size()) {
            if (nums1[i] < nums2[j]) {
                sum1 += nums1[i++];
            } else if (nums1[i] > nums2[j]) {
                sum2 += nums2[j++];
            } else {
                total += max(sum1, sum2) + nums1[i];
                sum1 = sum2 = 0;
                i++; j++;
            }
        }

        while (i < nums1.size()) sum1 += nums1[i++];
        while (j < nums2.size()) sum2 += nums2[j++];

        total += max(sum1, sum2);
        return total % MOD;
    }
};
```

---

##51 ****[Problem Link]https://leetcode.com/problems/maximum-alternating-subsequence-sum****  
**Approach:** Use DP to keep track of even and odd indexed subsequences.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    long long maxAlternatingSum(vector<int>& nums) {
        long long even = 0, odd = 0;
        for (int n : nums) {
            long long new_even = max(even, odd + n);
            long long new_odd = max(odd, even - n);
            even = new_even;
            odd = new_odd;
        }
        return even;
    }
};
```

---

##52 ****[Problem Link]https://leetcode.com/problems/special-binary-string****  
**Approach:** Recursively split valid blocks and sort them to get largest string.  
**Time Complexity:** O(n^2)

```cpp
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    string makeLargestSpecial(string s) {
        vector<string> subs;
        int count = 0, start = 0;
        for (int i = 0; i < s.size(); ++i) {
            count += s[i] == '1' ? 1 : -1;
            if (count == 0) {
                subs.push_back("1" + makeLargestSpecial(s.substr(start + 1, i - start - 1)) + "0");
                start = i + 1;
            }
        }
        sort(subs.rbegin(), subs.rend());
        string res;
        for (auto& sub : subs) res += sub;
        return res;
    }
};
```

---

##53 ****[Problem Link]https://leetcode.com/problems/check-if-word-is-valid-after-substitutions****  
**Approach:** Stack to simulate insert and reduce 'abc'.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    bool isValid(string s) {
        string stack;
        for (char c : s) {
            stack.push_back(c);
            if (stack.size() >= 3 && stack.substr(stack.size() - 3) == "abc") {
                stack.erase(stack.end() - 3, stack.end());
            }
        }
        return stack.empty();
    }
};
```

---

##54 ****[Problem Link]https://leetcode.com/problems/can-make-arithmetic-progression-from-sequence****  
**Approach:** Sort and check difference between adjacent elements.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    bool canMakeArithmeticProgression(vector<int>& arr) {
        sort(arr.begin(), arr.end());
        int diff = arr[1] - arr[0];
        for (int i = 2; i < arr.size(); ++i)
            if (arr[i] - arr[i - 1] != diff)
                return false;
        return true;
    }
};
```

---

##55 ****[Problem Link]https://leetcode.com/problems/super-washing-machines****  
**Approach:** Prefix sum to track imbalance and find max transfer needed.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

class Solution {
public:
    int findMinMoves(vector<int>& machines) {
        int total = accumulate(machines.begin(), machines.end(), 0);
        int n = machines.size();
        if (total % n != 0) return -1;

        int target = total / n, res = 0, balance = 0;
        for (int load : machines) {
            balance += load - target;
            res = max(res, max(abs(balance), load - target));
        }
        return res;
    }
};
```

---

##56 ****[Problem Link]https://leetcode.com/problems/count-of-matches-in-tournament****  
**Approach:** Keep dividing teams and adding number of matches.  
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int numberOfMatches(int n) {
        int matches = 0;
        while (n > 1) {
            matches += n / 2;
            n = (n + 1) / 2;
        }
        return matches;
    }
};
```

---

##57 ****[Problem Link]https://leetcode.com/problems/perfect-number****  
**Approach:** Check divisors up to sqrt(n) and sum them.  
**Time Complexity:** O(sqrt n)

```cpp
class Solution {
public:
    bool checkPerfectNumber(int num) {
        if (num <= 1) return false;
        int sum = 1;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                sum += i;
                if (i != num / i) sum += num / i;
            }
        }
        return sum == num;
    }
};
```

---

##58 ****[Problem Link]https://leetcode.com/problems/guess-number-higher-or-lower****  
**Approach:** Standard binary search.  
**Time Complexity:** O(log n)

```cpp
// Forward declaration of guess API.
int guess(int num);

class Solution {
public:
    int guessNumber(int n) {
        int low = 1, high = n;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            int res = guess(mid);
            if (res == 0) return mid;
            else if (res < 0) high = mid - 1;
            else low = mid + 1;
        }
        return -1;
    }
};
```

---

##59 ****[Problem Link]https://leetcode.com/problems/minimize-hamming-distance-after-swap-operations****  
**Approach:** Use DSU to group swappable indices and compare frequency.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class DSU {
public:
    vector<int> parent;
    DSU(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    void unite(int x, int y) {
        parent[find(x)] = find(y);
    }
};

class Solution {
public:
    int minimumHammingDistance(vector<int>& source, vector<int>& target, vector<vector<int>>& allowedSwaps) {
        int n = source.size();
        DSU dsu(n);
        for (auto& swap : allowedSwaps) {
            dsu.unite(swap[0], swap[1]);
        }

        unordered_map<int, unordered_map<int, int>> groups;
        for (int i = 0; i < n; ++i) {
            groups[dsu.find(i)][source[i]]++;
        }

        int res = 0;
        for (int i = 0; i < n; ++i) {
            int group = dsu.find(i);
            if (groups[group][target[i]] > 0) {
                groups[group][target[i]]--;
            } else {
                res++;
            }
        }
        return res;
    }
};
```

---

##60 ****[Problem Link]https://leetcode.com/problems/can-make-palindrome-from-substring****  
**Approach:** Use prefix frequency and count odd characters in substring.  
**Time Complexity:** O(q + 26 * n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<bool> canMakePaliQueries(string s, vector<vector<int>>& queries) {
        int n = s.size();
        vector<vector<int>> prefix(n + 1, vector<int>(26, 0));

        for (int i = 0; i < n; ++i) {
            prefix[i + 1] = prefix[i];
            prefix[i + 1][s[i] - 'a']++;
        }

        vector<bool> res;
        for (auto& q : queries) {
            int l = q[0], r = q[1], k = q[2], odds = 0;
            for (int i = 0; i < 26; ++i) {
                odds += (prefix[r + 1][i] - prefix[l][i]) % 2;
            }
            res.push_back(odds / 2 <= k);
        }
        return res;
    }
};
```

---

##61 ****[Problem Link]https://leetcode.com/problems/check-if-there-is-a-valid-path-in-a-grid****  
**Approach:** DFS or BFS based on tile connections.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    bool hasValidPath(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<bool>> visited(m, vector<bool>(n, false));

        vector<vector<int>> dirs = {
            {}, // 0
            {0,1,0,-1},    // 1: left, right
            {1,0,-1,0},    // 2: up, down
            {0,-1,1,0},    // 3: left, down
            {0,1,1,0},     // 4: right, down
            {-1,0,0,-1},   // 5: left, up
            {-1,0,0,1}     // 6: right, up
        };

        vector<vector<int>> valid = {
            {}, // 0
            {1,3,5}, // connect left/right
            {2,5,6}, // connect up/down
            {1,2,3}, // left-down
            {1,2,4}, // right-down
            {1,5,6}, // left-up
            {1,5,6}  // right-up
        };

        queue<pair<int,int>> q;
        q.push({0,0});
        visited[0][0] = true;

        while (!q.empty()) {
            auto [x, y] = q.front(); q.pop();
            if (x == m-1 && y == n-1) return true;

            int t = grid[x][y];
            for (int i = 0; i < 4; i += 2) {
                int nx = x + dirs[t][i];
                int ny = y + dirs[t][i+1];
                if (nx < 0 || ny < 0 || nx >= m || ny >= n || visited[nx][ny])
                    continue;
                int nt = grid[nx][ny];
                if (find(valid[t].begin(), valid[t].end(), nt) != valid[t].end()) {
                    visited[nx][ny] = true;
                    q.push({nx, ny});
                }
            }
        }
        return false;
    }
};
```

---

##62 ****[Problem Link]https://leetcode.com/problems/detect-cycles-in-2d-grid****  
**Approach:** DFS with parent tracking to detect cycles.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    bool containsCycle(vector<vector<char>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<bool>> visited(m, vector<bool>(n, false));

        function<bool(int,int,int,int,char)> dfs = [&](int x, int y, int px, int py, char c) {
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] != c)
                return false;
            if (visited[x][y]) return true;
            visited[x][y] = true;

            vector<pair<int,int>> dirs = {{0,1},{1,0},{0,-1},{-1,0}};
            for (auto [dx, dy] : dirs) {
                int nx = x + dx, ny = y + dy;
                if (nx == px && ny == py) continue;
                if (dfs(nx, ny, x, y, c)) return true;
            }

            return false;
        };

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (!visited[i][j] && dfs(i, j, -1, -1, grid[i][j]))
                    return true;

        return false;
    }
};
```

---

##63 ****[Problem Link]https://leetcode.com/problems/number-of-restricted-paths-from-first-to-last-node****  
**Approach:** Dijkstra followed by DP counting valid paths in decreasing distance.  
**Time Complexity:** O(E log V)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int countRestrictedPaths(int n, vector<vector<int>>& edges) {
        const int MOD = 1e9 + 7;
        vector<vector<pair<int,int>>> graph(n+1);
        for (auto& e : edges) {
            graph[e[0]].emplace_back(e[1], e[2]);
            graph[e[1]].emplace_back(e[0], e[2]);
        }

        vector<int> dist(n+1, INT_MAX);
        dist[n] = 0;
        priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
        pq.emplace(0, n);

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            for (auto& [v, w] : graph[u]) {
                if (dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                    pq.emplace(dist[v], v);
                }
            }
        }

        vector<int> dp(n+1, -1);
        function<int(int)> dfs = [&](int u) {
            if (u == n) return 1;
            if (dp[u] != -1) return dp[u];
            int ways = 0;
            for (auto& [v, w] : graph[u]) {
                if (dist[v] < dist[u])
                    ways = (ways + dfs(v)) % MOD;
            }
            return dp[u] = ways;
        };

        return dfs(1);
    }
};
```

---

##64 ****[Problem Link]https://leetcode.com/problems/camelcase-matching****  
**Approach:** Two pointer technique matching capital and lowercase.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<bool> camelMatch(vector<string>& queries, string pattern) {
        vector<bool> res;
        for (string& q : queries) {
            int i = 0;
            bool valid = true;
            for (char c : q) {
                if (i < pattern.size() && c == pattern[i]) {
                    i++;
                } else if (isupper(c)) {
                    valid = false;
                    break;
                }
            }
            res.push_back(valid && i == pattern.size());
        }
        return res;
    }
};
```

---

##65 ****[Problem Link]https://leetcode.com/problems/minimum-sideway-jumps****  
**Approach:** DP on position and lane with transition options.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minSideJumps(vector<int>& obs) {
        int n = obs.size() - 1;
        vector<int> dp = {1, 0, 1};
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (obs[i] == j + 1) dp[j] = 1e9;
            }
            for (int j = 0; j < 3; ++j) {
                if (obs[i] != j + 1) {
                    dp[j] = min(dp[j], min(dp[(j+1)%3], dp[(j+2)%3]) + 1);
                }
            }
        }
        return min({dp[0], dp[1], dp[2]});
    }
};
```

---

##66 ****[Problem Link]https://leetcode.com/problems/friends-of-appropriate-ages****  
**Approach:** Sort ages and use prefix sum to count valid friend requests.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int numFriendRequests(vector<int>& ages) {
        sort(ages.begin(), ages.end());
        int n = ages.size(), res = 0;
        for (int i = 0; i < n; ++i) {
            int age = ages[i];
            if (age < 15) continue;
            int low = lower_bound(ages.begin(), ages.end(), age / 2 + 7 + 1) - ages.begin();
            int high = upper_bound(ages.begin(), ages.end(), age) - ages.begin();
            res += high - low - 1;
        }
        return res;
    }
};
```

---

##67 ****[Problem Link]https://leetcode.com/problems/pizza-with-3n-slices****  
**Approach:** DP with circular slice exclusion.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxSizeSlices(vector<int>& slices) {
        return max(helper(slices, 0, slices.size() - 2), helper(slices, 1, slices.size() - 1));
    }

    int helper(vector<int>& slices, int start, int end) {
        int n = end - start + 1, k = n / 3;
        vector<vector<int>> dp(k + 1, vector<int>(n + 2, 0));
        for (int i = 1; i <= k; ++i) {
            for (int j = start; j <= end; ++j) {
                int idx = j - start + 1;
                dp[i][idx] = max(dp[i][idx - 1], dp[i - 1][max(0, idx - 2)] + slices[j]);
            }
        }
        return dp[k][n];
    }
};
```

---

##68 ****[Problem Link]https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps****  
**Approach:** DP with bounded position range.  
**Time Complexity:** O(steps * min(steps, arrLen))

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int numWays(int steps, int arrLen) {
        const int MOD = 1e9 + 7;
        int maxPos = min(steps / 2 + 1, arrLen);
        vector<vector<int>> dp(steps + 1, vector<int>(maxPos, 0));
        dp[0][0] = 1;

        for (int i = 1; i <= steps; ++i) {
            for (int j = 0; j < maxPos; ++j) {
                dp[i][j] = dp[i - 1][j];
                if (j > 0) dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % MOD;
                if (j < maxPos - 1) dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % MOD;
            }
        }
        return dp[steps][0];
    }
};
```

---

##69 ****[Problem Link]https://leetcode.com/problems/find-a-peak-element-ii****  
**Approach:** Binary search on columns using max element in row.  
**Time Complexity:** O(m log n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> findPeakGrid(vector<vector<int>>& mat) {
        int m = mat.size(), n = mat[0].size();
        int left = 0, right = n - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int maxRow = 0;
            for (int i = 0; i < m; ++i) {
                if (mat[i][mid] > mat[maxRow][mid])
                    maxRow = i;
            }

            if ((mid == 0 || mat[maxRow][mid] > mat[maxRow][mid - 1]) &&
                (mid == n - 1 || mat[maxRow][mid] > mat[maxRow][mid + 1]))
                return {maxRow, mid};

            if (mid > 0 && mat[maxRow][mid - 1] > mat[maxRow][mid])
                right = mid - 1;
            else
                left = mid + 1;
        }

        return {-1, -1};
    }
};
```

---

##70 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced****  
**Approach:** Count imbalance in brackets while traversing.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minSwaps(string s) {
        int balance = 0, maxImbalance = 0;
        for (char c : s) {
            balance += c == '[' ? 1 : -1;
            if (balance < 0) {
                maxImbalance = max(maxImbalance, -balance);
            }
        }
        return (maxImbalance + 1) / 2;
    }
};
```

---

##71 ****[Problem Link]https://leetcode.com/problems/range-sum-of-sorted-subarray-sums****  
**Approach:** Generate all subarray sums and sort.  
**Time Complexity:** O(n^2 log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int rangeSum(vector<int>& nums, int n, int left, int right) {
        const int MOD = 1e9 + 7;
        vector<int> sums;
        for (int i = 0; i < n; ++i) {
            int s = 0;
            for (int j = i; j < n; ++j) {
                s += nums[j];
                sums.push_back(s);
            }
        }
        sort(sums.begin(), sums.end());
        int res = 0;
        for (int i = left - 1; i < right; ++i) {
            res = (res + sums[i]) % MOD;
        }
        return res;
    }
};
```

---

##72 ****[Problem Link]https://leetcode.com/problems/split-a-string-into-the-max-number-of-unique-substrings****  
**Approach:** Backtracking with a set of visited substrings.  
**Time Complexity:** Exponential

```cpp
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int maxUniqueSplit(string s) {
        unordered_set<string> seen;
        return dfs(s, 0, seen);
    }

    int dfs(const string& s, int start, unordered_set<string>& seen) {
        if (start == s.size()) return 0;
        int res = 0;
        for (int end = start + 1; end <= s.size(); ++end) {
            string substr = s.substr(start, end - start);
            if (!seen.count(substr)) {
                seen.insert(substr);
                res = max(res, 1 + dfs(s, end, seen));
                seen.erase(substr);
            }
        }
        return res;
    }
};
```

---

##73 ****[Problem Link]https://leetcode.com/problems/reachable-nodes-in-subdivided-graph****  
**Approach:** Dijkstra with node use limit per edge.  
**Time Complexity:** O(E log V)

```cpp
#include <vector>
#include <unordered_map>
#include <queue>

using namespace std;

class Solution {
public:
    int reachableNodes(vector<vector<int>>& edges, int maxMoves, int n) {
        vector<unordered_map<int, int>> graph(n);
        for (auto& e : edges) {
            graph[e[0]][e[1]] = e[2];
            graph[e[1]][e[0]] = e[2];
        }

        vector<int> dist(n, -1);
        priority_queue<pair<int, int>> pq;
        pq.push({maxMoves, 0});

        while (!pq.empty()) {
            auto [moves, node] = pq.top(); pq.pop();
            if (dist[node] != -1) continue;
            dist[node] = moves;

            for (auto& [nei, cnt] : graph[node]) {
                int left = moves - cnt - 1;
                if (dist[nei] == -1 && left >= 0) {
                    pq.push({left, nei});
                }
            }
        }

        int res = 0;
        for (int d : dist)
            if (d >= 0) res++;

        for (auto& e : edges) {
            int a = dist[e[0]] >= 0 ? dist[e[0]] : 0;
            int b = dist[e[1]] >= 0 ? dist[e[1]] : 0;
            res += min(e[2], a + b);
        }

        return res;
    }
};
```

---

##74 ****[Problem Link]https://leetcode.com/problems/distance-between-bus-stops****  
**Approach:** Compute both directions and take the minimum.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <numeric>

using namespace std;

class Solution {
public:
    int distanceBetweenBusStops(vector<int>& dist, int start, int dest) {
        if (start > dest) swap(start, dest);
        int total = accumulate(dist.begin(), dist.end(), 0);
        int path = accumulate(dist.begin() + start, dist.begin() + dest, 0);
        return min(path, total - path);
    }
};
```

---

##75 ****[Problem Link]https://leetcode.com/problems/paint-house-iii****  
**Approach:** DP with memoization on house index, previous color, and neighborhood count.  
**Time Complexity:** O(m * n * t)

```cpp
#include <vector>
#include <cstring>

using namespace std;

class Solution {
public:
    int dp[101][21][101];
    const int INF = 1e8;

    int minCost(vector<int>& houses, vector<vector<int>>& cost, int m, int n, int target) {
        memset(dp, -1, sizeof(dp));
        int res = dfs(houses, cost, 0, 0, target, m, n);
        return res >= INF ? -1 : res;
    }

    int dfs(vector<int>& houses, vector<vector<int>>& cost, int i, int prev, int target, int m, int n) {
        if (target < 0) return INF;
        if (i == m) return target == 0 ? 0 : INF;
        if (dp[i][prev][target] != -1) return dp[i][prev][target];

        int res = INF;
        if (houses[i]) {
            res = dfs(houses, cost, i + 1, houses[i], target - (houses[i] != prev), m, n);
        } else {
            for (int color = 1; color <= n; ++color) {
                int c = cost[i][color - 1];
                res = min(res, c + dfs(houses, cost, i + 1, color, target - (color != prev), m, n));
            }
        }
        return dp[i][prev][target] = res;
    }
};
```

---

##76 ****[Problem Link]https://leetcode.com/problems/process-tasks-using-servers****  
**Approach:** Use two priority queues for free and busy servers.  
**Time Complexity:** O(n log k)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    vector<int> assignTasks(vector<int>& servers, vector<int>& tasks) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> free, busy;
        for (int i = 0; i < servers.size(); ++i)
            free.emplace(servers[i], i);

        vector<int> res;
        long long time = 0;
        for (int i = 0; i < tasks.size(); ++i) {
            time = max(time, (long long)i);
            while (!busy.empty() && busy.top().first <= time) {
                auto [_, id] = busy.top(); busy.pop();
                free.emplace(servers[id], id);
            }
            if (free.empty()) {
                time = busy.top().first;
                while (!busy.empty() && busy.top().first <= time) {
                    auto [_, id] = busy.top(); busy.pop();
                    free.emplace(servers[id], id);
                }
            }
            auto [w, id] = free.top(); free.pop();
            res.push_back(id);
            busy.emplace(time + tasks[i], id);
        }
        return res;
    }
};
```

---

##77 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-removable-characters****  
**Approach:** Binary search over removals and check subsequence validity.  
**Time Complexity:** O(n log k)

```cpp
#include <string>
#include <vector>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int maximumRemovals(string s, string p, vector<int>& removable) {
        int lo = 0, hi = removable.size();
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            unordered_set<int> removed(removable.begin(), removable.begin() + mid);
            if (isSubsequence(s, p, removed)) lo = mid + 1;
            else hi = mid - 1;
        }
        return hi;
    }

    bool isSubsequence(const string& s, const string& p, unordered_set<int>& removed) {
        int i = 0;
        for (int j = 0; j < s.size() && i < p.size(); ++j) {
            if (!removed.count(j) && s[j] == p[i]) i++;
        }
        return i == p.size();
    }
};
```

---

##78 ****[Problem Link]https://leetcode.com/problems/flip-columns-for-maximum-number-of-equal-rows****  
**Approach:** Normalize row with flips and use hashmap for frequency.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

class Solution {
public:
    int maxEqualRowsAfterFlips(vector<vector<int>>& matrix) {
        unordered_map<string, int> count;
        for (auto& row : matrix) {
            string key;
            int flip = row[0];
            for (int val : row) key += (val ^ flip) + '0';
            count[key]++;
        }
        int res = 0;
        for (auto& [_, freq] : count)
            res = max(res, freq);
        return res;
    }
};
```

---

##79 ****[Problem Link]https://leetcode.com/problems/rle-iterator****  
**Approach:** Consume counts lazily on next call.  
**Time Complexity:** O(n) for all calls

```cpp
#include <vector>

using namespace std;

class RLEIterator {
    vector<int> A;
    int index = 0;

public:
    RLEIterator(vector<int>& encoding) : A(encoding) {}

    int next(int n) {
        while (index < A.size() && n > A[index]) {
            n -= A[index];
            index += 2;
        }
        if (index >= A.size()) return -1;
        A[index] -= n;
        return A[index + 1];
    }
};
```

---

##80 ****[Problem Link]https://leetcode.com/problems/binary-prefix-divisible-by-5****  
**Approach:** Keep running modulo 5 value while iterating.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<bool> prefixesDivBy5(vector<int>& A) {
        vector<bool> res;
        int num = 0;
        for (int bit : A) {
            num = ((num << 1) + bit) % 5;
            res.push_back(num == 0);
        }
        return res;
    }
};
```

---

##81 ****[Problem Link]https://leetcode.com/problems/slowest-key****  
**Approach:** Track maximum duration and corresponding key.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    char slowestKey(vector<int>& releaseTimes, string keysPressed) {
        int maxDur = releaseTimes[0];
        char res = keysPressed[0];
        for (int i = 1; i < releaseTimes.size(); ++i) {
            int dur = releaseTimes[i] - releaseTimes[i - 1];
            if (dur > maxDur || (dur == maxDur && keysPressed[i] > res)) {
                maxDur = dur;
                res = keysPressed[i];
            }
        }
        return res;
    }
};
```

---

##82 ****[Problem Link]https://leetcode.com/problems/maximum-score-of-a-good-subarray****  
**Approach:** Expand window from index k while keeping min value.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maximumScore(vector<int>& nums, int k) {
        int i = k, j = k, minVal = nums[k], res = nums[k];
        int n = nums.size();
        while (i > 0 || j < n - 1) {
            if ((i > 0 ? nums[i - 1] : 0) > (j < n - 1 ? nums[j + 1] : 0))
                minVal = min(minVal, nums[--i]);
            else
                minVal = min(minVal, nums[++j]);
            res = max(res, minVal * (j - i + 1));
        }
        return res;
    }
};
```

---

##83 ****[Problem Link]https://leetcode.com/problems/minimize-maximum-pair-sum-in-array****  
**Approach:** Sort array and pair smallest with largest.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minPairSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int res = 0, n = nums.size();
        for (int i = 0; i < n / 2; ++i) {
            res = max(res, nums[i] + nums[n - 1 - i]);
        }
        return res;
    }
};
```

---

##84 ****[Problem Link]https://leetcode.com/problems/complex-number-multiplication****  
**Approach:** Parse and multiply real and imaginary parts.  
**Time Complexity:** O(1)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string complexNumberMultiply(string num1, string num2) {
        auto [a, b] = parse(num1);
        auto [c, d] = parse(num2);
        int real = a * c - b * d;
        int imag = a * d + b * c;
        return to_string(real) + "+" + to_string(imag) + "i";
    }

    pair<int, int> parse(string s) {
        int plus = s.find('+');
        int a = stoi(s.substr(0, plus));
        int b = stoi(s.substr(plus + 1, s.size() - plus - 2));
        return {a, b};
    }
};
```

---

##85 ****[Problem Link]https://leetcode.com/problems/maximum-non-negative-product-in-a-matrix****  
**Approach:** DP with min and max tracking due to negatives.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int maxProductPath(vector<vector<int>>& grid) {
        const int MOD = 1e9 + 7;
        int m = grid.size(), n = grid[0].size();
        vector<vector<long long>> maxDP(m, vector<long long>(n));
        vector<vector<long long>> minDP(m, vector<long long>(n));
        maxDP[0][0] = minDP[0][0] = grid[0][0];

        for (int i = 1; i < m; ++i) {
            maxDP[i][0] = minDP[i][0] = maxDP[i-1][0] * grid[i][0];
        }
        for (int j = 1; j < n; ++j) {
            maxDP[0][j] = minDP[0][j] = maxDP[0][j-1] * grid[0][j];
        }

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                long long a = maxDP[i-1][j], b = minDP[i-1][j];
                long long c = maxDP[i][j-1], d = minDP[i][j-1];
                if (grid[i][j] >= 0) {
                    maxDP[i][j] = max(a, c) * grid[i][j];
                    minDP[i][j] = min(b, d) * grid[i][j];
                } else {
                    maxDP[i][j] = min(b, d) * grid[i][j];
                    minDP[i][j] = max(a, c) * grid[i][j];
                }
            }
        }

        return maxDP[m-1][n-1] < 0 ? -1 : maxDP[m-1][n-1] % MOD;
    }
};
```

---

##86 ****[Problem Link]https://leetcode.com/problems/the-number-of-weak-characters-in-the-game****  
**Approach:** Sort by attack desc and defense asc; track max defense seen.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int numberOfWeakCharacters(vector<vector<int>>& properties) {
        sort(properties.begin(), properties.end(), [](auto& a, auto& b) {
            return a[0] == b[0] ? a[1] < b[1] : a[0] > b[0];
        });

        int res = 0, maxDef = 0;
        for (auto& p : properties) {
            if (p[1] < maxDef) res++;
            else maxDef = p[1];
        }
        return res;
    }
};
```

---

##87 ****[Problem Link]https://leetcode.com/problems/increasing-decreasing-string****  
**Approach:** Count sort letters and build string alternating direction.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string sortString(string s) {
        vector<int> count(26, 0);
        for (char c : s) count[c - 'a']++;
        string res;
        while (res.size() < s.size()) {
            for (int i = 0; i < 26; ++i)
                if (count[i]--) res += (char)('a' + i);
            for (int i = 25; i >= 0; --i)
                if (count[i]--) res += (char)('a' + i);
        }
        return res;
    }
};
```

---

##88 ****[Problem Link]https://leetcode.com/problems/find-longest-awesome-substring****  
**Approach:** Prefix XOR of character counts to detect palindromes.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int longestAwesome(string s) {
        unordered_map<int, int> seen;
        seen[0] = -1;
        int mask = 0, res = 0;
        for (int i = 0; i < s.size(); ++i) {
            mask ^= 1 << (s[i] - '0');
            if (seen.count(mask)) res = max(res, i - seen[mask]);
            for (int j = 0; j < 10; ++j) {
                int alt = mask ^ (1 << j);
                if (seen.count(alt)) res = max(res, i - seen[alt]);
            }
            if (!seen.count(mask)) seen[mask] = i;
        }
        return res;
    }
};
```

---

##89 ****[Problem Link]https://leetcode.com/problems/compare-strings-by-frequency-of-the-smallest-character****  
**Approach:** Count frequency for queries and words, then binary search.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    int f(string s) {
        char minChar = *min_element(s.begin(), s.end());
        return count(s.begin(), s.end(), minChar);
    }

    vector<int> numSmallerByFrequency(vector<string>& queries, vector<string>& words) {
        vector<int> wFreq;
        for (string& w : words)
            wFreq.push_back(f(w));
        sort(wFreq.begin(), wFreq.end());

        vector<int> res;
        for (string& q : queries) {
            int fq = f(q);
            int count = wFreq.end() - upper_bound(wFreq.begin(), wFreq.end(), fq);
            res.push_back(count);
        }
        return res;
    }
};
```

---

##90 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-flips-to-convert-binary-matrix-to-zero-matrix****  
**Approach:** BFS with bitmask representing grid state.  
**Time Complexity:** O(2^(m*n))

```cpp
#include <vector>
#include <queue>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int minFlips(vector<vector<int>>& mat) {
        int m = mat.size(), n = mat[0].size(), start = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (mat[i][j])
                    start |= 1 << (i * n + j);

        queue<pair<int, int>> q;
        unordered_set<int> visited;
        q.push({start, 0});
        visited.insert(start);

        vector<pair<int, int>> dirs = {{0,0},{0,1},{1,0},{0,-1},{-1,0}};

        while (!q.empty()) {
            auto [state, step] = q.front(); q.pop();
            if (state == 0) return step;

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    int next = state;
                    for (auto [dx, dy] : dirs) {
                        int x = i + dx, y = j + dy;
                        if (x >= 0 && y >= 0 && x < m && y < n) {
                            next ^= 1 << (x * n + y);
                        }
                    }
                    if (!visited.count(next)) {
                        visited.insert(next);
                        q.push({next, step + 1});
                    }
                }
            }
        }
        return -1;
    }
};
```

---

##91 ****[Problem Link]https://leetcode.com/problems/arithmetic-subarrays****  
**Approach:** Sort each subarray and check constant difference.  
**Time Complexity:** O(m * k log k)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<bool> checkArithmeticSubarrays(vector<int>& nums, vector<int>& l, vector<int>& r) {
        vector<bool> res;
        for (int i = 0; i < l.size(); ++i) {
            vector<int> sub(nums.begin() + l[i], nums.begin() + r[i] + 1);
            sort(sub.begin(), sub.end());
            bool valid = true;
            int diff = sub[1] - sub[0];
            for (int j = 2; j < sub.size(); ++j) {
                if (sub[j] - sub[j - 1] != diff) {
                    valid = false;
                    break;
                }
            }
            res.push_back(valid);
        }
        return res;
    }
};
```

---

##92 ****[Problem Link]https://leetcode.com/problems/form-largest-integer-with-digits-that-add-up-to-target****  
**Approach:** DP with digit consideration and maximum length building.  
**Time Complexity:** O(target * 9)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string largestNumber(vector<int>& cost, int target) {
        vector<string> dp(target + 1, "#");
        dp[0] = "";
        for (int t = 1; t <= target; ++t) {
            for (int d = 1; d <= 9; ++d) {
                int c = cost[d - 1];
                if (t >= c && dp[t - c] != "#") {
                    string cand = dp[t - c] + (char)('0' + d);
                    if (cand.size() > dp[t].size() || (cand.size() == dp[t].size() && cand > dp[t])) {
                        dp[t] = cand;
                    }
                }
            }
        }
        return dp[target] == "#" ? "0" : dp[target];
    }
};
```

---

##93 ****[Problem Link]https://leetcode.com/problems/number-of-substrings-with-only-1s****  
**Approach:** Count consecutive 1s and use formula for sum of counts.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int numSub(string s) {
        const int MOD = 1e9 + 7;
        long res = 0, count = 0;
        for (char c : s) {
            if (c == '1') count++;
            else {
                res += count * (count + 1) / 2;
                count = 0;
            }
        }
        res += count * (count + 1) / 2;
        return res % MOD;
    }
};
```

---

##94 ****[Problem Link]https://leetcode.com/problems/length-of-last-word****  
**Approach:** Traverse from end, skip spaces, count characters.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int lengthOfLastWord(string s) {
        int i = s.size() - 1, len = 0;
        while (i >= 0 && s[i] == ' ') i--;
        while (i >= 0 && s[i] != ' ') {
            len++;
            i--;
        }
        return len;
    }
};
```

---

##95 ****[Problem Link]https://leetcode.com/problems/string-matching-in-an-array****  
**Approach:** Check if one string is a substring of another.  
**Time Complexity:** O(n^2 * m)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<string> stringMatching(vector<string>& words) {
        vector<string> res;
        for (int i = 0; i < words.size(); ++i) {
            for (int j = 0; j < words.size(); ++j) {
                if (i != j && words[j].find(words[i]) != string::npos) {
                    res.push_back(words[i]);
                    break;
                }
            }
        }
        return res;
    }
};
```

---

##96 ****[Problem Link]https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold****  
**Approach:** Sliding window sum and compare average.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int numOfSubarrays(vector<int>& arr, int k, int threshold) {
        int sum = 0, count = 0;
        for (int i = 0; i < k; ++i) sum += arr[i];
        if (sum >= threshold * k) count++;

        for (int i = k; i < arr.size(); ++i) {
            sum += arr[i] - arr[i - k];
            if (sum >= threshold * k) count++;
        }

        return count;
    }
};
```

---

##97 ****[Problem Link]https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options****  
**Approach:** Use formula with modulo to count permutations.  
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int countOrders(int n) {
        const int MOD = 1e9 + 7;
        long res = 1;
        for (int i = 1; i <= n; ++i) {
            res = res * i % MOD;
            res = res * (2 * i - 1) % MOD;
        }
        return res;
    }
};
```

---

##98 ****[Problem Link]https://leetcode.com/problems/rotating-the-box****  
**Approach:** Simulate stone falling, then rotate matrix.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<char>> rotateTheBox(vector<vector<char>>& box) {
        int m = box.size(), n = box[0].size();
        for (int i = 0; i < m; ++i) {
            int empty = n - 1;
            for (int j = n - 1; j >= 0; --j) {
                if (box[i][j] == '*') {
                    empty = j - 1;
                } else if (box[i][j] == '#') {
                    swap(box[i][j], box[i][empty--]);
                }
            }
        }

        vector<vector<char>> res(n, vector<char>(m));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                res[j][m - 1 - i] = box[i][j];
        return res;
    }
};
```

---

##99 ****[Problem Link]https://leetcode.com/problems/check-if-a-string-can-break-another-string****  
**Approach:** Sort both and check element-wise dominance.  
**Time Complexity:** O(n log n)

```cpp
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    bool checkIfCanBreak(string s1, string s2) {
        sort(s1.begin(), s1.end());
        sort(s2.begin(), s2.end());
        return dominates(s1, s2) || dominates(s2, s1);
    }

    bool dominates(string& a, string& b) {
        for (int i = 0; i < a.size(); ++i)
            if (a[i] < b[i]) return false;
        return true;
    }
};
```

---

##100 ****[Problem Link]https://leetcode.com/problems/largest-1-bordered-square****  
**Approach:** Prefix sum of horizontal and vertical 1s to validate borders.  
**Time Complexity:** O(m * n * min(m, n))

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int largest1BorderedSquare(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size(), res = 0;
        vector<vector<int>> h(m, vector<int>(n)), v(m, vector<int>(n));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (grid[i][j]) {
                    h[i][j] = (j == 0 ? 0 : h[i][j - 1]) + 1;
                    v[i][j] = (i == 0 ? 0 : v[i - 1][j]) + 1;
                }

        for (int i = m - 1; i >= 0; --i)
            for (int j = n - 1; j >= 0; --j)
                for (int k = min(h[i][j], v[i][j]); k > 0; --k)
                    if (v[i][j - k + 1] >= k && h[i - k + 1][j] >= k) {
                        res = max(res, k);
                        break;
                    }

        return res * res;
    }
};
```

---
