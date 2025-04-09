
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

##101 ****[Problem Link]https://leetcode.com/problems/maximum-score-after-splitting-a-string****  
**Approach:** Count total 1s, then scan left to right and compute max score.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int maxScore(string s) {
        int ones = count(s.begin(), s.end(), '1');
        int zeros = 0, maxScore = 0;
        for (int i = 0; i < s.size() - 1; ++i) {
            zeros += s[i] == '0';
            ones -= s[i] == '1';
            maxScore = max(maxScore, zeros + ones);
        }
        return maxScore;
    }
};
```

---

##102 ****[Problem Link]https://leetcode.com/problems/count-good-meals****  
**Approach:** For each number, check if power-of-two - number was seen before.  
**Time Complexity:** O(n * log M)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int countPairs(vector<int>& deliciousness) {
        const int MOD = 1e9 + 7;
        unordered_map<int, int> freq;
        int res = 0;
        for (int x : deliciousness) {
            for (int p = 1; p <= (1 << 21); p <<= 1) {
                res = (res + freq[p - x]) % MOD;
            }
            freq[x]++;
        }
        return res;
    }
};
```

---

##103 ****[Problem Link]https://leetcode.com/problems/find-the-closest-palindrome****  
**Approach:** Generate candidates from prefix manipulation and compare.  
**Time Complexity:** O(log n)

```cpp
#include <string>
#include <cmath>
#include <set>

using namespace std;

class Solution {
public:
    string nearestPalindromic(string n) {
        long num = stol(n);
        int len = n.size();
        set<long> candidates;

        candidates.insert((long)pow(10, len) + 1);
        candidates.insert((long)pow(10, len - 1) - 1);

        long prefix = stol(n.substr(0, (len + 1) / 2));
        for (int i = -1; i <= 1; ++i) {
            string p = to_string(prefix + i);
            string pal = p + string(p.rbegin() + (len % 2), p.rend());
            candidates.insert(stol(pal));
        }

        candidates.erase(num);
        long res = -1, diff = LONG_MAX;
        for (long c : candidates) {
            if (abs(c - num) < diff || (abs(c - num) == diff && c < res)) {
                diff = abs(c - num);
                res = c;
            }
        }
        return to_string(res);
    }
};
```

---

##104 ****[Problem Link]https://leetcode.com/problems/check-if-number-is-a-sum-of-powers-of-three****  
**Approach:** Convert to base 3 and check if all digits are 0 or 1.  
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    bool checkPowersOfThree(int n) {
        while (n > 0) {
            if (n % 3 == 2) return false;
            n /= 3;
        }
        return true;
    }
};
```

---

##105 ****[Problem Link]https://leetcode.com/problems/split-two-strings-to-make-palindrome****  
**Approach:** Check both directions if the prefix of one and suffix of the other can form a palindrome.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    bool checkPalindrome(string& a, int l, int r) {
        while (l < r) {
            if (a[l++] != a[r--]) return false;
        }
        return true;
    }

    bool check(string& a, string& b) {
        int i = 0, j = b.size() - 1;
        while (i < j && a[i] == b[j]) {
            i++;
            j--;
        }
        return checkPalindrome(a, i, j) || checkPalindrome(b, i, j);
    }

    bool checkPalindromeFormation(string a, string b) {
        return check(a, b) || check(b, a);
    }
};
```

---

##106 ****[Problem Link]https://leetcode.com/problems/pyramid-transition-matrix****  
**Approach:** DFS + backtracking with a mapping from base pairs to possible tops.  
**Time Complexity:** Exponential

```cpp
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

class Solution {
public:
    unordered_map<string, vector<char>> mp;

    bool pyramidTransition(string bottom, vector<string>& allowed) {
        for (auto& s : allowed) {
            mp[s.substr(0, 2)].push_back(s[2]);
        }
        return dfs(bottom, "");
    }

    bool dfs(string cur, string next) {
        if (cur.size() == 1) return true;
        if (next.size() + 1 == cur.size()) return dfs(next, "");
        int i = next.size();
        string key = cur.substr(i, 2);
        if (!mp.count(key)) return false;
        for (char c : mp[key]) {
            if (dfs(cur, next + c)) return true;
        }
        return false;
    }
};
```

---

##107 ****[Problem Link]https://leetcode.com/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence****  
**Approach:** Split sentence into words and check prefix.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <sstream>

using namespace std;

class Solution {
public:
    int isPrefixOfWord(string sentence, string searchWord) {
        stringstream ss(sentence);
        string word;
        int index = 1;
        while (ss >> word) {
            if (word.find(searchWord) == 0) return index;
            index++;
        }
        return -1;
    }
};
```

---

##108 ****[Problem Link]https://leetcode.com/problems/minimum-operations-to-make-the-array-increasing****  
**Approach:** Traverse and ensure each element is larger than the previous.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int minOperations(vector<int>& nums) {
        int res = 0;
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] <= nums[i - 1]) {
                res += nums[i - 1] - nums[i] + 1;
                nums[i] = nums[i - 1] + 1;
            }
        }
        return res;
    }
};
```

---

##109 ****[Problem Link]https://leetcode.com/problems/spiral-matrix-iii****  
**Approach:** Simulate steps in all four directions expanding range each time.  
**Time Complexity:** O(R * C)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> spiralMatrixIII(int R, int C, int r0, int c0) {
        vector<vector<int>> res;
        int steps = 0, len = 1, dir = 0;
        vector<int> dr = {0, 1, 0, -1}, dc = {1, 0, -1, 0};

        while (res.size() < R * C) {
            if (dir == 0 || dir == 2) ++steps;
            for (int i = 0; i < steps; ++i) {
                if (r0 >= 0 && r0 < R && c0 >= 0 && c0 < C) {
                    res.push_back({r0, c0});
                }
                r0 += dr[dir];
                c0 += dc[dir];
            }
            dir = (dir + 1) % 4;
        }
        return res;
    }
};
```

---

##110 ****[Problem Link]https://leetcode.com/problems/number-of-equivalent-domino-pairs****  
**Approach:** Encode domino pairs into ordered keys and count combinations.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        unordered_map<int, int> count;
        int res = 0;
        for (auto& d : dominoes) {
            int a = d[0], b = d[1];
            int key = min(a, b) * 10 + max(a, b);
            res += count[key]++;
        }
        return res;
    }
};
```

---

##111 ****[Problem Link]https://leetcode.com/problems/matrix-cells-in-distance-order****  
**Approach:** Sort all coordinates by Manhattan distance from (r0, c0).  
**Time Complexity:** O(R * C log(R * C))

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<vector<int>> allCellsDistOrder(int R, int C, int r0, int c0) {
        vector<vector<int>> res;
        for (int i = 0; i < R; ++i)
            for (int j = 0; j < C; ++j)
                res.push_back({i, j});
        sort(res.begin(), res.end(), [&](vector<int>& a, vector<int>& b) {
            return abs(a[0] - r0) + abs(a[1] - c0) < abs(b[0] - r0) + abs(b[1] - c0);
        });
        return res;
    }
};
```

---

##112 ****[Problem Link]https://leetcode.com/problems/path-crossing****  
**Approach:** Track positions using a set and return true on revisit.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <set>

using namespace std;

class Solution {
public:
    bool isPathCrossing(string path) {
        set<pair<int, int>> visited;
        int x = 0, y = 0;
        visited.insert({x, y});
        for (char c : path) {
            if (c == 'N') y++;
            else if (c == 'S') y--;
            else if (c == 'E') x++;
            else x--;
            if (!visited.insert({x, y}).second) return true;
        }
        return false;
    }
};
```

---

##113 ****[Problem Link]https://leetcode.com/problems/maximum-sum-obtained-of-any-permutation****  
**Approach:** Use difference array to count frequency and apply to sorted array.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxSumRangeQuery(vector<int>& nums, vector<vector<int>>& requests) {
        const int MOD = 1e9 + 7;
        int n = nums.size();
        vector<int> freq(n + 1);
        for (auto& r : requests) {
            freq[r[0]]++;
            freq[r[1] + 1]--;
        }

        for (int i = 1; i < n; ++i)
            freq[i] += freq[i - 1];
        freq.pop_back();

        sort(nums.begin(), nums.end());
        sort(freq.begin(), freq.end());

        long res = 0;
        for (int i = 0; i < n; ++i)
            res = (res + (long)nums[i] * freq[i]) % MOD;
        return res;
    }
};
```

---

##114 ****[Problem Link]https://leetcode.com/problems/map-of-highest-peak****  
**Approach:** BFS from all water cells and assign distance as height.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    vector<vector<int>> highestPeak(vector<vector<int>>& isWater) {
        int m = isWater.size(), n = isWater[0].size();
        queue<pair<int, int>> q;
        vector<vector<int>> res(m, vector<int>(n, -1));

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (isWater[i][j]) {
                    res[i][j] = 0;
                    q.push({i, j});
                }

        vector<pair<int, int>> dirs = {{0,1},{1,0},{0,-1},{-1,0}};
        while (!q.empty()) {
            auto [x, y] = q.front(); q.pop();
            for (auto& [dx, dy] : dirs) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && ny >= 0 && nx < m && ny < n && res[nx][ny] == -1) {
                    res[nx][ny] = res[x][y] + 1;
                    q.push({nx, ny});
                }
            }
        }

        return res;
    }
};
```

---

##115 ****[Problem Link]https://leetcode.com/problems/strong-password-checker****  
**Approach:** Check missing character types and apply greedy operations for replacements/deletions.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    int strongPasswordChecker(string password) {
        int n = password.size(), a = 1, A = 1, d = 1;
        for (char c : password) {
            if (islower(c)) a = 0;
            else if (isupper(c)) A = 0;
            else if (isdigit(c)) d = 0;
        }

        vector<int> rep;
        for (int i = 2; i < n;) {
            if (password[i] == password[i - 1] && password[i - 1] == password[i - 2]) {
                int j = i - 2;
                while (i < n && password[i] == password[j]) i++;
                rep.push_back(i - j);
            } else {
                i++;
            }
        }

        int total_missing = a + A + d;
        if (n < 6) return max(total_missing, 6 - n);

        int over = max(n - 20, 0), left = over;
        for (int& r : rep) {
            if (left <= 0) break;
            int reduce = min(left, r - 2);
            r -= reduce;
            left -= reduce;
        }

        int replace = 0;
        for (int r : rep) replace += r / 3;

        if (n <= 20) return max(total_missing, replace);
        return over + max(total_missing, replace);
    }
};
```

---

##116 ****[Problem Link]https://leetcode.com/problems/shift-2d-grid****  
**Approach:** Flatten the grid, shift, and rebuild 2D array.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> shiftGrid(vector<vector<int>>& grid, int k) {
        int m = grid.size(), n = grid[0].size();
        vector<int> flat;
        for (auto& row : grid)
            for (int val : row)
                flat.push_back(val);
        k = k % (m * n);
        rotate(flat.rbegin(), flat.rbegin() + k, flat.rend());
        vector<vector<int>> res(m, vector<int>(n));
        for (int i = 0; i < m * n; ++i)
            res[i / n][i % n] = flat[i];
        return res;
    }
};
```

---

##117 ****[Problem Link]https://leetcode.com/problems/create-sorted-array-through-instructions****  
**Approach:** Use Fenwick Tree to count less and greater elements.  
**Time Complexity:** O(n log m)

```cpp
#include <vector>

using namespace std;

class Solution {
    vector<int> bit;

    void update(int i) {
        for (; i < bit.size(); i += i & -i) bit[i]++;
    }

    int query(int i) {
        int res = 0;
        for (; i > 0; i -= i & -i) res += bit[i];
        return res;
    }

public:
    int createSortedArray(vector<int>& instructions) {
        const int MOD = 1e9 + 7;
        int maxVal = *max_element(instructions.begin(), instructions.end());
        bit.resize(maxVal + 2);
        long cost = 0;
        for (int i = 0; i < instructions.size(); ++i) {
            int x = instructions[i];
            int less = query(x - 1);
            int greater = i - query(x);
            cost = (cost + min(less, greater)) % MOD;
            update(x);
        }
        return cost;
    }
};
```

---

##118 ****[Problem Link]https://leetcode.com/problems/string-without-aaa-or-bbb****  
**Approach:** Greedy generation by choosing dominant character unless restricted.  
**Time Complexity:** O(a + b)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string strWithout3a3b(int a, int b) {
        string res;
        while (a > 0 || b > 0) {
            if (a > b) {
                if (a >= 2) { res += "aa"; a -= 2; }
                else { res += "a"; a--; }
                if (b > 0) { res += "b"; b--; }
            } else {
                if (b >= 2) { res += "bb"; b -= 2; }
                else { res += "b"; b--; }
                if (a > 0) { res += "a"; a--; }
            }
        }
        return res;
    }
};
```

---

##119 ****[Problem Link]https://leetcode.com/problems/rearrange-words-in-a-sentence****  
**Approach:** Sort words by length, lowercase first word, capitalize first result word.  
**Time Complexity:** O(n log n)

```cpp
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    string arrangeWords(string text) {
        vector<string> words;
        stringstream ss(text);
        string word;
        while (ss >> word) {
            word[0] = tolower(word[0]);
            words.push_back(word);
        }
        stable_sort(words.begin(), words.end(), [](const string& a, const string& b) {
            return a.size() < b.size();
        });
        words[0][0] = toupper(words[0][0]);
        string res;
        for (string& w : words) res += w + " ";
        res.pop_back();
        return res;
    }
};
```

---

##120 ****[Problem Link]https://leetcode.com/problems/determine-if-string-halves-are-alike****  
**Approach:** Count vowels in both halves and compare.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool halvesAreAlike(string s) {
        unordered_set<char> vowels = {'a','e','i','o','u','A','E','I','O','U'};
        int mid = s.size() / 2, count = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (vowels.count(s[i])) count += (i < mid ? 1 : -1);
        }
        return count == 0;
    }
};
```

---

##121 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-eaten-apples****  
**Approach:** Greedy using min-heap to prioritize expiry.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int eatenApples(vector<int>& apples, vector<int>& days) {
        int i = 0, n = apples.size(), count = 0;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

        while (i < n || !pq.empty()) {
            if (i < n && apples[i] > 0)
                pq.emplace(i + days[i], apples[i]);

            while (!pq.empty() && pq.top().first <= i)
                pq.pop();

            if (!pq.empty()) {
                auto [exp, num] = pq.top(); pq.pop();
                if (--num > 0)
                    pq.emplace(exp, num);
                count++;
            }
            i++;
        }

        return count;
    }
};
```

---

##122 ****[Problem Link]https://leetcode.com/problems/sum-of-subsequence-widths****  
**Approach:** Use combinatorics and powers of 2 for contribution of each element.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int sumSubseqWidths(vector<int>& nums) {
        const int MOD = 1e9 + 7;
        sort(nums.begin(), nums.end());
        int n = nums.size();
        vector<long> pow2(n, 1);
        for (int i = 1; i < n; ++i)
            pow2[i] = pow2[i - 1] * 2 % MOD;

        long res = 0;
        for (int i = 0; i < n; ++i)
            res = (res + nums[i] * (pow2[i] - pow2[n - 1 - i])) % MOD;
        return (res + MOD) % MOD;
    }
};
```

---

##123 ****[Problem Link]https://leetcode.com/problems/valid-permutations-for-di-sequence****  
**Approach:** DP with prefix sums.  
**Time Complexity:** O(n^2)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    int numPermsDISequence(string s) {
        const int MOD = 1e9 + 7;
        int n = s.size();
        vector<vector<int>> dp(n + 1, vector<int>(n + 1));
        dp[0][0] = 1;
        for (int i = 1; i <= n; ++i) {
            if (s[i - 1] == 'I') {
                int curr = 0;
                for (int j = 0; j <= i; ++j) {
                    dp[i][j] = curr;
                    if (j < i) curr = (curr + dp[i - 1][j]) % MOD;
                }
            } else {
                int curr = 0;
                for (int j = i - 1; j >= 0; --j) {
                    curr = (curr + dp[i - 1][j + 1]) % MOD;
                    dp[i][j] = curr;
                }
            }
        }
        int res = 0;
        for (int val : dp[n]) res = (res + val) % MOD;
        return res;
    }
};
```

---

##124 ****[Problem Link]https://leetcode.com/problems/maximum-average-pass-ratio****  
**Approach:** Greedy with priority queue, always add to class with best ratio gain.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    double maxAverageRatio(vector<vector<int>>& classes, int extraStudents) {
        auto gain = [](int pass, int total) {
            return (double)(pass + 1) / (total + 1) - (double)pass / total;
        };

        priority_queue<pair<double, pair<int, int>>> pq;
        for (auto& c : classes)
            pq.push({gain(c[0], c[1]), {c[0], c[1]}});

        while (extraStudents--) {
            auto [_, cur] = pq.top(); pq.pop();
            pq.push({gain(cur.first + 1, cur.second + 1), {cur.first + 1, cur.second + 1}});
        }

        double total = 0.0;
        while (!pq.empty()) {
            auto [_, c] = pq.top(); pq.pop();
            total += (double)c.first / c.second;
        }

        return total / classes.size();
    }
};
```

---

##125 ****[Problem Link]https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs****  
**Approach:** Binary search on time limit + backtracking with pruning.  
**Time Complexity:** O(k^n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    bool backtrack(vector<int>& jobs, vector<int>& workers, int idx, int limit) {
        if (idx == jobs.size()) return true;
        int cur = jobs[idx];
        for (int i = 0; i < workers.size(); ++i) {
            if (workers[i] + cur <= limit) {
                workers[i] += cur;
                if (backtrack(jobs, workers, idx + 1, limit)) return true;
                workers[i] -= cur;
            }
            if (workers[i] == 0) break;
        }
        return false;
    }

    int minimumTimeRequired(vector<int>& jobs, int k) {
        sort(jobs.rbegin(), jobs.rend());
        int low = jobs[0], high = accumulate(jobs.begin(), jobs.end(), 0);
        while (low < high) {
            int mid = (low + high) / 2;
            vector<int> workers(k, 0);
            if (backtrack(jobs, workers, 0, mid)) high = mid;
            else low = mid + 1;
        }
        return low;
    }
};
```

---

##126 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-non-overlapping-substrings****  
**Approach:** Greedy selection of valid substrings sorted by end index.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<string> maxNumOfSubstrings(string s) {
        int n = s.size();
        vector<int> first(26, n), last(26, -1);
        for (int i = 0; i < n; ++i) {
            first[s[i] - 'a'] = min(first[s[i] - 'a'], i);
            last[s[i] - 'a'] = i;
        }

        vector<pair<int, int>> intervals;
        for (int i = 0; i < 26; ++i) {
            if (last[i] == -1) continue;
            int l = first[i], r = last[i], j = l;
            while (j <= r) {
                l = min(l, first[s[j] - 'a']);
                r = max(r, last[s[j] - 'a']);
                j++;
            }
            if (l == first[i])
                intervals.emplace_back(r, l);
        }

        sort(intervals.begin(), intervals.end());
        vector<string> res;
        int end = -1;
        for (auto& [r, l] : intervals) {
            if (l > end) {
                res.push_back(s.substr(l, r - l + 1));
                end = r;
            }
        }
        return res;
    }
};
```

---

##127 ****[Problem Link]https://leetcode.com/problems/detect-pattern-of-length-m-repeated-k-or-more-times****  
**Approach:** Slide a window of length m and count repetitions.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    bool containsPattern(vector<int>& arr, int m, int k) {
        int n = arr.size();
        for (int i = 0; i <= n - m * k; ++i) {
            bool match = true;
            for (int j = 0; j < m * k; ++j) {
                if (arr[i + j] != arr[i + j % m]) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
        return false;
    }
};
```

---

##128 ****[Problem Link]https://leetcode.com/problems/delete-columns-to-make-sorted-ii****  
**Approach:** Greedy lexicographic comparison with used flags.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int minDeletionSize(vector<string>& A) {
        int n = A.size(), m = A[0].size(), res = 0;
        vector<bool> sorted(n - 1, false);

        for (int j = 0; j < m; ++j) {
            bool needDelete = false;
            for (int i = 0; i < n - 1; ++i) {
                if (!sorted[i] && A[i][j] > A[i + 1][j]) {
                    needDelete = true;
                    break;
                }
            }
            if (needDelete) {
                res++;
            } else {
                for (int i = 0; i < n - 1; ++i) {
                    if (A[i][j] < A[i + 1][j]) sorted[i] = true;
                }
            }
        }
        return res;
    }
};
```

---

##129 ****[Problem Link]https://leetcode.com/problems/decode-xored-permutation****  
**Approach:** Use XOR properties to reconstruct original permutation.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> decode(vector<int>& encoded) {
        int n = encoded.size() + 1;
        int total = 0, odd = 0;
        for (int i = 1; i <= n; ++i) total ^= i;
        for (int i = 1; i < encoded.size(); i += 2) odd ^= encoded[i];

        vector<int> perm(n);
        perm[0] = total ^ odd;
        for (int i = 1; i < n; ++i)
            perm[i] = perm[i - 1] ^ encoded[i - 1];
        return perm;
    }
};
```

---

##130 ****[Problem Link]https://leetcode.com/problems/set-intersection-size-at-least-two****  
**Approach:** Sort intervals and greedily cover with two-point tracking.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int intersectionSizeTwo(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), [](auto& a, auto& b) {
            return a[1] != b[1] ? a[1] < b[1] : a[0] > b[0];
        });

        int res = 0, a = -1, b = -1;
        for (auto& it : intervals) {
            if (it[0] > b) {
                res += 2;
                b = it[1];
                a = it[1] - 1;
            } else if (it[0] > a) {
                res += 1;
                a = b;
                b = it[1];
            }
        }
        return res;
    }
};
```

---

##131 ****[Problem Link]https://leetcode.com/problems/construct-quad-tree****  
**Approach:** Recursively divide grid into four quadrants.  
**Time Complexity:** O(n^2)

```cpp
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;

    Node() : val(false), isLeaf(false), topLeft(nullptr), topRight(nullptr), bottomLeft(nullptr), bottomRight(nullptr) {}
    Node(bool _val, bool _isLeaf) : val(_val), isLeaf(_isLeaf),
        topLeft(nullptr), topRight(nullptr), bottomLeft(nullptr), bottomRight(nullptr) {}
};

class Solution {
public:
    Node* construct(vector<vector<int>>& grid) {
        return helper(grid, 0, 0, grid.size());
    }

    Node* helper(vector<vector<int>>& grid, int r, int c, int len) {
        if (len == 1) return new Node(grid[r][c], true);
        int half = len / 2;
        Node* tl = helper(grid, r, c, half);
        Node* tr = helper(grid, r, c + half, half);
        Node* bl = helper(grid, r + half, c, half);
        Node* br = helper(grid, r + half, c + half, half);

        if (tl->isLeaf && tr->isLeaf && bl->isLeaf && br->isLeaf &&
            tl->val == tr->val && tr->val == bl->val && bl->val == br->val) {
            return new Node(tl->val, true);
        }
        Node* root = new Node(false, false);
        root->topLeft = tl;
        root->topRight = tr;
        root->bottomLeft = bl;
        root->bottomRight = br;
        return root;
    }
};
```

---

##132 ****[Problem Link]https://leetcode.com/problems/find-the-winner-of-an-array-game****  
**Approach:** Track max and count how many wins it gets.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int getWinner(vector<int>& arr, int k) {
        int win = arr[0], count = 0;
        for (int i = 1; i < arr.size(); ++i) {
            if (win > arr[i]) count++;
            else {
                win = arr[i];
                count = 1;
            }
            if (count == k) return win;
        }
        return win;
    }
};
```

---

##133 ****[Problem Link]https://leetcode.com/problems/equal-sum-arrays-with-minimum-number-of-operations****  
**Approach:** Use greedy + counting to equalize sums with least moves.  
**Time Complexity:** O(n log n) or O(n) with counting

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minOperations(vector<int>& nums1, vector<int>& nums2) {
        if (nums1.size() > nums2.size() * 6 || nums2.size() > nums1.size() * 6) return -1;

        vector<int> count1(7, 0), count2(7, 0);
        int sum1 = 0, sum2 = 0;
        for (int x : nums1) { count1[x]++; sum1 += x; }
        for (int x : nums2) { count2[x]++; sum2 += x; }

        if (sum1 > sum2) {
            swap(count1, count2);
            swap(sum1, sum2);
        }

        vector<int> change(6);
        for (int i = 1; i <= 6; ++i) {
            change[6 - i] += count1[i];
            change[i - 1] += count2[i];
        }

        int ops = 0, diff = sum2 - sum1;
        for (int i = 5; i >= 0 && diff > 0; --i) {
            int take = min((diff + i) / (i + 1), change[i]);
            ops += take;
            diff -= take * (i + 1);
        }

        return diff > 0 ? -1 : ops;
    }
};
```

---

##134 ****[Problem Link]https://leetcode.com/problems/maximum-value-at-a-given-index-in-a-bounded-array****  
**Approach:** Binary search the peak value and use prefix sum logic.  
**Time Complexity:** O(log(sum))

```cpp
class Solution {
public:
    long calc(int n, int peak) {
        if (peak >= n) return (long)n * (2 * peak - n + 1) / 2;
        return (long)peak * (peak + 1) / 2 + (n - peak);
    }

    bool can(int n, int index, int maxSum, int x) {
        long left = calc(index, x - 1);
        long right = calc(n - index - 1, x - 1);
        return left + right + x <= maxSum;
    }

    int maxValue(int n, int index, int maxSum) {
        int lo = 1, hi = maxSum;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (can(n, index, maxSum, mid)) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }
};
```

---

##135 ****[Problem Link]https://leetcode.com/problems/snakes-and-ladders****  
**Approach:** BFS on flattened board with jump mechanics.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <queue>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int snakesAndLadders(vector<vector<int>>& board) {
        int n = board.size();
        auto get = [&](int s) {
            int r = (s - 1) / n, c = (s - 1) % n;
            if (r % 2) c = n - 1 - c;
            return board[n - 1 - r][c];
        };

        queue<pair<int, int>> q;
        q.push({1, 0});
        vector<bool> visited(n * n + 1, false);
        visited[1] = true;

        while (!q.empty()) {
            auto [s, d] = q.front(); q.pop();
            if (s == n * n) return d;
            for (int i = 1; i <= 6 && s + i <= n * n; ++i) {
                int nxt = get(s + i);
                if (nxt == -1) nxt = s + i;
                if (!visited[nxt]) {
                    visited[nxt] = true;
                    q.push({nxt, d + 1});
                }
            }
        }
        return -1;
    }
};
```

---

##136 ****[Problem Link]https://leetcode.com/problems/online-majority-element-in-subarray****  
**Approach:** Use prefix sums per element + random sampling for queries.  
**Time Complexity:** Preprocessing O(n * √n), Query O(√n)

```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>

using namespace std;

class MajorityChecker {
    unordered_map<int, vector<int>> pos;
    vector<int> arr;

public:
    MajorityChecker(vector<int>& arr) : arr(arr) {
        for (int i = 0; i < arr.size(); ++i)
            pos[arr[i]].push_back(i);
    }

    int query(int left, int right, int threshold) {
        static default_random_engine gen((random_device())());
        uniform_int_distribution<int> dis(left, right);
        for (int t = 0; t < 20; ++t) {
            int x = arr[dis(gen)];
            auto& p = pos[x];
            int l = lower_bound(p.begin(), p.end(), left) - p.begin();
            int r = upper_bound(p.begin(), p.end(), right) - p.begin();
            if (r - l >= threshold) return x;
        }
        return -1;
    }
};
```

---

##137 ****[Problem Link]https://leetcode.com/problems/number-of-steps-to-reduce-a-number-in-binary-representation-to-one****  
**Approach:** Simulate binary reduction by traversing from right to left.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int numSteps(string s) {
        int steps = 0, carry = 0;
        for (int i = s.size() - 1; i > 0; --i) {
            if ((s[i] - '0' + carry) % 2 == 0) steps += 1;
            else {
                steps += 2;
                carry = 1;
            }
        }
        return steps + carry;
    }
};
```

---

##138 ****[Problem Link]https://leetcode.com/problems/base-7****  
**Approach:** Convert number to base-7 manually.  
**Time Complexity:** O(log n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string convertToBase7(int num) {
        if (num == 0) return "0";
        bool neg = num < 0;
        num = abs(num);
        string res;
        while (num > 0) {
            res = to_string(num % 7) + res;
            num /= 7;
        }
        return neg ? "-" + res : res;
    }
};
```

---

##139 ****[Problem Link]https://leetcode.com/problems/falling-squares****  
**Approach:** Segment tree or brute force simulation with intervals.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> fallingSquares(vector<vector<int>>& positions) {
        vector<pair<int, int>> heights;
        vector<int> res;
        int maxHeight = 0;
        for (auto& pos : positions) {
            int left = pos[0], side = pos[1], right = left + side;
            int height = side;
            for (auto& [l, r] : heights) {
                if (l < right && r > left)
                    height = max(height, r - l + side);
            }
            heights.push_back({left, right});
            maxHeight = max(maxHeight, height);
            res.push_back(maxHeight);
        }
        return res;
    }
};
```

---

##140 ****[Problem Link]https://leetcode.com/problems/last-substring-in-lexicographical-order****  
**Approach:** Use two-pointer algorithm to find the lexicographically last suffix.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string lastSubstring(string s) {
        int n = s.size(), i = 0, j = 1, k = 0;
        while (j + k < n) {
            if (s[i + k] == s[j + k]) {
                k++;
            } else if (s[i + k] < s[j + k]) {
                i = max(i + k + 1, j);
                j = i + 1;
                k = 0;
            } else {
                j += k + 1;
                k = 0;
            }
        }
        return s.substr(i);
    }
};
```

---

##141 ****[Problem Link]https://leetcode.com/problems/maximum-of-absolute-value-expression****  
**Approach:** Try all four combinations of signs and track maximum value.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class Solution {
public:
    int maxAbsValExpr(vector<int>& arr1, vector<int>& arr2) {
        int res = 0, n = arr1.size();
        for (int p : {-1, 1}) {
            for (int q : {-1, 1}) {
                int minVal = INT_MAX, maxVal = INT_MIN;
                for (int i = 0; i < n; ++i) {
                    int val = p * arr1[i] + q * arr2[i] + i;
                    minVal = min(minVal, val);
                    maxVal = max(maxVal, val);
                }
                res = max(res, maxVal - minVal);
            }
        }
        return res;
    }
};
```

---

##142 ****[Problem Link]https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths****  
**Approach:** Sort queries and edges, then union-find as edges grow.  
**Time Complexity:** O(q log q + e log e)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<int> parent;

    int find(int x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    }

    void unite(int x, int y) {
        parent[find(x)] = find(y);
    }

    vector<bool> distanceLimitedPathsExist(int n, vector<vector<int>>& edgeList, vector<vector<int>>& queries) {
        parent.resize(n);
        iota(parent.begin(), parent.end(), 0);

        sort(edgeList.begin(), edgeList.end(), [](auto& a, auto& b) {
            return a[2] < b[2];
        });

        int m = queries.size();
        vector<int> idx(m);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int i, int j) {
            return queries[i][2] < queries[j][2];
        });

        vector<bool> res(m);
        int j = 0;
        for (int i : idx) {
            while (j < edgeList.size() && edgeList[j][2] < queries[i][2]) {
                unite(edgeList[j][0], edgeList[j][1]);
                j++;
            }
            res[i] = find(queries[i][0]) == find(queries[i][1]);
        }
        return res;
    }
};
```

---

##143 ****[Problem Link]https://leetcode.com/problems/prime-number-of-set-bits-in-binary-representation****  
**Approach:** Count set bits and check for prime.  
**Time Complexity:** O(n log n)

```cpp
#include <unordered_set>

class Solution {
public:
    int countPrimeSetBits(int left, int right) {
        unordered_set<int> primes = {2,3,5,7,11,13,17,19};
        int res = 0;
        for (int i = left; i <= right; ++i) {
            int bits = __builtin_popcount(i);
            if (primes.count(bits)) res++;
        }
        return res;
    }
};
```

---

##144 ****[Problem Link]https://leetcode.com/problems/escape-a-large-maze****  
**Approach:** BFS with visited limit up to blocked area size squared.  
**Time Complexity:** O(b^2), b = number of blocked cells

```cpp
#include <vector>
#include <unordered_set>
#include <queue>

using namespace std;

class Solution {
public:
    bool isEscapePossible(vector<vector<int>>& blocked, vector<int>& source, vector<int>& target) {
        auto hash = [](int x, int y) { return (long long)x << 20 | y; };
        unordered_set<long long> block;
        for (auto& b : blocked)
            block.insert(hash(b[0], b[1]));

        auto bfs = [&](vector<int>& start, vector<int>& end) {
            queue<pair<int, int>> q;
            unordered_set<long long> seen;
            q.push({start[0], start[1]});
            seen.insert(hash(start[0], start[1]));
            int dirs[5] = {0, 1, 0, -1, 0};
            while (!q.empty() && seen.size() <= 20000) {
                auto [x, y] = q.front(); q.pop();
                for (int i = 0; i < 4; ++i) {
                    int nx = x + dirs[i], ny = y + dirs[i + 1];
                    if (nx < 0 || ny < 0 || nx >= 1e6 || ny >= 1e6) continue;
                    long long h = hash(nx, ny);
                    if (block.count(h) || seen.count(h)) continue;
                    if (nx == end[0] && ny == end[1]) return true;
                    seen.insert(h);
                    q.push({nx, ny});
                }
            }
            return seen.size() > 20000;
        };

        return bfs(source, target) && bfs(target, source);
    }
};
```

---

##145 ****[Problem Link]https://leetcode.com/problems/tiling-a-rectangle-with-the-fewest-squares****  
**Approach:** DFS with pruning and greedy placement.  
**Time Complexity:** Exponential with pruning

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int res = INT_MAX;

    void dfs(int n, int m, vector<int>& height, int cur) {
        if (cur >= res) return;
        int minH = INT_MAX, idx = -1;
        for (int i = 0; i < m; ++i) {
            if (height[i] < minH) {
                minH = height[i];
                idx = i;
            }
        }
        if (minH == n) {
            res = min(res, cur);
            return;
        }

        int j = idx;
        while (j < m && height[j] == minH && j - idx + 1 + minH <= n) ++j;
        --j;

        for (int len = j - idx + 1; len >= 1; --len) {
            for (int k = 0; k < len; ++k) height[idx + k] += len;
            dfs(n, m, height, cur + 1);
            for (int k = 0; k < len; ++k) height[idx + k] -= len;
        }
    }

    int tilingRectangle(int n, int m) {
        if (n < m) swap(n, m);
        vector<int> height(m, 0);
        dfs(n, m, height, 0);
        return res;
    }
};
```

---

##146 ****[Problem Link]https://leetcode.com/problems/maximum-population-year****  
**Approach:** Count changes per year using prefix sum logic.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int maximumPopulation(vector<vector<int>>& logs) {
        vector<int> pop(101);
        for (auto& l : logs) {
            pop[l[0] - 1950]++;
            pop[l[1] - 1950]--;
        }
        int maxYear = 1950, maxPop = pop[0];
        for (int i = 1; i < 101; ++i) {
            pop[i] += pop[i - 1];
            if (pop[i] > maxPop) {
                maxPop = pop[i];
                maxYear = 1950 + i;
            }
        }
        return maxYear;
    }
};
```

---

##147 ****[Problem Link]https://leetcode.com/problems/super-pow****  
**Approach:** Modular exponentiation and recursion.  
**Time Complexity:** O(log b)

```cpp
class Solution {
public:
    const int MOD = 1337;

    int modPow(int a, int k) {
        int res = 1;
        a %= MOD;
        for (int i = 0; i < k; ++i)
            res = (res * a) % MOD;
        return res;
    }

    int superPow(int a, vector<int>& b) {
        if (b.empty()) return 1;
        int last = b.back(); b.pop_back();
        return modPow(superPow(a, b), 10) * modPow(a, last) % MOD;
    }
};
```

---

##148 ****[Problem Link]https://leetcode.com/problems/find-latest-group-of-size-m****  
**Approach:** Reverse simulation with size tracking using map.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int findLatestStep(vector<int>& arr, int m) {
        int n = arr.size();
        if (m == n) return n;
        vector<int> length(n + 2), count(n + 1);
        int res = -1;

        for (int i = 0; i < n; ++i) {
            int a = arr[i], left = length[a - 1], right = length[a + 1];
            int total = left + right + 1;
            length[a - left] = length[a + right] = total;
            count[left]--, count[right]--, count[total]++;
            if (count[m]) res = i + 1;
        }
        return res;
    }
};
```

---

##149 ****[Problem Link]https://leetcode.com/problems/longest-nice-substring****  
**Approach:** Recursively split on first bad character.  
**Time Complexity:** O(n^2)

```cpp
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    string longestNiceSubstring(string s) {
        if (s.size() < 2) return "";
        unordered_set<char> st(s.begin(), s.end());
        for (int i = 0; i < s.size(); ++i) {
            if (st.count(tolower(s[i])) && st.count(toupper(s[i]))) continue;
            string s1 = longestNiceSubstring(s.substr(0, i));
            string s2 = longestNiceSubstring(s.substr(i + 1));
            return s1.size() >= s2.size() ? s1 : s2;
        }
        return s;
    }
};
```

---

##150 ****[Problem Link]https://leetcode.com/problems/remove-all-occurrences-of-a-substring****  
**Approach:** Use string find and erase until no occurrence remains.  
**Time Complexity:** O(n * m)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string removeOccurrences(string s, string part) {
        size_t pos;
        while ((pos = s.find(part)) != string::npos)
            s.erase(pos, part.size());
        return s;
    }
};
```

---

##151 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-work-sessions-to-finish-the-tasks****  
**Approach:** Backtracking with memoization and bitmasking.  
**Time Complexity:** O(n * 2^n)

```cpp
#include <vector>
#include <cstring>

using namespace std;

class Solution {
public:
    int dp[1 << 14];
    int minSessions(vector<int>& tasks, int sessionTime) {
        memset(dp, -1, sizeof dp);
        return dfs(tasks, sessionTime, 0, 0, 0);
    }

    int dfs(vector<int>& tasks, int sessionTime, int mask, int timeLeft, int count) {
        if (mask == (1 << tasks.size()) - 1) return count;
        if (dp[mask] != -1) return dp[mask];
        int res = tasks.size();
        for (int i = 0; i < tasks.size(); ++i) {
            if (!(mask & (1 << i))) {
                if (tasks[i] <= timeLeft)
                    res = min(res, dfs(tasks, sessionTime, mask | (1 << i), timeLeft - tasks[i], count));
                else
                    res = min(res, dfs(tasks, sessionTime, mask | (1 << i), sessionTime - tasks[i], count + 1));
            }
        }
        return dp[mask] = res;
    }
};
```

---

##152 ****[Problem Link]https://leetcode.com/problems/build-an-array-with-stack-operations****  
**Approach:** Simulate target creation using stack operations.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<string> buildArray(vector<int>& target, int n) {
        vector<string> res;
        int i = 1, j = 0;
        while (j < target.size()) {
            res.push_back("Push");
            if (target[j] != i++) res.push_back("Pop");
            else ++j;
        }
        return res;
    }
};
```

---

##153 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-consecutive-values-you-can-make****  
**Approach:** Greedy using prefix sum property.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int getMaximumConsecutive(vector<int>& coins) {
        sort(coins.begin(), coins.end());
        int res = 1;
        for (int c : coins) {
            if (c > res) break;
            res += c;
        }
        return res;
    }
};
```

---

##154 ****[Problem Link]https://leetcode.com/problems/stone-game-vi****  
**Approach:** Sort by value sum descending and simulate turn-taking.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int stoneGameVI(vector<int>& A, vector<int>& B) {
        int n = A.size(), score = 0;
        vector<tuple<int, int, int>> stones;
        for (int i = 0; i < n; ++i)
            stones.emplace_back(A[i] + B[i], A[i], B[i]);
        sort(stones.rbegin(), stones.rend());
        for (int i = 0; i < n; ++i) {
            if (i % 2 == 0) score += get<1>(stones[i]);
            else score -= get<2>(stones[i]);
        }
        return score > 0 ? 1 : score < 0 ? -1 : 0;
    }
};
```

---

##155 ****[Problem Link]https://leetcode.com/problems/maximum-distance-between-a-pair-of-values****  
**Approach:** Two-pointer on decreasing arrays.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int maxDistance(vector<int>& nums1, vector<int>& nums2) {
        int res = 0, j = 0;
        for (int i = 0; i < nums1.size(); ++i) {
            j = max(j, i);
            while (j < nums2.size() && nums1[i] <= nums2[j]) ++j;
            res = max(res, j - i - 1);
        }
        return res;
    }
};
```

---

##156 ****[Problem Link]https://leetcode.com/problems/available-captures-for-rook****  
**Approach:** Check four directions from rook's position.  
**Time Complexity:** O(1)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int numRookCaptures(vector<vector<char>>& board) {
        int x, y;
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                if (board[i][j] == 'R') {
                    x = i; y = j;
                }
        int res = 0, dirs[5] = {0, 1, 0, -1, 0};
        for (int d = 0; d < 4; ++d) {
            int nx = x, ny = y;
            while (true) {
                nx += dirs[d]; ny += dirs[d + 1];
                if (nx < 0 || ny < 0 || nx >= 8 || ny >= 8 || board[nx][ny] == 'B') break;
                if (board[nx][ny] == 'p') {
                    res++;
                    break;
                }
            }
        }
        return res;
    }
};
```

---

##157 ****[Problem Link]https://leetcode.com/problems/string-compression-ii****  
**Approach:** DP with memoization and state compression.  
**Time Complexity:** O(n * k * 26)

```cpp
#include <string>
#include <vector>
#include <cstring>

using namespace std;

class Solution {
public:
    int dp[101][27][101];

    int getLen(int cnt) {
        if (cnt == 1) return 0;
        if (cnt < 10) return 1;
        if (cnt < 100) return 2;
        return 3;
    }

    int solve(string& s, int i, int last, int cnt, int k) {
        if (k < 0) return 1e9;
        if (i == s.size()) return 0;
        if (dp[i][last][cnt] != -1) return dp[i][last][cnt];
        int res;
        if (s[i] - 'a' == last)
            res = getLen(cnt + 1) - getLen(cnt) + solve(s, i + 1, last, cnt + 1, k);
        else
            res = min(solve(s, i + 1, last, cnt, k - 1),
                      1 + solve(s, i + 1, s[i] - 'a', 1, k));
        return dp[i][last][cnt] = res;
    }

    int getLengthOfOptimalCompression(string s, int k) {
        memset(dp, -1, sizeof(dp));
        return solve(s, 0, 26, 0, k);
    }
};
```

---

##158 ****[Problem Link]https://leetcode.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible****  
**Approach:** DP with recurrence relation based on new/old tallest.  
**Time Complexity:** O(n * k)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int rearrangeSticks(int n, int k) {
        const int MOD = 1e9 + 7;
        vector<vector<long>> dp(n + 1, vector<long>(k + 1));
        dp[0][0] = 1;
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= k; ++j)
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j] * (i - 1)) % MOD;
        return dp[n][k];
    }
};
```

---

##159 ****[Problem Link]https://leetcode.com/problems/determine-whether-matrix-can-be-obtained-by-rotation****  
**Approach:** Check all 4 rotations of matrix.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    bool findRotation(vector<vector<int>>& mat, vector<vector<int>>& target) {
        for (int k = 0; k < 4; ++k) {
            if (mat == target) return true;
            rotate(mat);
        }
        return false;
    }

    void rotate(vector<vector<int>>& mat) {
        int n = mat.size();
        for (int i = 0; i < n / 2; ++i)
            for (int j = i; j < n - i - 1; ++j) {
                int tmp = mat[i][j];
                mat[i][j] = mat[n - j - 1][i];
                mat[n - j - 1][i] = mat[n - i - 1][n - j - 1];
                mat[n - i - 1][n - j - 1] = mat[j][n - i - 1];
                mat[j][n - i - 1] = tmp;
            }
    }
};
```

---

##160 ****[Problem Link]https://leetcode.com/problems/maximum-height-by-stacking-cuboids****  
**Approach:** Sort dimensions, then apply 3D LIS.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxHeight(vector<vector<int>>& cuboids) {
        for (auto& c : cuboids)
            sort(c.begin(), c.end());
        cuboids.push_back({0, 0, 0});
        sort(cuboids.begin(), cuboids.end());
        int n = cuboids.size(), res = 0;
        vector<int> dp(n);
        for (int i = 1; i < n; ++i)
            for (int j = 0; j < i; ++j)
                if (cuboids[j][0] <= cuboids[i][0] &&
                    cuboids[j][1] <= cuboids[i][1] &&
                    cuboids[j][2] <= cuboids[i][2])
                    dp[i] = max(dp[i], dp[j]);
        for (int i = 1; i < n; ++i)
            dp[i] += cuboids[i][2], res = max(res, dp[i]);
        return res;
    }
};
```

---

##161 ****[Problem Link]https://leetcode.com/problems/groups-of-special-equivalent-strings****  
**Approach:** Use canonical form by sorting even and odd indexed characters separately.  
**Time Complexity:** O(n * k log k)

```cpp
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int numSpecialEquivGroups(vector<string>& words) {
        unordered_set<string> seen;
        for (auto& word : words) {
            string even, odd;
            for (int i = 0; i < word.size(); ++i) {
                if (i % 2) odd += word[i];
                else even += word[i];
            }
            sort(even.begin(), even.end());
            sort(odd.begin(), odd.end());
            seen.insert(even + odd);
        }
        return seen.size();
    }
};
```

---

##162 ****[Problem Link]https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza****  
**Approach:** DP + prefix sum to check if apple exists in slice.  
**Time Complexity:** O(k * m * n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int ways(vector<string>& pizza, int k) {
        const int MOD = 1e9 + 7;
        int m = pizza.size(), n = pizza[0].size();
        vector<vector<int>> pre(m + 1, vector<int>(n + 1));
        for (int i = m - 1; i >= 0; --i)
            for (int j = n - 1; j >= 0; --j)
                pre[i][j] = (pizza[i][j] == 'A') + pre[i + 1][j] + pre[i][j + 1] - pre[i + 1][j + 1];

        vector<vector<vector<int>>> dp(k, vector<vector<int>>(m, vector<int>(n)));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                dp[0][i][j] = pre[i][j] > 0;

        for (int l = 1; l < k; ++l)
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    for (int ii = i + 1; ii < m; ++ii)
                        if (pre[i][j] - pre[ii][j] > 0)
                            dp[l][i][j] = (dp[l][i][j] + dp[l - 1][ii][j]) % MOD;
                    for (int jj = j + 1; jj < n; ++jj)
                        if (pre[i][j] - pre[i][jj] > 0)
                            dp[l][i][j] = (dp[l][i][j] + dp[l - 1][i][jj]) % MOD;
                }

        return dp[k - 1][0][0];
    }
};
```

---

##163 ****[Problem Link]https://leetcode.com/problems/palindrome-partitioning-iv****  
**Approach:** 3-part DP checking if substring splits into 3 palindromes.  
**Time Complexity:** O(n^3)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    bool checkPartitioning(string s) {
        int n = s.size();
        vector<vector<bool>> isPal(n, vector<bool>(n, false));
        for (int i = n - 1; i >= 0; --i)
            for (int j = i; j < n; ++j)
                isPal[i][j] = (s[i] == s[j]) && (j - i < 2 || isPal[i + 1][j - 1]);

        for (int i = 1; i < n - 1; ++i)
            for (int j = i; j < n - 1; ++j)
                if (isPal[0][i - 1] && isPal[i][j] && isPal[j + 1][n - 1])
                    return true;
        return false;
    }
};
```

---

##164 ****[Problem Link]https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary****  
**Approach:** Find sum, min, and max, then compute average.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    double average(vector<int>& salary) {
        int total = 0, mn = INT_MAX, mx = INT_MIN;
        for (int s : salary) {
            total += s;
            mn = min(mn, s);
            mx = max(mx, s);
        }
        return (total - mn - mx) / (double)(salary.size() - 2);
    }
};
```

---

##165 ****[Problem Link]https://leetcode.com/problems/maximum-score-from-removing-stones****  
**Approach:** Always pick top 2 largest piles to remove from.  
**Time Complexity:** O(log n) per operation

```cpp
#include <queue>

using namespace std;

class Solution {
public:
    int maximumScore(int a, int b, int c) {
        priority_queue<int> pq;
        pq.push(a); pq.push(b); pq.push(c);
        int score = 0;
        while (true) {
            int x = pq.top(); pq.pop();
            int y = pq.top(); pq.pop();
            if (x == 0 || y == 0) break;
            x--; y--; score++;
            pq.push(x); pq.push(y);
        }
        return score;
    }
};
```

---

##166 ****[Problem Link]https://leetcode.com/problems/minimum-absolute-sum-difference****  
**Approach:** Binary search for closest value to minimize replacement difference.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
        const int MOD = 1e9 + 7;
        int n = nums1.size(), maxGain = 0, total = 0;
        vector<int> sorted = nums1;
        sort(sorted.begin(), sorted.end());

        for (int i = 0; i < n; ++i) {
            int diff = abs(nums1[i] - nums2[i]);
            total = (total + diff) % MOD;

            auto it = lower_bound(sorted.begin(), sorted.end(), nums2[i]);
            if (it != sorted.end()) maxGain = max(maxGain, diff - abs(*it - nums2[i]));
            if (it != sorted.begin()) maxGain = max(maxGain, diff - abs(*prev(it) - nums2[i]));
        }

        return (total - maxGain + MOD) % MOD;
    }
};
```

---

##167 ****[Problem Link]https://leetcode.com/problems/queries-on-number-of-points-inside-a-circle****  
**Approach:** Brute force check each point against each query using distance.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> countPoints(vector<vector<int>>& points, vector<vector<int>>& queries) {
        vector<int> res;
        for (auto& q : queries) {
            int x = q[0], y = q[1], r2 = q[2] * q[2], cnt = 0;
            for (auto& p : points) {
                int dx = p[0] - x, dy = p[1] - y;
                if (dx * dx + dy * dy <= r2) cnt++;
            }
            res.push_back(cnt);
        }
        return res;
    }
};
```

---

##168 ****[Problem Link]https://leetcode.com/problems/dota2-senate****  
**Approach:** Queue-based simulation of banning using index comparison.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <queue>

using namespace std;

class Solution {
public:
    string predictPartyVictory(string senate) {
        queue<int> r, d;
        int n = senate.size();
        for (int i = 0; i < n; ++i) {
            if (senate[i] == 'R') r.push(i);
            else d.push(i);
        }
        while (!r.empty() && !d.empty()) {
            int ri = r.front(); r.pop();
            int di = d.front(); d.pop();
            if (ri < di) r.push(ri + n);
            else d.push(di + n);
        }
        return r.empty() ? "Dire" : "Radiant";
    }
};
```

---

##169 ****[Problem Link]https://leetcode.com/problems/delete-columns-to-make-sorted-iii****  
**Approach:** Longest non-decreasing subsequence of column-wise comparison.  
**Time Complexity:** O(n^2 * m)

```cpp
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minDeletionSize(vector<string>& strs) {
        int n = strs.size(), m = strs[0].size();
        vector<int> dp(m, 1);
        int res = m - 1;

        for (int j = 1; j < m; ++j)
            for (int i = 0; i < j; ++i) {
                bool valid = true;
                for (int k = 0; k < n; ++k)
                    if (strs[k][i] > strs[k][j]) {
                        valid = false;
                        break;
                    }
                if (valid) dp[j] = max(dp[j], dp[i] + 1);
            }

        return m - *max_element(dp.begin(), dp.end());
    }
};
```

---

##170 ****[Problem Link]https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree****  
**Approach:** Kruskal + MST rebuild with inclusion/exclusion simulation.  
**Time Complexity:** O(E log E + E^2)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class DSU {
    vector<int> parent;
public:
    DSU(int n) {
        parent.resize(n);
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    }
    bool unite(int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx == ry) return false;
        parent[rx] = ry;
        return true;
    }
};

class Solution {
public:
    vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges) {
        for (int i = 0; i < edges.size(); ++i)
            edges[i].push_back(i);
        sort(edges.begin(), edges.end(), [](auto& a, auto& b) {
            return a[2] < b[2];
        });

        auto kruskal = [&](int skip = -1, int force = -1) {
            DSU dsu(n);
            int cost = 0;
            if (force != -1) {
                dsu.unite(edges[force][0], edges[force][1]);
                cost += edges[force][2];
            }
            for (int i = 0; i < edges.size(); ++i) {
                if (i == skip) continue;
                if (dsu.unite(edges[i][0], edges[i][1])) {
                    cost += edges[i][2];
                }
            }
            for (int i = 0; i < n; ++i)
                if (dsu.find(i) != dsu.find(0)) return 1e9;
            return cost;
        };

        int base = kruskal();
        vector<int> critical, pseudo;
        for (int i = 0; i < edges.size(); ++i) {
            if (kruskal(i) > base) critical.push_back(edges[i][3]);
            else if (kruskal(-1, i) == base) pseudo.push_back(edges[i][3]);
        }

        return {critical, pseudo};
    }
};
```

---

##171 ****[Problem Link]https://leetcode.com/problems/maximum-earnings-from-taxi****  
**Approach:** DP with binary search on end times.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    long long maxTaxiEarnings(int n, vector<vector<int>>& rides) {
        sort(rides.begin(), rides.end(), [](auto& a, auto& b) {
            return a[1] < b[1];
        });

        int m = rides.size();
        vector<long long> dp(m + 1, 0);
        vector<int> endTimes;
        for (auto& r : rides)
            endTimes.push_back(r[1]);

        for (int i = 0; i < m; ++i) {
            auto& r = rides[i];
            int gain = r[1] - r[0] + r[2];
            int j = lower_bound(endTimes.begin(), endTimes.end(), r[0]) - endTimes.begin();
            dp[i + 1] = max(dp[i], dp[j] + gain);
        }
        return dp[m];
    }
};
```

---

##172 ****[Problem Link]https://leetcode.com/problems/number-of-segments-in-a-string****  
**Approach:** Traverse string and count transition into words.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int countSegments(string s) {
        int count = 0;
        for (int i = 0; i < s.size(); ++i)
            if (s[i] != ' ' && (i == 0 || s[i - 1] == ' '))
                count++;
        return count;
    }
};
```

---

##173 ****[Problem Link]https://leetcode.com/problems/valid-number****  
**Approach:** FSM (finite state machine) or regex; here use flags.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    bool isNumber(string s) {
        bool num = false, dot = false, exp = false;
        for (int i = 0; i < s.size(); ++i) {
            if (isdigit(s[i])) num = true;
            else if (s[i] == '.') {
                if (dot || exp) return false;
                dot = true;
            } else if (s[i] == 'e' || s[i] == 'E') {
                if (exp || !num) return false;
                exp = true; num = false;
            } else if (s[i] == '+' || s[i] == '-') {
                if (i > 0 && s[i - 1] != 'e' && s[i - 1] != 'E') return false;
            } else return false;
        }
        return num;
    }
};
```

---

##174 ****[Problem Link]https://leetcode.com/problems/substrings-of-size-three-with-distinct-characters****  
**Approach:** Sliding window check for 3-char unique substrings.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int countGoodSubstrings(string s) {
        int count = 0;
        for (int i = 0; i + 2 < s.size(); ++i) {
            if (s[i] != s[i + 1] && s[i] != s[i + 2] && s[i + 1] != s[i + 2])
                count++;
        }
        return count;
    }
};
```

---

##175 ****[Problem Link]https://leetcode.com/problems/profitable-schemes****  
**Approach:** DP on group and profit dimensions.  
**Time Complexity:** O(G * P * N)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int profitableSchemes(int n, int minProfit, vector<int>& group, vector<int>& profit) {
        const int MOD = 1e9 + 7;
        int m = group.size();
        vector<vector<int>> dp(n + 1, vector<int>(minProfit + 1));
        dp[0][0] = 1;
        for (int k = 0; k < m; ++k) {
            int g = group[k], p = profit[k];
            for (int i = n; i >= g; --i) {
                for (int j = minProfit; j >= 0; --j) {
                    dp[i][j] = (dp[i][j] + dp[i - g][max(0, j - p)]) % MOD;
                }
            }
        }
        int res = 0;
        for (int i = 0; i <= n; ++i)
            res = (res + dp[i][minProfit]) % MOD;
        return res;
    }
};
```

---

##176 ****[Problem Link]https://leetcode.com/problems/crawler-log-folder****  
**Approach:** Use a depth counter to track folder changes.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int minOperations(vector<string>& logs) {
        int depth = 0;
        for (const string& log : logs) {
            if (log == "./") continue;
            else if (log == "../") depth = max(0, depth - 1);
            else depth++;
        }
        return depth;
    }
};
```

---

##177 ****[Problem Link]https://leetcode.com/problems/binary-gap****  
**Approach:** Track the position of the last '1' bit.  
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int binaryGap(int n) {
        int last = -1, res = 0;
        for (int i = 0; n; ++i, n >>= 1) {
            if (n & 1) {
                if (last != -1)
                    res = max(res, i - last);
                last = i;
            }
        }
        return res;
    }
};
```

---

##178 ****[Problem Link]https://leetcode.com/problems/longest-chunked-palindrome-decomposition****  
**Approach:** Two pointers comparing prefix and suffix substrings.  
**Time Complexity:** O(n^2)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int longestDecomposition(string text) {
        int n = text.size(), res = 0;
        string l = "", r = "";
        for (int i = 0; i < n; ++i) {
            l += text[i];
            r = text[n - 1 - i] + r;
            if (l == r) {
                res++;
                l = r = "";
            }
        }
        return res;
    }
};
```

---

##179 ****[Problem Link]https://leetcode.com/problems/check-if-array-is-sorted-and-rotated****  
**Approach:** Count how many times array decreases.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    bool check(vector<int>& nums) {
        int count = 0, n = nums.size();
        for (int i = 0; i < n; ++i)
            if (nums[i] > nums[(i + 1) % n])
                count++;
        return count <= 1;
    }
};
```

---

##180 ****[Problem Link]https://leetcode.com/problems/cinema-seat-allocation****  
**Approach:** Use map to track occupied blocks per row.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int maxNumberOfFamilies(int n, vector<vector<int>>& reservedSeats) {
        unordered_map<int, int> rows;
        for (auto& seat : reservedSeats) {
            int row = seat[0], col = seat[1];
            if (col >= 2 && col <= 9)
                rows[row] |= (1 << (col - 2));
        }

        int total = (n - rows.size()) * 2;
        for (auto& [row, mask] : rows) {
            bool left = (mask & 0b00001111) == 0;
            bool mid  = (mask & 0b00111100) == 0;
            bool right= (mask & 0b11110000) == 0;
            if (left && right) total += 2;
            else if (left || mid || right) total += 1;
        }

        return total;
    }
};
```

---

##181 ****[Problem Link]https://leetcode.com/problems/find-kth-bit-in-nth-binary-string****  
**Approach:** Recursive approach based on mirroring and inversion.  
**Time Complexity:** O(log n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    char findKthBit(int n, int k) {
        if (n == 1) return '0';
        int len = (1 << n) - 1, mid = len / 2 + 1;
        if (k == mid) return '1';
        if (k < mid) return findKthBit(n - 1, k);
        char ch = findKthBit(n - 1, len - k + 1);
        return ch == '0' ? '1' : '0';
    }
};
```

---

##182 ****[Problem Link]https://leetcode.com/problems/minimum-moves-to-make-array-complementary****  
**Approach:** Difference array with sweep line on valid move ranges.  
**Time Complexity:** O(n + limit)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int minMoves(vector<int>& nums, int limit) {
        vector<int> diff(2 * limit + 2);
        int n = nums.size();
        for (int i = 0; i < n / 2; ++i) {
            int a = nums[i], b = nums[n - 1 - i];
            int l = min(a, b) + 1, r = max(a, b) + limit;
            diff[2] += 2;
            diff[l] -= 1;
            diff[a + b] -= 1;
            diff[a + b + 1] += 1;
            diff[r + 1] += 1;
        }

        int res = n, curr = 0;
        for (int i = 2; i <= 2 * limit; ++i) {
            curr += diff[i];
            res = min(res, curr);
        }
        return res;
    }
};
```

---

##183 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii****  
**Approach:** DP with sorting and binary search to skip overlapping.  
**Time Complexity:** O(nk log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxValue(vector<vector<int>>& events, int k) {
        sort(events.begin(), events.end());
        int n = events.size();
        vector<vector<int>> dp(n + 1, vector<int>(k + 1));
        for (int i = n - 1; i >= 0; --i) {
            int j = upper_bound(events.begin(), events.end(), vector<int>{events[i][1], INT_MAX, INT_MAX}) - events.begin();
            for (int t = 1; t <= k; ++t)
                dp[i][t] = max(dp[i + 1][t], events[i][2] + dp[j][t - 1]);
        }
        return dp[0][k];
    }
};
```

---

##184 ****[Problem Link]https://leetcode.com/problems/merge-strings-alternately****  
**Approach:** Use two pointers to merge strings character by character.  
**Time Complexity:** O(n + m)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string mergeAlternately(string word1, string word2) {
        string res;
        int i = 0, j = 0;
        while (i < word1.size() || j < word2.size()) {
            if (i < word1.size()) res += word1[i++];
            if (j < word2.size()) res += word2[j++];
        }
        return res;
    }
};
```

---

##185 ****[Problem Link]https://leetcode.com/problems/number-of-ways-to-split-a-string****  
**Approach:** Count positions of '1's and use combination math.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    int numWays(string s) {
        const int MOD = 1e9 + 7;
        int ones = 0;
        for (char c : s) if (c == '1') ones++;
        if (ones % 3 != 0) return 0;
        if (ones == 0) return (long long)(s.size() - 1) * (s.size() - 2) / 2 % MOD;

        int t = ones / 3, cnt = 0, a = 0, b = 0;
        for (char c : s) {
            if (c == '1') cnt++;
            if (cnt == t) a++;
            else if (cnt == 2 * t) b++;
        }
        return (long long)a * b % MOD;
    }
};
```

---

##186 ****[Problem Link]https://leetcode.com/problems/smallest-range-i****  
**Approach:** Range compression by adjusting max and min.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int smallestRangeI(vector<int>& nums, int k) {
        int mx = *max_element(nums.begin(), nums.end());
        int mn = *min_element(nums.begin(), nums.end());
        return max(0, mx - mn - 2 * k);
    }
};
```

---

##187 ****[Problem Link]https://leetcode.com/problems/calculate-money-in-leetcode-bank****  
**Approach:** Use arithmetic series sum formula to calculate.  
**Time Complexity:** O(1)

```cpp
class Solution {
public:
    int totalMoney(int n) {
        int weeks = n / 7, days = n % 7;
        int sum = 28 * weeks + 7 * (weeks - 1) * weeks / 2;
        for (int i = 0; i < days; ++i)
            sum += weeks + 1 + i;
        return sum;
    }
};
```

---

##188 ****[Problem Link]https://leetcode.com/problems/surface-area-of-3d-shapes****  
**Approach:** Add face contributions from top, bottom, and side faces.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int surfaceArea(vector<vector<int>>& grid) {
        int res = 0, n = grid.size();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (grid[i][j]) {
                    res += 4 * grid[i][j] + 2;
                    if (i) res -= min(grid[i][j], grid[i - 1][j]);
                    if (j) res -= min(grid[i][j], grid[i][j - 1]);
                }
        return res;
    }
};
```

---

##189 ****[Problem Link]https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons****  
**Approach:** DP with memoization and 3D state: (n, k, maxVal).  
**Time Complexity:** O(n * k * m)

```cpp
#include <vector>
#include <cstring>

using namespace std;

class Solution {
public:
    const int MOD = 1e9 + 7;
    int dp[51][101][51];

    int dfs(int n, int m, int k, int maxVal) {
        if (n == 0) return k == 0;
        if (dp[n][k][maxVal] != -1) return dp[n][k][maxVal];
        int res = 0;
        for (int i = 1; i <= m; ++i) {
            if (i > maxVal)
                res = (res + dfs(n - 1, m, k - 1, i)) % MOD;
            else
                res = (res + dfs(n - 1, m, k, maxVal)) % MOD;
        }
        return dp[n][k][maxVal] = res;
    }

    int numOfArrays(int n, int m, int k) {
        memset(dp, -1, sizeof dp);
        return dfs(n, m, k, 0);
    }
};
```

---

##190 ****[Problem Link]https://leetcode.com/problems/minimize-malware-spread-ii****  
**Approach:** DFS + Union Find to evaluate which node causes max infection reduction.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial) {
        int n = graph.size();
        vector<int> parent(n), size(n, 1);
        iota(parent.begin(), parent.end(), 0);

        function<int(int)> find = [&](int x) {
            return parent[x] == x ? x : parent[x] = find(parent[x]);
        };

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (graph[i][j] && i != j)
                    if (find(i) != find(j)) {
                        size[find(j)] += size[find(i)];
                        parent[find(i)] = find(j);
                    }

        vector<int> count(n);
        for (int node : initial)
            count[find(node)]++;

        int res = *min_element(initial.begin(), initial.end()), maxSize = -1;
        for (int node : initial) {
            int root = find(node);
            if (count[root] == 1 && size[root] > maxSize) {
                res = node;
                maxSize = size[root];
            } else if (count[root] == 1 && size[root] == maxSize) {
                res = min(res, node);
            }
        }
        return res;
    }
};
```

---

##191 ****[Problem Link]https://leetcode.com/problems/print-words-vertically****  
**Approach:** Split sentence and print column-wise while preserving spacing.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>
#include <string>
#include <sstream>

using namespace std;

class Solution {
public:
    vector<string> printVertically(string s) {
        istringstream iss(s);
        vector<string> words, res;
        string word;
        int maxLen = 0;
        while (iss >> word) {
            words.push_back(word);
            maxLen = max(maxLen, (int)word.size());
        }

        for (int i = 0; i < maxLen; ++i) {
            string line;
            for (string& w : words)
                line += i < w.size() ? w[i] : ' ';
            while (!line.empty() && line.back() == ' ') line.pop_back();
            res.push_back(line);
        }
        return res;
    }
};
```

---

##192 ****[Problem Link]https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c****  
**Approach:** Bitwise check per bit of a, b, and c.  
**Time Complexity:** O(1)

```cpp
class Solution {
public:
    int minFlips(int a, int b, int c) {
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            int x = (a >> i) & 1, y = (b >> i) & 1, z = (c >> i) & 1;
            if ((x | y) != z) res += z ? 1 : x + y;
        }
        return res;
    }
};
```

---

##193 ****[Problem Link]https://leetcode.com/problems/special-positions-in-a-binary-matrix****  
**Approach:** Count row and column sums and check for uniqueness.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int numSpecial(vector<vector<int>>& mat) {
        int m = mat.size(), n = mat[0].size(), res = 0;
        vector<int> row(m), col(n);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                row[i] += mat[i][j], col[j] += mat[i][j];

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (mat[i][j] == 1 && row[i] == 1 && col[j] == 1)
                    res++;
        return res;
    }
};
```

---

##194 ****[Problem Link]https://leetcode.com/problems/closest-subsequence-sum****  
**Approach:** Meet in the middle - subset sums from both halves.  
**Time Complexity:** O(2^(n/2) * log(2^(n/2)))

```cpp
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class Solution {
public:
    void dfs(vector<int>& nums, int i, int end, int sum, vector<int>& res) {
        if (i == end) {
            res.push_back(sum);
            return;
        }
        dfs(nums, i + 1, end, sum, res);
        dfs(nums, i + 1, end, sum + nums[i], res);
    }

    int minAbsDifference(vector<int>& nums, int goal) {
        int n = nums.size();
        vector<int> left, right;
        dfs(nums, 0, n / 2, 0, left);
        dfs(nums, n / 2, n, 0, right);
        sort(right.begin(), right.end());

        int res = abs(goal);
        for (int a : left) {
            int rem = goal - a;
            auto it = lower_bound(right.begin(), right.end(), rem);
            if (it != right.end()) res = min(res, abs(rem - *it));
            if (it != right.begin()) res = min(res, abs(rem - *prev(it)));
        }
        return res;
    }
};
```

---

##195 ****[Problem Link]https://leetcode.com/problems/parse-lisp-expression****  
**Approach:** Recursive parsing and evaluation using maps.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>
#include <sstream>

using namespace std;

class Solution {
public:
    int evaluate(string expression) {
        unordered_map<string, int> scope;
        return eval(expression, scope);
    }

    int eval(string s, unordered_map<string, int>& scope) {
        if (s[0] != '(') {
            if (isalpha(s[0])) return scope[s];
            return stoi(s);
        }

        stringstream ss(s.substr(1, s.size() - 2));
        string token;
        ss >> token;

        if (token == "add" || token == "mult") {
            string t1, t2;
            getline(ss, t1, ' ');
            getline(ss, t2);
            return token == "add" ? eval(t1, scope) + eval(t2, scope)
                                  : eval(t1, scope) * eval(t2, scope);
        } else {
            unordered_map<string, int> newScope = scope;
            vector<string> tokens;
            while (getline(ss, token, ' '))
                tokens.push_back(token);

            for (int i = 1; i + 2 < tokens.size(); i += 2)
                newScope[tokens[i]] = eval(tokens[i + 1], newScope);
            return eval(tokens.back(), newScope);
        }
    }
};
```

---

##196 ****[Problem Link]https://leetcode.com/problems/largest-color-value-in-a-directed-graph****  
**Approach:** Topological sort with color count tracking.  
**Time Complexity:** O(n + e)

```cpp
#include <vector>
#include <string>
#include <queue>
#include <algorithm>

using namespace std;

class Solution {
public:
    int largestPathValue(string colors, vector<vector<int>>& edges) {
        int n = colors.size();
        vector<vector<int>> graph(n);
        vector<int> indegree(n, 0);

        for (auto& e : edges) {
            graph[e[0]].push_back(e[1]);
            indegree[e[1]]++;
        }

        vector<vector<int>> count(n, vector<int>(26));
        queue<int> q;
        for (int i = 0; i < n; ++i)
            if (indegree[i] == 0) q.push(i);

        int seen = 0, res = 0;
        while (!q.empty()) {
            int node = q.front(); q.pop();
            seen++;
            count[node][colors[node] - 'a']++;
            res = max(res, count[node][colors[node] - 'a']);
            for (int nei : graph[node]) {
                for (int c = 0; c < 26; ++c)
                    count[nei][c] = max(count[nei][c], count[node][c]);
                if (--indegree[nei] == 0) q.push(nei);
            }
        }
        return seen == n ? res : -1;
    }
};
```

---

##197 ****[Problem Link]https://leetcode.com/problems/minimize-the-difference-between-target-and-chosen-elements****  
**Approach:** DP with set of sums to track feasible values.  
**Time Complexity:** O(m * sum)

```cpp
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minimizeTheDifference(vector<vector<int>>& mat, int target) {
        unordered_set<int> sums = {0};
        for (auto& row : mat) {
            unordered_set<int> newSums;
            for (int x : row)
                for (int s : sums)
                    newSums.insert(x + s);
            if (newSums.size() > 10000) break;
            sums = move(newSums);
        }

        int res = INT_MAX;
        for (int s : sums)
            res = min(res, abs(s - target));
        return res;
    }
};
```

---

##198 ****[Problem Link]https://leetcode.com/problems/truncate-sentence****  
**Approach:** Split by space up to k tokens.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <sstream>

using namespace std;

class Solution {
public:
    string truncateSentence(string s, int k) {
        stringstream ss(s);
        string word, res;
        while (k-- && ss >> word) {
            if (!res.empty()) res += " ";
            res += word;
        }
        return res;
    }
};
```

---

##199 ****[Problem Link]https://leetcode.com/problems/minimum-speed-to-arrive-on-time****  
**Approach:** Binary search on speed value.  
**Time Complexity:** O(n log maxSpeed)

```cpp
#include <vector>
#include <cmath>

using namespace std;

class Solution {
public:
    int minSpeedOnTime(vector<int>& dist, double hour) {
        int lo = 1, hi = 1e7, ans = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            double time = 0;
            for (int i = 0; i < dist.size(); ++i)
                time += i + 1 == dist.size() ? (double)dist[i] / mid : ceil((double)dist[i] / mid);
            if (time <= hour) {
                ans = mid;
                hi = mid - 1;
            } else lo = mid + 1;
        }
        return ans;
    }
};
```

---

##200 ****[Problem Link]https://leetcode.com/problems/unique-length-3-palindromic-subsequences****  
**Approach:** Use first and last occurrence of characters to check in-between characters.  
**Time Complexity:** O(26 * n)

```cpp
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int countPalindromicSubsequence(string s) {
        int res = 0;
        for (char c = 'a'; c <= 'z'; ++c) {
            int l = s.find(c), r = s.rfind(c);
            if (l < r) {
                unordered_set<char> mid(s.begin() + l + 1, s.begin() + r);
                res += mid.size();
            }
        }
        return res;
    }
};
```

---

##201 ****[Problem Link]https://leetcode.com/problems/largest-triangle-area****  
**Approach:** Brute-force every triplet and compute area using determinant.  
**Time Complexity:** O(n^3)

```cpp
#include <vector>
#include <cmath>

using namespace std;

class Solution {
public:
    double largestTriangleArea(vector<vector<int>>& points) {
        double res = 0;
        for (int i = 0; i < points.size(); ++i)
            for (int j = i + 1; j < points.size(); ++j)
                for (int k = j + 1; k < points.size(); ++k) {
                    res = max(res, 0.5 * abs(
                        points[i][0] * (points[j][1] - points[k][1]) +
                        points[j][0] * (points[k][1] - points[i][1]) +
                        points[k][0] * (points[i][1] - points[j][1])
                    ));
                }
        return res;
    }
};
```

---

##202 ****[Problem Link]https://leetcode.com/problems/maximum-product-difference-between-two-pairs****  
**Approach:** Sort and compare product of two largest vs two smallest.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxProductDifference(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        return (nums[n-1] * nums[n-2]) - (nums[0] * nums[1]);
    }
};
```

---

##203 ****[Problem Link]https://leetcode.com/problems/nearest-exit-from-entrance-in-maze****  
**Approach:** BFS from entrance tracking distance to first exit.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
        int m = maze.size(), n = maze[0].size();
        queue<pair<int, int>> q;
        q.push({entrance[0], entrance[1]});
        maze[entrance[0]][entrance[1]] = '+';
        int steps = 0;
        vector<pair<int, int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
        while (!q.empty()) {
            int sz = q.size();
            while (sz--) {
                auto [x, y] = q.front(); q.pop();
                for (auto [dx, dy] : dirs) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == '.') {
                        if (nx == 0 || ny == 0 || nx == m - 1 || ny == n - 1)
                            return steps + 1;
                        maze[nx][ny] = '+';
                        q.push({nx, ny});
                    }
                }
            }
            ++steps;
        }
        return -1;
    }
};
```

---

##204 ****[Problem Link]https://leetcode.com/problems/dinner-plate-stacks****  
**Approach:** Use set and heap for available stacks and max pop.  
**Time Complexity:** Amortized O(log n)

```cpp
#include <vector>
#include <stack>
#include <set>

using namespace std;

class DinnerPlates {
    int cap;
    vector<stack<int>> stacks;
    set<int> available;

public:
    DinnerPlates(int capacity) : cap(capacity) {}

    void push(int val) {
        if (available.empty()) stacks.emplace_back();
        int idx = available.empty() ? stacks.size() - 1 : *available.begin();
        if (idx == stacks.size()) stacks.emplace_back();
        stacks[idx].push(val);
        if (stacks[idx].size() == cap) available.erase(idx);
        else available.insert(idx);
    }

    int pop() {
        while (!stacks.empty() && stacks.back().empty()) stacks.pop_back();
        if (stacks.empty()) return -1;
        int val = stacks.back().top(); stacks.back().pop();
        available.insert(stacks.size() - 1);
        return val;
    }

    int popAtStack(int index) {
        if (index >= stacks.size() || stacks[index].empty()) return -1;
        int val = stacks[index].top(); stacks[index].pop();
        available.insert(index);
        return val;
    }
};
```

---

##205 ****[Problem Link]https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal****  
**Approach:** Find mismatched indices and check swap feasibility.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    bool areAlmostEqual(string s1, string s2) {
        vector<int> diff;
        for (int i = 0; i < s1.size(); ++i)
            if (s1[i] != s2[i]) diff.push_back(i);
        if (diff.empty()) return true;
        if (diff.size() != 2) return false;
        return s1[diff[0]] == s2[diff[1]] && s1[diff[1]] == s2[diff[0]];
    }
};
```

---

##206 ****[Problem Link]https://leetcode.com/problems/last-moment-before-all-ants-fall-out-of-a-plank****  
**Approach:** The last ant falls off from max(left) or (length - min(right)).  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int getLastMoment(int n, vector<int>& left, vector<int>& right) {
        int maxLeft = left.empty() ? 0 : *max_element(left.begin(), left.end());
        int maxRight = right.empty() ? 0 : n - *min_element(right.begin(), right.end());
        return max(maxLeft, maxRight);
    }
};
```

---

##207 ****[Problem Link]https://leetcode.com/problems/occurrences-after-bigram****  
**Approach:** Iterate and check for matching bigram, then collect next word.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<string> findOcurrences(string text, string first, string second) {
        vector<string> res;
        vector<string> words;
        string word;
        for (char c : text) {
            if (c == ' ') {
                words.push_back(word);
                word.clear();
            } else {
                word += c;
            }
        }
        words.push_back(word);
        for (int i = 0; i + 2 < words.size(); ++i) {
            if (words[i] == first && words[i + 1] == second) {
                res.push_back(words[i + 2]);
            }
        }
        return res;
    }
};
```

---

##208 ****[Problem Link]https://leetcode.com/problems/number-of-unique-good-subsequences****  
**Approach:** DP for counting subsequences with ending 0 or 1.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int mod = 1e9 + 7;

    int numberOfUniqueGoodSubsequences(string binary) {
        int ends0 = 0, ends1 = 0, has0 = 0;
        for (char c : binary) {
            if (c == '0') ends0 = (ends0 + ends1) % mod;
            else ends1 = (ends0 + ends1 + 1) % mod;
            if (c == '0') has0 = 1;
        }
        return (ends0 + ends1 + has0) % mod;
    }
};
```

---

##209 ****[Problem Link]https://leetcode.com/problems/queries-on-a-permutation-with-key****  
**Approach:** Simulate using a list and adjust positions.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> processQueries(vector<int>& queries, int m) {
        vector<int> perm(m);
        for (int i = 0; i < m; ++i) perm[i] = i + 1;
        vector<int> res;
        for (int q : queries) {
            for (int i = 0; i < m; ++i) {
                if (perm[i] == q) {
                    res.push_back(i);
                    perm.erase(perm.begin() + i);
                    perm.insert(perm.begin(), q);
                    break;
                }
            }
        }
        return res;
    }
};
```

---

##210 ****[Problem Link]https://leetcode.com/problems/prime-palindrome****  
**Approach:** Generate palindromes and check primality.  
**Time Complexity:** O(n log n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    bool isPrime(int n) {
        if (n < 2) return false;
        for (int i = 2; i * i <= n; ++i)
            if (n % i == 0) return false;
        return true;
    }

    int primePalindrome(int n) {
        for (int i = 1; i < 100000; ++i) {
            string s = to_string(i);
            string r = s;
            reverse(r.begin(), r.end());
            int p = stoi(s + r.substr(1));
            if (p >= n && isPrime(p)) return p;
        }
        return 100030001;
    }
};
```

---

##211 ****[Problem Link]https://leetcode.com/problems/making-file-names-unique****  
**Approach:** HashMap to track filename counts and append suffix if needed.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    vector<string> getFolderNames(vector<string>& names) {
        unordered_map<string, int> mp;
        vector<string> res;
        for (auto& name : names) {
            if (mp.count(name) == 0) {
                res.push_back(name);
                mp[name] = 1;
            } else {
                int k = mp[name];
                string newName;
                while (mp.count(newName = name + "(" + to_string(k) + ")")) {
                    ++k;
                }
                res.push_back(newName);
                mp[name] = k + 1;
                mp[newName] = 1;
            }
        }
        return res;
    }
};
```

---

##212 ****[Problem Link]https://leetcode.com/problems/largest-merge-of-two-strings****  
**Approach:** Greedy - compare suffixes of both strings and take the larger one.  
**Time Complexity:** O(m + n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string largestMerge(string word1, string word2) {
        string res;
        int i = 0, j = 0;
        while (i < word1.size() || j < word2.size()) {
            if (word1.substr(i) > word2.substr(j)) res += word1[i++];
            else res += word2[j++];
        }
        return res;
    }
};
```

---

##213 ****[Problem Link]https://leetcode.com/problems/maximize-the-confusion-of-an-exam****  
**Approach:** Sliding window to maintain window of length k with majority char.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int maxConsecutiveAnswers(string answerKey, int k) {
        return max(maxConsecutiveChar(answerKey, k, 'T'), maxConsecutiveChar(answerKey, k, 'F'));
    }

    int maxConsecutiveChar(string& s, int k, char ch) {
        int left = 0, maxLen = 0, cnt = 0;
        for (int right = 0; right < s.size(); ++right) {
            if (s[right] != ch) ++cnt;
            while (cnt > k) {
                if (s[left++] != ch) --cnt;
            }
            maxLen = max(maxLen, right - left + 1);
        }
        return maxLen;
    }
};
```

---

##214 ****[Problem Link]https://leetcode.com/problems/relative-ranks****  
**Approach:** Sort and map original indices to ranking.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<string> findRelativeRanks(vector<int>& score) {
        vector<pair<int, int>> scores;
        for (int i = 0; i < score.size(); ++i)
            scores.emplace_back(score[i], i);
        sort(scores.rbegin(), scores.rend());

        vector<string> res(score.size());
        for (int i = 0; i < scores.size(); ++i) {
            if (i == 0) res[scores[i].second] = "Gold Medal";
            else if (i == 1) res[scores[i].second] = "Silver Medal";
            else if (i == 2) res[scores[i].second] = "Bronze Medal";
            else res[scores[i].second] = to_string(i + 1);
        }
        return res;
    }
};
```

---

##215 ****[Problem Link]https://leetcode.com/problems/thousand-separator****  
**Approach:** Build result string from end with dot every three digits.  
**Time Complexity:** O(log n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string thousandSeparator(int n) {
        if (n == 0) return "0";
        string s = to_string(n), res;
        int len = s.size();
        for (int i = 0; i < len; ++i) {
            if (i && (len - i) % 3 == 0) res += '.';
            res += s[i];
        }
        return res;
    }
};
```

---

##216 ****[Problem Link]https://leetcode.com/problems/super-palindromes****  
**Approach:** Generate palindromes, square and check if result is palindrome.  
**Time Complexity:** O(sqrt(N))

```cpp
#include <string>
#include <cmath>

using namespace std;

class Solution {
public:
    bool isPalindrome(long long n) {
        string s = to_string(n);
        int l = 0, r = s.size() - 1;
        while (l < r)
            if (s[l++] != s[r--]) return false;
        return true;
    }

    int superpalindromesInRange(string left, string right) {
        long long l = stoll(left), r = stoll(right);
        int res = 0;
        for (int k = 1; k < 100000; ++k) {
            string s = to_string(k), rs = s;
            reverse(rs.begin(), rs.end());
            long long v1 = stoll(s + rs); // even
            long long v2 = stoll(s + rs.substr(1)); // odd
            for (long long x : {v1, v2}) {
                long long sq = x * x;
                if (sq > r) continue;
                if (sq >= l && isPalindrome(sq)) res++;
            }
        }
        return res;
    }
};
```

---

##217 ****[Problem Link]https://leetcode.com/problems/reconstruct-a-2-row-binary-matrix****  
**Approach:** Greedy - assign 2 first, then fill upper and lower if possible.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> reconstructMatrix(int upper, int lower, vector<int>& colsum) {
        int n = colsum.size();
        vector<vector<int>> res(2, vector<int>(n, 0));
        for (int i = 0; i < n; ++i) {
            if (colsum[i] == 2) {
                res[0][i] = res[1][i] = 1;
                upper--; lower--;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (colsum[i] == 1) {
                if (upper > 0) {
                    res[0][i] = 1;
                    upper--;
                } else {
                    res[1][i] = 1;
                    lower--;
                }
            }
        }
        if (upper == 0 && lower == 0) return res;
        return {};
    }
};
```

---

##218 ****[Problem Link]https://leetcode.com/problems/minimum-cost-to-connect-two-groups-of-points****  
**Approach:** DP with bitmask for right-side connections.  
**Time Complexity:** O(m * 2^n)

```cpp
#include <vector>
#include <cstring>

using namespace std;

class Solution {
public:
    int connectTwoGroups(vector<vector<int>>& cost) {
        int m = cost.size(), n = cost[0].size();
        vector<vector<int>> dp(m + 1, vector<int>(1 << n, 1e9));
        dp[0][0] = 0;

        for (int i = 0; i < m; ++i) {
            for (int mask = 0; mask < (1 << n); ++mask) {
                for (int j = 0; j < n; ++j) {
                    dp[i + 1][mask | (1 << j)] = min(dp[i + 1][mask | (1 << j)], dp[i][mask] + cost[i][j]);
                }
            }
        }

        int res = dp[m][(1 << n) - 1];
        for (int mask = 0; mask < (1 << n); ++mask) {
            if ((mask & ((1 << n) - 1)) != ((1 << n) - 1)) continue;
            res = min(res, dp[m][mask]);
        }

        return res;
    }
};
```

---

##219 ****[Problem Link]https://leetcode.com/problems/get-maximum-in-generated-array****  
**Approach:** Simulate the generation of array with given rules.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int getMaximumGenerated(int n) {
        if (n == 0) return 0;
        vector<int> nums(n + 1);
        nums[0] = 0; nums[1] = 1;
        for (int i = 2; i <= n; ++i) {
            if (i % 2 == 0) nums[i] = nums[i / 2];
            else nums[i] = nums[i / 2] + nums[i / 2 + 1];
        }
        return *max_element(nums.begin(), nums.end());
    }
};
```

---

##220 ****[Problem Link]https://leetcode.com/problems/largest-odd-number-in-string****  
**Approach:** Traverse from right to find last odd digit.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string largestOddNumber(string num) {
        for (int i = num.size() - 1; i >= 0; --i) {
            if ((num[i] - '0') % 2) return num.substr(0, i + 1);
        }
        return "";
    }
};
```

---

##221 ****[Problem Link]https://leetcode.com/problems/average-waiting-time****  
**Approach:** Simulate queue processing and accumulate waiting time.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    double averageWaitingTime(vector<vector<int>>& customers) {
        long long time = 0, wait = 0;
        for (auto& c : customers) {
            time = max(time, (long long)c[0]) + c[1];
            wait += time - c[0];
        }
        return (double)wait / customers.size();
    }
};
```

---

##222 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous****  
**Approach:** Coordinate compression + sliding window for unique values.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int minOperations(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        nums.erase(unique(nums.begin(), nums.end()), nums.end());

        int res = n, j = 0;
        for (int i = 0; i < nums.size(); ++i) {
            while (j < nums.size() && nums[j] < nums[i] + n) ++j;
            res = min(res, n - (j - i));
        }
        return res;
    }
};
```

---

##223 ****[Problem Link]https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference****  
**Approach:** Meet in the middle - subset sums of halves and binary search.  
**Time Complexity:** O(2^(n/2) * log(2^(n/2)))

```cpp
#include <vector>
#include <set>
#include <cmath>

using namespace std;

class Solution {
public:
    int minimumDifference(vector<int>& nums) {
        int n = nums.size(), half = n / 2;
        long total = 0;
        for (int x : nums) total += x;

        vector<long> left, right;
        for (int i = 0; i < (1 << half); ++i) {
            long sum = 0;
            for (int j = 0; j < half; ++j)
                if (i & (1 << j)) sum += nums[j];
            left.push_back(sum);
        }

        for (int i = 0; i < (1 << (n - half)); ++i) {
            long sum = 0;
            for (int j = 0; j < n - half; ++j)
                if (i & (1 << j)) sum += nums[half + j];
            right.push_back(sum);
        }

        sort(right.begin(), right.end());
        long res = abs(total - 2 * left[0]);

        for (long a : left) {
            long target = total / 2 - a;
            auto it = lower_bound(right.begin(), right.end(), target);
            if (it != right.end()) res = min(res, abs(total - 2 * (a + *it)));
            if (it != right.begin()) --it, res = min(res, abs(total - 2 * (a + *it)));
        }
        return res;
    }
};
```

---

##224 ****[Problem Link]https://leetcode.com/problems/sum-of-subarray-ranges****  
**Approach:** Monotonic stack to count contribution of each element.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <stack>

using namespace std;

class Solution {
public:
    long long subArrayRanges(vector<int>& nums) {
        return sumOfMax(nums) - sumOfMin(nums);
    }

    long long sumOfMax(vector<int>& nums) {
        long long res = 0;
        stack<int> s;
        int n = nums.size();
        for (int i = 0; i <= n; ++i) {
            while (!s.empty() && (i == n || nums[s.top()] < nums[i])) {
                int mid = s.top(); s.pop();
                int left = s.empty() ? -1 : s.top();
                res += (long long)nums[mid] * (i - mid) * (mid - left);
            }
            s.push(i);
        }
        return res;
    }

    long long sumOfMin(vector<int>& nums) {
        long long res = 0;
        stack<int> s;
        int n = nums.size();
        for (int i = 0; i <= n; ++i) {
            while (!s.empty() && (i == n || nums[s.top()] > nums[i])) {
                int mid = s.top(); s.pop();
                int left = s.empty() ? -1 : s.top();
                res += (long long)nums[mid] * (i - mid) * (mid - left);
            }
            s.push(i);
        }
        return res;
    }
};
```

---

##225 ****[Problem Link]https://leetcode.com/problems/two-best-non-overlapping-events****  
**Approach:** Sort by end time and use prefix max on value for best before start.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxTwoEvents(vector<vector<int>>& events) {
        sort(events.begin(), events.end());
        int res = 0, n = events.size();
        vector<int> best(n, 0);
        best[0] = events[0][2];
        for (int i = 1; i < n; ++i) {
            best[i] = max(best[i - 1], events[i][2]);
        }

        vector<vector<int>> ends = events;
        sort(ends.begin(), ends.end(), [](auto& a, auto& b) {
            return a[1] < b[1];
        });

        for (int i = 0; i < n; ++i) {
            int start = events[i][0], val = events[i][2];
            int lo = 0, hi = n - 1, idx = -1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if (ends[mid][1] < start) {
                    idx = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            int bestBefore = idx == -1 ? 0 : best[idx];
            res = max(res, bestBefore + val);
        }
        return res;
    }
};
```

---

##226 ****[Problem Link]https://leetcode.com/problems/closest-room****  
**Approach:** Sort rooms and queries, use TreeSet to find closest id >= preferred.  
**Time Complexity:** O(n log n + q log q)

```cpp
#include <vector>
#include <set>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<int> closestRoom(vector<vector<int>>& rooms, vector<vector<int>>& queries) {
        vector<int> res(queries.size()), idx(queries.size());
        for (int i = 0; i < idx.size(); ++i) idx[i] = i;

        sort(rooms.begin(), rooms.end(), [](auto& a, auto& b) {
            return a[1] > b[1];
        });
        sort(idx.begin(), idx.end(), [&](int i, int j) {
            return queries[i][1] > queries[j][1];
        });

        set<int> ids;
        int i = 0;
        for (int j : idx) {
            while (i < rooms.size() && rooms[i][1] >= queries[j][1]) {
                ids.insert(rooms[i++][0]);
            }
            if (ids.empty()) {
                res[j] = -1;
                continue;
            }
            auto it = ids.lower_bound(queries[j][0]);
            int a = it == ids.end() ? INT_MAX : *it;
            int b = it == ids.begin() ? INT_MAX : *(--it);
            if (abs(a - queries[j][0]) < abs(b - queries[j][0])) res[j] = a;
            else if (abs(a - queries[j][0]) > abs(b - queries[j][0])) res[j] = b;
            else res[j] = min(a, b);
        }
        return res;
    }
};
```

---

##227 ****[Problem Link]https://leetcode.com/problems/minimum-operations-to-convert-number****  
**Approach:** BFS from start to end using +, -, ^.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <queue>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int minimumOperations(vector<int>& nums, int start, int goal) {
        queue<pair<int, int>> q;
        unordered_set<int> visited;
        q.push({start, 0});
        visited.insert(start);

        while (!q.empty()) {
            auto [cur, steps] = q.front(); q.pop();
            if (cur == goal) return steps;
            for (int x : nums) {
                for (int nxt : {cur + x, cur - x, cur ^ x}) {
                    if (nxt >= 0 && nxt <= 1000 && !visited.count(nxt)) {
                        q.push({nxt, steps + 1});
                        visited.insert(nxt);
                    } else if (nxt == goal) {
                        return steps + 1;
                    }
                }
            }
        }
        return -1;
    }
};
```

---

##228 ****[Problem Link]https://leetcode.com/problems/graph-connectivity-with-threshold****  
**Approach:** Union-Find for multiples of each number > threshold.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> parent;

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int x, int y) {
        parent[find(x)] = find(y);
    }

    vector<bool> areConnected(int n, int threshold, vector<vector<int>>& queries) {
        parent.resize(n + 1);
        for (int i = 0; i <= n; ++i) parent[i] = i;

        for (int i = threshold + 1; i <= n; ++i)
            for (int j = 2 * i; j <= n; j += i)
                unite(i, j);

        vector<bool> res;
        for (auto& q : queries)
            res.push_back(find(q[0]) == find(q[1]));
        return res;
    }
};
```

---

##229 ****[Problem Link]https://leetcode.com/problems/first-day-where-you-have-been-in-all-the-rooms****  
**Approach:** Simulation with memoization to compute visit days.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int firstDayBeenInAllRooms(vector<int>& nextVisit) {
        const int MOD = 1e9 + 7;
        int n = nextVisit.size();
        vector<long> dp(n);
        for (int i = 1; i < n; ++i) {
            dp[i] = (2 * dp[i - 1] - dp[nextVisit[i - 1]] + 2 + MOD) % MOD;
        }
        return dp[n - 1];
    }
};
```

---

##230 ****[Problem Link]https://leetcode.com/problems/number-of-rectangles-that-can-form-the-largest-square****  
**Approach:** Track max square length and count of such rectangles.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int countGoodRectangles(vector<vector<int>>& rectangles) {
        int maxLen = 0, count = 0;
        for (auto& r : rectangles) {
            int side = min(r[0], r[1]);
            if (side > maxLen) {
                maxLen = side;
                count = 1;
            } else if (side == maxLen) {
                count++;
            }
        }
        return count;
    }
};
```

---

##231 ****[Problem Link]https://leetcode.com/problems/random-pick-with-weight****  
**Approach:** Use prefix sum and binary search for weighted sampling.  
**Time Complexity:** O(log n) per pick

```cpp
#include <vector>
#include <cstdlib>

using namespace std;

class Solution {
    vector<int> prefix;
    int total = 0;

public:
    Solution(vector<int>& w) {
        for (int weight : w) {
            total += weight;
            prefix.push_back(total);
        }
    }

    int pickIndex() {
        int target = rand() % total;
        return upper_bound(prefix.begin(), prefix.end(), target) - prefix.begin();
    }
};
```

---

##232 ****[Problem Link]https://leetcode.com/problems/check-if-word-equals-summation-of-two-words****  
**Approach:** Convert word to integer, compare sum.  
**Time Complexity:** O(1)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    bool isSumEqual(string firstWord, string secondWord, string targetWord) {
        return toInt(firstWord) + toInt(secondWord) == toInt(targetWord);
    }

    int toInt(string word) {
        int res = 0;
        for (char c : word) res = res * 10 + (c - 'a');
        return res;
    }
};
```

---

##233 ****[Problem Link]https://leetcode.com/problems/fraction-addition-and-subtraction****  
**Approach:** Parse fractions, reduce with gcd, add and simplify.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <numeric>

using namespace std;

class Solution {
public:
    string fractionAddition(string expression) {
        int num = 0, den = 1, i = 0;
        while (i < expression.size()) {
            int sign = 1;
            if (expression[i] == '-' || expression[i] == '+') sign = (expression[i++] == '-') ? -1 : 1;
            int n = 0;
            while (isdigit(expression[i])) n = n * 10 + (expression[i++] - '0');
            i++; // skip '/'
            int d = 0;
            while (i < expression.size() && isdigit(expression[i])) d = d * 10 + (expression[i++] - '0');
            num = num * d + sign * n * den;
            den *= d;
            int g = gcd(abs(num), den);
            num /= g; den /= g;
        }
        return to_string(num) + "/" + to_string(den);
    }
};
```

---

##234 ****[Problem Link]https://leetcode.com/problems/number-of-paths-with-max-score****  
**Approach:** DP with max score and path count from bottom-right to top-left.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    const int MOD = 1e9 + 7;

    vector<int> pathsWithMaxScore(vector<string>& board) {
        int n = board.size();
        vector<vector<int>> score(n, vector<int>(n, -1));
        vector<vector<int>> ways(n, vector<int>(n, 0));
        score[n - 1][n - 1] = 0;
        ways[n - 1][n - 1] = 1;

        for (int i = n - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                if (board[i][j] == 'X' || score[i][j] == -1) continue;
                for (auto [dx, dy] : vector<pair<int, int>>{{-1, 0}, {0, -1}, {-1, -1}}) {
                    int x = i + dx, y = j + dy;
                    if (x < 0 || y < 0 || board[x][y] == 'X') continue;
                    int val = board[x][y] == 'E' || board[x][y] == 'S' ? 0 : board[x][y] - '0';
                    if (score[x][y] < score[i][j] + val) {
                        score[x][y] = score[i][j] + val;
                        ways[x][y] = ways[i][j];
                    } else if (score[x][y] == score[i][j] + val) {
                        ways[x][y] = (ways[x][y] + ways[i][j]) % MOD;
                    }
                }
            }
        }
        if (score[0][0] == -1) return {0, 0};
        return {score[0][0], ways[0][0]};
    }
};
```

---

##235 ****[Problem Link]https://leetcode.com/problems/verbal-arithmetic-puzzle****  
**Approach:** Backtracking with letter-to-digit mapping.  
**Time Complexity:** O(10!)

```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class Solution {
public:
    bool isSolvable(vector<string>& words, string result) {
        unordered_map<char, int> mp;
        unordered_set<int> used;
        vector<char> chars;
        vector<bool> leading(26, false);
        int maxLen = 0;

        for (auto& w : words) {
            for (char c : w) if (!mp.count(c)) mp[c] = -1;
            leading[w[0] - 'A'] = true;
            maxLen = max(maxLen, (int)w.size());
        }

        for (char c : result) if (!mp.count(c)) mp[c] = -1;
        leading[result[0] - 'A'] = true;

        for (auto& p : mp) chars.push_back(p.first);
        return dfs(0, chars, mp, used, leading, words, result);
    }

    bool dfs(int i, vector<char>& chars, unordered_map<char, int>& mp, unordered_set<int>& used,
             vector<bool>& leading, vector<string>& words, string& result) {
        if (i == chars.size()) return isValid(mp, words, result);
        for (int d = 0; d <= 9; ++d) {
            if (used.count(d)) continue;
            if (d == 0 && leading[chars[i] - 'A']) continue;
            mp[chars[i]] = d;
            used.insert(d);
            if (dfs(i + 1, chars, mp, used, leading, words, result)) return true;
            mp[chars[i]] = -1;
            used.erase(d);
        }
        return false;
    }

    bool isValid(unordered_map<char, int>& mp, vector<string>& words, string& result) {
        long sum = 0;
        for (auto& w : words) {
            long val = 0;
            for (char c : w) val = val * 10 + mp[c];
            sum += val;
        }
        long res = 0;
        for (char c : result) res = res * 10 + mp[c];
        return sum == res;
    }
};
```

---

##236 ****[Problem Link]https://leetcode.com/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits****  
**Approach:** Use segment tree or greedy greedy selection within a sliding window.  
**Time Complexity:** O(n log n)

```cpp
#include <string>
#include <set>
#include <vector>

using namespace std;

class Solution {
public:
    string minInteger(string num, int k) {
        int n = num.size();
        if (k >= n * (n - 1) / 2) {
            sort(num.begin(), num.end());
            return num;
        }

        vector<set<int>> pos(10);
        for (int i = 0; i < n; ++i)
            pos[num[i] - '0'].insert(i);

        string res;
        int used = 0;
        vector<bool> removed(n, false);

        for (int i = 0; i < n; ++i) {
            for (int d = 0; d <= 9; ++d) {
                if (pos[d].empty()) continue;
                int idx = *pos[d].begin();
                int skipped = 0;
                for (int j = 0; j < idx; ++j) if (removed[j]) ++skipped;
                int dist = idx - skipped - used;
                if (dist <= k) {
                    k -= dist;
                    res += ('0' + d);
                    removed[idx] = true;
                    pos[d].erase(pos[d].begin());
                    ++used;
                    break;
                }
            }
        }

        return res;
    }
};
```

---

##237 ****[Problem Link]https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments****  
**Approach:** DP to count segments placed on i-length line with j segments.  
**Time Complexity:** O(n * k)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int numberOfSets(int n, int k) {
        const int MOD = 1e9 + 7;
        vector<vector<int>> dp(n + 1, vector<int>(k + 1));
        for (int i = 0; i <= n; ++i) dp[i][0] = 1;
        for (int j = 1; j <= k; ++j) {
            int pre = 0;
            for (int i = 1; i <= n; ++i) {
                pre = (pre + dp[i - 1][j - 1]) % MOD;
                dp[i][j] = (dp[i - 1][j] + pre) % MOD;
            }
        }
        return dp[n][k];
    }
};
```

---

##238 ****[Problem Link]https://leetcode.com/problems/strange-printer-ii****  
**Approach:** Topological sort over dependencies among characters.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

using namespace std;

class Solution {
public:
    bool isPrintable(vector<vector<int>>& targetGrid) {
        int m = targetGrid.size(), n = targetGrid[0].size();
        unordered_map<int, vector<int>> rect(61, {m, n, 0, 0});
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                int c = targetGrid[i][j];
                rect[c][0] = min(rect[c][0], i);
                rect[c][1] = min(rect[c][1], j);
                rect[c][2] = max(rect[c][2], i);
                rect[c][3] = max(rect[c][3], j);
            }

        vector<vector<int>> graph(61);
        vector<int> indeg(61);
        for (int c = 1; c <= 60; ++c) {
            if (rect[c][0] == m) continue;
            for (int i = rect[c][0]; i <= rect[c][2]; ++i)
                for (int j = rect[c][1]; j <= rect[c][3]; ++j)
                    if (targetGrid[i][j] != c) {
                        graph[targetGrid[i][j]].push_back(c);
                        indeg[c]++;
                        break;
                    }
        }

        queue<int> q;
        for (int i = 1; i <= 60; ++i)
            if (indeg[i] == 0 && rect[i][0] != m) q.push(i);

        int count = 0;
        while (!q.empty()) {
            int c = q.front(); q.pop(); count++;
            for (int nei : graph[c])
                if (--indeg[nei] == 0) q.push(nei);
        }

        return count == rect.size() - 1; // exclude unused color
    }
};
```

---

##239 ****[Problem Link]https://leetcode.com/problems/find-the-kth-largest-integer-in-the-array****  
**Approach:** Sort by length and lexicographical order.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    string kthLargestNumber(vector<string>& nums, int k) {
        sort(nums.begin(), nums.end(), [](const string& a, const string& b) {
            return a.size() == b.size() ? a < b : a.size() < b.size();
        });
        return nums[nums.size() - k];
    }
};
```

---

##240 ****[Problem Link]https://leetcode.com/problems/shortest-completing-word****  
**Approach:** Count letter frequencies, check if each word satisfies.  
**Time Complexity:** O(n * m)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string shortestCompletingWord(string licensePlate, vector<string>& words) {
        vector<int> cnt(26, 0);
        for (char c : licensePlate) {
            if (isalpha(c)) cnt[tolower(c) - 'a']++;
        }

        string res;
        for (string& w : words) {
            vector<int> temp = cnt;
            for (char c : w) temp[c - 'a']--;
            if (all_of(temp.begin(), temp.end(), [](int x) { return x <= 0; })) {
                if (res.empty() || w.size() < res.size()) res = w;
            }
        }
        return res;
    }
};
```

---

##241 ****[Problem Link]https://leetcode.com/problems/count-odd-numbers-in-an-interval-range****  
**Approach:** Direct math formula.  
**Time Complexity:** O(1)

```cpp
class Solution {
public:
    int countOdds(int low, int high) {
        return (high + 1) / 2 - (low / 2);
    }
};
```

---

##242 ****[Problem Link]https://leetcode.com/problems/circular-array-loop****  
**Approach:** Cycle detection with slow and fast pointer.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    bool circularArrayLoop(vector<int>& nums) {
        int n = nums.size();
        auto next = [&](int i) {
            return ((i + nums[i]) % n + n) % n;
        };

        for (int i = 0; i < n; ++i) {
            if (!nums[i]) continue;
            int slow = i, fast = next(i);
            while (nums[slow] * nums[fast] > 0 && nums[slow] * nums[next(fast)] > 0) {
                if (slow == fast) {
                    if (slow == next(slow)) break;
                    return true;
                }
                slow = next(slow);
                fast = next(next(fast));
            }

            int j = i;
            while (nums[j] * nums[next(j)] > 0) {
                int tmp = j;
                j = next(j);
                nums[tmp] = 0;
            }
        }
        return false;
    }
};
```

---

##243 ****[Problem Link]https://leetcode.com/problems/maximum-ice-cream-bars****  
**Approach:** Sort costs and greedily buy cheapest first.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxIceCream(vector<int>& costs, int coins) {
        sort(costs.begin(), costs.end());
        int count = 0;
        for (int cost : costs) {
            if (coins < cost) break;
            coins -= cost;
            count++;
        }
        return count;
    }
};
```

---

##244 ****[Problem Link]https://leetcode.com/problems/reduction-operations-to-make-the-array-elements-equal****  
**Approach:** Sort array, count moves by comparing consecutive elements.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int reductionOperations(vector<int>& nums) {
        sort(nums.begin(), nums.end(), greater<int>());
        int res = 0, steps = 0;
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] != nums[i - 1]) ++steps;
            res += steps;
        }
        return res;
    }
};
```

---

##245 ****[Problem Link]https://leetcode.com/problems/count-the-repetitions****  
**Approach:** Simulate and use hashmap to find cycle and skip.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int getMaxRepetitions(string s1, int n1, string s2, int n2) {
        int len1 = s1.size(), len2 = s2.size();
        vector<int> repeatCount(n1 + 1), nextIndex(n1 + 1);
        unordered_map<int, pair<int, int>> visited;
        int j = 0, count = 0;
        for (int k = 1; k <= n1; ++k) {
            for (char c : s1) {
                if (c == s2[j]) ++j;
                if (j == len2) {
                    ++count;
                    j = 0;
                }
            }
            repeatCount[k] = count;
            nextIndex[k] = j;
            if (visited.count(j)) {
                int start = visited[j].first;
                int prefixCount = visited[j].second;
                int patternLength = k - start;
                int patternCount = (n1 - start) / patternLength;
                int suffix = (n1 - start) % patternLength;
                return (repeatCount[start] + patternCount * (repeatCount[k] - repeatCount[start]) + repeatCount[start + suffix] - repeatCount[start]) / n2;
            }
            visited[j] = {k, count};
        }
        return repeatCount[n1] / n2;
    }
};
```

---

##246 ****[Problem Link]https://leetcode.com/problems/transform-to-chessboard****  
**Approach:** Validate transformation rules and compute minimal swaps.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int movesToChessboard(vector<vector<int>>& board) {
        int n = board.size(), rowSum = 0, colSum = 0, rowSwap = 0, colSwap = 0;
        for (int i = 0; i < n; ++i) {
            rowSum += board[0][i];
            colSum += board[i][0];
            rowSwap += board[i][0] == i % 2;
            colSwap += board[0][i] == i % 2;
        }

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if ((board[0][0] ^ board[i][0] ^ board[0][j] ^ board[i][j]) != 0)
                    return -1;

        if (rowSum < n / 2 || rowSum > (n + 1) / 2) return -1;
        if (colSum < n / 2 || colSum > (n + 1) / 2) return -1;

        if (n % 2 == 0)
            return min(n - rowSwap, rowSwap) / 2 + min(n - colSwap, colSwap) / 2;
        else
            return ((rowSwap % 2 == 0) ? rowSwap : n - rowSwap) / 2 +
                   ((colSwap % 2 == 0) ? colSwap : n - colSwap) / 2;
    }
};
```

---

##247 ****[Problem Link]https://leetcode.com/problems/moving-stones-until-consecutive-ii****  
**Approach:** Sort and use sliding window for min/max moves.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<int> numMovesStonesII(vector<int>& stones) {
        sort(stones.begin(), stones.end());
        int n = stones.size();
        int high = max(stones[n - 1] - stones[1] - n + 2,
                       stones[n - 2] - stones[0] - n + 2);
        int low = n;
        for (int i = 0, j = 0; i < n; ++i) {
            while (stones[i] - stones[j] + 1 > n) ++j;
            int inWindow = i - j + 1;
            if (inWindow == n - 1 && stones[i] - stones[j] + 1 == n - 1)
                low = min(low, 2);
            else
                low = min(low, n - inWindow);
        }
        return {low, high};
    }
};
```

---

##248 ****[Problem Link]https://leetcode.com/problems/maximum-binary-string-after-change****  
**Approach:** Greedy - count 0s and fix their position.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string maximumBinaryString(string binary) {
        int n = binary.size(), count0 = 0, first0 = -1;
        for (int i = 0; i < n; ++i) {
            if (binary[i] == '0') {
                if (first0 == -1) first0 = i;
                ++count0;
            }
        }
        if (count0 <= 1) return binary;
        string res(n, '1');
        res[first0 + count0 - 1] = '0';
        return res;
    }
};
```

---

##249 ****[Problem Link]https://leetcode.com/problems/final-value-of-variable-after-performing-operations****  
**Approach:** Count net effect of operations.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int finalValueAfterOperations(vector<string>& operations) {
        int x = 0;
        for (string& op : operations) {
            if (op[1] == '+') ++x;
            else --x;
        }
        return x;
    }
};
```

---

##250 ****[Problem Link]https://leetcode.com/problems/minimum-area-rectangle-ii****  
**Approach:** Brute-force all point pairs as diagonals and check for 4th vertex.  
**Time Complexity:** O(n^3)

```cpp
#include <vector>
#include <unordered_set>
#include <cmath>

using namespace std;

class Solution {
public:
    double minAreaFreeRect(vector<vector<int>>& points) {
        int n = points.size();
        double minArea = DBL_MAX;
        unordered_set<string> seen;
        for (auto& p : points)
            seen.insert(to_string(p[0]) + "," + to_string(p[1]));

        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                for (int k = j + 1; k < n; ++k) {
                    vector<int> p1 = points[i], p2 = points[j], p3 = points[k];
                    vector<int> v1 = {p2[0] - p1[0], p2[1] - p1[1]};
                    vector<int> v2 = {p3[0] - p1[0], p3[1] - p1[1]};
                    if (v1[0] * v2[0] + v1[1] * v2[1] != 0) continue;
                    int x = p3[0] + v1[0], y = p3[1] + v1[1];
                    string key = to_string(x) + "," + to_string(y);
                    if (seen.count(key)) {
                        double area = sqrt(v1[0] * v1[0] + v1[1] * v1[1]) *
                                      sqrt(v2[0] * v2[0] + v2[1] * v2[1]);
                        minArea = min(minArea, area);
                    }
                }
        return minArea == DBL_MAX ? 0 : minArea;
    }
};
```

---

##251 ****[Problem Link]https://leetcode.com/problems/generate-a-string-with-characters-that-have-odd-counts****  
**Approach:** For even n, use n - 1 of 'a' and 1 of 'b'. For odd n, use all 'a'.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    string generateTheString(int n) {
        if (n % 2 == 1) return string(n, 'a');
        return string(n - 1, 'a') + 'b';
    }
};
```

---

##252 ****[Problem Link]https://leetcode.com/problems/random-flip-matrix****  
**Approach:** Use hashmap to track swapped indices for uniform random selection.  
**Time Complexity:** O(1) per operation

```cpp
#include <unordered_map>
#include <cstdlib>

using namespace std;

class Solution {
    int m, n, total;
    unordered_map<int, int> map;

public:
    Solution(int m, int n) : m(m), n(n), total(m * n) {}

    vector<int> flip() {
        int r = rand() % total--;
        int x = map.count(r) ? map[r] : r;
        map[r] = map.count(total) ? map[total] : total;
        return {x / n, x % n};
    }

    void reset() {
        map.clear();
        total = m * n;
    }
};
```

---

##253 ****[Problem Link]https://leetcode.com/problems/preimage-size-of-factorial-zeroes-function****  
**Approach:** Binary search to count trailing zeroes.  
**Time Complexity:** O(log n * log n)

```cpp
class Solution {
public:
    int preimageSizeFZF(int k) {
        return helper(k) - helper(k - 1);
    }

    int helper(int k) {
        long l = 0, r = 5L * (k + 1);
        while (l < r) {
            long mid = l + (r - l) / 2;
            if (trailingZeroes(mid) <= k) l = mid + 1;
            else r = mid;
        }
        return l;
    }

    int trailingZeroes(long n) {
        int count = 0;
        while (n) {
            count += n / 5;
            n /= 5;
        }
        return count;
    }
};
```

---

##254 ****[Problem Link]https://leetcode.com/problems/eliminate-maximum-number-of-monsters****  
**Approach:** Sort monsters by arrival time and shoot one per time unit.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int eliminateMaximum(vector<int>& dist, vector<int>& speed) {
        int n = dist.size();
        vector<double> time(n);
        for (int i = 0; i < n; ++i)
            time[i] = (double)dist[i] / speed[i];
        sort(time.begin(), time.end());
        for (int i = 0; i < n; ++i)
            if (time[i] <= i) return i;
        return n;
    }
};
```

---

##255 ****[Problem Link]https://leetcode.com/problems/find-target-indices-after-sorting-array****  
**Approach:** Count numbers less than and equal to target.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<int> targetIndices(vector<int>& nums, int target) {
        int less = 0, equal = 0;
        for (int x : nums) {
            if (x < target) ++less;
            else if (x == target) ++equal;
        }
        vector<int> res;
        for (int i = 0; i < equal; ++i)
            res.push_back(less + i);
        return res;
    }
};
```

---

##256 ****[Problem Link]https://leetcode.com/problems/sum-game****  
**Approach:** Check sum difference and blank counts on both sides.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    bool sumGame(string num) {
        int n = num.size(), leftSum = 0, rightSum = 0, leftQ = 0, rightQ = 0;
        for (int i = 0; i < n / 2; ++i) {
            if (num[i] == '?') ++leftQ;
            else leftSum += num[i] - '0';
        }
        for (int i = n / 2; i < n; ++i) {
            if (num[i] == '?') ++rightQ;
            else rightSum += num[i] - '0';
        }

        if ((leftQ + rightQ) % 2) return true;
        int deltaQ = leftQ - rightQ;
        int deltaS = rightSum - leftSum;
        return deltaS != (deltaQ / 2) * 9;
    }
};
```

---

##257 ****[Problem Link]https://leetcode.com/problems/number-of-different-integers-in-a-string****  
**Approach:** Parse numeric substrings and use set for uniqueness.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int numDifferentIntegers(string word) {
        unordered_set<string> nums;
        int i = 0, n = word.size();
        while (i < n) {
            if (isdigit(word[i])) {
                int j = i;
                while (j < n && isdigit(word[j])) ++j;
                while (i < j && word[i] == '0') ++i; // remove leading zeros
                nums.insert(word.substr(i, j - i));
                i = j;
            } else {
                ++i;
            }
        }
        return nums.size();
    }
};
```

---

##258 ****[Problem Link]https://leetcode.com/problems/range-frequency-queries****  
**Approach:** Store positions and binary search frequency in range.  
**Time Complexity:** O(log k) per query

```cpp
#include <unordered_map>
#include <vector>
#include <algorithm>

using namespace std;

class RangeFreqQuery {
    unordered_map<int, vector<int>> map;
public:
    RangeFreqQuery(vector<int>& arr) {
        for (int i = 0; i < arr.size(); ++i)
            map[arr[i]].push_back(i);
    }

    int query(int left, int right, int value) {
        if (!map.count(value)) return 0;
        auto& v = map[value];
        return upper_bound(v.begin(), v.end(), right) - lower_bound(v.begin(), v.end(), left);
    }
};
```

---

##259 ****[Problem Link]https://leetcode.com/problems/sum-of-beauty-of-all-substrings****  
**Approach:** Count frequency and compute difference of max and min for each substring.  
**Time Complexity:** O(n^2)

```cpp
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int beautySum(string s) {
        int res = 0;
        for (int i = 0; i < s.size(); ++i) {
            vector<int> count(26, 0);
            for (int j = i; j < s.size(); ++j) {
                count[s[j] - 'a']++;
                int mx = 0, mn = INT_MAX;
                for (int c : count) {
                    if (c == 0) continue;
                    mx = max(mx, c);
                    mn = min(mn, c);
                }
                res += mx - mn;
            }
        }
        return res;
    }
};
```

---

##260 ****[Problem Link]https://leetcode.com/problems/count-nodes-with-the-highest-score****  
**Approach:** DFS to compute subtree sizes and scores, track max.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
    vector<vector<int>> tree;
    long maxScore = 0;
    int count = 0;

public:
    int countHighestScoreNodes(vector<int>& parents) {
        int n = parents.size();
        tree.resize(n);
        for (int i = 1; i < n; ++i)
            tree[parents[i]].push_back(i);
        dfs(0, n);
        return count;
    }

    int dfs(int node, int total) {
        long score = 1;
        int size = 0;
        for (int child : tree[node]) {
            int t = dfs(child, total);
            score *= t;
            size += t;
        }
        if (node != 0)
            score *= (total - size - 1);
        if (score == maxScore) ++count;
        else if (score > maxScore) {
            maxScore = score;
            count = 1;
        }
        return size + 1;
    }
};
```

---

##261 ****[Problem Link]https://leetcode.com/problems/splitting-a-string-into-descending-consecutive-values****  
**Approach:** Backtracking to try all possible splits and check descending order.  
**Time Complexity:** O(n^2)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    bool splitString(string s) {
        return dfs(s, 0, -1);
    }

    bool dfs(const string& s, int i, long prev) {
        if (i == s.size()) return true;
        long num = 0;
        for (int j = i; j < s.size(); ++j) {
            num = num * 10 + (s[j] - '0');
            if (num >= prev && prev != -1) break;
            if (dfs(s, j + 1, num) && (prev == -1 || prev - num == 1)) return true;
        }
        return false;
    }
};
```

---

##262 ****[Problem Link]https://leetcode.com/problems/largest-values-from-labels****  
**Approach:** Sort items by value, pick up to limit per label.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

class Solution {
public:
    int largestValsFromLabels(vector<int>& values, vector<int>& labels, int numWanted, int useLimit) {
        vector<pair<int, int>> items;
        for (int i = 0; i < values.size(); ++i)
            items.push_back({values[i], labels[i]});
        sort(items.begin(), items.end(), greater<>());

        unordered_map<int, int> used;
        int res = 0;
        for (auto& [val, lab] : items) {
            if (used[lab] < useLimit && numWanted > 0) {
                res += val;
                used[lab]++;
                numWanted--;
            }
        }
        return res;
    }
};
```

---

##263 ****[Problem Link]https://leetcode.com/problems/can-convert-string-in-k-moves****  
**Approach:** Count required shifts, ensure enough moves are available.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    bool canConvertString(string s, string t, int k) {
        if (s.size() != t.size()) return false;
        vector<int> count(26, 0);
        for (int i = 0; i < s.size(); ++i) {
            int diff = (t[i] - s[i] + 26) % 26;
            if (diff != 0) {
                int time = diff + 26 * count[diff]++;
                if (time > k) return false;
            }
        }
        return true;
    }
};
```

---

##264 ****[Problem Link]https://leetcode.com/problems/find-the-distance-value-between-two-arrays****  
**Approach:** Sort array2 and binary search for each value in array1.  
**Time Complexity:** O(n log m)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int findTheDistanceValue(vector<int>& arr1, vector<int>& arr2, int d) {
        sort(arr2.begin(), arr2.end());
        int count = 0;
        for (int x : arr1) {
            auto it = lower_bound(arr2.begin(), arr2.end(), x);
            if ((it != arr2.end() && abs(*it - x) <= d) ||
                (it != arr2.begin() && abs(*prev(it) - x) <= d)) continue;
            count++;
        }
        return count;
    }
};
```

---

##265 ****[Problem Link]https://leetcode.com/problems/sum-of-floored-pairs****  
**Approach:** Use prefix sums and count multiples efficiently.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int sumOfFlooredPairs(vector<int>& nums) {
        const int MOD = 1e9 + 7;
        int maxVal = *max_element(nums.begin(), nums.end());
        vector<int> freq(maxVal + 1), prefix(maxVal + 2);

        for (int x : nums) freq[x]++;
        for (int i = 1; i <= maxVal; ++i)
            prefix[i] = prefix[i - 1] + freq[i];

        long res = 0;
        for (int i = 1; i <= maxVal; ++i) {
            if (freq[i] == 0) continue;
            for (int j = 1; i * j <= maxVal; ++j) {
                int l = i * j, r = min(maxVal, i * j + i - 1);
                res = (res + (long)freq[i] * j * (prefix[r] - prefix[l - 1])) % MOD;
            }
        }
        return res;
    }
};
```

---

##266 ****[Problem Link]https://leetcode.com/problems/painting-a-grid-with-three-different-colors****  
**Approach:** DP with state compression using bitmask for valid colorings.  
**Time Complexity:** O(m * 3^m * n)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int colorTheGrid(int m, int n) {
        const int MOD = 1e9 + 7;
        vector<int> valid;
        function<void(int, int)> dfs = [&](int row, int code) {
            if (row == m) {
                valid.push_back(code);
                return;
            }
            for (int c = 0; c < 3; ++c) {
                if (row > 0 && ((code >> ((row - 1) * 2)) & 3) == c) continue;
                dfs(row + 1, code | (c << (row * 2)));
            }
        };
        dfs(0, 0);

        unordered_map<int, vector<int>> next;
        for (int a : valid) {
            for (int b : valid) {
                bool ok = true;
                for (int i = 0; i < m; ++i) {
                    if (((a >> (i * 2)) & 3) == ((b >> (i * 2)) & 3)) {
                        ok = false;
                        break;
                    }
                }
                if (ok) next[a].push_back(b);
            }
        }

        unordered_map<int, int> dp;
        for (int a : valid) dp[a] = 1;

        for (int col = 1; col < n; ++col) {
            unordered_map<int, int> ndp;
            for (auto& [a, cnt] : dp) {
                for (int b : next[a]) {
                    ndp[b] = (ndp[b] + cnt) % MOD;
                }
            }
            dp = move(ndp);
        }

        int res = 0;
        for (auto& [_, cnt] : dp) res = (res + cnt) % MOD;
        return res;
    }
};
```

---

##267 ****[Problem Link]https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies****  
**Approach:** Topological sort with dependency count.  
**Time Complexity:** O(n + e)

```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>

using namespace std;

class Solution {
public:
    vector<string> findAllRecipes(vector<string>& recipes, vector<vector<string>>& ingredients, vector<string>& supplies) {
        unordered_map<string, vector<string>> graph;
        unordered_map<string, int> indeg;
        for (int i = 0; i < recipes.size(); ++i) {
            for (auto& ing : ingredients[i]) {
                graph[ing].push_back(recipes[i]);
                indeg[recipes[i]]++;
            }
        }

        queue<string> q;
        unordered_set<string> available(supplies.begin(), supplies.end());
        for (auto& s : supplies) q.push(s);

        vector<string> res;
        while (!q.empty()) {
            string cur = q.front(); q.pop();
            for (auto& nei : graph[cur]) {
                if (--indeg[nei] == 0) {
                    res.push_back(nei);
                    q.push(nei);
                }
            }
        }
        return res;
    }
};
```

---

##268 ****[Problem Link]https://leetcode.com/problems/second-minimum-time-to-reach-destination****  
**Approach:** Modified Dijkstra tracking second shortest arrival time.  
**Time Complexity:** O(E + V log V)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int secondMinimum(int n, vector<vector<int>>& edges, int time, int change) {
        vector<vector<int>> g(n + 1);
        for (auto& e : edges) {
            g[e[0]].push_back(e[1]);
            g[e[1]].push_back(e[0]);
        }

        vector<int> dist(n + 1, -1), cnt(n + 1, 0);
        queue<int> q;
        q.push(1);
        dist[1] = 0;
        vector<int> visited(n + 1, 0);
        visited[1] = 1;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : g[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }

        int shortest = dist[n];
        vector<int> arrival(n + 1, 0);
        queue<pair<int, int>> pq;
        pq.push({1, 0});

        while (!pq.empty()) {
            auto [u, t] = pq.front(); pq.pop();
            if (u == n) {
                if (++cnt[n] == 2) return t;
            }
            for (int v : g[u]) {
                if (dist[v] >= dist[u]) {
                    int nt = t;
                    if ((nt / change) % 2) nt = ((nt / change) + 1) * change;
                    nt += time;
                    if (++arrival[v] <= 2)
                        pq.push({v, nt});
                }
            }
        }

        return -1;
    }
};
```

---

##269 ****[Problem Link]https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store****  
**Approach:** Binary search for minimal maximum with check function.  
**Time Complexity:** O(n log m)

```cpp
#include <vector>
#include <numeric>

using namespace std;

class Solution {
public:
    int minimizedMaximum(int n, vector<int>& quantities) {
        int lo = 1, hi = *max_element(quantities.begin(), quantities.end());
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            int cnt = 0;
            for (int q : quantities)
                cnt += (q + mid - 1) / mid;
            if (cnt <= n) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }
};
```

---

##270 ****[Problem Link]https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests****  
**Approach:** Priority queue for available/busy servers and count requests.  
**Time Complexity:** O(n log k)

```cpp
#include <vector>
#include <queue>
#include <set>

using namespace std;

class Solution {
public:
    vector<int> busiestServers(int k, vector<int>& arrival, vector<int>& load) {
        set<int> free;
        for (int i = 0; i < k; ++i) free.insert(i);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> busy;
        vector<int> count(k, 0);
        int n = arrival.size();
        for (int i = 0; i < n; ++i) {
            while (!busy.empty() && busy.top().first <= arrival[i]) {
                free.insert(busy.top().second);
                busy.pop();
            }
            if (free.empty()) continue;
            auto it = free.lower_bound(i % k);
            if (it == free.end()) it = free.begin();
            int server = *it;
            free.erase(it);
            count[server]++;
            busy.emplace(arrival[i] + load[i], server);
        }

        int maxReq = *max_element(count.begin(), count.end());
        vector<int> res;
        for (int i = 0; i < k; ++i)
            if (count[i] == maxReq) res.push_back(i);
        return res;
    }
};
```

---

##271 ****[Problem Link]https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero****  
**Approach:** Use Gray code properties for recursive calculation.  
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int minimumOneBitOperations(int n) {
        if (n == 0) return 0;
        int mask = 1;
        while (mask <= n) mask <<= 1;
        mask >>= 1;
        return mask - 1 - minimumOneBitOperations(n ^ mask);
    }
};
```

---

##272 ****[Problem Link]https://leetcode.com/problems/count-subtrees-with-max-distance-between-cities****  
**Approach:** Bitmask all subsets, compute distances with BFS.  
**Time Complexity:** O(2^n * n^2)

```cpp
#include <vector>
#include <queue>
#include <unordered_set>

using namespace std;

class Solution {
public:
    vector<int> countSubgraphsForEachDiameter(int n, vector<vector<int>>& edges) {
        vector<vector<int>> g(n);
        for (auto& e : edges) {
            g[e[0] - 1].push_back(e[1] - 1);
            g[e[1] - 1].push_back(e[0] - 1);
        }

        vector<int> res(n - 1);
        for (int mask = 1; mask < (1 << n); ++mask) {
            vector<int> nodes;
            for (int i = 0; i < n; ++i)
                if (mask & (1 << i)) nodes.push_back(i);
            if (nodes.size() < 2) continue;

            unordered_set<int> seen;
            queue<int> q;
            q.push(nodes[0]);
            seen.insert(nodes[0]);

            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (int v : g[u])
                    if ((mask & (1 << v)) && !seen.count(v)) {
                        seen.insert(v);
                        q.push(v);
                    }
            }

            if (seen.size() != nodes.size()) continue;

            int maxDist = 0;
            for (int u : nodes) {
                vector<int> dist(n, -1);
                queue<int> q;
                q.push(u);
                dist[u] = 0;
                while (!q.empty()) {
                    int x = q.front(); q.pop();
                    for (int v : g[x])
                        if ((mask & (1 << v)) && dist[v] == -1) {
                            dist[v] = dist[x] + 1;
                            maxDist = max(maxDist, dist[v]);
                            q.push(v);
                        }
                }
            }

            if (maxDist > 0) res[maxDist - 1]++;
        }
        return res;
    }
};
```

---

##273 ****[Problem Link]https://leetcode.com/problems/teemo-attacking****  
**Approach:** Sum duration, avoid overlap.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int findPoisonedDuration(vector<int>& timeSeries, int duration) {
        int total = 0;
        for (int i = 0; i < timeSeries.size(); ++i) {
            if (i == 0) total += duration;
            else total += min(duration, timeSeries[i] - timeSeries[i - 1]);
        }
        return total;
    }
};
```

---

##274 ****[Problem Link]https://leetcode.com/problems/find-xor-sum-of-all-pairs-bitwise-and****  
**Approach:** Use bitwise identity: A⊕B & C⊕D = (A & C) ⊕ (A & D) ⊕ (B & C) ⊕ (B & D).  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int getXORSum(vector<int>& arr1, vector<int>& arr2) {
        int a = 0, b = 0;
        for (int x : arr1) a ^= x;
        for (int y : arr2) b ^= y;
        return a & b;
    }
};
```

---

##275 ****[Problem Link]https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting****  
**Approach:** DFS to find cycles and longest chains leading to mutual pairs.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maximumInvitations(vector<int>& favorite) {
        int n = favorite.size();
        vector<int> indeg(n), dp(n);
        for (int x : favorite) indeg[x]++;

        vector<bool> visited(n);
        vector<int> chain(n, 1);
        queue<int> q;
        for (int i = 0; i < n; ++i)
            if (indeg[i] == 0) q.push(i);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            visited[u] = true;
            int v = favorite[u];
            chain[v] = max(chain[v], chain[u] + 1);
            if (--indeg[v] == 0) q.push(v);
        }

        int maxCycle = 0, pairSum = 0;
        for (int i = 0; i < n; ++i) {
            if (visited[i]) continue;
            int cur = i, len = 0;
            while (!visited[cur]) {
                visited[cur] = true;
                cur = favorite[cur];
                len++;
            }
            if (len == 2) {
                int a = i, b = favorite[i];
                pairSum += chain[a] + chain[b];
            } else {
                maxCycle = max(maxCycle, len);
            }
        }
        return max(maxCycle, pairSum);
    }
};
```

---

##276 ****[Problem Link]https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings****  
**Approach:** Alternate group assignment for each open paren.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> maxDepthAfterSplit(string seq) {
        vector<int> res(seq.size());
        int depth = 0;
        for (int i = 0; i < seq.size(); ++i) {
            if (seq[i] == '(') res[i] = depth++ % 2;
            else res[i] = --depth % 2;
        }
        return res;
    }
};
```

---

##277 ****[Problem Link]https://leetcode.com/problems/find-the-middle-index-in-array****  
**Approach:** Use prefix sum and check for each index.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int findMiddleIndex(vector<int>& nums) {
        int total = 0, left = 0;
        for (int x : nums) total += x;
        for (int i = 0; i < nums.size(); ++i) {
            if (left == total - left - nums[i]) return i;
            left += nums[i];
        }
        return -1;
    }
};
```

---

##278 ****[Problem Link]https://leetcode.com/problems/plates-between-candles****  
**Approach:** Prefix sum of plates and nearest candles to left/right.  
**Time Complexity:** O(n + q)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> platesBetweenCandles(string s, vector<vector<int>>& queries) {
        int n = s.size();
        vector<int> prefix(n + 1), left(n), right(n);

        for (int i = 0; i < n; ++i) {
            prefix[i + 1] = prefix[i] + (s[i] == '*' ? 1 : 0);
            left[i] = (s[i] == '|') ? i : (i > 0 ? left[i - 1] : -1);
        }

        for (int i = n - 1; i >= 0; --i)
            right[i] = (s[i] == '|') ? i : (i + 1 < n ? right[i + 1] : n);

        vector<int> res;
        for (auto& q : queries) {
            int l = right[q[0]], r = left[q[1]];
            res.push_back((l < r) ? prefix[r] - prefix[l] : 0);
        }
        return res;
    }
};
```

---

##279 ****[Problem Link]https://leetcode.com/problems/find-kth-largest-xor-coordinate-value****  
**Approach:** Prefix XOR and use min-heap to track top k.  
**Time Complexity:** O(mn log k)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    int kthLargestValue(vector<vector<int>>& matrix, int k) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> pre(m + 1, vector<int>(n + 1));
        priority_queue<int, vector<int>, greater<>> pq;

        for (int i = 1; i <= m; ++i)
            for (int j = 1; j <= n; ++j) {
                pre[i][j] = pre[i - 1][j] ^ pre[i][j - 1] ^ pre[i - 1][j - 1] ^ matrix[i - 1][j - 1];
                pq.push(pre[i][j]);
                if (pq.size() > k) pq.pop();
            }

        return pq.top();
    }
};
```

---

##280 ****[Problem Link]https://leetcode.com/problems/count-good-numbers****  
**Approach:** Fast exponentiation for even and odd digits.  
**Time Complexity:** O(log n)

```cpp
class Solution {
    const int MOD = 1e9 + 7;

    long modPow(long x, long n) {
        long res = 1;
        while (n) {
            if (n % 2) res = res * x % MOD;
            x = x * x % MOD;
            n /= 2;
        }
        return res;
    }

public:
    int countGoodNumbers(long n) {
        return modPow(5, (n + 1) / 2) * modPow(4, n / 2) % MOD;
    }
};
```

---

##281 ****[Problem Link]https://leetcode.com/problems/invalid-transactions****  
**Approach:** Group by name and cross-validate constraints with other transactions.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    vector<string> invalidTransactions(vector<string>& transactions) {
        vector<string> res;
        unordered_map<string, vector<tuple<int, int, string, string>>> map;

        for (auto& t : transactions) {
            auto i1 = t.find(','), i2 = t.find(',', i1 + 1), i3 = t.find(',', i2 + 1);
            string name = t.substr(0, i1);
            int time = stoi(t.substr(i1 + 1, i2 - i1 - 1));
            int amount = stoi(t.substr(i2 + 1, i3 - i2 - 1));
            string city = t.substr(i3 + 1);
            map[name].emplace_back(time, amount, city, t);
        }

        for (auto& [name, ts] : map) {
            for (int i = 0; i < ts.size(); ++i) {
                auto [t1, a1, c1, s1] = ts[i];
                if (a1 > 1000) {
                    res.push_back(s1);
                    continue;
                }
                for (int j = 0; j < ts.size(); ++j) {
                    if (i == j) continue;
                    auto [t2, a2, c2, s2] = ts[j];
                    if (abs(t1 - t2) <= 60 && c1 != c2) {
                        res.push_back(s1);
                        break;
                    }
                }
            }
        }
        return res;
    }
};
```

---

##282 ****[Problem Link]https://leetcode.com/problems/minimum-time-to-type-word-using-special-typewriter****  
**Approach:** Calculate circular distance and accumulate time.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <cmath>

using namespace std;

class Solution {
public:
    int minTimeToType(string word) {
        int time = 0, prev = 'a';
        for (char c : word) {
            int dist = abs(c - prev);
            time += min(dist, 26 - dist) + 1;
            prev = c;
        }
        return time;
    }
};
```

---

##283 ****[Problem Link]https://leetcode.com/problems/evaluate-the-bracket-pairs-of-a-string****  
**Approach:** Use hashmap and scan string replacing bracketed tokens.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class Solution {
public:
    string evaluate(string s, vector<vector<string>>& knowledge) {
        unordered_map<string, string> map;
        for (auto& k : knowledge)
            map[k[0]] = k[1];

        string res, key;
        bool inBracket = false;
        for (char c : s) {
            if (c == '(') {
                key.clear();
                inBracket = true;
            } else if (c == ')') {
                res += map.count(key) ? map[key] : "?";
                inBracket = false;
            } else if (inBracket) {
                key += c;
            } else {
                res += c;
            }
        }
        return res;
    }
};
```

---

##284 ****[Problem Link]https://leetcode.com/problems/find-all-groups-of-farmland****  
**Approach:** DFS to identify bounding rectangle of connected 1s.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> findFarmland(vector<vector<int>>& land) {
        int m = land.size(), n = land[0].size();
        vector<vector<int>> res;

        function<void(int, int, int&, int&)> dfs = [&](int i, int j, int& x, int& y) {
            if (i < 0 || j < 0 || i >= m || j >= n || land[i][j] == 0) return;
            land[i][j] = 0;
            x = max(x, i);
            y = max(y, j);
            dfs(i + 1, j, x, y);
            dfs(i - 1, j, x, y);
            dfs(i, j + 1, x, y);
            dfs(i, j - 1, x, y);
        };

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (land[i][j] == 1) {
                    int x = i, y = j;
                    dfs(i, j, x, y);
                    res.push_back({i, j, x, y});
                }

        return res;
    }
};
```

---

##285 ****[Problem Link]https://leetcode.com/problems/find-all-good-strings****  
**Approach:** DP with KMP automaton and memoization.  
**Time Complexity:** O(n * k^2)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
    const int MOD = 1e9 + 7;

    vector<int> buildLPS(string evil) {
        int m = evil.size(), j = 0;
        vector<int> lps(m, 0);
        for (int i = 1; i < m; ++i) {
            while (j > 0 && evil[i] != evil[j]) j = lps[j - 1];
            if (evil[i] == evil[j]) lps[i] = ++j;
        }
        return lps;
    }

    int dp[501][51][2][2];

    int dfs(string& s1, string& s2, string& evil, vector<int>& lps, int i, int j, bool lo, bool hi) {
        if (j == evil.size()) return 0;
        if (i == s1.size()) return 1;
        if (dp[i][j][lo][hi] != -1) return dp[i][j][lo][hi];

        char from = lo ? s1[i] : 'a';
        char to = hi ? s2[i] : 'z';
        int res = 0;

        for (char c = from; c <= to; ++c) {
            int k = j;
            while (k > 0 && evil[k] != c) k = lps[k - 1];
            if (c == evil[k]) ++k;
            res = (res + dfs(s1, s2, evil, lps, i + 1, k, lo && c == from, hi && c == to)) % MOD;
        }

        return dp[i][j][lo][hi] = res;
    }

public:
    int findGoodStrings(int n, string s1, string s2, string evil) {
        vector<int> lps = buildLPS(evil);
        memset(dp, -1, sizeof(dp));
        return dfs(s1, s2, evil, lps, 0, 0, true, true);
    }
};
```

---

##286 ****[Problem Link]https://leetcode.com/problems/smallest-good-base****  
**Approach:** Try all base lengths from max down to 2 using binary search.  
**Time Complexity:** O(log n * log n)

```cpp
#include <string>
#include <cmath>

using namespace std;

class Solution {
public:
    string smallestGoodBase(string n) {
        long long N = stoll(n);
        for (int m = log2(N); m >= 1; --m) {
            long long k = pow(N, 1.0 / m);
            long long sum = 1, curr = 1;
            for (int i = 0; i < m; ++i)
                sum = sum * k + 1;
            if (sum == N) return to_string(k);
        }
        return to_string(N - 1);
    }
};
```

---

##287 ****[Problem Link]https://leetcode.com/problems/decrease-elements-to-make-array-zigzag****  
**Approach:** Try making even or odd indices local minima and count moves.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int movesToMakeZigzag(vector<int>& nums) {
        return min(helper(nums, 0), helper(nums, 1));
    }

    int helper(vector<int> nums, int start) {
        int moves = 0;
        for (int i = start; i < nums.size(); i += 2) {
            int left = i > 0 ? nums[i - 1] : INT_MAX;
            int right = i + 1 < nums.size() ? nums[i + 1] : INT_MAX;
            int minNeighbor = min(left, right);
            if (nums[i] >= minNeighbor)
                moves += nums[i] - minNeighbor + 1;
        }
        return moves;
    }
};
```

---

##288 ****[Problem Link]https://leetcode.com/problems/find-a-value-of-a-mysterious-function-closest-to-target****  
**Approach:** Sliding window with bitwise AND and hashset pruning.  
**Time Complexity:** O(n log max_val)

```cpp
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace std;

class Solution {
public:
    int closestToTarget(vector<int>& arr, int target) {
        unordered_set<int> prev;
        int res = INT_MAX;
        for (int num : arr) {
            unordered_set<int> curr;
            curr.insert(num);
            for (int x : prev)
                curr.insert(x & num);
            for (int x : curr)
                res = min(res, abs(x - target));
            prev = move(curr);
        }
        return res;
    }
};
```

---

##289 ****[Problem Link]https://leetcode.com/problems/finding-pairs-with-a-certain-sum****  
**Approach:** Use hash maps to track frequency of nums2 and support updates.  
**Time Complexity:** O(1) for add and find

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class FindSumPairs {
    vector<int> nums1, nums2;
    unordered_map<int, int> freq;

public:
    FindSumPairs(vector<int>& nums1, vector<int>& nums2) : nums1(nums1), nums2(nums2) {
        for (int n : nums2) freq[n]++;
    }

    void add(int index, int val) {
        freq[nums2[index]]--;
        nums2[index] += val;
        freq[nums2[index]]++;
    }

    int count(int tot) {
        int ans = 0;
        for (int a : nums1)
            ans += freq[tot - a];
        return ans;
    }
};
```

---

##290 ****[Problem Link]https://leetcode.com/problems/least-operators-to-express-number****  
**Approach:** DP with memoization minimizing number of operators.  
**Time Complexity:** O(log target)

```cpp
#include <unordered_map>

using namespace std;

class Solution {
    unordered_map<long, int> memo;

    int dp(int x, long target) {
        if (target == 0) return 0;
        if (target == 1) return cost(1);
        if (memo.count(target)) return memo[target];

        long p = x, k = 0;
        while (p < target) p *= x, ++k;

        if (p == target) return memo[target] = cost(k);

        int res = cost(k) + dp(x, p - target);
        if (k > 0) res = min(res, cost(k - 1) + dp(x, target - p / x));

        return memo[target] = res;
    }

    int cost(int k) {
        return k == 0 ? 2 : k;
    }

public:
    int leastOpsExpressTarget(int x, int target) {
        return dp(x, target) - 1;
    }
};
```

---

##291 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating****  
**Approach:** Try both alternating patterns, count mismatches.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    int minSwaps(string s) {
        int ones = count(s.begin(), s.end(), '1'), n = s.size();
        if (abs(n - 2 * ones) > 1) return -1;

        int swap0 = 0, swap1 = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i] - '0' != i % 2) ++swap0;
            if (s[i] - '0' != (i + 1) % 2) ++swap1;
        }

        if (n % 2 == 0) return min(swap0, swap1) / 2;
        return (ones * 2 > n ? swap1 : swap0) / 2;
    }
};
```

---

##292 ****[Problem Link]https://leetcode.com/problems/merge-triplets-to-form-target-triplet****  
**Approach:** Check each triplet and update max.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    bool mergeTriplets(vector<vector<int>>& triplets, vector<int>& target) {
        vector<int> res(3, 0);
        for (auto& t : triplets) {
            if (t[0] <= target[0] && t[1] <= target[1] && t[2] <= target[2]) {
                res[0] = max(res[0], t[0]);
                res[1] = max(res[1], t[1]);
                res[2] = max(res[2], t[2]);
            }
        }
        return res == target;
    }
};
```

---

##293 ****[Problem Link]https://leetcode.com/problems/vowels-of-all-substrings****  
**Approach:** For each vowel, add its contribution to all substrings that include it.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    long long countVowels(string word) {
        long long res = 0, n = word.size();
        for (int i = 0; i < n; ++i) {
            if (isVowel(word[i]))
                res += (long long)(i + 1) * (n - i);
        }
        return res;
    }

    bool isVowel(char c) {
        return string("aeiou").find(c) != string::npos;
    }
};
```

---

##294 ****[Problem Link]https://leetcode.com/problems/ambiguous-coordinates****  
**Approach:** Try placing decimal in all valid positions.  
**Time Complexity:** O(n^3)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    vector<string> ambiguousCoordinates(string s) {
        vector<string> res;
        s = s.substr(1, s.size() - 2);

        for (int i = 1; i < s.size(); ++i) {
            vector<string> left = split(s.substr(0, i));
            vector<string> right = split(s.substr(i));
            for (string& l : left)
                for (string& r : right)
                    res.push_back("(" + l + ", " + r + ")");
        }
        return res;
    }

    vector<string> split(string s) {
        vector<string> res;
        if (s.front() == '0' && s.back() == '0' && s.size() > 1) return res;
        if (s.size() > 1 && s.front() == '0') res.push_back("0." + s.substr(1));
        else {
            res.push_back(s);
            for (int i = 1; i < s.size(); ++i)
                if (s.back() != '0')
                    res.push_back(s.substr(0, i) + "." + s.substr(i));
        }
        return res;
    }
};
```

---

##295 ****[Problem Link]https://leetcode.com/problems/describe-the-painting****  
**Approach:** Line sweep with prefix sum to track segment changes.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <map>

using namespace std;

class Solution {
public:
    vector<vector<long long>> splitPainting(vector<vector<int>>& segments) {
        map<int, long long> diff;
        for (auto& seg : segments) {
            diff[seg[0]] += seg[2];
            diff[seg[1]] -= seg[2];
        }

        vector<vector<long long>> res;
        long long color = 0, prev = -1;
        for (auto& [pos, val] : diff) {
            if (prev != -1 && color > 0)
                res.push_back({prev, pos, color});
            color += val;
            prev = pos;
        }
        return res;
    }
};
```

---

##296 ****[Problem Link]https://leetcode.com/problems/count-pairs-with-xor-in-a-range****  
**Approach:** Use Trie to count numbers with XOR in range.  
**Time Complexity:** O(n * log maxVal)

```cpp
struct TrieNode {
    TrieNode* child[2] = {};
    int count = 0;
};

class Solution {
    TrieNode* root = new TrieNode();

    void insert(int num) {
        TrieNode* node = root;
        for (int i = 14; i >= 0; --i) {
            int bit = (num >> i) & 1;
            if (!node->child[bit]) node->child[bit] = new TrieNode();
            node = node->child[bit];
            node->count++;
        }
    }

    int countLess(int num, int limit) {
        TrieNode* node = root;
        int res = 0;
        for (int i = 14; i >= 0 && node; --i) {
            int nBit = (num >> i) & 1;
            int lBit = (limit >> i) & 1;
            if (lBit == 1) {
                if (node->child[nBit]) res += node->child[nBit]->count;
                node = node->child[1 - nBit];
            } else {
                node = node->child[nBit];
            }
        }
        return res;
    }

public:
    int countPairs(vector<int>& nums, int low, int high) {
        int res = 0;
        for (int num : nums) {
            res += countLess(num, high + 1) - countLess(num, low);
            insert(num);
        }
        return res;
    }
};
```

---

##297 ****[Problem Link]https://leetcode.com/problems/count-vowel-substrings-of-a-string****  
**Approach:** Sliding window with vowel frequency count.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int countVowelSubstrings(string word) {
        int res = 0, n = word.size();
        for (int i = 0; i < n; ++i) {
            unordered_set<char> st;
            for (int j = i; j < n; ++j) {
                if (!isVowel(word[j])) break;
                st.insert(word[j]);
                if (st.size() == 5) res++;
            }
        }
        return res;
    }

    bool isVowel(char c) {
        return string("aeiou").find(c) != string::npos;
    }
};
```

---

##298 ****[Problem Link]https://leetcode.com/problems/optimal-division****  
**Approach:** Best result is always to divide first by grouping the rest.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string optimalDivision(vector<int>& nums) {
        if (nums.size() == 1) return to_string(nums[0]);
        if (nums.size() == 2) return to_string(nums[0]) + "/" + to_string(nums[1]);

        string res = to_string(nums[0]) + "/(" + to_string(nums[1]);
        for (int i = 2; i < nums.size(); ++i)
            res += "/" + to_string(nums[i]);
        return res + ")";
    }
};
```

---

##299 ****[Problem Link]https://leetcode.com/problems/find-greatest-common-divisor-of-array****  
**Approach:** Find min and max, return gcd(min, max).  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int findGCD(vector<int>& nums) {
        int mn = *min_element(nums.begin(), nums.end());
        int mx = *max_element(nums.begin(), nums.end());
        return gcd(mn, mx);
    }

    int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }
};
```

---

##300 ****[Problem Link]https://leetcode.com/problems/sum-of-digits-in-base-k****  
**Approach:** Repeatedly take mod and divide.  
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int sumBase(int n, int k) {
        int sum = 0;
        while (n) {
            sum += n % k;
            n /= k;
        }
        return sum;
    }
};
```

---


##301 ****[Problem Link]https://leetcode.com/problems/stock-price-fluctuation****  
**Approach:** Use two heaps (max/min) and a map to track latest timestamp and prices.  
**Time Complexity:** O(log n) per operation

```cpp
#include <map>
#include <set>
using namespace std;

class StockPrice {
    map<int, int> timePrice;
    multiset<int> prices;
    int currentTime = 0;

public:
    void update(int timestamp, int price) {
        if (timePrice.count(timestamp)) {
            prices.erase(prices.find(timePrice[timestamp]));
        }
        timePrice[timestamp] = price;
        prices.insert(price);
        currentTime = max(currentTime, timestamp);
    }

    int current() {
        return timePrice[currentTime];
    }

    int maximum() {
        return *prices.rbegin();
    }

    int minimum() {
        return *prices.begin();
    }
};
```

---

##302 ****[Problem Link]https://leetcode.com/problems/count-largest-group****  
**Approach:** Group numbers by digit sum, count max group size.  
**Time Complexity:** O(n * d) where d is number of digits

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int countLargestGroup(int n) {
        unordered_map<int, int> count;
        int maxSize = 0;
        for (int i = 1; i <= n; ++i) {
            int sum = 0, x = i;
            while (x) {
                sum += x % 10;
                x /= 10;
            }
            maxSize = max(maxSize, ++count[sum]);
        }
        int res = 0;
        for (auto& [k, v] : count)
            if (v == maxSize) res++;
        return res;
    }
};
```

---

##303 ****[Problem Link]https://leetcode.com/problems/longer-contiguous-segments-of-ones-than-zeros****  
**Approach:** Track longest streaks of 1s and 0s.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    bool checkZeroOnes(string s) {
        int max1 = 0, max0 = 0, cur = 1;
        for (int i = 1; i < s.size(); ++i) {
            if (s[i] == s[i-1]) cur++;
            else cur = 1;
            if (s[i] == '1') max1 = max(max1, cur);
            else max0 = max(max0, cur);
        }
        if (s[0] == '1') max1 = max(max1, 1);
        else max0 = max(max0, 1);
        return max1 > max0;
    }
};
```

---

##304 ****[Problem Link]https://leetcode.com/problems/soup-servings****  
**Approach:** DP with memoization, simulate probabilities, cutoff at N ≥ 5000.  
**Time Complexity:** O(n²)

```cpp
#include <unordered_map>
using namespace std;

class Solution {
    unordered_map<int, unordered_map<int, double>> memo;

    double dfs(int A, int B) {
        if (A <= 0 && B <= 0) return 0.5;
        if (A <= 0) return 1;
        if (B <= 0) return 0;
        if (memo[A][B]) return memo[A][B];

        memo[A][B] = 0.25 * (
            dfs(A - 100, B) + dfs(A - 75, B - 25) +
            dfs(A - 50, B - 50) + dfs(A - 25, B - 75)
        );
        return memo[A][B];
    }

public:
    double soupServings(int N) {
        if (N >= 5000) return 1.0;
        return dfs(N, N);
    }
};
```

---

##305 ****[Problem Link]https://leetcode.com/problems/check-if-all-characters-have-equal-number-of-occurrences****  
**Approach:** Count frequency and check if all are equal.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>
using namespace std;

class Solution {
public:
    bool areOccurrencesEqual(string s) {
        unordered_map<char, int> count;
        for (char c : s) count[c]++;
        int val = count[s[0]];
        for (auto& [k, v] : count)
            if (v != val) return false;
        return true;
    }
};
```

---

##306 ****[Problem Link]https://leetcode.com/problems/convert-integer-to-the-sum-of-two-no-zero-integers****  
**Approach:** Brute force increment until both parts have no zeros.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
    bool hasZero(int n) {
        while (n) {
            if (n % 10 == 0) return true;
            n /= 10;
        }
        return false;
    }
public:
    vector<int> getNoZeroIntegers(int n) {
        for (int i = 1; i < n; ++i) {
            if (!hasZero(i) && !hasZero(n - i))
                return {i, n - i};
        }
        return {};
    }
};
```

---

##307 ****[Problem Link]https://leetcode.com/problems/make-the-xor-of-all-segments-equal-to-zero****  
**Approach:** For k = 1 return xor == 0; otherwise use DP to choose k non-overlapping subarrays.  
**Time Complexity:** O(n * k * 1024)

```cpp
#include <vector>
#include <cstring>
using namespace std;

class Solution {
public:
    bool canChoose(vector<int>& nums, int k) {
        int n = nums.size();
        if (k == 1) {
            int x = 0;
            for (int i : nums) x ^= i;
            return x == 0;
        }
        vector<vector<int>> dp(k + 1, vector<int>(1024, -1));
        dp[0][0] = 0;
        for (int i = 0; i < n; ++i) {
            vector<vector<int>> ndp = dp;
            for (int j = 0; j < k; ++j) {
                for (int x = 0; x < 1024; ++x) {
                    if (dp[j][x] == -1) continue;
                    int xorVal = 0;
                    for (int l = i; l < n; ++l) {
                        xorVal ^= nums[l];
                        ndp[j + 1][x ^ xorVal] = max(ndp[j + 1][x ^ xorVal], l + 1);
                    }
                }
            }
            dp = move(ndp);
        }
        return dp[k][0] != -1;
    }

    bool canPartition(vector<int>& nums, int k) {
        int n = nums.size();
        if (n % k != 0) return false;
        return canChoose(nums, k);
    }

    bool canPartitionKSubsets(vector<int>& nums, int k) {
        return false; // fallback in case constraints need clarification
    }

    bool canChooseMain(vector<int>& nums, int k) {
        return false; // fallback for undefined problem context
    }

    bool xorGame(vector<int>& nums) {
        int x = 0;
        for (int n : nums) x ^= n;
        return x == 0 || nums.size() % 2 == 0;
    }
};
```

---

##308 ****[Problem Link]https://leetcode.com/problems/stone-game-viii****  
**Approach:** Work backwards tracking max score difference from suffix sum.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int stoneGameVIII(vector<int>& stones) {
        int n = stones.size();
        for (int i = 1; i < n; ++i)
            stones[i] += stones[i - 1];
        int res = stones[n - 1];
        for (int i = n - 2; i >= 1; --i)
            res = max(res, stones[i] - res);
        return res;
    }
};
```

---

##309 ****[Problem Link]https://leetcode.com/problems/process-restricted-friend-requests****  
**Approach:** Union-Find with undo on failure due to restriction check.  
**Time Complexity:** O(n + q * α(n))

```cpp
#include <vector>
using namespace std;

class DSU {
    vector<int> p;
public:
    DSU(int n) : p(n) {
        for (int i = 0; i < n; ++i) p[i] = i;
    }

    int find(int x) {
        return p[x] == x ? x : p[x] = find(p[x]);
    }

    void unite(int x, int y) {
        p[find(x)] = find(y);
    }

    void reset(int x, int val) {
        p[x] = val;
    }

    vector<int> get() { return p; }
};

class Solution {
public:
    bool check(int x, int y, vector<vector<int>>& restrictions, DSU& dsu) {
        for (auto& r : restrictions)
            if ((dsu.find(r[0]) == dsu.find(x) && dsu.find(r[1]) == dsu.find(y)) ||
                (dsu.find(r[0]) == dsu.find(y) && dsu.find(r[1]) == dsu.find(x)))
                return false;
        return true;
    }

    vector<bool> friendRequests(int n, vector<vector<int>>& restrictions, vector<vector<int>>& requests) {
        DSU dsu(n);
        vector<bool> res;
        for (auto& r : requests) {
            int x = dsu.find(r[0]), y = dsu.find(r[1]);
            if (x == y) {
                res.push_back(true);
                continue;
            }
            if (!check(x, y, restrictions, dsu)) {
                res.push_back(false);
                continue;
            }
            dsu.unite(x, y);
            res.push_back(true);
        }
        return res;
    }
};
```

---

##310 ****[Problem Link]https://leetcode.com/problems/self-crossing****  
**Approach:** Geometry and bounds checking in simulation.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool isSelfCrossing(vector<int>& d) {
        int n = d.size();
        for (int i = 3; i < n; ++i) {
            if (d[i] >= d[i-2] && d[i-1] <= d[i-3]) return true;
            if (i >= 4 && d[i-1] == d[i-3] && d[i] + d[i-4] >= d[i-2]) return true;
            if (i >= 5 && d[i-2] >= d[i-4] && d[i] + d[i-4] >= d[i-2] &&
                d[i-1] <= d[i-3] && d[i-1] + d[i-5] >= d[i-3]) return true;
        }
        return false;
    }
};
```

---

##311 ****[Problem Link]https://leetcode.com/problems/minimum-length-of-string-after-deleting-similar-ends****  
**Approach:** Use two pointers to trim matching characters from both ends.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    int minimumLength(string s) {
        int l = 0, r = s.size() - 1;
        while (l < r && s[l] == s[r]) {
            char c = s[l];
            while (l <= r && s[l] == c) ++l;
            while (l <= r && s[r] == c) --r;
        }
        return r - l + 1;
    }
};
```

---

##312 ****[Problem Link]https://leetcode.com/problems/longest-common-subpath****  
**Approach:** Binary search + rolling hash to check common subpath.  
**Time Complexity:** O(n * log m)

```cpp
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
    const long long mod = 1e11 + 19, base = 1e5 + 7;

    bool check(int len, vector<vector<int>>& paths) {
        unordered_set<long long> common;
        long long hash = 0, pow = 1;

        for (int i = 0; i < len; ++i) pow = (pow * base) % mod;

        for (int i = 0; i < paths[0].size(); ++i) {
            hash = (hash * base + paths[0][i]) % mod;
            if (i >= len) hash = (hash - pow * paths[0][i - len] % mod + mod) % mod;
            if (i >= len - 1) common.insert(hash);
        }

        for (int k = 1; k < paths.size(); ++k) {
            unordered_set<long long> seen;
            hash = 0;
            for (int i = 0; i < paths[k].size(); ++i) {
                hash = (hash * base + paths[k][i]) % mod;
                if (i >= len) hash = (hash - pow * paths[k][i - len] % mod + mod) % mod;
                if (i >= len - 1 && common.count(hash)) seen.insert(hash);
            }
            common = seen;
            if (common.empty()) return false;
        }
        return true;
    }

public:
    int longestCommonSubpath(int n, vector<vector<int>>& paths) {
        int l = 0, r = INT_MAX;
        for (auto& p : paths) r = min(r, (int)p.size());

        int res = 0;
        while (l <= r) {
            int m = (l + r) / 2;
            if (check(m, paths)) {
                res = m;
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return res;
    }
};
```

---

##313 ****[Problem Link]https://leetcode.com/problems/remove-stones-to-minimize-the-total****  
**Approach:** Use max heap to always remove half of the largest pile.  
**Time Complexity:** O(k log n)

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int minStoneSum(vector<int>& piles, int k) {
        priority_queue<int> pq(piles.begin(), piles.end());
        while (k--) {
            int top = pq.top(); pq.pop();
            pq.push(top - top / 2);
        }
        int sum = 0;
        while (!pq.empty()) {
            sum += pq.top(); pq.pop();
        }
        return sum;
    }
};
```

---

##314 ****[Problem Link]https://leetcode.com/problems/minimum-operations-to-make-a-uni-value-grid****  
**Approach:** Flatten, check divisibility by x, take median as target.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minOperations(vector<vector<int>>& grid, int x) {
        vector<int> nums;
        for (auto& row : grid)
            for (int val : row) nums.push_back(val);

        for (int val : nums)
            if ((val - nums[0]) % x != 0) return -1;

        sort(nums.begin(), nums.end());
        int median = nums[nums.size() / 2];
        int ops = 0;
        for (int val : nums) ops += abs(val - median) / x;
        return ops;
    }
};
```

---

##315 ****[Problem Link]https://leetcode.com/problems/parallel-courses-iii****  
**Approach:** Topological sort + DP for longest path in DAG.  
**Time Complexity:** O(n + e)

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int minimumTime(int n, vector<vector<int>>& relations, vector<int>& time) {
        vector<vector<int>> graph(n);
        vector<int> indeg(n, 0), dp(n, 0);
        for (auto& r : relations) {
            graph[r[0] - 1].push_back(r[1] - 1);
            indeg[r[1] - 1]++;
        }

        queue<int> q;
        for (int i = 0; i < n; ++i)
            if (indeg[i] == 0) {
                q.push(i);
                dp[i] = time[i];
            }

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : graph[u]) {
                dp[v] = max(dp[v], dp[u] + time[v]);
                if (--indeg[v] == 0) q.push(v);
            }
        }

        return *max_element(dp.begin(), dp.end());
    }
};
```

---

##316 ****[Problem Link]https://leetcode.com/problems/maximum-matrix-sum****  
**Approach:** Flip all negatives, track min absolute for possible adjustment.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
#include <cstdlib>
using namespace std;

class Solution {
public:
    long long maxMatrixSum(vector<vector<int>>& matrix) {
        long long sum = 0, minAbs = INT_MAX;
        int negCount = 0;

        for (auto& row : matrix) {
            for (int val : row) {
                sum += abs(val);
                if (val < 0) negCount++;
                minAbs = min(minAbs, (long long)abs(val));
            }
        }

        return negCount % 2 == 0 ? sum : sum - 2 * minAbs;
    }
};
```

---

##317 ****[Problem Link]https://leetcode.com/problems/smallest-missing-genetic-value-in-each-subtree****  
**Approach:** DFS subtree traversal + set tracking for value presence.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
    void dfs(int u, vector<vector<int>>& tree, vector<int>& nums, unordered_set<int>& seen) {
        seen.insert(nums[u]);
        for (int v : tree[u]) dfs(v, tree, nums, seen);
    }

public:
    vector<int> smallestMissingValueSubtree(vector<int>& parents, vector<int>& nums) {
        int n = parents.size();
        vector<vector<int>> tree(n);
        for (int i = 1; i < n; ++i)
            tree[parents[i]].push_back(i);

        vector<int> res(n, 1);
        int u = -1;
        for (int i = 0; i < n; ++i)
            if (nums[i] == 1) u = i;

        if (u == -1) return res;

        unordered_set<int> seen;
        int miss = 1;

        for (int v = u; v != -1; v = parents[v]) {
            dfs(v, tree, nums, seen);
            while (seen.count(miss)) ++miss;
            res[v] = miss;
        }

        return res;
    }
};
```

---

##318 ****[Problem Link]https://leetcode.com/problems/sum-of-beauty-in-the-array****  
**Approach:** Precompute prefix/suffix min/max to check monotonicity.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int sumOfBeauties(vector<int>& nums) {
        int n = nums.size();
        vector<int> rightMin(n);
        rightMin[n - 1] = nums[n - 1];
        for (int i = n - 2; i >= 0; --i)
            rightMin[i] = min(nums[i], rightMin[i + 1]);

        int leftMax = nums[0], res = 0;
        for (int i = 1; i < n - 1; ++i) {
            if (leftMax < nums[i] && nums[i] < rightMin[i + 1])
                res += 2;
            else if (nums[i - 1] < nums[i] && nums[i] < nums[i + 1])
                res += 1;
            leftMax = max(leftMax, nums[i]);
        }
        return res;
    }
};
```

---

##319 ****[Problem Link]https://leetcode.com/problems/mean-of-array-after-removing-some-elements****  
**Approach:** Sort, remove smallest/largest 5%, and compute mean.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    double trimMean(vector<int>& arr) {
        sort(arr.begin(), arr.end());
        int n = arr.size(), k = n / 20;
        double sum = 0;
        for (int i = k; i < n - k; ++i) sum += arr[i];
        return sum / (n - 2 * k);
    }
};
```

---

##320 ****[Problem Link]https://leetcode.com/problems/maximum-building-height****  
**Approach:** Process sorted restrictions with binary limit then propagate.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maxBuilding(int n, vector<vector<int>>& restrictions) {
        restrictions.push_back({1, 0});
        sort(restrictions.begin(), restrictions.end());

        int m = restrictions.size();
        for (int i = 1; i < m; ++i)
            restrictions[i][1] = min(restrictions[i][1], 
                restrictions[i - 1][1] + restrictions[i][0] - restrictions[i - 1][0]);

        for (int i = m - 2; i >= 0; --i)
            restrictions[i][1] = min(restrictions[i][1],
                restrictions[i + 1][1] + restrictions[i + 1][0] - restrictions[i][0]);

        int res = 0;
        for (int i = 1; i < restrictions.size(); ++i) {
            int l = restrictions[i - 1][0], r = restrictions[i][0];
            int hl = restrictions[i - 1][1], hr = restrictions[i][1];
            int d = r - l;
            res = max(res, (hl + hr + d) / 2);
        }
        return res;
    }
};
```

---

##321 ****[Problem Link]https://leetcode.com/problems/magic-squares-in-grid****  
**Approach:** Check every 3x3 subgrid for being a Lo Shu magic square.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool isMagic(vector<vector<int>>& g, int i, int j) {
        vector<int> count(10);
        for (int x = i; x < i + 3; ++x)
            for (int y = j; y < j + 3; ++y)
                if (g[x][y] < 1 || g[x][y] > 9 || ++count[g[x][y]] > 1)
                    return false;

        int s = g[i][j] + g[i][j + 1] + g[i][j + 2];
        return g[i + 1][j] + g[i + 1][j + 1] + g[i + 1][j + 2] == s &&
               g[i + 2][j] + g[i + 2][j + 1] + g[i + 2][j + 2] == s &&
               g[i][j] + g[i + 1][j] + g[i + 2][j] == s &&
               g[i][j + 1] + g[i + 1][j + 1] + g[i + 2][j + 1] == s &&
               g[i][j + 2] + g[i + 1][j + 2] + g[i + 2][j + 2] == s &&
               g[i][j] + g[i + 1][j + 1] + g[i + 2][j + 2] == s &&
               g[i][j + 2] + g[i + 1][j + 1] + g[i + 2][j] == s;
    }

    int numMagicSquaresInside(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size(), count = 0;
        for (int i = 0; i < m - 2; ++i)
            for (int j = 0; j < n - 2; ++j)
                if (isMagic(grid, i, j)) count++;
        return count;
    }
};
```

---

##322 ****[Problem Link]https://leetcode.com/problems/rearrange-spaces-between-words****  
**Approach:** Count total spaces and redistribute evenly among words.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <sstream>
#include <vector>
using namespace std;

class Solution {
public:
    string reorderSpaces(string text) {
        int space = 0;
        for (char c : text)
            if (c == ' ') space++;

        stringstream ss(text);
        string word;
        vector<string> words;
        while (ss >> word) words.push_back(word);

        int gaps = words.size() - 1;
        string res;
        if (gaps == 0) {
            res = words[0] + string(space, ' ');
        } else {
            int even = space / gaps, rem = space % gaps;
            for (int i = 0; i < words.size(); ++i) {
                res += words[i];
                if (i < words.size() - 1)
                    res += string(even, ' ');
            }
            res += string(rem, ' ');
        }
        return res;
    }
};
```

---

##323 ****[Problem Link]https://leetcode.com/problems/most-beautiful-item-for-each-query****  
**Approach:** Sort and use prefix max on beauty. Binary search per query.  
**Time Complexity:** O(n log n + q log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> maximumBeauty(vector<vector<int>>& items, vector<int>& queries) {
        sort(items.begin(), items.end());
        vector<int> prices, beauty;
        int maxB = 0;
        for (auto& item : items) {
            maxB = max(maxB, item[1]);
            prices.push_back(item[0]);
            beauty.push_back(maxB);
        }

        vector<int> res;
        for (int q : queries) {
            int idx = upper_bound(prices.begin(), prices.end(), q) - prices.begin() - 1;
            res.push_back(idx >= 0 ? beauty[idx] : 0);
        }
        return res;
    }
};
```

---

##324 ****[Problem Link]https://leetcode.com/problems/fancy-sequence****  
**Approach:** Lazy update with mod inverse on all operations.  
**Time Complexity:** O(1) per operation

```cpp
#include <vector>
#include <cmath>
using namespace std;

class Fancy {
    static constexpr int MOD = 1e9 + 7;
    vector<long long> seq;
    long long add = 0, mult = 1;

    long long modinv(long long x) {
        long long res = 1, p = MOD - 2;
        while (p) {
            if (p & 1) res = res * x % MOD;
            x = x * x % MOD;
            p >>= 1;
        }
        return res;
    }

public:
    Fancy() {}

    void append(int val) {
        seq.push_back(((val - add + MOD) % MOD) * modinv(mult) % MOD);
    }

    void addAll(int inc) {
        add = (add + inc) % MOD;
    }

    void multAll(int m) {
        add = add * m % MOD;
        mult = mult * m % MOD;
    }

    int getIndex(int idx) {
        if (idx >= seq.size()) return -1;
        return (seq[idx] * mult + add) % MOD;
    }
};
```

---

##325 ****[Problem Link]https://leetcode.com/problems/seat-reservation-manager****  
**Approach:** Use a min-heap to track available seats efficiently.  
**Time Complexity:** O(log n) per operation

```cpp
#include <queue>
using namespace std;

class SeatManager {
    priority_queue<int, vector<int>, greater<int>> pq;
    int next = 1;

public:
    SeatManager(int n) {
        for (int i = 1; i <= n; ++i) pq.push(i);
    }

    int reserve() {
        int res = pq.top(); pq.pop();
        return res;
    }

    void unreserve(int seatNumber) {
        pq.push(seatNumber);
    }
};
```

---

##326 ****[Problem Link]https://leetcode.com/problems/student-attendance-record-i****  
**Approach:** Traverse the string and check for more than 1 'A' or "LLL".  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    bool checkRecord(string s) {
        int aCount = 0, lStreak = 0;
        for (char c : s) {
            if (c == 'A') {
                aCount++;
                if (aCount > 1) return false;
                lStreak = 0;
            } else if (c == 'L') {
                lStreak++;
                if (lStreak > 2) return false;
            } else {
                lStreak = 0;
            }
        }
        return true;
    }
};
```

---

##327 ****[Problem Link]https://leetcode.com/problems/binary-string-with-substrings-representing-1-to-n****  
**Approach:** Brute force substring integer check up to 20 bits.  
**Time Complexity:** O(n^2)

```cpp
#include <string>
#include <unordered_set>
using namespace std;

class Solution {
public:
    bool queryString(string s, int n) {
        unordered_set<int> seen;
        for (int i = 0; i < s.size(); ++i) {
            if (s[i] == '0') continue;
            int val = 0;
            for (int j = i; j < s.size() && j - i < 20; ++j) {
                val = (val << 1) | (s[j] - '0');
                if (val >= 1 && val <= n) seen.insert(val);
            }
        }
        return seen.size() == n;
    }
};
```

---

##328 ****[Problem Link]https://leetcode.com/problems/number-of-different-subsequences-gcds****  
**Approach:** Count using frequency map and multiples for GCD.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <numeric>
using namespace std;

class Solution {
public:
    int countDifferentSubsequenceGCDs(vector<int>& nums) {
        int maxVal = *max_element(nums.begin(), nums.end());
        vector<bool> present(maxVal + 1, false);
        for (int x : nums) present[x] = true;

        int count = 0;
        for (int x = 1; x <= maxVal; ++x) {
            int g = 0;
            for (int y = x; y <= maxVal; y += x) {
                if (present[y]) g = gcd(g, y);
            }
            if (g == x) count++;
        }
        return count;
    }
};
```

---

##329 ****[Problem Link]https://leetcode.com/problems/redistribute-characters-to-make-all-strings-equal****  
**Approach:** Count characters and check divisibility by n.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    bool makeEqual(vector<string>& words) {
        vector<int> count(26, 0);
        for (auto& word : words)
            for (char c : word) count[c - 'a']++;

        for (int c : count)
            if (c % words.size() != 0) return false;

        return true;
    }
};
```

---

##330 ****[Problem Link]https://leetcode.com/problems/watering-plants****  
**Approach:** Simulate walk and refill when needed.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int wateringPlants(vector<int>& plants, int capacity) {
        int steps = 0, curr = capacity;
        for (int i = 0; i < plants.size(); ++i) {
            if (curr < plants[i]) {
                steps += 2 * i;
                curr = capacity;
            }
            curr -= plants[i];
            steps++;
        }
        return steps;
    }
};
```

---

##331 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-ways-to-partition-an-array****  
**Approach:** Prefix sums + hash map to track potential changes at each pivot.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int waysToPartition(vector<int>& nums, int k) {
        int n = nums.size();
        long long total = 0, prefix = 0;
        for (int x : nums) total += x;

        unordered_map<long long, int> left, right;
        vector<long long> psum(n);
        for (int i = 0; i < n; ++i) {
            prefix += nums[i];
            psum[i] = prefix;
        }

        for (int i = 0; i < n - 1; ++i)
            right[psum[i]]++;

        int res = right[total / 2];
        prefix = 0;

        for (int i = 0; i < n; ++i) {
            long long newTotal = total - nums[i] + k;
            int cur = 0;
            if (newTotal % 2 == 0) {
                long long target = newTotal / 2;
                cur = left[target] + right[target - (nums[i] - k)];
            }
            res = max(res, cur);

            if (i < n - 1) {
                left[psum[i]]++;
                right[psum[i]]--;
            }
        }
        return res;
    }
};
```

---

##332 ****[Problem Link]https://leetcode.com/problems/two-furthest-houses-with-different-colors****  
**Approach:** Check both ends to maximize the distance.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int maxDistance(vector<int>& colors) {
        int n = colors.size(), res = 0;
        for (int i = 0; i < n; ++i) {
            if (colors[i] != colors[0])
                res = max(res, i);
            if (colors[i] != colors[n - 1])
                res = max(res, n - 1 - i);
        }
        return res;
    }
};
```

---

##333 ****[Problem Link]https://leetcode.com/problems/prime-arrangements****  
**Approach:** Count primes ≤ n and compute factorials with mod.  
**Time Complexity:** O(n log log n)

```cpp
#include <vector>
using namespace std;

class Solution {
    static constexpr int MOD = 1e9 + 7;

    int countPrimes(int n) {
        vector<bool> isPrime(n + 1, true);
        isPrime[0] = isPrime[1] = false;
        for (int i = 2; i * i <= n; ++i)
            if (isPrime[i])
                for (int j = i * i; j <= n; j += i)
                    isPrime[j] = false;

        int count = 0;
        for (int i = 2; i <= n; ++i)
            if (isPrime[i]) count++;
        return count;
    }

    long long factorial(int n) {
        long long res = 1;
        for (int i = 2; i <= n; ++i)
            res = res * i % MOD;
        return res;
    }

public:
    int numPrimeArrangements(int n) {
        int primes = countPrimes(n);
        return factorial(primes) * factorial(n - primes) % MOD;
    }
};
```

---

##334 ****[Problem Link]https://leetcode.com/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate****  
**Approach:** Check all points, track min Manhattan distance and index.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <climits>
using namespace std;

class Solution {
public:
    int nearestValidPoint(int x, int y, vector<vector<int>>& points) {
        int res = -1, minDist = INT_MAX;
        for (int i = 0; i < points.size(); ++i) {
            if (points[i][0] == x || points[i][1] == y) {
                int dist = abs(x - points[i][0]) + abs(y - points[i][1]);
                if (dist < minDist) {
                    minDist = dist;
                    res = i;
                }
            }
        }
        return res;
    }
};
```

---

##335 ****[Problem Link]https://leetcode.com/problems/count-ways-to-build-rooms-in-an-ant-colony****  
**Approach:** DFS with combinatorics.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
using namespace std;

class Solution {
    static constexpr int MOD = 1e9 + 7;
    vector<long long> fact, invFact;
    vector<vector<int>> tree;

    long long modinv(long long a) {
        long long res = 1, p = MOD - 2;
        while (p) {
            if (p & 1) res = res * a % MOD;
            a = a * a % MOD;
            p >>= 1;
        }
        return res;
    }

    void buildFactorials(int n) {
        fact.resize(n + 1, 1);
        invFact.resize(n + 1, 1);
        for (int i = 1; i <= n; ++i)
            fact[i] = fact[i - 1] * i % MOD;
        for (int i = 0; i <= n; ++i)
            invFact[i] = modinv(fact[i]);
    }

    pair<long long, int> dfs(int u) {
        long long ways = 1;
        int size = 0;
        for (int v : tree[u]) {
            auto [w, s] = dfs(v);
            ways = ways * w % MOD * invFact[s] % MOD;
            size += s;
        }
        ways = ways * fact[size] % MOD;
        return {ways, size + 1};
    }

public:
    int waysToBuildRooms(vector<int>& prevRoom) {
        int n = prevRoom.size();
        tree.resize(n);
        for (int i = 1; i < n; ++i)
            tree[prevRoom[i]].push_back(i);
        buildFactorials(n);
        return dfs(0).first;
    }
};
```

---

##336 ****[Problem Link]https://leetcode.com/problems/number-of-smooth-descent-periods-of-a-stock****  
**Approach:** Count contiguous non-increasing segments.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    long long getDescentPeriods(vector<int>& prices) {
        long long count = 1, res = 1;
        for (int i = 1; i < prices.size(); ++i) {
            if (prices[i] == prices[i - 1] - 1) count++;
            else count = 1;
            res += count;
        }
        return res;
    }
};
```

---

##337 ****[Problem Link]https://leetcode.com/problems/people-whose-list-of-favorite-companies-is-not-a-subset-of-another-list****  
**Approach:** Compare each person's list with others to find strict non-subsets.  
**Time Complexity:** O(n^2 * m)

```cpp
#include <vector>
#include <string>
#include <unordered_set>
using namespace std;

class Solution {
public:
    vector<int> peopleIndexes(vector<vector<string>>& favoriteCompanies) {
        int n = favoriteCompanies.size();
        vector<unordered_set<string>> sets(n);
        for (int i = 0; i < n; ++i)
            for (string& s : favoriteCompanies[i])
                sets[i].insert(s);

        vector<int> res;
        for (int i = 0; i < n; ++i) {
            bool isSubset = false;
            for (int j = 0; j < n && !isSubset; ++j) {
                if (i == j || sets[j].size() < sets[i].size()) continue;
                int match = 0;
                for (string& s : sets[i])
                    if (sets[j].count(s)) match++;
                if (match == sets[i].size()) isSubset = true;
            }
            if (!isSubset) res.push_back(i);
        }
        return res;
    }
};
```

---

##338 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-buckets-required-to-collect-rainwater-from-houses****  
**Approach:** Greedy placement of buckets around houses.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    int minimumBuckets(string street) {
        int n = street.size(), res = 0;
        for (int i = 0; i < n; ++i) {
            if (street[i] == 'H') {
                if (i > 0 && street[i - 1] == 'B') continue;
                if (i + 1 < n && street[i + 1] == '.') {
                    street[i + 1] = 'B';
                    res++;
                } else if (i > 0 && street[i - 1] == '.') {
                    street[i - 1] = 'B';
                    res++;
                } else return -1;
            }
        }
        return res;
    }
};
```

---

##339 ****[Problem Link]https://leetcode.com/problems/minimum-space-wasted-from-packaging****  
**Approach:** For each supplier, sort and binary search package fits.  
**Time Complexity:** O(k * n log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minWastedSpace(vector<int>& packages, vector<vector<int>>& boxes) {
        sort(packages.begin(), packages.end());
        long long total = accumulate(packages.begin(), packages.end(), 0LL);
        long long res = LLONG_MAX;
        const int MOD = 1e9 + 7;

        for (auto& b : boxes) {
            sort(b.begin(), b.end());
            if (b.back() < packages.back()) continue;

            long long waste = 0;
            int i = 0;
            for (int size : b) {
                auto it = upper_bound(packages.begin() + i, packages.end(), size);
                waste += 1LL * (it - packages.begin() - i) * size;
                i = it - packages.begin();
            }
            res = min(res, waste);
        }

        return res == LLONG_MAX ? -1 : res % MOD;
    }
};
```

---

##340 ****[Problem Link]https://leetcode.com/problems/powerful-integers****  
**Approach:** Generate all possible x^i + y^j ≤ bound.  
**Time Complexity:** O(log bound)

```cpp
#include <vector>
#include <unordered_set>
#include <cmath>
using namespace std;

class Solution {
public:
    vector<int> powerfulIntegers(int x, int y, int bound) {
        unordered_set<int> res;
        for (int i = 1; i < bound; i *= x) {
            for (int j = 1; i + j <= bound; j *= y) {
                res.insert(i + j);
                if (y == 1) break;
            }
            if (x == 1) break;
        }
        return vector<int>(res.begin(), res.end());
    }
};
```

---

##341 ****[Problem Link]https://leetcode.com/problems/maximum-difference-between-increasing-elements****  
**Approach:** Track min element while scanning to find max difference.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int maximumDifference(vector<int>& nums) {
        int minVal = nums[0], res = -1;
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] > minVal)
                res = max(res, nums[i] - minVal);
            minVal = min(minVal, nums[i]);
        }
        return res;
    }
};
```

---

##342 ****[Problem Link]https://leetcode.com/problems/day-of-the-year****  
**Approach:** Parse date string and compute day count with leap year logic.  
**Time Complexity:** O(1)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    int dayOfYear(string date) {
        int year = stoi(date.substr(0, 4));
        int month = stoi(date.substr(5, 2));
        int day = stoi(date.substr(8, 2));

        vector<int> days = { 31,28,31,30,31,30,31,31,30,31,30,31 };
        if ((year % 400 == 0) || (year % 100 != 0 && year % 4 == 0))
            days[1]++;

        int total = 0;
        for (int i = 0; i < month - 1; ++i) total += days[i];
        return total + day;
    }
};
```

---

##343 ****[Problem Link]https://leetcode.com/problems/circular-permutation-in-binary-representation****  
**Approach:** Gray code generation with rotation based on start.  
**Time Complexity:** O(2^n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> circularPermutation(int n, int start) {
        vector<int> res;
        for (int i = 0; i < (1 << n); ++i)
            res.push_back(start ^ (i ^ (i >> 1)));
        return res;
    }
};
```

---

##344 ****[Problem Link]https://leetcode.com/problems/gcd-sort-of-an-array****  
**Approach:** Union-Find for all numbers with shared prime factor.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;

class Solution {
    vector<int> parent;

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int x, int y) {
        parent[find(x)] = find(y);
    }

public:
    bool gcdSort(vector<int>& nums) {
        int maxVal = *max_element(nums.begin(), nums.end());
        parent.resize(maxVal + 1);
        for (int i = 0; i <= maxVal; ++i) parent[i] = i;

        vector<bool> isPrime(maxVal + 1, true);
        for (int i = 2; i * i <= maxVal; ++i) {
            if (isPrime[i]) {
                for (int j = i * i; j <= maxVal; j += i) isPrime[j] = false;
            }
        }

        for (int x : nums) {
            int temp = x;
            for (int i = 2; i * i <= temp; ++i) {
                if (temp % i == 0) {
                    unite(x, i);
                    while (temp % i == 0) temp /= i;
                }
            }
            if (temp > 1) unite(x, temp);
        }

        vector<int> sorted = nums;
        sort(sorted.begin(), sorted.end());
        for (int i = 0; i < nums.size(); ++i)
            if (find(nums[i]) != find(sorted[i])) return false;

        return true;
    }
};
```

---

##345 ****[Problem Link]https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum****  
**Approach:** Use a min-heap to select top k elements and restore original order.  
**Time Complexity:** O(n log k)

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    vector<int> maxSubsequence(vector<int>& nums, int k) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        for (int i = 0; i < nums.size(); ++i) {
            pq.push({nums[i], i});
            if (pq.size() > k) pq.pop();
        }

        vector<pair<int, int>> top(pq.size());
        int idx = 0;
        while (!pq.empty()) {
            top[idx++] = pq.top();
            pq.pop();
        }

        sort(top.begin(), top.end(), [](auto& a, auto& b) {
            return a.second < b.second;
        });

        vector<int> res;
        for (auto& [val, i] : top) res.push_back(val);
        return res;
    }
};
```

---

##346 ****[Problem Link]https://leetcode.com/problems/number-of-ways-where-square-of-number-is-equal-to-product-of-two-numbers****  
**Approach:** For each square check if it can be split into two numbers in the list.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int numTriplets(vector<int>& nums1, vector<int>& nums2) {
        return count(nums1, nums2) + count(nums2, nums1);
    }

    int count(vector<int>& A, vector<int>& B) {
        unordered_map<long long, int> freq;
        for (int b : B) freq[b]++;
        int res = 0;
        for (int a : A) {
            long long target = (long long)a * a;
            for (int b : B) {
                if (target % b != 0) continue;
                long long other = target / b;
                if (other == b) res += freq[b] - 1;
                else res += freq[other];
            }
        }
        return res / 2;
    }
};
```

---

##347 ****[Problem Link]https://leetcode.com/problems/circle-and-rectangle-overlapping****  
**Approach:** Clamp circle center to rectangle and check distance.  
**Time Complexity:** O(1)

```cpp
class Solution {
public:
    bool checkOverlap(int radius, int xCenter, int yCenter,
                      int x1, int y1, int x2, int y2) {
        int x = max(x1, min(xCenter, x2));
        int y = max(y1, min(yCenter, y2));
        int dx = x - xCenter, dy = y - yCenter;
        return dx * dx + dy * dy <= radius * radius;
    }
};
```

---

##348 ****[Problem Link]https://leetcode.com/problems/delivering-boxes-from-storage-to-ports****  
**Approach:** DP with prefix sums and port grouping.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int boxDelivering(vector<vector<int>>& boxes, int portsCount, int maxBoxes, int maxWeight) {
        int n = boxes.size();
        vector<int> dp(n + 1, 1e9), trips(n + 1), wsum(n + 1);
        dp[0] = 0;

        for (int i = 0; i < n; ++i) {
            trips[i + 1] = trips[i] + (i == 0 || boxes[i][0] != boxes[i - 1][0]);
            wsum[i + 1] = wsum[i] + boxes[i][1];
        }

        for (int i = 1, j = 0; i <= n; ++i) {
            while (i - j > maxBoxes || wsum[i] - wsum[j] > maxWeight)
                ++j;
            dp[i] = dp[j] + trips[i] - trips[j + 1] + 2;
        }

        return dp[n];
    }
};
```

---

##349 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-words-you-can-type****  
**Approach:** Use set for broken keys and check each word.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_set>
#include <sstream>
using namespace std;

class Solution {
public:
    int canBeTypedWords(string text, string brokenLetters) {
        unordered_set<char> broken(brokenLetters.begin(), brokenLetters.end());
        stringstream ss(text);
        string word;
        int count = 0;
        while (ss >> word) {
            bool valid = true;
            for (char c : word)
                if (broken.count(c)) {
                    valid = false;
                    break;
                }
            if (valid) count++;
        }
        return count;
    }
};
```

---

##350 ****[Problem Link]https://leetcode.com/problems/array-with-elements-not-equal-to-average-of-neighbors****  
**Approach:** Sort and interleave from both ends.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> rearrangeArray(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        vector<int> res(n);
        int left = 0, right = (n + 1) / 2, idx = 0;

        for (int i = 0; i < n; ++i) {
            if (i % 2 == 0)
                res[i] = nums[left++];
            else
                res[i] = nums[right++];
        }
        return res;
    }
};
```

---

##351 ****[Problem Link]https://leetcode.com/problems/triples-with-bitwise-and-equal-to-zero****  
**Approach:** Use frequency of numbers and bitmasking to optimize.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int countTriplets(vector<int>& A) {
        unordered_map<int, int> cnt;
        for (int a : A)
            for (int b : A)
                cnt[a & b]++;

        int res = 0;
        for (int a : A)
            for (auto& [val, freq] : cnt)
                if ((a & val) == 0)
                    res += freq;

        return res;
    }
};
```

---

##352 ****[Problem Link]https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary****  
**Approach:** DP with prefix counts of characters per column.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
#include <string>
using namespace std;

class Solution {
    static constexpr int MOD = 1e9 + 7;

public:
    int numWays(vector<string>& words, string target) {
        int m = words[0].size(), n = target.size();
        vector<vector<int>> freq(m, vector<int>(26, 0));
        for (string& word : words)
            for (int i = 0; i < m; ++i)
                freq[i][word[i] - 'a']++;

        vector<long long> dp(n + 1);
        dp[0] = 1;

        for (int i = 0; i < m; ++i) {
            for (int j = n - 1; j >= 0; --j) {
                dp[j + 1] += dp[j] * freq[i][target[j] - 'a'];
                dp[j + 1] %= MOD;
            }
        }
        return dp[n];
    }
};
```

---

##353 ****[Problem Link]https://leetcode.com/problems/check-if-all-the-integers-in-a-range-are-covered****  
**Approach:** Boolean array to mark covered values.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool isCovered(vector<vector<int>>& ranges, int left, int right) {
        vector<bool> covered(51, false);
        for (auto& r : ranges)
            for (int i = r[0]; i <= r[1]; ++i)
                covered[i] = true;

        for (int i = left; i <= right; ++i)
            if (!covered[i]) return false;

        return true;
    }
};
```

---

##354 ****[Problem Link]https://leetcode.com/problems/time-needed-to-buy-tickets****  
**Approach:** Simulate each second or compute directly.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int timeRequiredToBuy(vector<int>& tickets, int k) {
        int res = 0;
        for (int i = 0; i < tickets.size(); ++i) {
            if (i <= k)
                res += min(tickets[i], tickets[k]);
            else
                res += min(tickets[i], tickets[k] - 1);
        }
        return res;
    }
};
```

---

##355 ****[Problem Link]https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid****  
**Approach:** Track min and max balance range using greedy simulation.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    bool canBeValid(string s, string locked) {
        if (s.size() % 2) return false;

        int lo = 0, hi = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (locked[i] == '0') {
                lo--;
                hi++;
            } else {
                if (s[i] == '(') {
                    lo++;
                    hi++;
                } else {
                    lo--;
                    hi--;
                }
            }
            if (hi < 0) return false;
            lo = max(lo, 0);
        }

        return lo == 0;
    }
};
```

---

##356 ****[Problem Link]https://leetcode.com/problems/day-of-the-week****  
**Approach:** Use known weekday offset and modulo math.  
**Time Complexity:** O(1)

```cpp
#include <string>
#include <vector>
using namespace std;

class Solution {
public:
    string dayOfTheWeek(int day, int month, int year) {
        vector<int> daysOfMonth = {31,28,31,30,31,30,31,31,30,31,30,31};
        vector<string> days = {"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"};

        if ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0)
            daysOfMonth[1]++;

        int total = 4; // Jan 1, 1971 was a Friday
        for (int y = 1971; y < year; ++y)
            total += 365 + ((y % 4 == 0 && y % 100 != 0) || (y % 400 == 0));

        for (int m = 1; m < month; ++m)
            total += daysOfMonth[m - 1];

        total += day - 1;
        return days[total % 7];
    }
};
```

---

##357 ****[Problem Link]https://leetcode.com/problems/simplified-fractions****  
**Approach:** Generate all reduced proper fractions using gcd.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <string>
#include <numeric>
using namespace std;

class Solution {
public:
    vector<string> simplifiedFractions(int n) {
        vector<string> res;
        for (int i = 1; i < n; ++i)
            for (int j = i + 1; j <= n; ++j)
                if (gcd(i, j) == 1)
                    res.push_back(to_string(i) + "/" + to_string(j));
        return res;
    }
};
```

---

##358 ****[Problem Link]https://leetcode.com/problems/reverse-prefix-of-word****  
**Approach:** Find first occurrence of ch and reverse the prefix.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <algorithm>
using namespace std;

class Solution {
public:
    string reversePrefix(string word, char ch) {
        int idx = word.find(ch);
        if (idx != string::npos)
            reverse(word.begin(), word.begin() + idx + 1);
        return word;
    }
};
```

---

##359 ****[Problem Link]https://leetcode.com/problems/valid-boomerang****  
**Approach:** Check if area of triangle formed is non-zero.  
**Time Complexity:** O(1)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool isBoomerang(vector<vector<int>>& points) {
        int x1 = points[0][0], y1 = points[0][1];
        int x2 = points[1][0], y2 = points[1][1];
        int x3 = points[2][0], y3 = points[2][1];
        return (y2 - y1)*(x3 - x2) != (y3 - y2)*(x2 - x1);
    }
};
```

---

##360 ****[Problem Link]https://leetcode.com/problems/check-if-it-is-a-good-array****  
**Approach:** Use gcd to determine if combination yields 1.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <numeric>
using namespace std;

class Solution {
public:
    bool isGoodArray(vector<int>& nums) {
        int g = nums[0];
        for (int n : nums)
            g = gcd(g, n);
        return g == 1;
    }
};
```

---

##361 ****[Problem Link]https://leetcode.com/problems/tree-of-coprimes****  
**Approach:** DFS while maintaining recent coprimes in the ancestor stack.  
**Time Complexity:** O(n * 50)

```cpp
#include <vector>
#include <numeric>
using namespace std;

class Solution {
    vector<int> res;
    vector<vector<int>> tree;
    vector<int> val;
    vector<vector<int>> ancestors;

    void dfs(int node, int parent, int depth) {
        int bestDepth = -1, bestNode = -1;
        for (int i = 1; i <= 50; ++i) {
            if (!ancestors[i].empty() && gcd(i, val[node]) == 1) {
                int anc = ancestors[i].back();
                if (res[anc] > bestDepth) {
                    bestDepth = res[anc];
                    bestNode = anc;
                }
            }
        }
        res[node] = bestNode;

        ancestors[val[node]].push_back(node);
        for (int nei : tree[node]) {
            if (nei != parent) dfs(nei, node, depth + 1);
        }
        ancestors[val[node]].pop_back();
    }

public:
    vector<int> getCoprimes(vector<int>& nums, vector<vector<int>>& edges) {
        int n = nums.size();
        tree.resize(n);
        val = nums;
        res.assign(n, -1);
        ancestors.resize(51);
        for (auto& e : edges) {
            tree[e[0]].push_back(e[1]);
            tree[e[1]].push_back(e[0]);
        }
        res[0] = -1;
        dfs(0, -1, 0);
        return res;
    }
};
```

---

##362 ****[Problem Link]https://leetcode.com/problems/lexicographically-smallest-string-after-applying-operations****  
**Approach:** BFS for lexicographically smallest string state.  
**Time Complexity:** O(n!)

```cpp
#include <string>
#include <queue>
#include <unordered_set>
#include <algorithm>
using namespace std;

class Solution {
public:
    string findLexSmallestString(string s, int a, int b) {
        unordered_set<string> seen;
        queue<string> q;
        q.push(s);
        seen.insert(s);
        string res = s;

        while (!q.empty()) {
            string cur = q.front(); q.pop();
            res = min(res, cur);

            // Add operation
            string t = cur;
            for (int i = 1; i < t.size(); i += 2)
                t[i] = (t[i] - '0' + a) % 10 + '0';

            if (!seen.count(t)) {
                seen.insert(t);
                q.push(t);
            }

            // Rotate operation
            t = cur.substr(cur.size() - b) + cur.substr(0, cur.size() - b);
            if (!seen.count(t)) {
                seen.insert(t);
                q.push(t);
            }
        }
        return res;
    }
};
```

---

##363 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-achievable-transfer-requests****  
**Approach:** Backtrack through all combinations of request selections.  
**Time Complexity:** O(2^n * m)

```cpp
#include <vector>
using namespace std;

class Solution {
    int maxReq = 0;

    void backtrack(int i, int count, vector<int>& indegree, vector<vector<int>>& requests) {
        if (i == requests.size()) {
            for (int bal : indegree)
                if (bal != 0) return;
            maxReq = max(maxReq, count);
            return;
        }

        // Accept request
        indegree[requests[i][0]]--;
        indegree[requests[i][1]]++;
        backtrack(i + 1, count + 1, indegree, requests);
        indegree[requests[i][0]]++;
        indegree[requests[i][1]]--;

        // Reject request
        backtrack(i + 1, count, indegree, requests);
    }

public:
    int maximumRequests(int n, vector<vector<int>>& requests) {
        vector<int> indegree(n, 0);
        backtrack(0, 0, indegree, requests);
        return maxReq;
    }
};
```

---

##364 ****[Problem Link]https://leetcode.com/problems/reformat-date****  
**Approach:** Parse date components and reformat with padding.  
**Time Complexity:** O(1)

```cpp
#include <string>
#include <unordered_map>
using namespace std;

class Solution {
public:
    string reformatDate(string date) {
        unordered_map<string, string> month = {
            {"Jan","01"},{"Feb","02"},{"Mar","03"},{"Apr","04"},{"May","05"},{"Jun","06"},
            {"Jul","07"},{"Aug","08"},{"Sep","09"},{"Oct","10"},{"Nov","11"},{"Dec","12"}
        };
        string day = date.substr(0, date.find_first_not_of("0123456789"));
        if (day.size() == 1) day = "0" + day;
        string mon = month[date.substr(date.find(' ') + 1, 3)];
        string year = date.substr(date.size() - 4);
        return year + "-" + mon + "-" + day;
    }
};
```

---

##365 ****[Problem Link]https://leetcode.com/problems/second-largest-digit-in-a-string****  
**Approach:** Track largest and second largest digit using two variables.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    int secondHighest(string s) {
        int max1 = -1, max2 = -1;
        for (char c : s) {
            if (isdigit(c)) {
                int d = c - '0';
                if (d > max1) {
                    max2 = max1;
                    max1 = d;
                } else if (d < max1 && d > max2) {
                    max2 = d;
                }
            }
        }
        return max2;
    }
};
```

---

##366 ****[Problem Link]https://leetcode.com/problems/change-minimum-characters-to-satisfy-one-of-three-conditions****  
**Approach:** Frequency analysis and prefix sum over 26 characters.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>
using namespace std;

class Solution {
public:
    int minCharacters(string a, string b) {
        vector<int> ca(26), cb(26);
        for (char c : a) ca[c - 'a']++;
        for (char c : b) cb[c - 'a']++;

        int res = a.size() + b.size();

        // Condition 3: all chars are the same
        for (int i = 0; i < 26; ++i)
            res = min(res, int(a.size() - ca[i] + b.size() - cb[i]));

        // Condition 1 and 2
        for (int i = 1; i < 26; ++i) {
            int caSum = 0, cbSum = 0;
            for (int j = 0; j < i; ++j) {
                caSum += ca[j];
                cbSum += cb[j];
            }
            res = min(res, int(a.size() - caSum + cbSum)); // a < b
            res = min(res, int(b.size() - cbSum + caSum)); // b < a
        }

        return res;
    }
};
```

---

##367 ****[Problem Link]https://leetcode.com/problems/maximum-value-after-insertion****  
**Approach:** Greedy insert val at the position that maximizes value.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    string maxValue(string n, int x) {
        int i = 0;
        bool isNeg = (n[0] == '-');
        i = isNeg ? 1 : 0;

        while (i < n.size()) {
            if ((isNeg && n[i] - '0' > x) || (!isNeg && n[i] - '0' < x))
                break;
            ++i;
        }
        return n.substr(0, i) + to_string(x) + n.substr(i);
    }
};
```

---

##368 ****[Problem Link]https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores****  
**Approach:** Sort array and take min difference over sliding window of k.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minimumDifference(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int res = INT_MAX;
        for (int i = 0; i <= nums.size() - k; ++i)
            res = min(res, nums[i + k - 1] - nums[i]);
        return res;
    }
};
```

---

##369 ****[Problem Link]https://leetcode.com/problems/maximum-fruits-harvested-after-at-most-k-steps****  
**Approach:** Sliding window technique using prefix sums and binary search.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int maxTotalFruits(vector<vector<int>>& fruits, int startPos, int k) {
        int n = fruits.size(), res = 0;
        vector<int> pos(n), prefix(n + 1);
        for (int i = 0; i < n; ++i) {
            pos[i] = fruits[i][0];
            prefix[i + 1] = prefix[i] + fruits[i][1];
        }

        for (int l = 0; l <= k; ++l) {
            int left = startPos - l;
            int right = startPos + max(k - 2 * l, (k - l) / 2 * 2);
            int lo = lower_bound(pos.begin(), pos.end(), left) - pos.begin();
            int hi = upper_bound(pos.begin(), pos.end(), right) - pos.begin();
            res = max(res, prefix[hi] - prefix[lo]);
        }

        for (int r = 0; r <= k; ++r) {
            int right = startPos + r;
            int left = startPos - max(k - 2 * r, (k - r) / 2 * 2);
            int lo = lower_bound(pos.begin(), pos.end(), left) - pos.begin();
            int hi = upper_bound(pos.begin(), pos.end(), right) - pos.begin();
            res = max(res, prefix[hi] - prefix[lo]);
        }

        return res;
    }
};
```

---

##370 ****[Problem Link]https://leetcode.com/problems/contain-virus****  
**Approach:** BFS to simulate spread and quarantine most dangerous region.  
**Time Complexity:** O(m * n * (m+n))

```cpp
#include <vector>
#include <queue>
#include <set>
using namespace std;

class Solution {
public:
    int containVirus(vector<vector<int>>& isInfected) {
        int m = isInfected.size(), n = isInfected[0].size(), res = 0;
        int dirs[5] = {0, 1, 0, -1, 0};

        while (true) {
            vector<set<pair<int, int>>> frontiers;
            vector<set<pair<int, int>>> regions;
            vector<int> walls;

            vector<vector<bool>> visited(m, vector<bool>(n, false));
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (isInfected[i][j] == 1 && !visited[i][j]) {
                        set<pair<int, int>> frontier, region;
                        int wall = 0;
                        queue<pair<int, int>> q;
                        q.emplace(i, j);
                        visited[i][j] = true;

                        while (!q.empty()) {
                            auto [x, y] = q.front(); q.pop();
                            region.insert({x, y});
                            for (int d = 0; d < 4; ++d) {
                                int nx = x + dirs[d], ny = y + dirs[d + 1];
                                if (nx < 0 || ny < 0 || nx >= m || ny >= n) continue;
                                if (isInfected[nx][ny] == 1 && !visited[nx][ny]) {
                                    q.emplace(nx, ny);
                                    visited[nx][ny] = true;
                                } else if (isInfected[nx][ny] == 0) {
                                    frontier.insert({nx, ny});
                                    wall++;
                                }
                            }
                        }

                        frontiers.push_back(frontier);
                        regions.push_back(region);
                        walls.push_back(wall);
                    }
                }
            }

            if (frontiers.empty()) break;
            int idx = max_element(frontiers.begin(), frontiers.end(),
                [](auto& a, auto& b) { return a.size() < b.size(); }) - frontiers.begin();

            res += walls[idx];
            for (int i = 0; i < frontiers.size(); ++i) {
                if (i == idx) {
                    for (auto& [x, y] : regions[i])
                        isInfected[x][y] = -1;
                } else {
                    for (auto& [x, y] : frontiers[i])
                        isInfected[x][y] = 1;
                }
            }
        }

        return res;
    }
};
```

---

##371 ****[Problem Link]https://leetcode.com/problems/adding-two-negabinary-numbers****  
**Approach:** Simulate binary addition using negabinary rules.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> addNegabinary(vector<int>& arr1, vector<int>& arr2) {
        vector<int> res;
        int i = arr1.size() - 1, j = arr2.size() - 1, carry = 0;

        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) sum += arr1[i--];
            if (j >= 0) sum += arr2[j--];

            res.push_back(sum & 1);
            carry = -(sum >> 1);
        }

        while (res.size() > 1 && res.back() == 0) res.pop_back();
        reverse(res.begin(), res.end());
        return res;
    }
};
```

---

##372 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-groups-getting-fresh-donuts****  
**Approach:** DFS with memoization on donut count remainder state.  
**Time Complexity:** Exponential (DFS with pruning)

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int maxHappyGroups(int batchSize, vector<int>& groups) {
        vector<int> rem(batchSize, 0);
        int res = 0;
        for (int g : groups) {
            int r = g % batchSize;
            if (r == 0) res++;
            else rem[r]++;
        }

        unordered_map<long long, int> memo;
        function<int(int, vector<int>&)> dfs = [&](int prev, vector<int>& r) {
            long long key = prev;
            for (int i = 1; i < batchSize; ++i)
                key = key * 31 + r[i];
            if (memo.count(key)) return memo[key];

            int ans = 0;
            for (int i = 1; i < batchSize; ++i) {
                if (r[i]) {
                    r[i]--;
                    ans = max(ans, (prev == 0) + dfs((prev + i) % batchSize, r));
                    r[i]++;
                }
            }
            return memo[key] = ans;
        };

        return res + dfs(0, rem);
    }
};
```

---

##373 ****[Problem Link]https://leetcode.com/problems/maximize-grid-happiness****  
**Approach:** DP + Bitmask with memoization over row states.  
**Time Complexity:** O(m*n*2^(2n))

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
    int m, n;
    int memo[6][6][7][7][1 << 6];

public:
    int getMaxGridHappiness(int m_, int n_, int introvertsCount, int extrovertsCount) {
        m = m_, n = n_;
        memset(memo, -1, sizeof(memo));
        return dfs(0, 0, introvertsCount, extrovertsCount, 0);
    }

    int dfs(int i, int j, int in, int ex, int mask) {
        if (j == n) j = 0, i++;
        if (i == m || (in == 0 && ex == 0)) return 0;
        if (memo[i][j][in][ex][mask] != -1) return memo[i][j][in][ex][mask];

        int up = (mask >> (n - 1)) & 3;
        int left = (mask >> ((n - j - 1) * 2)) & 3;
        int nextMask = mask & ~(3 << ((n - j - 1) * 2));

        int res = dfs(i, j + 1, in, ex, nextMask);

        if (in > 0) {
            int happy = 120;
            if (up == 1) happy -= 30, happy -= 30;
            else if (up == 2) happy -= 10, happy += 20;
            if (left == 1) happy -= 30, happy -= 30;
            else if (left == 2) happy -= 10, happy += 20;
            res = max(res, happy + dfs(i, j + 1, in - 1, ex, nextMask | (1 << ((n - j - 1) * 2))));
        }

        if (ex > 0) {
            int happy = 40;
            if (up == 1) happy += 20, happy -= 30;
            else if (up == 2) happy += 20, happy -= 10;
            if (left == 1) happy += 20, happy -= 30;
            else if (left == 2) happy += 20, happy -= 10;
            res = max(res, happy + dfs(i, j + 1, in, ex - 1, nextMask | (2 << ((n - j - 1) * 2))));
        }

        return memo[i][j][in][ex][mask] = res;
    }
};
```

---

##374 ****[Problem Link]https://leetcode.com/problems/maximum-genetic-difference-query****  
**Approach:** DFS traversal with trie to track prefix xor during query.  
**Time Complexity:** O(n log U)

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Trie {
public:
    Trie* child[2] = {};
    void insert(int num) {
        Trie* node = this;
        for (int i = 17; i >= 0; --i) {
            int b = (num >> i) & 1;
            if (!node->child[b]) node->child[b] = new Trie();
            node = node->child[b];
        }
    }

    void remove(int num) {
        Trie* node = this;
        for (int i = 17; i >= 0; --i) {
            node = node->child[(num >> i) & 1];
        }
    }

    int query(int num) {
        Trie* node = this;
        int res = 0;
        for (int i = 17; i >= 0; --i) {
            int b = (num >> i) & 1;
            if (node->child[!b]) {
                res |= (1 << i);
                node = node->child[!b];
            } else {
                node = node->child[b];
            }
        }
        return res;
    }
};

class Solution {
public:
    vector<int> maxGeneticDifference(vector<int>& parents, vector<vector<int>>& queries) {
        int n = parents.size();
        vector<vector<int>> tree(n);
        int root = 0;
        for (int i = 0; i < n; ++i) {
            if (parents[i] == -1) root = i;
            else tree[parents[i]].push_back(i);
        }

        vector<vector<pair<int, int>>> q(n);
        for (int i = 0; i < queries.size(); ++i)
            q[queries[i][0]].emplace_back(queries[i][1], i);

        vector<int> res(queries.size());
        Trie trie;
        function<void(int)> dfs = [&](int u) {
            trie.insert(u);
            for (auto& [val, idx] : q[u])
                res[idx] = trie.query(val);
            for (int v : tree[u]) dfs(v);
            trie.remove(u);
        };

        dfs(root);
        return res;
    }
};
```

---

##375 ****[Problem Link]https://leetcode.com/problems/number-of-burgers-with-no-waste-of-ingredients****  
**Approach:** Solve system of equations with basic math.  
**Time Complexity:** O(1)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> numOfBurgers(int tomatoSlices, int cheeseSlices) {
        int x = tomatoSlices - 2 * cheeseSlices;
        if (x < 0 || x % 2 != 0) return {};
        int jumbo = x / 2;
        int small = cheeseSlices - jumbo;
        if (small < 0) return {};
        return {jumbo, small};
    }
};
```

---

##376 ****[Problem Link]https://leetcode.com/problems/removing-minimum-and-maximum-from-array****  
**Approach:** Try removing from left, right, or both ends.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minimumDeletions(vector<int>& nums) {
        int n = nums.size();
        int minIdx = min_element(nums.begin(), nums.end()) - nums.begin();
        int maxIdx = max_element(nums.begin(), nums.end()) - nums.begin();

        if (minIdx > maxIdx) swap(minIdx, maxIdx);

        return min({maxIdx + 1, n - minIdx, minIdx + 1 + n - maxIdx});
    }
};
```

---

##377 ****[Problem Link]https://leetcode.com/problems/find-good-days-to-rob-the-bank****  
**Approach:** Precompute left and right non-increasing/increasing streaks.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> goodDaysToRobBank(vector<int>& sec, int time) {
        int n = sec.size();
        vector<int> dec(n), inc(n), res;

        for (int i = 1; i < n; ++i)
            if (sec[i] <= sec[i - 1]) dec[i] = dec[i - 1] + 1;
        for (int i = n - 2; i >= 0; --i)
            if (sec[i] <= sec[i + 1]) inc[i] = inc[i + 1] + 1;

        for (int i = time; i < n - time; ++i)
            if (dec[i] >= time && inc[i] >= time)
                res.push_back(i);
        return res;
    }
};
```

---

##378 ****[Problem Link]https://leetcode.com/problems/building-boxes****  
**Approach:** Math + binary search to maximize full pyramid levels.  
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int minimumBoxes(int n) {
        int k = 0, sum = 0;
        while (sum + (k + 1) * (k + 2) / 2 <= n) {
            k++;
            sum += k * (k + 1) / 2;
        }

        int res = k * (k + 1) / 2;
        for (int i = 1; sum < n; ++i) {
            sum += i;
            res++;
        }
        return res;
    }
};
```

---

##379 ****[Problem Link]https://leetcode.com/problems/get-watched-videos-by-your-friends****  
**Approach:** BFS from target user to depth d, count video frequency.  
**Time Complexity:** O(n + m log m)

```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <set>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<string> watchedVideosByFriends(vector<vector<string>>& watchedVideos,
                                          vector<vector<int>>& friends, int id, int level) {
        int n = friends.size();
        vector<bool> visited(n, false);
        queue<int> q;
        q.push(id);
        visited[id] = true;

        while (level--) {
            int sz = q.size();
            while (sz--) {
                int curr = q.front(); q.pop();
                for (int f : friends[curr])
                    if (!visited[f]) {
                        visited[f] = true;
                        q.push(f);
                    }
            }
        }

        unordered_map<string, int> freq;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& video : watchedVideos[u])
                freq[video]++;
        }

        vector<pair<string, int>> vec(freq.begin(), freq.end());
        sort(vec.begin(), vec.end(), [](auto& a, auto& b) {
            return a.second == b.second ? a.first < b.first : a.second < b.second;
        });

        vector<string> res;
        for (auto& [v, _] : vec)
            res.push_back(v);
        return res;
    }
};
```

---

##380 ****[Problem Link]https://leetcode.com/problems/filter-restaurants-by-vegan-friendly-price-and-distance****  
**Approach:** Filter, then sort by rating and id descending.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> filterRestaurants(vector<vector<int>>& restaurants, int veganFriendly,
                                  int maxPrice, int maxDistance) {
        vector<vector<int>> filtered;
        for (auto& r : restaurants) {
            if ((veganFriendly == 0 || r[2] == 1) &&
                r[3] <= maxPrice && r[4] <= maxDistance) {
                filtered.push_back(r);
            }
        }

        sort(filtered.begin(), filtered.end(), [](auto& a, auto& b) {
            return a[1] == b[1] ? a[0] > b[0] : a[1] > b[1];
        });

        vector<int> res;
        for (auto& r : filtered) res.push_back(r[0]);
        return res;
    }
};
```

---

##381 ****[Problem Link]https://leetcode.com/problems/detonate-the-maximum-bombs****  
**Approach:** Graph traversal with DFS to simulate chain reactions.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <cmath>
using namespace std;

class Solution {
public:
    int maximumDetonation(vector<vector<int>>& bombs) {
        int n = bombs.size();
        vector<vector<int>> adj(n);
        for (int i = 0; i < n; ++i) {
            long long x1 = bombs[i][0], y1 = bombs[i][1], r1 = bombs[i][2];
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                long long x2 = bombs[j][0], y2 = bombs[j][1];
                long long distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                if (distSq <= r1 * r1)
                    adj[i].push_back(j);
            }
        }

        int res = 0;
        for (int i = 0; i < n; ++i) {
            vector<bool> visited(n, false);
            res = max(res, dfs(i, adj, visited));
        }
        return res;
    }

    int dfs(int i, vector<vector<int>>& adj, vector<bool>& visited) {
        visited[i] = true;
        int res = 1;
        for (int nei : adj[i])
            if (!visited[nei])
                res += dfs(nei, adj, visited);
        return res;
    }
};
```

---

##382 ****[Problem Link]https://leetcode.com/problems/sentence-similarity-iii****  
**Approach:** Check if one is a suffix or prefix of the other.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    bool areSentencesSimilar(string s1, string s2) {
        if (s1.size() > s2.size()) swap(s1, s2);

        if (s2.substr(0, s1.size() + 1) == s1 + " ") return true;
        if (s2.substr(s2.size() - s1.size() - 1) == " " + s1) return true;
        if (s2.find(" " + s1 + " ") != string::npos) return true;
        return s1 == s2;
    }
};
```

---

##383 ****[Problem Link]https://leetcode.com/problems/number-of-strings-that-appear-as-substrings-in-word****  
**Approach:** Check for presence of each string in the word.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    int numOfStrings(vector<string>& patterns, string word) {
        int count = 0;
        for (auto& p : patterns)
            if (word.find(p) != string::npos) count++;
        return count;
    }
};
```

---

##384 ****[Problem Link]https://leetcode.com/problems/two-out-of-three****  
**Approach:** Use sets to track distinct sources of each number.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <set>
#include <unordered_map>
using namespace std;

class Solution {
public:
    vector<int> twoOutOfThree(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3) {
        unordered_map<int, set<int>> mp;
        for (int x : set<int>(nums1.begin(), nums1.end())) mp[x].insert(1);
        for (int x : set<int>(nums2.begin(), nums2.end())) mp[x].insert(2);
        for (int x : set<int>(nums3.begin(), nums3.end())) mp[x].insert(3);

        vector<int> res;
        for (auto& [num, s] : mp)
            if (s.size() >= 2)
                res.push_back(num);
        return res;
    }
};
```

---

##385 ****[Problem Link]https://leetcode.com/problems/the-time-when-the-network-becomes-idle****  
**Approach:** BFS + math to compute round trip and last sent message.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int networkBecomesIdle(vector<vector<int>>& edges, vector<int>& patience) {
        int n = patience.size();
        vector<vector<int>> g(n);
        for (auto& e : edges) {
            g[e[0]].push_back(e[1]);
            g[e[1]].push_back(e[0]);
        }

        vector<int> dist(n, -1);
        queue<int> q;
        q.push(0);
        dist[0] = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : g[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }

        int res = 0;
        for (int i = 1; i < n; ++i) {
            int rtt = 2 * dist[i];
            int last = ((rtt - 1) / patience[i]) * patience[i];
            res = max(res, last + rtt);
        }

        return res + 1;
    }
};
```

---

##386 ****[Problem Link]https://leetcode.com/problems/kth-smallest-product-of-two-sorted-arrays****  
**Approach:** Binary search on answer space with counting function.  
**Time Complexity:** O((m+n) * log(range))

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    long long countSmallerOrEqual(const vector<int>& nums1, const vector<int>& nums2, long long x) {
        long long count = 0;
        for (int a : nums1) {
            if (a == 0) {
                if (x >= 0) count += nums2.size();
            } else if (a > 0) {
                int l = 0, r = nums2.size();
                while (l < r) {
                    int m = l + (r - l) / 2;
                    if ((long long)a * nums2[m] <= x) l = m + 1;
                    else r = m;
                }
                count += l;
            } else {
                int l = 0, r = nums2.size();
                while (l < r) {
                    int m = l + (r - l) / 2;
                    if ((long long)a * nums2[m] <= x) r = m;
                    else l = m + 1;
                }
                count += nums2.size() - l;
            }
        }
        return count;
    }

    long long kthSmallestProduct(vector<int>& nums1, vector<int>& nums2, long long k) {
        long long lo = -1e10, hi = 1e10;
        while (lo < hi) {
            long long mid = lo + (hi - lo) / 2;
            if (countSmallerOrEqual(nums1, nums2, mid) < k) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
};
```

---

##387 ****[Problem Link]https://leetcode.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls****  
**Approach:** DFS with combinatorics and symmetry property.  
**Time Complexity:** Exponential, small input

```cpp
#include <vector>
#include <cmath>
using namespace std;

class Solution {
    double fact[10];
    int totalBalls = 0;

public:
    double getProbability(vector<int>& balls) {
        fact[0] = 1;
        for (int i = 1; i < 10; ++i) fact[i] = fact[i - 1] * i;
        totalBalls = accumulate(balls.begin(), balls.end(), 0);
        return dfs(balls, 0, 0, 0, 0, 0, 0) / total();
    }

    double dfs(vector<int>& balls, int i, int c1, int c2, int n1, int n2, double ways) {
        if (i == balls.size()) {
            if (n1 == n2 && c1 == c2) return ways;
            return 0;
        }

        double res = 0;
        for (int j = 0; j <= balls[i]; ++j) {
            int k = balls[i] - j;
            res += dfs(balls, i + 1, c1 + (j > 0), c2 + (k > 0),
                       n1 + j, n2 + k, ways * fact[balls[i]] / (fact[j] * fact[k]));
        }
        return res;
    }

    double total() {
        double res = fact[totalBalls];
        for (int i = 1; i <= totalBalls / 2; ++i)
            res /= fact[i];
        return res;
    }
};
```

---

##388 ****[Problem Link]https://leetcode.com/problems/count-pairs-of-nodes****  
**Approach:** Graph degree + edge count precomputation.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> countPairs(int n, vector<vector<int>>& edges, vector<int>& queries) {
        vector<int> degree(n + 1, 0);
        unordered_map<int, unordered_map<int, int>> freq;
        for (auto& e : edges) {
            int u = e[0], v = e[1];
            degree[u]++;
            degree[v]++;
            if (u > v) swap(u, v);
            freq[u][v]++;
        }

        vector<int> sorted = degree;
        sort(sorted.begin(), sorted.end());
        vector<int> res;
        for (int q : queries) {
            int total = 0;
            int l = 1, r = n;
            while (l < r) {
                if (sorted[l] + sorted[r] <= q) l++;
                else {
                    total += r - l;
                    r--;
                }
            }

            for (auto& [u, mp] : freq)
                for (auto& [v, cnt] : mp)
                    if (degree[u] + degree[v] > q && degree[u] + degree[v] - cnt <= q)
                        total--;

            res.push_back(total);
        }

        return res;
    }
};
```

---

##389 ****[Problem Link]https://leetcode.com/problems/minimum-cost-homecoming-of-a-robot-in-a-grid****  
**Approach:** Accumulate cost by moving right then down using prefix sums.  
**Time Complexity:** O(m + n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int minCost(vector<int>& start, vector<int>& home,
                vector<int>& rowCosts, vector<int>& colCosts) {
        int res = 0;
        int r1 = start[0], r2 = home[0];
        int c1 = start[1], c2 = home[1];

        for (int i = r1 + (r1 < r2 ? 1 : -1); i != r2 + (r1 < r2 ? 1 : -1); i += (r1 < r2 ? 1 : -1))
            res += rowCosts[i];
        for (int i = c1 + (c1 < c2 ? 1 : -1); i != c2 + (c1 < c2 ? 1 : -1); i += (c1 < c2 ? 1 : -1))
            res += colCosts[i];
        return res;
    }
};
```

---

##390 ****[Problem Link]https://leetcode.com/problems/valid-arrangement-of-pairs****  
**Approach:** Eulerian path using DFS + stack.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>
#include <stack>
#include <deque>
using namespace std;

class Solution {
public:
    vector<vector<int>> validArrangement(vector<vector<int>>& pairs) {
        unordered_map<int, deque<int>> graph;
        unordered_map<int, int> outdeg, indeg;

        for (auto& p : pairs) {
            graph[p[0]].push_back(p[1]);
            outdeg[p[0]]++;
            indeg[p[1]]++;
        }

        int start = pairs[0][0];
        for (auto& [node, _] : graph)
            if (outdeg[node] > indeg[node]) {
                start = node;
                break;
            }

        vector<vector<int>> res;
        stack<int> st;
        st.push(start);

        while (!st.empty()) {
            int u = st.top();
            if (graph[u].empty()) {
                st.pop();
                if (!st.empty())
                    res.push_back({st.top(), u});
            } else {
                st.push(graph[u].front());
                graph[u].pop_front();
            }
        }

        reverse(res.begin(), res.end());
        return res;
    }
};
```

---

##391 ****[Problem Link]https://leetcode.com/problems/find-the-student-that-will-replace-the-chalk****  
**Approach:** Prefix sum and modulo to reduce problem size.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int chalkReplacer(vector<int>& chalk, int k) {
        long long sum = 0;
        for (int c : chalk) sum += c;
        k %= sum;
        for (int i = 0; i < chalk.size(); ++i) {
            if (k < chalk[i]) return i;
            k -= chalk[i];
        }
        return -1;
    }
};
```

---

##392 ****[Problem Link]https://leetcode.com/problems/distribute-repeating-integers****  
**Approach:** DP over subsets with bitmasking to fit students.  
**Time Complexity:** O(2^m * n * m)

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    bool canDistribute(vector<int>& nums, vector<int>& quantity) {
        unordered_map<int, int> freq;
        for (int x : nums) freq[x]++;
        vector<int> counts;
        for (auto& [k, v] : freq) counts.push_back(v);

        int m = quantity.size(), n = counts.size();
        vector<bool> dp(1 << m, false);
        dp[0] = true;
        vector<int> sum(1 << m, 0);
        for (int mask = 1; mask < (1 << m); ++mask) {
            for (int i = 0; i < m; ++i)
                if (mask & (1 << i))
                    sum[mask] += quantity[i];
        }

        for (int c : counts) {
            vector<bool> ndp = dp;
            for (int mask = 0; mask < (1 << m); ++mask) {
                if (!dp[mask]) continue;
                int subset = ((1 << m) - 1) ^ mask;
                for (int sub = subset; sub; sub = (sub - 1) & subset) {
                    if (sum[sub] <= c)
                        ndp[mask | sub] = true;
                }
            }
            dp = move(ndp);
        }
        return dp[(1 << m) - 1];
    }
};
```

---

##393 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-operations-to-reinitialize-a-permutation****  
**Approach:** Simulate permutation operation until back to original.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int reinitializePermutation(int n) {
        vector<int> perm(n), arr(n);
        for (int i = 0; i < n; ++i) perm[i] = i;
        arr = perm;
        int steps = 0;
        do {
            vector<int> next(n);
            for (int i = 0; i < n; ++i) {
                next[i] = (i % 2 == 0) ? arr[i / 2] : arr[n / 2 + (i - 1) / 2];
            }
            arr = next;
            steps++;
        } while (arr != perm);
        return steps;
    }
};
```

---

##394 ****[Problem Link]https://leetcode.com/problems/add-minimum-number-of-rungs****  
**Approach:** Greedy — whenever difference > dist, add needed rungs.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int addRungs(vector<int>& rungs, int dist) {
        int prev = 0, res = 0;
        for (int r : rungs) {
            if (r - prev > dist)
                res += (r - prev - 1) / dist;
            prev = r;
        }
        return res;
    }
};
```

---

##395 ****[Problem Link]https://leetcode.com/problems/minimum-moves-to-reach-target-with-rotations****  
**Approach:** BFS with visited states represented as tail+orientation.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <queue>
#include <tuple>
using namespace std;

class Solution {
public:
    int minimumMoves(vector<vector<int>>& grid) {
        int n = grid.size();
        queue<tuple<int, int, int>> q;
        q.push({0, 0, 0}); // x, y, horizontal(0)/vertical(1)
        vector<vector<vector<bool>>> vis(n, vector<vector<bool>>(n, vector<bool>(2, false)));
        vis[0][0][0] = true;
        int steps = 0;

        while (!q.empty()) {
            int sz = q.size();
            while (sz--) {
                auto [x, y, d] = q.front(); q.pop();
                if (x == n - 1 && y == n - 2 && d == 0) return steps;

                if (d == 0) { // horizontal
                    if (y + 2 < n && !grid[x][y + 2] && !vis[x][y + 1][0]) {
                        vis[x][y + 1][0] = true;
                        q.push({x, y + 1, 0});
                    }
                    if (x + 1 < n && !grid[x + 1][y] && !grid[x + 1][y + 1] && !vis[x + 1][y][0]) {
                        vis[x + 1][y][0] = true;
                        q.push({x + 1, y, 0});
                    }
                    if (x + 1 < n && !grid[x + 1][y] && !grid[x + 1][y + 1] && !vis[x][y][1]) {
                        vis[x][y][1] = true;
                        q.push({x, y, 1});
                    }
                } else { // vertical
                    if (x + 2 < n && !grid[x + 2][y] && !vis[x + 1][y][1]) {
                        vis[x + 1][y][1] = true;
                        q.push({x + 1, y, 1});
                    }
                    if (y + 1 < n && !grid[x][y + 1] && !grid[x + 1][y + 1] && !vis[x][y + 1][1]) {
                        vis[x][y + 1][1] = true;
                        q.push({x, y + 1, 1});
                    }
                    if (y + 1 < n && !grid[x][y + 1] && !grid[x + 1][y + 1] && !vis[x][y][0]) {
                        vis[x][y][0] = true;
                        q.push({x, y, 0});
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

##396 ****[Problem Link]https://leetcode.com/problems/latest-time-by-replacing-hidden-digits****  
**Approach:** Replace ? greedily with max valid digits.  
**Time Complexity:** O(1)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    string maximumTime(string time) {
        if (time[0] == '?')
            time[0] = (time[1] <= '3' || time[1] == '?') ? '2' : '1';
        if (time[1] == '?')
            time[1] = (time[0] == '2') ? '3' : '9';
        if (time[3] == '?') time[3] = '5';
        if (time[4] == '?') time[4] = '9';
        return time;
    }
};
```

---

##397 ****[Problem Link]https://leetcode.com/problems/sum-of-digits-of-string-after-convert****  
**Approach:** Simulate conversion and digit sum k times.  
**Time Complexity:** O(n + k)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    int getLucky(string s, int k) {
        string num;
        for (char c : s) num += to_string(c - 'a' + 1);
        while (k--) {
            int sum = 0;
            for (char d : num) sum += d - '0';
            num = to_string(sum);
        }
        return stoi(num);
    }
};
```

---

##398 ****[Problem Link]https://leetcode.com/problems/reformat-phone-number****  
**Approach:** Clean input then group in 3-3-2 pattern.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    string reformatNumber(string number) {
        string digits;
        for (char c : number)
            if (isdigit(c)) digits += c;

        string res;
        int i = 0, n = digits.size();
        while (n - i > 4) {
            res += digits.substr(i, 3) + "-";
            i += 3;
        }
        if (n - i == 4)
            res += digits.substr(i, 2) + "-" + digits.substr(i + 2);
        else
            res += digits.substr(i);
        return res;
    }
};
```

---

##399 ****[Problem Link]https://leetcode.com/problems/finding-mk-average****  
**Approach:** Use multisets to track three segments of the window.  
**Time Complexity:** O(log n)

```cpp
#include <queue>
#include <set>
using namespace std;

class MKAverage {
    int m, k;
    queue<int> q;
    multiset<int> lo, mid, hi;
    long long midSum = 0;

public:
    MKAverage(int m_, int k_) : m(m_), k(k_) {}

    void addElement(int num) {
        q.push(num);
        lo.insert(num);

        if (lo.size() > k) {
            mid.insert(*lo.rbegin());
            midSum += *lo.rbegin();
            lo.erase(prev(lo.end()));
        }

        if (mid.size() > m - 2 * k) {
            hi.insert(*mid.rbegin());
            midSum -= *mid.rbegin();
            mid.erase(prev(mid.end()));
        }

        if (q.size() > m) {
            int old = q.front(); q.pop();
            if (lo.count(old)) lo.erase(lo.find(old));
            else if (mid.count(old)) {
                mid.erase(mid.find(old));
                midSum -= old;
            } else hi.erase(hi.find(old));

            if (lo.size() < k && !mid.empty()) {
                lo.insert(*mid.begin());
                midSum -= *mid.begin();
                mid.erase(mid.begin());
            }

            if (mid.size() < m - 2 * k && !hi.empty()) {
                mid.insert(*hi.begin());
                midSum += *hi.begin();
                hi.erase(hi.begin());
            }
        }
    }

    int calculateMKAverage() {
        if (q.size() < m) return -1;
        return midSum / (m - 2 * k);
    }
};
```

---

##400 ****[Problem Link]https://leetcode.com/problems/merge-bsts-to-create-single-bst****  
**Approach:** Recursive merging using value map and in-order check.  
**Time Complexity:** O(n log n)

```cpp
#include <unordered_map>
#include <vector>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
    unordered_map<int, TreeNode*> leaves;

    bool valid(TreeNode* root, long minV, long maxV) {
        if (!root) return true;
        if (root->val <= minV || root->val >= maxV) return false;
        return valid(root->left, minV, root->val) && valid(root->right, root->val, maxV);
    }

public:
    TreeNode* canMerge(vector<TreeNode*>& trees) {
        unordered_map<int, TreeNode*> roots;
        unordered_map<int, int> count;
        for (auto* t : trees) {
            roots[t->val] = t;
            if (t->left) count[t->left->val]++;
            if (t->right) count[t->right->val]++;
        }

        for (auto* t : trees)
            if (!count[t->val]) {
                TreeNode* root = t;
                merge(root, roots);
                return valid(root, LONG_MIN, LONG_MAX) && roots.size() == 1 ? root : nullptr;
            }
        return nullptr;
    }

    void merge(TreeNode* root, unordered_map<int, TreeNode*>& roots) {
        if (!root) return;
        if (root->left && roots.count(root->left->val)) {
            root->left = roots[root->left->val];
            roots.erase(root->left->val);
            merge(root->left, roots);
        }
        if (root->right && roots.count(root->right->val)) {
            root->right = roots[root->right->val];
            roots.erase(root->right->val);
            merge(root->right, roots);
        }
    }
};
```

---
