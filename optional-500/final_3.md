
##101 ****[Problem Link]https://leetcode.com/problems/largest-triangle-area****  
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

##102 ****[Problem Link]https://leetcode.com/problems/maximum-product-difference-between-two-pairs****  
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

##103 ****[Problem Link]https://leetcode.com/problems/nearest-exit-from-entrance-in-maze****  
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

##104 ****[Problem Link]https://leetcode.com/problems/dinner-plate-stacks****  
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

##105 ****[Problem Link]https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal****  
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

##106 ****[Problem Link]https://leetcode.com/problems/last-moment-before-all-ants-fall-out-of-a-plank****  
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

##107 ****[Problem Link]https://leetcode.com/problems/occurrences-after-bigram****  
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

##108 ****[Problem Link]https://leetcode.com/problems/number-of-unique-good-subsequences****  
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

##109 ****[Problem Link]https://leetcode.com/problems/queries-on-a-permutation-with-key****  
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

##110 ****[Problem Link]https://leetcode.com/problems/prime-palindrome****  
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

##111 ****[Problem Link]https://leetcode.com/problems/making-file-names-unique****  
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

##112 ****[Problem Link]https://leetcode.com/problems/largest-merge-of-two-strings****  
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

##113 ****[Problem Link]https://leetcode.com/problems/maximize-the-confusion-of-an-exam****  
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

##114 ****[Problem Link]https://leetcode.com/problems/relative-ranks****  
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

##115 ****[Problem Link]https://leetcode.com/problems/thousand-separator****  
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

##116 ****[Problem Link]https://leetcode.com/problems/super-palindromes****  
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

##117 ****[Problem Link]https://leetcode.com/problems/reconstruct-a-2-row-binary-matrix****  
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

##118 ****[Problem Link]https://leetcode.com/problems/minimum-cost-to-connect-two-groups-of-points****  
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

##119 ****[Problem Link]https://leetcode.com/problems/get-maximum-in-generated-array****  
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

##120 ****[Problem Link]https://leetcode.com/problems/largest-odd-number-in-string****  
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

##121 ****[Problem Link]https://leetcode.com/problems/average-waiting-time****  
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

##122 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous****  
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

##123 ****[Problem Link]https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference****  
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

##124 ****[Problem Link]https://leetcode.com/problems/sum-of-subarray-ranges****  
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

##125 ****[Problem Link]https://leetcode.com/problems/two-best-non-overlapping-events****  
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

##126 ****[Problem Link]https://leetcode.com/problems/closest-room****  
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

##127 ****[Problem Link]https://leetcode.com/problems/minimum-operations-to-convert-number****  
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

##128 ****[Problem Link]https://leetcode.com/problems/graph-connectivity-with-threshold****  
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

##129 ****[Problem Link]https://leetcode.com/problems/first-day-where-you-have-been-in-all-the-rooms****  
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

##130 ****[Problem Link]https://leetcode.com/problems/number-of-rectangles-that-can-form-the-largest-square****  
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

##131 ****[Problem Link]https://leetcode.com/problems/random-pick-with-weight****  
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

##132 ****[Problem Link]https://leetcode.com/problems/check-if-word-equals-summation-of-two-words****  
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

##133 ****[Problem Link]https://leetcode.com/problems/fraction-addition-and-subtraction****  
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

##134 ****[Problem Link]https://leetcode.com/problems/number-of-paths-with-max-score****  
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

##135 ****[Problem Link]https://leetcode.com/problems/verbal-arithmetic-puzzle****  
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

##136 ****[Problem Link]https://leetcode.com/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits****  
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

##137 ****[Problem Link]https://leetcode.com/problems/number-of-sets-of-k-non-overlapping-line-segments****  
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

##138 ****[Problem Link]https://leetcode.com/problems/strange-printer-ii****  
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

##139 ****[Problem Link]https://leetcode.com/problems/find-the-kth-largest-integer-in-the-array****  
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

##140 ****[Problem Link]https://leetcode.com/problems/shortest-completing-word****  
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

##141 ****[Problem Link]https://leetcode.com/problems/count-odd-numbers-in-an-interval-range****  
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

##142 ****[Problem Link]https://leetcode.com/problems/circular-array-loop****  
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

##143 ****[Problem Link]https://leetcode.com/problems/maximum-ice-cream-bars****  
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

##144 ****[Problem Link]https://leetcode.com/problems/reduction-operations-to-make-the-array-elements-equal****  
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

##145 ****[Problem Link]https://leetcode.com/problems/count-the-repetitions****  
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

##146 ****[Problem Link]https://leetcode.com/problems/transform-to-chessboard****  
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

##147 ****[Problem Link]https://leetcode.com/problems/moving-stones-until-consecutive-ii****  
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

##148 ****[Problem Link]https://leetcode.com/problems/maximum-binary-string-after-change****  
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

##149 ****[Problem Link]https://leetcode.com/problems/final-value-of-variable-after-performing-operations****  
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

##150 ****[Problem Link]https://leetcode.com/problems/minimum-area-rectangle-ii****  
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

##151 ****[Problem Link]https://leetcode.com/problems/generate-a-string-with-characters-that-have-odd-counts****  
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

##152 ****[Problem Link]https://leetcode.com/problems/random-flip-matrix****  
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

##153 ****[Problem Link]https://leetcode.com/problems/preimage-size-of-factorial-zeroes-function****  
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

##154 ****[Problem Link]https://leetcode.com/problems/eliminate-maximum-number-of-monsters****  
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

##155 ****[Problem Link]https://leetcode.com/problems/find-target-indices-after-sorting-array****  
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

##156 ****[Problem Link]https://leetcode.com/problems/sum-game****  
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

##157 ****[Problem Link]https://leetcode.com/problems/number-of-different-integers-in-a-string****  
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

##158 ****[Problem Link]https://leetcode.com/problems/range-frequency-queries****  
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

##159 ****[Problem Link]https://leetcode.com/problems/sum-of-beauty-of-all-substrings****  
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

##160 ****[Problem Link]https://leetcode.com/problems/count-nodes-with-the-highest-score****  
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

##161 ****[Problem Link]https://leetcode.com/problems/splitting-a-string-into-descending-consecutive-values****  
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

##162 ****[Problem Link]https://leetcode.com/problems/largest-values-from-labels****  
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

##163 ****[Problem Link]https://leetcode.com/problems/can-convert-string-in-k-moves****  
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

##164 ****[Problem Link]https://leetcode.com/problems/find-the-distance-value-between-two-arrays****  
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

##165 ****[Problem Link]https://leetcode.com/problems/sum-of-floored-pairs****  
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

##166 ****[Problem Link]https://leetcode.com/problems/painting-a-grid-with-three-different-colors****  
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

##167 ****[Problem Link]https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies****  
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

##168 ****[Problem Link]https://leetcode.com/problems/second-minimum-time-to-reach-destination****  
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

##169 ****[Problem Link]https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store****  
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

##170 ****[Problem Link]https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests****  
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

##171 ****[Problem Link]https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero****  
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

##172 ****[Problem Link]https://leetcode.com/problems/count-subtrees-with-max-distance-between-cities****  
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

##173 ****[Problem Link]https://leetcode.com/problems/teemo-attacking****  
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

##174 ****[Problem Link]https://leetcode.com/problems/find-xor-sum-of-all-pairs-bitwise-and****  
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

##175 ****[Problem Link]https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting****  
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

##176 ****[Problem Link]https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings****  
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

##177 ****[Problem Link]https://leetcode.com/problems/find-the-middle-index-in-array****  
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

##178 ****[Problem Link]https://leetcode.com/problems/plates-between-candles****  
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

##179 ****[Problem Link]https://leetcode.com/problems/find-kth-largest-xor-coordinate-value****  
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

##180 ****[Problem Link]https://leetcode.com/problems/count-good-numbers****  
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

##181 ****[Problem Link]https://leetcode.com/problems/invalid-transactions****  
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

##182 ****[Problem Link]https://leetcode.com/problems/minimum-time-to-type-word-using-special-typewriter****  
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

##183 ****[Problem Link]https://leetcode.com/problems/evaluate-the-bracket-pairs-of-a-string****  
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

##184 ****[Problem Link]https://leetcode.com/problems/find-all-groups-of-farmland****  
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

##185 ****[Problem Link]https://leetcode.com/problems/find-all-good-strings****  
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

##186 ****[Problem Link]https://leetcode.com/problems/smallest-good-base****  
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

##187 ****[Problem Link]https://leetcode.com/problems/decrease-elements-to-make-array-zigzag****  
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

##188 ****[Problem Link]https://leetcode.com/problems/find-a-value-of-a-mysterious-function-closest-to-target****  
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

##189 ****[Problem Link]https://leetcode.com/problems/finding-pairs-with-a-certain-sum****  
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

##190 ****[Problem Link]https://leetcode.com/problems/least-operators-to-express-number****  
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

##191 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating****  
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

##192 ****[Problem Link]https://leetcode.com/problems/merge-triplets-to-form-target-triplet****  
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

##193 ****[Problem Link]https://leetcode.com/problems/vowels-of-all-substrings****  
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

##194 ****[Problem Link]https://leetcode.com/problems/ambiguous-coordinates****  
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

##195 ****[Problem Link]https://leetcode.com/problems/describe-the-painting****  
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

##196 ****[Problem Link]https://leetcode.com/problems/count-pairs-with-xor-in-a-range****  
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

##197 ****[Problem Link]https://leetcode.com/problems/count-vowel-substrings-of-a-string****  
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

##198 ****[Problem Link]https://leetcode.com/problems/optimal-division****  
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

##199 ****[Problem Link]https://leetcode.com/problems/find-greatest-common-divisor-of-array****  
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

##200 ****[Problem Link]https://leetcode.com/problems/sum-of-digits-in-base-k****  
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
