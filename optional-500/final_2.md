
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
