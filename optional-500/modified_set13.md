##1 ****[Problem Link]https://leetcode.com/problems/maximize-grid-happiness****
```cpp
// Your solution here
int dp[1<<5][1<<5][7][7][6][6][3];
int m,n;
int rec(int i, int j, int previ, int preve, int last, int lefti, int lefte) {
    if (lefti<0 || lefte<0) return -1e9;
    if (j==n) return rec(i+1,0,previ,preve,last,lefti,lefte);
    if (i==m) return 0;

    int &ans=dp[previ][preve][lefti][lefte][i][j][last];
    if (ans!=-1) return ans;

    ans=-1e9;
    int ihappy=120, ehappy=40;
    if (i && (previ>>j)&1) { ihappy-=60; ehappy-=10; } 
    else if (i && (preve>>j)&1) { ihappy-=10; ehappy+=40; }
    if (j && last==1) { ihappy-=60; ehappy-=10; }
    else if (j && last==2) { ihappy-=10; ehappy+=40; }
    ans=max(ans, rec(i,j+1,previ&(~(1<<j)),preve&(~(1<<j)),0,lefti,lefte));
    ans=max(ans, rec(i,j+1,previ|(1<<j),preve&(~(1<<j)),1,lefti-1,lefte)+ihappy);
    ans=max(ans, rec(i,j+1,previ&(~(1<<j)),preve|(1<<j),2,lefti,lefte-1)+ehappy);
    return ans;
}

class Solution {
public:
int getMaxGridHappiness(int m_, int n_, int introvertsCount, int extrovertsCount) {
    m=m_,n=n_;
    if (n>m) swap(m,n);
    memset(dp,-1,sizeof(dp));
    return rec(0,0,0,0,0,introvertsCount,extrovertsCount);
}
};
```
##2 ****[Problem Link]https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies****
```cpp
// Your solution here
class Solution {
public:
    vector<int> sortItems(int n, int m, vector<int>& group, vector<vector<int>>& beforeItems) {
        for (int i = 0; i < n; i++)
            if (group[i] == -1)
                group[i] = m++;

        vector<vector<int>> itemGraph(n), groupGraph(m);
        vector<int> itemIndegree(n), groupIndegree(m);

        for (int i = 0; i < n; i++) {
            for (int b : beforeItems[i]) {
                itemGraph[b].push_back(i);
                itemIndegree[i]++;
                if (group[i] != group[b]) {
                    groupGraph[group[b]].push_back(group[i]);
                    groupIndegree[group[i]]++;
                }
            }
        }

        auto topSort = [](vector<vector<int>>& graph, vector<int>& indegree) {
            vector<int> res;
            queue<int> q;
            for (int i = 0; i < indegree.size(); i++)
                if (indegree[i] == 0) q.push(i);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                res.push_back(u);
                for (int v : graph[u])
                    if (--indegree[v] == 0) q.push(v);
            }
            return res.size() == indegree.size() ? res : vector<int>();
        };

        vector<int> itemOrder = topSort(itemGraph, itemIndegree);
        vector<int> groupOrder = topSort(groupGraph, groupIndegree);
        if (itemOrder.empty() || groupOrder.empty()) return {};

        unordered_map<int, vector<int>> groupedItems;
        for (int item : itemOrder)
            groupedItems[group[item]].push_back(item);

        vector<int> res;
        for (int grp : groupOrder)
            for (int item : groupedItems[grp])
                res.push_back(item);

        return res;
    }
};
```
##3 ****[Problem Link]https://leetcode.com/problems/smallest-missing-genetic-value-in-each-subtree****
```cpp
// Your solution here
class Solution {
public:
    int n;
    vector<int> adj[100001];
    vector<int> nums, par;
    int vis[100002];

    void dfs(int u) {
        vis[nums[u]] = 1;
        for (int v : adj[u]) {
            if (!vis[nums[v]]) dfs(v);
        }
    }

    vector<int> smallestMissingValueSubtree(vector<int>& _par, vector<int>& _nums) {
        nums = std::move(_nums);
        par = std::move(_par);
        n = nums.size();
        vector<int> ans(n, 1);

        for (int i = 0; i < n; i++) {
            if (par[i] != -1)
                adj[par[i]].push_back(i);
        }
        int u = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) {
                u = i; break;
            }
        }
        int lo = 1;
        while (u != -1) {
            dfs(u);
            while (vis[lo]) lo++;
            ans[u] = lo;
            u = par[u];
        }
        return ans;
    }
};
```
##4 ****[Problem Link]https://leetcode.com/problems/least-operators-to-express-number****
```cpp
// Your solution here
class Solution {
public:
    int leastOpsExpressTarget(int x, int target) {
        // Memoization table to store results for sub-targets
        unordered_map<long long, int> memo;
        
        // Recursive lambda function (DFS with memo) to compute cost for a given value t
        function<int(long long)> dfs = [&](long long t) -> int {
            if (t == 0) {
                return 0;  // 0 can be formed as x - x, but we count that in the caller context
            }
            if (t < x) {
                // If t is smaller than x, choose the cheaper between:
                // adding 1 (x/x) t times vs. subtracting from x.
                int opsAdd = 2 * (int)t - 1;       // cost = t divisions for 1's + (t-1) additions
                int opsSub = 2 * (int)(x - t);    // cost = (x-t) divisions for 1's + (x-t) - 1 additions + 1 subtraction
                return min(opsAdd, opsSub);
            }
            if (memo.count(t)) {
                return memo[t];  // return cached result if already computed
            }
            
            // Find largest power of x (say x^k) that is <= t
            long long p = 1;
            int k = 0;
            while (p * x <= t) {
                p *= x;
                ++k;
            }
            
            if (p == t) {
                // t is exactly x^k, needs k-1 multiplications (e.g., x*x*...*x with k terms has k-1 operators)
                memo[t] = k - 1;
                return k - 1;
            }
            
            // Compute cost if we use p = x^k as part of the expression
            long long remainder = t - p;
            int costUseLower = dfs(remainder) + k;  
            // (k includes k-1 multiplications for x^k and 1 addition to add the remainder)
            
            // Compute cost if we use the next power p*x = x^(k+1) and subtract the excess
            long long leftOver = p * x - t;  // this is the difference we would subtract
            int costUseHigher = INT_MAX;
            if (leftOver < t) {
                costUseHigher = dfs(leftOver) + k + 1;  
                // (k+1 includes k multiplications for x^(k+1) and 1 subtraction operator)
            }
            
            int result = min(costUseLower, costUseHigher);
            memo[t] = result;
            return result;
        };
        
        return dfs(target);
    }
};
```
##5 ****[Problem Link]https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero****
```cpp
// Your solution here
class Solution {
public:
    int minimumOneBitOperations(int n) {
        if (n == 0) return 0;
        int k = 0;
        while ((1 << (k + 1)) <= n) ++k;
        return (1 << (k + 1)) - 1 - minimumOneBitOperations(n ^ (1 << k));
    }
};
```

##6 ****[Problem Link]https://leetcode.com/problems/second-minimum-time-to-reach-destination****
```cpp
class Solution {
public:
    int secondMinimum(int n, vector<vector<int>>& edges, int time, int change) {
        vector<vector<int>> adj(n + 1);

        for (auto& edge : edges) {
            adj[edge[0]].push_back(edge[1]);
            adj[edge[1]].push_back(edge[0]);
        }

        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minHeap;
        vector<int> dist1(n + 1, INT_MAX), dist2(n + 1, INT_MAX), freq(n + 1, 0);

        minHeap.push({0, 1});

        while (!minHeap.empty()) {
            auto [timeTaken, node] = minHeap.top();
            minHeap.pop();

            freq[node]++;

            if (freq[node] == 2 && node == n) {
                return timeTaken;
            }

            for (auto neighbor : adj[node]) {
                int newTimeTaken = timeTaken;

                if ((newTimeTaken / change) % 2) {
                    newTimeTaken = change * (newTimeTaken / change + 1) + time;
                } else {
                    newTimeTaken = newTimeTaken + time;
                }

                if (dist1[neighbor] > newTimeTaken) {
                    dist2[neighbor] = dist1[neighbor];
                    dist1[neighbor] = newTimeTaken;
                    minHeap.push({newTimeTaken, neighbor});
                } else if (dist2[neighbor] > newTimeTaken && dist1[neighbor] != newTimeTaken) {
                    dist2[neighbor] = newTimeTaken;
                    minHeap.push({newTimeTaken, neighbor});
                }
            }
        }
        return 0;
    }
};
```

##7 ****[Problem Link]https://leetcode.com/problems/verbal-arithmetic-puzzle****
```cpp
class Solution {
    int map[256];
    int visited[10];
public:
    bool isSolvable(vector<string>& words, string result) {
        for(int i=0; i<256; i++)
            map[i] = -1;
        reverse(result.begin(), result.end());
        for(auto& word: words) {
            if(word.size()>result.size())
                return false;
            reverse(word.begin(), word.end());
        }
        return dfs(words, result, 0, 0, 0);
    }
    bool dfs(vector<string>& words, string result, int j, int i, int sum) {
        if( j == result.size() ) {
            if( sum != 0 )
                return false;
            if(result.size()>1 && map[result.back()]==0) //result has leading zero.
                return false;
            return true;
        };
        if( i == words.size() ) { //result row
            if(map[ result[j] ] != -1) {//already mapped.
                if(map[ result[j] ] != sum%10)
                    return false;
                else
                    return dfs(words, result, j+1, 0, sum/10);
            } else {
                if(visited[sum%10] == 1)
                    return false;
                map[result[j]] = sum%10;
                visited[sum%10] = 1;
                if(dfs(words, result, j+1, 0, sum/10))
                    return true;
                map[result[j]] = -1;
                visited[sum%10] = 0;
                return false;
            }
        };

        if( j >= words[i].length() )
            return dfs(words, result, j, i+1, sum);
        char ch = words[i][j];
        if(map[ch]!=-1) {
            if(words[i].size()>1 && j==words.size()-1 && map[ch]==0) //leading zero
                return false;
            return dfs(words, result, j, i+1, sum+map[ch]);
        } else {
            for(int d=0; d<=9; d++) {
                if(visited[d])
                    continue;
                if(d==0 && words[i].size()>1 && j==words[i].size()-1)
                    continue;
                map[ch] = d;
                visited[d] = 1;
                if(dfs(words, result, j, i+1, sum+d))
                    return true;
                map[ch] = -1;
                visited[d] = 0;
            }
            return false;
        }
        return true;
    }
};
```
##8 ****[Problem Link]https://leetcode.com/problems/minimum-operations-to-convert-number****
```cpp
// Your solution here
class Solution {
public:
    int minimumOperations(vector<int>& nums, int start, int goal) 
    {
        queue<int> que;
        que.push(start); 
        vector<int> visit(1001, 0); 
        int level = 0;
        
        while(!que.empty()) {
            int size = que.size(); 
            for(int i=0; i<size; i++) {
                int curr = que.front(); 
                que.pop();
                if(curr == goal)
                    return level; 
                if(curr<0 || curr>1000 || visit[curr] == 1)
                    continue; 
                visit[curr] = 1; 
                for(int i=0; i< nums.size(); i++) {
                    que.push(curr + nums[i]);
                    que.push(curr - nums[i]); 
                    que.push(curr ^ nums[i]);                    
                }
            }
            level++; 
        }
        return -1; 
    }
};
```
##9 ****[Problem Link]https://leetcode.com/problems/two-best-non-overlapping-events****
```cpp
// Your solution here
```
##10 ****[Problem Link]https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference****
```cpp
// Your solution here
```
##11 ****[Problem Link]https://leetcode.com/problems/minimum-cost-to-connect-two-groups-of-points****
```cpp
// Your solution here
```
##12 ****[Problem Link]https://leetcode.com/problems/parse-lisp-expression****
```cpp
// Your solution here
```
##13 ****[Problem Link]https://leetcode.com/problems/minimize-malware-spread-ii****
```cpp
// Your solution here
```
##14 ****[Problem Link]https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons****
```cpp
// Your solution here
```
##15 ****[Problem Link]https://leetcode.com/problems/string-compression-ii****
```cpp
// Your solution here
```
##16 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-work-sessions-to-finish-the-tasks****
```cpp
// Your solution here
```
##17 ****[Problem Link]https://leetcode.com/problems/falling-squares****
```cpp
// Your solution here
```
##18 ****[Problem Link]https://leetcode.com/problems/equal-sum-arrays-with-minimum-number-of-operations****
```cpp
// Your solution here
```
##19 ****[Problem Link]https://leetcode.com/problems/construct-quad-tree****
```cpp
// Your solution here
```
##20 ****[Problem Link]https://leetcode.com/problems/valid-permutations-for-di-sequence****
```cpp
// Your solution here
```
##21 ****[Problem Link]https://leetcode.com/problems/string-without-aaa-or-bbb****
```cpp
// Your solution here
```
##22 ****[Problem Link]https://leetcode.com/problems/strong-password-checker****
```cpp
// Your solution here
```
##23 ****[Problem Link]https://leetcode.com/problems/pyramid-transition-matrix****
```cpp
// Your solution here
```
##24 ****[Problem Link]https://leetcode.com/problems/increasing-decreasing-string****
```cpp
// Your solution here
```
##25 ****[Problem Link]https://leetcode.com/problems/pizza-with-3n-slices****
```cpp
// Your solution here
```
##26 ****[Problem Link]https://leetcode.com/problems/check-if-there-is-a-valid-path-in-a-grid****
```cpp
// Your solution here
```
##27 ****[Problem Link]https://leetcode.com/problems/stamping-the-sequence****
```cpp
// Your solution here
```
##28 ****[Problem Link]https://leetcode.com/problems/maximum-students-taking-exam****
```cpp
// Your solution here
```
##29 ****[Problem Link]https://leetcode.com/problems/max-dot-product-of-two-subsequences****
```cpp
// Your solution here
```
##30 ****[Problem Link]https://leetcode.com/problems/poor-pigs****
```cpp
// Your solution here
```
##31 ****[Problem Link]https://leetcode.com/problems/smallest-sufficient-team****
```cpp
// Your solution here
```
##32 ****[Problem Link]https://leetcode.com/problems/ugly-number-iii****
```cpp
// Your solution here
```
##33 ****[Problem Link]https://leetcode.com/problems/swap-for-longest-repeated-character-substring****
```cpp
// Your solution here
```
##34 ****[Problem Link]https://leetcode.com/problems/find-and-replace-in-string****
```cpp
// Your solution here
```
##35 ****[Problem Link]https://leetcode.com/problems/swap-adjacent-in-lr-string****
```cpp
// Your solution here
```
##36 ****[Problem Link]https://leetcode.com/problems/k-th-smallest-prime-fraction****
```cpp
// Your solution here
```
##37 ****[Problem Link]https://leetcode.com/problems/race-car****
```cpp
// Your solution here
```
##38 ****[Problem Link]https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor****
```cpp
// Your solution here
```
##39 ****[Problem Link]https://leetcode.com/problems/binary-number-with-alternating-bits****
```cpp
// Your solution here
```
##40 ****[Problem Link]https://leetcode.com/problems/3sum-with-multiplicity****
```cpp
// Your solution here
```
##41 ****[Problem Link]https://leetcode.com/problems/count-unique-characters-of-all-substrings-of-a-given-string****
```cpp
// Your solution here
```
##42 ****[Problem Link]https://leetcode.com/problems/peeking-iterator****
```cpp
// Your solution here
```
##43 ****[Problem Link]https://leetcode.com/problems/shortest-path-with-alternating-colors****
```cpp
// Your solution here
```
##44 ****[Problem Link]https://leetcode.com/problems/second-minimum-node-in-a-binary-tree****
```cpp
// Your solution here
```
##45 ****[Problem Link]https://leetcode.com/problems/length-of-longest-fibonacci-subsequence****
```cpp
// Your solution here
```
