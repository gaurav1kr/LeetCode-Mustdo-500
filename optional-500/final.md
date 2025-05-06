
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
class Solution {
public:
    int maxTwoEvents(vector<vector<int>>& events) {
        int n = events.size();
        
        sort(events.begin(), events.end(), [](const vector<int>& a, const vector<int>& b) {
            return a[0] < b[0];
        });
        
        vector<int> suffixMax(n);
        suffixMax[n - 1] = events[n - 1][2];
        
        for (int i = n - 2; i >= 0; --i) {
            suffixMax[i] = max(events[i][2], suffixMax[i + 1]);
        }
        
        int maxSum = 0;
        
        for (int i = 0; i < n; ++i) {
            int left = i + 1, right = n - 1;
            int nextEventIndex = -1;
            
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (events[mid][0] > events[i][1]) {
                    nextEventIndex = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            
            if (nextEventIndex != -1) {
                maxSum = max(maxSum, events[i][2] + suffixMax[nextEventIndex]);
            }
            
            maxSum = max(maxSum, events[i][2]);
        }
        
        return maxSum;
    }
};
```
##10 ****[Problem Link]https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference****
```cpp
class Solution {
public:
    int minimumDifference(vector<int>& nums) {
        vector<int> left, right;
        int sum = 0;
        int n = nums.size() / 2;
        for(int i = 0; i < 2 * n; i++) {
            if(i < n)
                left.push_back(nums[i]);
            else
                right.push_back(nums[i]);
            sum += nums[i];
        }
        vector<vector<int>> sumLeft(n + 1), sumRight(n + 1); //sumRight[i] stores all possible sum of i elements in the Right array
        for(int i = 1; i < 1 << n; i++) {
            int count = 0;
            int subSetSum = 0;
            for(int j = 0; j < n; j++) {
                if((1 << j ) & i){
                    count++;
                    subSetSum += left[j];
                }
            }
            sumLeft[count].push_back(subSetSum);
            subSetSum = 0;
            count = 0;
            for(int j = 0; j < n; j++) {
                if((1 << j ) & i){
                    count++;
                    subSetSum += right[j];
                }
            }
            sumRight[count].push_back(subSetSum);
        }
        sumLeft[0].push_back(0);
        sumRight[0].push_back(0);
        for(auto it: sumLeft){
            sort(it.begin(), it.end());
        }
        int ans = INT_MAX;
        for(int i = 0; i <= n; i++) {
                sort(sumRight[n-i].begin(), sumRight[n-i].end());
            for(auto iSumLeft : sumLeft[i]) {
                int fi = (sum - 2*iSumLeft) / 2;
                auto it = lower_bound(sumRight[n-i].begin(), sumRight[n-i].end(), fi) - sumRight[n-i].begin();// try to find the closest element in the sumRight[n - i] array such that our answer is close to 0
                if(it != sumRight[n-i].size())
                    ans = min(ans, abs(sum - 2*((sumRight[n-i][it]) + iSumLeft)));
                if(it != 0){
                    ans = min(ans, abs(sum-2*(iSumLeft+(sumRight[n-i][it-1]))));}
            }
        }
        return ans;
    }
};
```
##11 ****[Problem Link]https://leetcode.com/problems/minimum-cost-to-connect-two-groups-of-points****
```cpp
class Solution {
public:
    int rec (int ind,int mask,vector<vector<int>>& cost,vector<vector<int>>&dp,vector<int>&mini)
    {
       if (dp[ind][mask]!=-1) return dp[ind][mask];
       int res=ind>=cost.size()?0:1e8;
       if (ind>=cost.size())
       {
           for (int j=0;j<cost[0].size();j++)
           {
               if ((mask&(1<<j))==0)
               {
                   res+=mini[j];
               }
           }
       }
       else {
           for (int j=0;j<cost[0].size();j++)
           {
               res=min(res,cost[ind][j]+rec(ind+1,mask | (1<<j),cost,dp,mini));
           }
       }
       return dp[ind][mask]=res;

    }
    int connectTwoGroups(vector<vector<int>>& cost) 
    {
        vector<vector<int>>dp(13,vector<int>(4096,-1));
        vector<int>mini(cost[0].size(),INT_MAX);
        for (int j=0;j<cost[0].size();j++)
        {
            for (int i=0;i<cost.size();i++)
            {
                mini[j]=min(mini[j],cost[i][j]);
            }
        }
        return rec(0,0,cost,dp,mini);
    }
};
```
##12 ****[Problem Link]https://leetcode.com/problems/parse-lisp-expression****
```cpp
class Solution {
public:

    string read_inst(string& exp, int& i){
        string inst;
        if (exp[i] == '-') inst += exp[i++];
        while(i<size(exp) && isalnum(exp[i])){
            inst += exp[i++];
        }
        return inst;
    }
    void restore_vars(unordered_map<string, int>& vars,unordered_map<string, int>& new_vars){
        for(auto& [var_name, prev_val]: new_vars){
            if (prev_val == INT_MAX)
                vars.erase(var_name);
            else
                vars[var_name] = prev_val;
        }
    }

    // Returns value of expression starts in indx i
    // expression is a variable, integer or one of ["let","add","mult"]
    int eval_aux(string& exp,unordered_map<string, int>& vars, int& i){
        int n = size(exp);
        if (exp[i] == '(' ) i++;
        
        while(i< n){
            string inst = read_inst(exp,i);
            
            // integer exp
            if (inst[0] == '-' || isdigit(inst[0]))
                return stoi(inst);
            
            //  variable exp
            if(vars.count(inst)){
                return vars[inst];
            }
            if (inst == "let"){
                unordered_map<string, int> new_vars;

                while (true){
                    string var_name = read_inst(exp, ++i);
                    // last operand reached
                    if (var_name.empty() || exp[i] == ')'){
                        int return_val;
                        if (var_name.empty()) // new parentheses expression
                            return_val =  eval_aux(exp,vars, i);
                        else                  // a variable name or an integer
                            return_val = islower(var_name[0]) ? vars[var_name] : stoi(var_name);
                        ++i; // skip ')'
                        restore_vars(vars, new_vars);
                        return return_val;
                    }
                    // store a new variable
                    if (new_vars.count(var_name) == 0)
                        new_vars[var_name] = vars.count(var_name) ? vars[var_name] : INT_MAX;
                    vars[var_name] = eval_aux(exp,vars, ++i);
                }

            }
            // instruction is "add" or "mult"
            else{
                int arg1 = eval_aux(exp, vars, ++i);
                int arg2 = eval_aux(exp, vars, ++i);
                ++i; // skip ')'
                return (inst == "add") ? arg1+arg2 : arg1*arg2;

            }
        }
        return -1; // should not reach here
    }

    int evaluate(string exp) {
        int i=0;
        unordered_map<string, int> vars;
        return eval_aux(exp,vars,i);
    }
};
```
##13 ****[Problem Link]https://leetcode.com/problems/minimize-malware-spread-ii****
```cpp

class Solution {
public:
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial) {
        int n=graph.size();
        vector<vector<int>> adj(graph.size());
        for(int i=0;i<graph.size();i++){
            for(int j=0;j<graph[0].size();j++){
                if(i!=j && graph[i][j]){
                    adj[i].push_back(j);
                }
            }
        }

        sort(initial.begin(), initial.end());

        int min_infected=INT_MAX;
        int ans=0;
        vector<int> base(n,0);
        for(auto &i:initial){
            base[i]=1;
        }

        for(auto &i:initial){
            vector<int> dp=base;
            dp[i]=0;
            vector<int> isvisited(n,false);
            queue<int> Q;
            for(auto &node:initial){
                if(node!=i && isvisited[node]==false){
                    Q.push(node);
                    isvisited[node]=true;
                }
            }
            while(!Q.empty()){
                int node=Q.front();
                Q.pop();
                for(auto &edge:adj[node]){
                    if(!isvisited[edge] && i!=edge){
                        isvisited[edge]=true;
                        Q.push(edge);
                        dp[edge]=1;
                    }
                }
            }
            int s=accumulate(dp.begin(),dp.end(),0);
            if(s<min_infected){
                min_infected=s;
                ans=i;
            }
        }


        return ans;


    }
};
```
##14 ****[Problem Link]https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons****
```cpp
class Solution {
public:
    int numOfArrays(int n, int m, int k) {
        const int MOD = 1000000007;
        vector<vector<vector<long long>>> dp(n + 1, vector<vector<long long>>(m + 1, vector<long long>(k + 1, 0)));
        
        // Initialize dp[1][i][1] to 1, as there's only one way to create
        // a single-element array with any number from 1 to m.
        for (int i = 1; i <= m; i++) {
            dp[1][i][1] = 1;
        }
        
        // Fill in the dp table using dynamic programming.
        for (int len = 2; len <= n; len++) {
            for (int maxVal = 1; maxVal <= m; maxVal++) {
                for (int cost = 1; cost <= k; cost++) {
                    // dp[len][maxVal][cost] can be calculated by summing up
                    // dp[len-1][i][cost-1] for i < maxVal and dp[len-1][maxVal][cost] * maxVal.
                    long long sum = 0;
                    for (int i = 1; i < maxVal; i++) {
                        sum = (sum + dp[len - 1][i][cost - 1]) % MOD;
                    }
                    dp[len][maxVal][cost] = (dp[len - 1][maxVal][cost] * maxVal + sum) % MOD;
                }
            }
        }
        
        // Sum up the possibilities for the final state dp[n][1..m][k].
        long long ans = 0;
        for (int i = 1; i <= m; i++) {
            ans = (ans + dp[n][i][k]) % MOD;
        }
        
        return ans;
    }
};
```
##15 ****[Problem Link]https://leetcode.com/problems/string-compression-ii****
```cpp
class Solution {
public:
    int getLengthOfOptimalCompression(string s, int k) {
        int n = s.size();
        int m = k;

        int dp[110][110] = {};
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j <= i && j <= m; ++j) {
                int need_remove = 0;
                int group_count = 0;
                dp[i][j] = INT_MAX;
                if (j) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                for (int k = i; k >= 1; --k) {
                    if (s[k - 1] != s[i - 1]) {
                        need_remove += 1;
                    } else {
                        group_count += 1;
                    }

                    if (need_remove > j) {
                        break;
                    }

                    if (group_count == 1) {
                        dp[i][j] = min(dp[i][j], dp[k - 1][j - need_remove] + 1);
                    } else if (group_count < 10) {
                        dp[i][j] = min(dp[i][j], dp[k - 1][j - need_remove] + 2);
                    } else if (group_count < 100) {
                        dp[i][j] = min(dp[i][j], dp[k - 1][j - need_remove] + 3);
                    } else {
                        dp[i][j] = min(dp[i][j], dp[k - 1][j - need_remove] + 4);
                    }
                }
            }
        }

        return dp[n][m];
    }
};
```
##16 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-work-sessions-to-finish-the-tasks****
```cpp
class Solution {
public:
    int minSessions(vector<int>& tasks, int sessionTime) {
        sort(tasks.rbegin(), tasks.rend());
        int left = 1, right = tasks.size();

        while (left < right) {
            int mid = (left + right) / 2;
            vector<int> sessions(mid, 0);
            if (canPartition(tasks, sessions, 0, sessionTime)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
    
    bool canPartition(vector<int>& tasks, vector<int>& sessions, int index, int sessionTime) {
        if (index == tasks.size()) return true;
        unordered_set<int> seen;

        for (int& session : sessions) {
            if (seen.count(session)) continue;
            if (session + tasks[index] <= sessionTime) {
                session += tasks[index];
                if (canPartition(tasks, sessions, index + 1, sessionTime)) return true;
                session -= tasks[index];
                
                seen.insert(session);
                if (session == 0) break;
            }
        }

        return false;
    }
};
```

##17 ****[Problem Link]https://leetcode.com/problems/falling-squares****
```cpp
class Solution {
public:
    map < long long , long long > segment;
    map < long long , long long > lazy;
void lazy_prog ( long long seed , long long low , long long high )
{
    if ( lazy[seed] )
    {
        segment[seed] = lazy[seed];
        if ( low != high )
        {
            lazy[seed*2+1] = lazy[seed];
            lazy[seed*2+2] = lazy[seed];
        }
        lazy[seed] = 0;
        
    }
}
long long findhighest ( long long seed , long long blow , long long bhigh , long long low , long long high )
{
    lazy_prog ( seed , blow , bhigh );
    if ( blow >= low && bhigh <= high ) return segment[seed];
    if ( blow > high || bhigh < low ) return 0;
    long long mid = ( blow + bhigh ) / 2;
    return max ( findhighest( seed*2 + 1 , blow , mid , low , high )
                , findhighest ( seed*2 + 2 , mid + 1, bhigh , low , high ) );
}
void add ( long long seed , long long blow , long long bhigh , long long low , long long high , long long key )
{
    lazy_prog ( seed , blow , bhigh );
    if ( blow >= low && bhigh <= high ){
        segment[seed] = key;
        lazy[seed] = key;
        return;
    }
    if ( blow > high || bhigh < low ) return;
    
    long long mid = ( blow + bhigh ) / 2;
    add ( seed*2 + 1 , blow , mid , low , high , key );
    add ( seed*2 + 2 , mid + 1 , bhigh , low , high , key );
    segment[seed] = max(segment[seed*2+1] , segment[seed*2+2] );
}
    vector<int> fallingSquares(vector<vector<int>>& positions) {
        vector < int > result ;
        long long l , r, height;
        for ( int i = 0 ; i < positions.size() ; i ++ )
        {
            for ( int j = 0 ; j < 2 ; j ++ )
            {
                if ( j == 0 ) l = positions[i][j];
                else {
                    r = l + positions[i][j] - 1;
                    height = positions[i][j];
                }
            }
            long long highest = findhighest ( 0 , 0 , 101000005, l , r );
            add ( 0 , 0 , 101000005, l , r, highest + height );
            result.push_back(segment[0]);
        }
        return result ;
    }
};
```

##18 ****[Problem Link]https://leetcode.com/problems/equal-sum-arrays-with-minimum-number-of-operations****
```cpp
class Solution {
public:
    int minOperations(vector<int>& nums1, vector<int>& nums2) {

        int n1 = nums1.size();

        int n2 = nums2.size();

        // find the sum of nums1

        int sum1 = accumulate(nums1.begin(), nums1.end(), 0);

        // find the sum of nums2

        int sum2 = accumulate(nums2.begin(), nums2.end(), 0);

        // declare a max heap

        priority_queue<int> pq;

        // case 1 :- if sum1 > sum2 

        // we have to check maximum decrement for each element of nums1

        // we have to check maximum increment for each element of nums2

        if(sum1 > sum2)
        {
            // find max. decrement for each element of nums1 and push into pq

            for(int i = 0; i < n1; i++)
            {
                pq.push(nums1[i] - 1);
            }

            // find max. increment for each element of nums2 and push into pq

            for(int i = 0; i < n2; i++)
            {
                pq.push(6 - nums2[i]);
            }
        }

        // case 2 :- if sum1 < sum2 

        // we have to check maximum decrement for each element of nums2

        // we have to check maximum increment for each element of nums1

        else
        {
            // find max. increment for each element of nums1 and push into pq

            for(int i = 0; i < n1; i++)
            {
                pq.push(6 - nums1[i]);
            }

            // find the max. decrement for each element of nums2 and push into pq

            for(int i = 0; i < n2; i++)
            {
                pq.push(nums2[i] - 1);
            }
        }

        int diff = abs(sum1 - sum2);

        int count = 0;

        // now run the loop while diff becomes 0

        while(diff > 0 && pq.size())
        {
            int top = pq.top();

            pq.pop();

            // update diff

            diff -= min(diff, top);

            // increment operation

            count++;
        }

        if(diff > 0)
        {
            return -1;
        }
        else
        {
            return count;
        }
    }
};
```
##19 ****[Problem Link]https://leetcode.com/problems/construct-quad-tree****
```cpp

Node* zero = new Node(false, true);
Node* one = new Node(true, true);

class Solution {
public:
    Node* create(vector<vector<int>>& grid, int x, int y, int n) {
        if (n == 1) {
            return grid[x][y] == 1 ? one : zero;
        }
        int mid = n / 2;
        Node *topLeft = create(grid, x, y, mid);
        Node *topRight = create(grid, x, y+mid, mid);
        Node *botLeft = create(grid, x+mid, y, mid);
        Node *botRight = create(grid, x+mid, y+mid, mid);
        if (topLeft == topRight and botLeft == botRight and topLeft == botLeft) {
            return topLeft;
        }
        return new Node(false, false, topLeft, topRight, botLeft, botRight);
    }
    
    Node* construct(vector<vector<int>>& grid) {
        return create(grid, 0, 0, grid.size());    
    }
};
```
##20 ****[Problem Link]https://leetcode.com/problems/valid-permutations-for-di-sequence****
```cpp
class Solution {
public:
    #define ll long long
    const int mod = 1e9+7;
    int solve(int ind, int prev, string s, int n,vector<int>&vis,vector<vector<int>>&dp){
        if(ind>=n)return 1;
        if(dp[ind][prev]!=-1)return dp[ind][prev];

        ll ans=0;
        for(int i=0;i<=n;i++){
            if(i!=prev && vis[i]==0){
                if(s[ind]=='D' && i<prev){
                    vis[i]=1;
                    ans=(ans+solve(ind+1,i,s,n,vis,dp))%mod;
                    vis[i]=0;
                }
                if(s[ind]=='I' && i>prev){
                    vis[i]=1;
                    ans= (ans+solve(ind+1,i,s,n,vis,dp))%mod;
                    vis[i]=0;
                }
            }
        }
        return dp[ind][prev]=ans%mod;
    }
    int numPermsDISequence(string s) {
        
        int n = s.length();
        ll ans=0;
        vector<vector<int>>dp(n+1,vector<int>(n+1,-1));
        for(int i=0;i<=n;i++){
            vector<int>vis(n+2,0);
            vis[i]=1;
            
            ans= (ans+solve(0,i,s,n,vis, dp))%mod;
        }
        return ans;

    }
};
```
##21 ****[Problem Link]https://leetcode.com/problems/string-without-aaa-or-bbb****
```cpp
class Solution {
public:
    string strWithout3a3b(int a, int b) {
       string ans="";  //String to store the answer
        int counta=0,countb=0; // Counter to check that a and b should not be greater than two;
        int total=a+b;         //No of times the loop will run;
        for(int i=0;i<total;i++)
        {
            if((b>=a && countb<2) || (counta==2 && b>0)) //If b is greater than a and count of b is less than 2 || if count of a ==2 and b is greater than 2 add 'a';
            {
                ans+='b';
                b--;    // decrement given count of b;

                countb++; //increment count of b;

                counta=0; // make the count of a to 0 , if we don't do that then the length of the string will remain 3 because counta<2 || countb<2 condition will never become true after the string size becomes three that's why we are making counta=0 and countb=0 in every condition;
            }
            else if((a>=b && counta<2) || (countb==2 && a>0))
            {
                ans+='a';
                a--;
                counta++;
                countb=0;
            }
        }
        return ans; // Return the answer;
    }
};
```
##22 ****[Problem Link]https://leetcode.com/problems/strong-password-checker****
```cpp
class Solution {
public:
    int strongPasswordChecker(string password) {
        int n = password.size();
        
        bool hasLower = false, hasUpper = false, hasDigit = false;
        int missingTypes = 3; // Initially assume we are missing all three types

        for (char c : password) {
            if (islower(c)) hasLower = true;
            if (isupper(c)) hasUpper = true;
            if (isdigit(c)) hasDigit = true;
        }

        if (hasLower) missingTypes--;
        if (hasUpper) missingTypes--;
        if (hasDigit) missingTypes--;
        
        // Count sequences of three or more consecutive repeating characters
        int changeCount = 0;
        vector<int> repeats; // lengths of sequences of repeating characters
        
        for (int i = 2; i < n; i++) {
            if (password[i] == password[i-1] && password[i] == password[i-2]) {
                int len = 2;
                while (i < n && password[i] == password[i-1]) {
                    len++;
                    i++;
                }
                repeats.push_back(len);
            }
        }
        
        int totalChange = 0, totalDelete = 0;
        
        for (int len : repeats) {
            changeCount += len / 3;
        }
        
        if (n < 6) {
            return max(missingTypes, 6 - n);
        } else if (n <= 20) {
            return max(missingTypes, changeCount);
        } else {
            totalDelete = n - 20;
            for (int& len : repeats) {
                if (totalDelete <= 0) break;
                if (len % 3 == 0) {
                    int toDelete = min(totalDelete, 1);
                    totalDelete -= toDelete;
                    len -= toDelete;
                    changeCount -= toDelete;
                }
            }
            for (int& len : repeats) {
                if (totalDelete <= 0) break;
                if (len % 3 == 1) {
                    int toDelete = min(totalDelete, 2);
                    totalDelete -= toDelete;
                    len -= toDelete;
                    changeCount -= toDelete / 2;
                }
            }
            for (int& len : repeats) {
                if (totalDelete <= 0) break;
                int toDelete = min(totalDelete, len - 2);
                totalDelete -= toDelete;
                len -= toDelete;
                changeCount -= toDelete / 3;
            }
            return n - 20 + max(missingTypes, changeCount);
        }
    }
};
```
##23 ****[Problem Link]https://leetcode.com/problems/pyramid-transition-matrix****
```cpp
class Solution {
public:
    map<string, vector<string>> vec;
    unordered_map<string, bool> memo;

    bool un_fun(string s) {
        if (s.length() == 1) return true;

        if (memo.find(s) != memo.end()) {
            return memo[s];
        }

        bool ans = false;

        queue<pair<string, int>> q;
        vector<string> p;

        string pfx = s.substr(0, 2);

        for (int i = 0; i < vec[pfx].size(); i++) {
            q.push({ vec[pfx][i], 0 });
        }

        while (!q.empty()) {
            pair<string, int> top = q.front();
            q.pop();

            if (top.second == s.length() - 2) {
                p.push_back(top.first);
            }

            int st_idx = top.second + 1;
            string prev = top.first;

            pfx = s.substr(st_idx, 2);

            for (int i = 0; i < vec[pfx].size(); i++) {
                q.push({ prev + vec[pfx][i], st_idx });
            }
        }

        for (int i = 0; i < p.size(); i++) {
            ans = ans || un_fun(p[i]);
        }

        memo[s] = ans;
        return ans;
    }

    bool pyramidTransition(string bottom, vector<string> &allowed) {
        for (int i = 0; i < allowed.size(); i++) {
            string p = "";
            p += allowed[i][2];
            vec[allowed[i].substr(0, 2)].push_back(p);
        }

        return un_fun(bottom);
    }
};
```
##24 ****[Problem Link]https://leetcode.com/problems/increasing-decreasing-string****
```cpp
class Solution {
public:
    string ans;
    void CharErase(map<char, int> &mp, char ch)
    {
        if(mp.find(ch) != mp.end())
        {
            ans += ch;
            mp[ch]--;
            if(not mp[ch]) mp.erase(ch);
        }
    }
    string sortString(string s) 
    {
        map<char, int> mp;
        for(char ch:s) mp[ch]++;
        
        while(mp.size())
        {
            for(char ch='a'; ch <= 'z'; ch++)
                CharErase(mp, ch);
            for(char ch='z'; ch >= 'a'; ch--)
                CharErase(mp, ch);
        }
        return ans;
    }
};
```

##25 ****[Problem Link]https://leetcode.com/problems/pizza-with-3n-slices****
```cpp
class Solution {
public:

    int recursion(vector<int>& slices, int i, int last, vector<vector<int>> &dp, int n)
    {
        if(i >= slices.size() or i>last or n == 0) return 0;
        if(dp[i][n] != -1) return dp[i][n];

        int take = slices[i] + recursion(slices, i+2, last, dp, n-1);
        int notake = recursion(slices, i+1, last, dp, n); 

        return dp[i][n] = max(take, notake);
    }

    int maxSizeSlices(vector<int>& slices) 
    {
        int n = slices.size();
        vector<vector<int>> dp1(n+1,vector<int>(n/3+1,-1)), dp2(n+1,vector<int>(n/3+1,-1));

        int ans = max(recursion(slices, 0, n-2, dp1, n/3),
                      recursion(slices, 1, n-1, dp2, n/3));
        return ans;
    }
};
```

##26 ****[Problem Link]https://leetcode.com/problems/check-if-there-is-a-valid-path-in-a-grid****
```cpp
class Solution {
    int m, n;
    vector<vector<int>> directions {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}
    }; // down, up, right, left
    enum {
        down = 0, up = 1, right = 2, left = 3
    };
    bool atValidPosition(int x, int y)
    {
        return x >= 0 && y >= 0 && x < m && y < n;
    }
public:
    bool hasValidPath(vector<vector<int>>& grid) {
        m = grid.size(), n = grid[0].size();

        // we need to make sure the first street can connect downward or rightward
        int initType {grid[0][0]};
        if (initType == 5) return false;
        if (initType == 1 || initType == 6) return explore(grid, right);
        if (initType == 2 || initType == 3) return explore(grid, down);
        return explore(grid, down) || explore(grid, right);
    }
    bool explore(const vector<vector<int>>& grid, int dir)
    {
        int x {}, y {};
        vector<vector<int>> visited(m, vector(n, 0)); // to avoid cycles
        while (atValidPosition(x, y))
        {
            if (x == m-1 && y == n-1) return true; // end
            if (visited[x][y]) return false;
            visited[x][y] = 1; 

            int dx {directions[dir][0]}, dy {directions[dir][1]};
            x += dx, y += dy;

            if (!atValidPosition(x, y)) return false;

            // after making sure the next position is valid
            // we need to check if we can make to the next position actually

            // given current direction and next type of street
            // we can determine if next is reachable from current street
            // if it is reachable, we may change current direction according
            // to the type of the next street 
            if ((dir = validMove(dir, grid[x][y])) == -1) return false;
        }
        return false;
    }
    int validMove(int dir, int nextType) 
    {   // if move is invalid return -1
        switch (nextType) // down = 0, up = 1, right = 2, left = 3
        {
        case 1: // accept right or left direction, and do not change direction
            if (dir == right || dir == left) return dir; 
            break;

        case 2: // accept down or up direction, and do not change direction
            if (dir == down || dir == up) return dir;
            break;

        case 3: // accept right and return down, or accept up and return left
            if (dir == right) return down;
            if (dir == up) return left;
            break;

        case 4: // accept left and return down, or accept up and return right
            if (dir == left) return down;
            if (dir == up) return right;
            break;

        case 5: // accept down and return left, or accept right and return up
            if (dir == down) return left;
            if (dir == right) return up;
            break;
        
        case 6: // accept down and return right, or accept left and return up
            if (dir == down) return right;
            if (dir == left) return up;
            break;

        default:
            break;
        }
        return -1;
    }
};
```

##27 ****[Problem Link]https://leetcode.com/problems/stamping-the-sequence****
```cpp
class Solution {
public:
    //replace character by '?'
    int replace(int pos,string&stamp,string&target){
        int m=stamp.size();
        int cnt=0;
        for(int i=0;i<m;i++){
            if(target[pos+i]!='?'){
                target[pos+i]='?';
                cnt++;
            }
        }
        return cnt;
    }
    // check whether that part from index pos can be replaced
    bool canReplace(int pos,string&stamp, string&target){
        int n=target.size();
        int m=stamp.size();
        for(int i=0;i<m;i++){
            if(stamp[i]!=target[pos+i] && target[pos+i]!='?'){
                return false;
            }
        }
            return true;
    }
    vector<int> movesToStamp(string stamp, string target) {
        int n=target.size();
        vector<int>ans;
        int m=stamp.size();

        vector<bool>vis(n,false);
        int cnt=0;
        while(cnt!=n){
            bool change=false;
            //traverse target string and check whether it can be replaced..
            for(int i=0;i<=n-m;i++){
                if(!vis[i] && canReplace(i,stamp,target)){ 
                    vis[i]=true;//mark as visited
                    change=true;
                    ans.push_back(i);
                    cnt+=replace(i,stamp,target);//count no. of replacement
                    if(cnt==n){
                        break;
                    }
                }
            }
            // not replace any character then return empty array
            if(!change){
                return {};
            }
        }

        reverse(ans.begin(),ans.end());//reverse ,as we go from last to first 
        return ans;
    }
};
```

##28 ****[Problem Link]https://leetcode.com/problems/maximum-students-taking-exam****
```cpp
class Solution {
public:
    int maxStudents(vector<vector<char>>& seats) {
        int m = seats.size();
        int n = seats[0].size();
        
        vector<int>arr; // vector to store mask
        
        //calculating mask column-wise
        for(int i = 0; i < n; ++i){
            int num = 0;
            for(int j = 0; j < m; ++j){
                num += pow(2, j)*(seats[j][i] == '.');
            }
            arr.push_back(num);
        }
        
        //dp[i][j] = maximum number that can be placed till ith column if number of students in ith column is equal to number of set bits in j, provided it does not violate the rules.
        vector<vector<int>>dp(n, vector<int>(1<<m, -1));
        
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < (1 << m); ++j){
                //student should only sit on not-broken seat.
               if((arr[i] | j) == arr[i]){
                   //for the first column
                   if(i == 0){
                        dp[i][j] =  __builtin_popcount(j);
                        continue;
                    }
                   
                   //for the rest column
                    for(int k = 0; k < (1 << m); ++k){
                        if(dp[i-1][k] != -1){
                            
                            //students should not be sharing adjacent seat or the adjacent corners.
                            if(!(j & k) && !((j<<1) & k) && !((j >> 1) & k)){
                                dp[i][j] = max(dp[i][j], dp[i-1][k] + __builtin_popcount(j));
                            }
                        }
                    }
               }
            }
        }
        
        return *max_element(dp[n-1].begin(), dp[n-1].end());
    }
};
```
##29 ****[Problem Link]https://leetcode.com/problems/max-dot-product-of-two-subsequences****
```cpp
class Solution {
public:
    int maxDotProduct(std::vector<int>& nums1, std::vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        std::vector<int> current(n + 1, INT_MIN), previous(n + 1, INT_MIN);

        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                int curr_product = nums1[i-1] * nums2[j-1];
                current[j] = std::max({curr_product, previous[j], current[j-1], curr_product + std::max(0, previous[j-1])});
            }
            std::swap(current, previous);
        }
        return previous[n];
    }
};
```

##30 ****[Problem Link]https://leetcode.com/problems/poor-pigs****
```cpp
class Solution {
public:
    int poorPigs(int buckets, int timeDetect, int timeTest) {
        return ceil(log2(buckets)/log2(int(timeTest/timeDetect)+1));
    }
};
```

##31 ****[Problem Link]https://leetcode.com/problems/smallest-sufficient-team****
```cpp
class Solution {
public:
    long long dp[1<<17];
    long long int maxm=(1UL << 61)-1;
    long long helper(unordered_map<string,int>&mp,vector<vector<string>>&people,int mask)
    {
        int sz=mp.size();
        if(mask==((1<<sz)-1))
        {
            return 0ll;
        }
        if(dp[mask]!=-1)return dp[mask];
        long long ans=maxm;
        for(long long i=0;i<(int)people.size();i++)
        {
          
            int newmask=mask;
            long long curr=0;
            for(int j=0;j<(int)people[i].size();j++)
            {
               
                if(mp.count(people[i][j]))
                {
                    int idx=mp[people[i][j]];
                    if((newmask & (1 <<idx ))==0)
                    {newmask=(newmask | (1<<idx));}
                }
            }
            if(newmask > mask)
            {
                curr=(curr | (1UL<<i));
                curr|=helper(mp,people,newmask);
                if(__builtin_popcountll(curr) < __builtin_popcountll(ans))          {
                    ans=curr;
                }
                
            }
        }
        return dp[mask]=ans;
    }
    vector<int> smallestSufficientTeam(vector<string>& req, vector<vector<string>>& people) 
    {
        memset(dp,-1,sizeof(dp));
        unordered_map<string,int>mp;
        for(int i=0;i<(int)req.size();i++)
        mp[req[i]]=i;
        long long ans=helper(mp,people,0);
        vector<int>res;
        for(int i=0;i<=60;i++)
        {
            if(ans & (1UL << i))
            res.push_back(i);
        }
        return res;
    }
};
```

##32 ****[Problem Link]https://leetcode.com/problems/ugly-number-iii****
```cpp
class Solution {
public:
    int nthUglyNumber(int n, int A, int B, int C) {
        // Logic (Binary Search)
        // We can find the kth ugly element by binary searching it's position
        // No upto N divisible by a = N / a
        // No upto N divisible by b = N / b
        // No upto N divisible by c = N / c
        // No upto N divisible by both (a,b) = N / Lcm(a,b)
        // No upto N divisible by both (b,c) = N / Lcm(b,c)
        // No upto N divisible by both (c,a) = N / Lcm(c,a)
        // No upto N divisible by (a,b,c) = N / Lcm(a,b,c)

        // We can find the position of any ugly number by this
        // We already have range of ugly number from constraint
        
        long long s = 1;
        long long e = 2 * 1e9;
        long long ans;
        // Type converting
        long a = long(A);
        long b = long(B);
        long c = long(C);

        while(s <= e){
            long long mid = s + (e - s) / 2;
            long long count = 0;
            // Finding the position of ugly number
            // (A U B U C) = A + B + C - (A ∩ B) - (B ∩ C) - (C ∩ A) + (A ∩ B ∩ C)
            count += mid / a;
            count += mid / b;
            count += mid / c;
            count -= mid / lcm(a,b);
            count -= mid / lcm(b,c);
            count -= mid / lcm(c,a);
            count += mid / lcm(a,lcm(b,c));

            if(count >= n){
                ans = mid;
                e = mid - 1;
            }
            else{
                s = mid + 1;
            }
        }

        return ans;
    }
};
```

##33 ****[Problem Link]https://leetcode.com/problems/swap-for-longest-repeated-character-substring****
```cpp
class Solution {
public:
    int maxRepOpt1(string text) {
        int freq[26] = {0};
        for(char c : text) freq[c - 'a']++;

        int ans = 0, n = text.size();
        for(int i = 0; i < n; i++) {
            int j = i, diff = 0, cnt = 0;
            while(j < n && (text[j] == text[i] || diff == 0) && cnt < freq[text[i] - 'a']){
                if(text[j] != text[i]) diff++;
                cnt++;
                j++;
            }
            if(cnt < freq[text[i] - 'a'] && diff == 0) cnt++;
            ans = max(ans, cnt);
        }
        return ans;
    }
};
```

##34 ****[Problem Link]https://leetcode.com/problems/find-and-replace-in-string****
```cpp
class Solution {
public:
    string findReplaceString(string s, vector<int>& indices, vector<string>& sources, vector<string>& targets) {
        int n = s.size();
        unordered_map<int, int> st; 
        for (int i = 0; i < indices.size(); ++i) {
            if(st.find(indices[i]) !=st.end()){
                int oldIdx = st[indices[i]];
                string substring = s.substr(indices[i] ,sources[i].size());
                if(substring == sources[i]){
                    st[indices[i]] = i; 
                }
            }else 
                st[indices[i]] = i; 
        }
       
        int j = 0;
        string ans = "";
        while(j<n){
            if(st.count(j)){
                int i = st[j];
                string substring = s.substr(j ,sources[i].size());
                if(substring == sources[i]){
                    ans+=targets[i];
                    j+=sources[i].size();
                    continue;
                }
            }ans+=s[j];
            j++;
        }
        return ans;
    }
};
```

##35 ****[Problem Link]https://leetcode.com/problems/swap-adjacent-in-lr-string****
```cpp
class Solution {
public:

    bool canTransform(string start, string end) {
        // If lengths are different, transformation is impossible
        if (start.length() != end.length()) {
            return false;
        }
        
        // Create strings without 'X' to check if character sequences match
        string startChars, endChars;
        for (char c : start) {
            if (c != 'X') startChars += c;
        }
        for (char c : end) {
            if (c != 'X') endChars += c;
        }
        
        // If the sequences without 'X' don't match, return false
        if (startChars != endChars) {
            return false;
        }
        
        // Initialize pointers for both strings
        int p1 = 0, p2 = 0;
        int n = start.size();
        
        while (p1 < n && p2 < n) {
            // Skip 'X' in both strings
            while (p1 < n && start[p1] == 'X') p1++;
            while (p2 < n && end[p2] == 'X') p2++;
            
            // If both pointers reach the end, the strings are transformable
            if (p1 == n && p2 == n) {
                return true;
            }
            
            // If only one pointer reaches the end, strings aren't transformable
            if (p1 == n || p2 == n) {
                return false;
            }
            
            // If current characters don't match, transformation is impossible
            if (start[p1] != end[p2]) {
                return false;
            }
            
            // 'L' should not move right (should be left or same position)
            if (start[p1] == 'L' && p1 < p2) {
                return false; 
            }
            // 'R' should not move left (should be right or same position)
            if (start[p1] == 'R' && p1 > p2) {
                return false; 
            }
            
            // Move to the next character in both strings
            p1++;
            p2++;
        }
        
        // If all checks pass, the transformation is possible
        return true;
    }
};
```
##36 ****[Problem Link]https://leetcode.com/problems/k-th-smallest-prime-fraction****
```cpp
class Solution {
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        int n = arr.size();
        priority_queue<pair<double,pair<int,int>>> pq;

        for(int i=0;i<n;i++){
            for(int j=i+1;j<n;j++){
                double x = arr[i]/(arr[j]*1.0);
                
                if(pq.size() == k){
                    if(x < pq.top().first){
                        pq.pop();
                        pq.push({x,{arr[i],arr[j]}});
                    }
                }
                else{
                    pq.push({x,{arr[i],arr[j]}});
                }
            }
        }

        return {pq.top().second.first,pq.top().second.second};
    }
};
```

##37 ****[Problem Link]https://leetcode.com/problems/race-car****
```cpp
#define ll long long
#define p pair<ll,pair<ll,ll>>
class Solution {
public:
    int racecar(int target) {
        priority_queue<p,vector<p>,greater<p>>pq;
        map<pair<ll,ll>,bool>visited;

        pq.push({0,{0,1}}); // {moves,{pos,speed}}

        while(pq.size()){
            ll moves = pq.top().first;
            ll pos = pq.top().second.first;
            ll speed = pq.top().second.second;
            pq.pop();

            if(pos == target)
                return moves;
            
            if(pos>=INT_MAX)
                continue;
            
            if(!visited[{pos,speed}]){
                visited[{pos,speed}] = 1;

                // either we can accelerate
                pq.push({moves+1,{pos+speed,speed*2}});

                // or put reverse
                // if(speed>0){
                //     pq.push({moves+1,{pos,-1}});
                // } else{
                //     pq.push({moves+1,{pos,1}});
                // }

                if ((pos + speed > target && speed > 0) || (pos + speed < target && speed < 0)) {
                     pq.push({moves+1,{pos,speed>0?-1:1}});
                }
            }
        }

        return -1;
    }
};
```
##38 ****[Problem Link]https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor****
```cpp
class Solution {
public:
    int countTriplets(vector<int>& arr) {
        int n = arr.size();
        int count = 0;
        for(int i=0;i<n-1;i++){
            int x = arr[i];
            for(int j=i+1;j<n;j++){
                x = x^arr[j];
                if(x == 0)count += (j-i);
            }
        }
        return count;
    }
};
```

##39 ****[Problem Link]https://leetcode.com/problems/binary-number-with-alternating-bits****
```cpp
class Solution {
public:
    bool hasAlternatingBits(int n) {
        int cnt = 0;
        int c = n;
        while(n > 0){
            int rem = n % 2;
            if(c % 2 == 0){
                if(rem == 0 && cnt % 2 == 0){
                    cnt++;
                }
                else if(rem == 1 && cnt % 2 == 1){
                    cnt++;
                }
                else{
                    return false;
                }
            }
            else{
                if(rem == 1 && cnt % 2 == 0){
                    cnt++;
                }
                else if(rem == 0 && cnt % 2 == 1){
                    cnt++;
                }
                else{
                    return false;
                }
            }
            n /= 2;
        }
            return true;
        }
        
};
```

##40 ****[Problem Link]https://leetcode.com/problems/3sum-with-multiplicity****
```cpp
//Solution 01:
class Solution {
public:
    int threeSumMulti(vector<int>& arr, int target) {
        int n = arr.size(), mod = 1e9+7, count = 0;
        unordered_map<int, int> mp;
        
        for(int i=0; i<n; i++){
            count = (count + mp[target - arr[i]]) % mod;
            
            for(int j=0; j<i; j++){
                mp[arr[i] + arr[j]]++;
            }
        }
        
        return count;
    }
};
```
##41 ****[Problem Link]https://leetcode.com/problems/count-unique-characters-of-all-substrings-of-a-given-string****
```cpp
class Solution {
public:
    int uniqueLetterString(string s) {
        int n = s.size();
        vector<int> next(26,n);
        vector<int> nextIdx(n,n);
        vector<int> pre(26,-1);
        int ans = 0;
        for(int i = n - 1 ; i >= 0 ; i--){
            int pos = s[i] - 'A';
            if(next[pos] != n) nextIdx[i] = next[pos];   
            next[pos] = i;
        }
        for(int i = 0 ; i < n ; i++){
            int j = pre[s[i] - 'A'];
            ans += (i - j) * (nextIdx[i] - i);
            pre[s[i] - 'A'] = i;
        }

        return ans;
    }
};
```
##42 ****[Problem Link]https://leetcode.com/problems/peeking-iterator****
```cpp
class PeekingIterator : public Iterator {
    bool hasPeeked;
    int peekedElem;
public:
	PeekingIterator(const vector<int>& num) : Iterator(num) {
        hasPeeked = false;
	}

	int peek() {
        peekedElem = hasPeeked?peekedElem:Iterator::next();
        hasPeeked = true;
        return peekedElem;
	}

	int next() {
	    int nextElem = hasPeeked?peekedElem:Iterator::next();
	    hasPeeked = false;
	    return nextElem;
	}

	bool hasNext() const {
	    return hasPeeked||Iterator::hasNext();
	}
};  
```

##43 ****[Problem Link]https://leetcode.com/problems/shortest-path-with-alternating-colors****
```cpp
class Solution {
public:

    void bfs(int src, int n, vector<int> &dist, vector<vector<int>>& redAdj, vector<vector<int>>& blueAdj) {
        
        vector<vector<int>> vis(n, vector<int>(2, 0)); 
        queue<pair<int, int>> q; 

        dist[src] = 0;
        q.push({src, 0}); 
        q.push({src, 1}); 

        int level = 0;

        while (!q.empty()) {
            int size = q.size();
            while (size--) {
                auto [node, color] = q.front();
                q.pop();

                vector<int>& neighbors = (color == 0) ? redAdj[node] : blueAdj[node];

                for (int ngh : neighbors) {
                    if (!vis[ngh][color]) {
                        vis[ngh][color] = 1;
                        dist[ngh] = min(dist[ngh], level + 1);
                        q.push({ngh, 1 - color}); 
                    }
                }
            }
            level++;
        }
    }

    vector<int> shortestAlternatingPaths(int n, vector<vector<int>>& redEdges, vector<vector<int>>& blueEdges) {
        
        vector<vector<int>> redAdj(n), blueAdj(n);
        for (auto &edge : redEdges) redAdj[edge[0]].push_back(edge[1]);
        for (auto &edge : blueEdges) blueAdj[edge[0]].push_back(edge[1]);

        vector<int> dist(n, INT_MAX);
        bfs(0, n, dist, redAdj, blueAdj);

        for (int &d : dist) {
            if (d == INT_MAX) d = -1;
        }

        return dist;
    }
};
```

##44 ****[Problem Link]https://leetcode.com/problems/second-minimum-node-in-a-binary-tree****
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int sm = -1;
    int diff = INT_MAX;
    void rec(TreeNode* root,int tar){
        if(root == NULL) return;
        if(root->left){
            if(root->left->val != tar && diff > root->left->val - tar){
                sm = root->left->val;
                diff = root->left->val - tar;
            }
            rec(root->left,tar);
        }
        if(root->right){
            if(root->right->val != tar && diff > root->right->val - tar){
                sm = root->right->val;
                diff = root->right->val - tar;
            }
            rec(root->right,tar);
        }
    }

    int findSecondMinimumValue(TreeNode* root) {
        //one root
        if(root == NULL || (root->left == NULL && root->right == NULL)){
            return -1;
        }
        rec(root,root->val);
        return sm;
    }
};
```

##45 ****[Problem Link]https://leetcode.com/problems/length-of-longest-fibonacci-subsequence****
```cpp
class Solution {
public:
    int lenLongestFibSubseq(vector<int>& arr) {
        int n = arr.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        int maxLen = 0;

        for (int curr = 2; curr < n; curr++) {
            int start = 0, end = curr - 1;
            while (start < end) {
                int pairSum = arr[start] + arr[end];
                if (pairSum > arr[curr]) {
                    end--;
                } else if (pairSum < arr[curr]) {
                    start++;
                } else {
                    dp[end][curr] = dp[start][end] + 1;
                    maxLen = max(dp[end][curr], maxLen);
                    end--;
                    start++;
                }
            }
        }
        return maxLen == 0 ? 0 : maxLen + 2;
    }
};
```
