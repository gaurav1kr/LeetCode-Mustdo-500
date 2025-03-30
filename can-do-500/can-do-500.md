##1 ****[Problem Link]https://leetcode.com/problems/maximum-product-of-word-lengths****  
**Approach:** Use bitmasking to represent character presence in words. Compare only words with no common characters to find the maximum product.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int maxProduct(vector<string>& words) {
        int n = words.size(), maxProd = 0;
        vector<int> masks(n, 0);
        
        for (int i = 0; i < n; i++) {
            for (char c : words[i])
                masks[i] |= (1 << (c - 'a'));
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((masks[i] & masks[j]) == 0) {
                    maxProd = max(maxProd, (int)(words[i].size() * words[j].size()));
                }
            }
        }
        
        return maxProd;
    }
};
```

---

##2 ****[Problem Link]https://leetcode.com/problems/max-chunks-to-make-sorted****  
**Approach:** Track the maximum value encountered so far. A chunk ends when the max value equals the index.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        int maxVal = 0, chunks = 0;
        for (int i = 0; i < arr.size(); i++) {
            maxVal = max(maxVal, arr[i]);
            if (maxVal == i) chunks++;
        }
        return chunks;
    }
};
```

---

##3 ****[Problem Link]https://leetcode.com/problems/flip-string-to-monotone-increasing****  
**Approach:** Use prefix sums to track the number of flips needed at each position.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int minFlipsMonoIncr(string s) {
        int ones = 0, flips = 0;
        for (char c : s) {
            if (c == '1') ones++;
            else flips = min(flips + 1, ones);
        }
        return flips;
    }
};
```

---

##4 ****[Problem Link]https://leetcode.com/problems/sort-array-by-parity-ii****  
**Approach:** Use two pointers to place even numbers at even indices and odd numbers at odd indices.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> sortArrayByParityII(vector<int>& nums) {
        int i = 0, j = 1, n = nums.size();
        while (i < n && j < n) {
            while (i < n && nums[i] % 2 == 0) i += 2;
            while (j < n && nums[j] % 2 == 1) j += 2;
            if (i < n && j < n) swap(nums[i], nums[j]);
        }
        return nums;
    }
};
```

---

##5 ****[Problem Link]https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters****  
**Approach:** Use backtracking to explore all possible concatenations and maximize length.  
**Time Complexity:** O(2^n)

```cpp
#include <vector>
#include <string>
#include <unordered_set>

using namespace std;

class Solution {
public:
    int maxLength(vector<string>& arr) {
        return backtrack(arr, "", 0);
    }

    int backtrack(vector<string>& arr, string current, int index) {
        unordered_set<char> charSet(current.begin(), current.end());
        if (charSet.size() != current.size()) return 0;
        
        int maxLen = current.size();
        for (int i = index; i < arr.size(); i++) {
            maxLen = max(maxLen, backtrack(arr, current + arr[i], i + 1));
        }
        return maxLen;
    }
};
```

---

##6 ****[Problem Link]https://leetcode.com/problems/n-ary-tree-level-order-traversal****  
**Approach:** Use BFS with a queue to process each level.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <queue>

using namespace std;

class Solution {
public:
    vector<vector<int>> levelOrder(Node* root) {
        if (!root) return {};
        vector<vector<int>> result;
        queue<Node*> q;
        q.push(root);
        
        while (!q.empty()) {
            vector<int> level;
            int size = q.size();
            for (int i = 0; i < size; i++) {
                Node* node = q.front(); q.pop();
                level.push_back(node->val);
                for (Node* child : node->children)
                    q.push(child);
            }
            result.push_back(level);
        }
        return result;
    }
};
```

---

##7 ****[Problem Link]https://leetcode.com/problems/k-diff-pairs-in-an-array****  
**Approach:** Use a hash map to count occurrences and check for valid pairs.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int findPairs(vector<int>& nums, int k) {
        if (k < 0) return 0;
        unordered_map<int, int> numCount;
        int count = 0;
        
        for (int num : nums) numCount[num]++;
        for (auto& [num, freq] : numCount) {
            if ((k > 0 && numCount.count(num + k)) || (k == 0 && freq > 1)) count++;
        }
        
        return count;
    }
};
```

---

##8 ****[Problem Link]https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree****  
**Approach:** Use a stack to simulate the pre-order traversal process.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <stack>

using namespace std;

class Solution {
public:
    bool isValidSerialization(string preorder) {
        int slots = 1;
        int n = preorder.size();
        
        for (int i = 0; i < n; i++) {
            if (preorder[i] == ',') continue;
            if (--slots < 0) return false;
            if (preorder[i] != '#') slots += 2;
            while (i < n && preorder[i] != ',') i++;
        }
        
        return slots == 0;
    }
};
```

---

##9 ****[Problem Link]https://leetcode.com/problems/robot-return-to-origin****  
**Approach:** Track net movement in x and y directions.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    bool judgeCircle(string moves) {
        int x = 0, y = 0;
        for (char move : moves) {
            if (move == 'U') y++;
            else if (move == 'D') y--;
            else if (move == 'L') x--;
            else if (move == 'R') x++;
        }
        return x == 0 && y == 0;
    }
};
```

---

##10 ****[Problem Link]https://leetcode.com/problems/count-sorted-vowel-strings****  
**Approach:** Use dynamic programming to count valid vowel strings ending in each vowel.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int countVowelStrings(int n) {
        vector<int> dp(5, 1);
        for (int i = 2; i <= n; i++) {
            for (int j = 3; j >= 0; j--) {
                dp[j] += dp[j + 1];
            }
        }
        return dp[0] + dp[1] + dp[2] + dp[3] + dp[4];
    }
};
```
##11 ****[Problem Link]https://leetcode.com/problems/minimum-domino-rotations-for-equal-row****  
**Approach:** Try making all dominos match either A[0] or B[0] using rotations.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int minDominoRotations(vector<int>& tops, vector<int>& bottoms) {
        int rotationsA = check(tops[0], tops, bottoms);
        int rotationsB = check(bottoms[0], tops, bottoms);
        
        if (rotationsA == -1) return rotationsB;
        if (rotationsB == -1) return rotationsA;
        return min(rotationsA, rotationsB);
    }

    int check(int x, vector<int>& tops, vector<int>& bottoms) {
        int topRotations = 0, bottomRotations = 0;
        for (int i = 0; i < tops.size(); i++) {
            if (tops[i] != x && bottoms[i] != x) return -1;
            if (tops[i] != x) topRotations++;
            if (bottoms[i] != x) bottomRotations++;
        }
        return min(topRotations, bottomRotations);
    }
};
```

---

##12 ****[Problem Link]https://leetcode.com/problems/longest-continuous-increasing-subsequence****  
**Approach:** Iterate through the array while keeping track of the maximum increasing sequence length.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        if (nums.empty()) return 0;
        
        int maxLen = 1, currentLen = 1;
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] > nums[i - 1]) currentLen++;
            else currentLen = 1;
            maxLen = max(maxLen, currentLen);
        }
        return maxLen;
    }
};
```

---

##13 ****[Problem Link]https://leetcode.com/problems/longest-duplicate-substring****  
**Approach:** Use binary search combined with a rolling hash to detect duplicate substrings efficiently.  
**Time Complexity:** O(n log n)

```cpp
#define ull unsigned long long

class Solution {
public:
    
    string ans="";
    
    bool solve(int len, string &s, ull power){
        int start = 0, end = len;
        unordered_set<ull> st;
        
        ull curHash = 0;
        for(int i=0; i<len; ++i){
            curHash = (curHash*131 + (s[i]));
        }
        
        st.insert(curHash);
        for(int j=len; j<s.size(); ++j){
            curHash = ((curHash - power*(s[start]))) ;
            curHash = (curHash*131);
            curHash = (curHash + (s[j]));
            start++;
            
            if(st.find(curHash) != st.end()){
                string curS = s.substr(start,len);
                if(curS.size()>ans.size()){
                    ans = curS;
                } 
                return true;
            }
            st.insert(curHash);
        } 
        return false;
    }
    
    void binary(int l, int r, string &s, vector<ull>& power){
        if(l>r) return;
        int mid = l+(r-l)/2;
        if(solve(mid+1,s,power[mid])){
            l=mid+1;
        }else{
            r=mid-1;
        }
        binary(l,r,s,power);
    }

    string longestDupSubstring(string s) {
        int n = s.size();
        vector<ull> power(n,1);
        for(int i=1;i<n;++i){ 
            power[i]=(power[i-1]*131);
        }
        
        binary(0,n-1,s,power);
        return ans;
    }
};
```

---

##14 ****[Problem Link]https://leetcode.com/problems/split-a-string-in-balanced-strings****  
**Approach:** Count occurrences of 'L' and 'R' and track balance.  
**Time Complexity:** O(n)

```cpp
#include <string>

using namespace std;

class Solution {
public:
    int balancedStringSplit(string s) {
        int balance = 0, count = 0;
        for (char c : s) {
            balance += (c == 'L') ? 1 : -1;
            if (balance == 0) count++;
        }
        return count;
    }
};
```

---

##15 ****[Problem Link]https://leetcode.com/problems/add-digits****  
**Approach:** Use digital root formula for O(1) complexity.  
**Time Complexity:** O(1)

```cpp
class Solution {
public:
    int addDigits(int num) {
        return 1 + (num - 1) % 9;
    }
};
```

---

##16 ****[Problem Link]https://leetcode.com/problems/max-increase-to-keep-city-skyline****  
**Approach:** Compute the max heights for each row and column and adjust heights accordingly.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int maxIncreaseKeepingSkyline(vector<vector<int>>& grid) {
        int n = grid.size();
        vector<int> rowMax(n, 0), colMax(n, 0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                rowMax[i] = max(rowMax[i], grid[i][j]);
                colMax[j] = max(colMax[j], grid[i][j]);
            }
        }

        int increase = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                increase += min(rowMax[i], colMax[j]) - grid[i][j];
            }
        }

        return increase;
    }
};
```

---

##17 ****[Problem Link]https://leetcode.com/problems/di-string-match****  
**Approach:** Use a greedy two-pointer approach to fill values.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<int> diStringMatch(string s) {
        int low = 0, high = s.size();
        vector<int> result;
        for (char c : s) {
            if (c == 'I') result.push_back(low++);
            else result.push_back(high--);
        }
        result.push_back(low);
        return result;
    }
};
```

---

##18 ****[Problem Link]https://leetcode.com/problems/push-dominoes****  
**Approach:** Simulate forces from left and right.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    string pushDominoes(string dominoes) {
        int n = dominoes.size();
        vector<int> forces(n, 0);

        int force = 0;
        for (int i = 0; i < n; i++) {
            if (dominoes[i] == 'R') force = n;
            else if (dominoes[i] == 'L') force = 0;
            else force = max(force - 1, 0);
            forces[i] += force;
        }

        force = 0;
        for (int i = n - 1; i >= 0; i--) {
            if (dominoes[i] == 'L') force = n;
            else if (dominoes[i] == 'R') force = 0;
            else force = max(force - 1, 0);
            forces[i] -= force;
        }

        string result;
        for (int f : forces) {
            if (f > 0) result += 'R';
            else if (f < 0) result += 'L';
            else result += '.';
        }
        return result;
    }
};
```


---

##19 ****[Problem Link]https://leetcode.com/problems/leaf-similar-trees****  
**Approach:** Perform DFS on both trees and compare their leaf sequences.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;


class Solution {
public:
    bool leafSimilar(TreeNode* root1, TreeNode* root2) {
        vector<int> leaves1, leaves2;
        getLeaves(root1, leaves1);
        getLeaves(root2, leaves2);
        return leaves1 == leaves2;
    }

    void getLeaves(TreeNode* node, vector<int>& leaves) {
        if (!node) return;
        if (!node->left && !node->right) leaves.push_back(node->val);
        getLeaves(node->left, leaves);
        getLeaves(node->right, leaves);
    }
};
```

---

##20 ****[Problem Link]https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays****  
**Approach:** Use DP to keep track of maximum subarray sums at different positions.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> maxSumOfThreeSubarrays(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> sum(n + 1, 0), left(n, 0), right(n, n - k);
        
        for (int i = 0; i < n; i++) sum[i + 1] = sum[i] + nums[i];

        int maxSum = sum[k] - sum[0];
        for (int i = k; i < n; i++) {
            if (sum[i + 1] - sum[i + 1 - k] > maxSum) {
                left[i] = i + 1 - k;
                maxSum = sum[i + 1] - sum[i + 1 - k];
            } else {
                left[i] = left[i - 1];
            }
        }

        maxSum = sum[n] - sum[n - k];
        for (int i = n - k - 1; i >= 0; i--) {
            if (sum[i + k] - sum[i] >= maxSum) {
                right[i] = i;
                maxSum = sum[i + k] - sum[i];
            } else {
                right[i] = right[i + 1];
            }
        }

        vector<int> result;
        int maxTotal = 0;
        for (int i = k; i <= n - 2 * k; i++) {
            int l = left[i - 1], r = right[i + k];
            int total = (sum[i + k] - sum[i]) + (sum[l + k] - sum[l]) + (sum[r + k] - sum[r]);
            if (total > maxTotal) {
                maxTotal = total;
                result = {l, i, r};
            }
        }
        
        return result;
    }
};
```

##21 ****[Problem Link]https://leetcode.com/problems/text-justification****  
**Approach:** Use a greedy approach to add words while keeping track of spaces for full justification.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        vector<string> result;
        int n = words.size(), index = 0;
        
        while (index < n) {
            int totalChars = words[index].size();
            int last = index + 1;
            while (last < n && totalChars + 1 + words[last].size() <= maxWidth) {
                totalChars += 1 + words[last].size();
                last++;
            }

            string line = words[index];
            int gaps = last - index - 1;

            if (last == n || gaps == 0) {
                for (int i = index + 1; i < last; i++) {
                    line += " " + words[i];
                }
                while (line.size() < maxWidth) {
                    line += " ";
                }
            } else {
                int spaces = (maxWidth - totalChars + gaps) / gaps;
                int extra = (maxWidth - totalChars + gaps) % gaps;

                for (int i = index + 1; i < last; i++) {
                    line += string(spaces + (i - index <= extra ? 1 : 0), ' ') + words[i];
                }
            }

            result.push_back(line);
            index = last;
        }

        return result;
    }
};
```

---

##22 ****[Problem Link]https://leetcode.com/problems/sum-of-all-odd-length-subarrays****  
**Approach:** Use mathematical contribution of each element in odd-length subarrays.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int sumOddLengthSubarrays(vector<int>& arr) {
        int sum = 0, n = arr.size();
        for (int i = 0; i < n; i++) {
            int contribution = ((i + 1) * (n - i) + 1) / 2;
            sum += arr[i] * contribution;
        }
        return sum;
    }
};
```

---

##23 ****[Problem Link]https://leetcode.com/problems/kth-smallest-number-in-multiplication-table****  
**Approach:** Use binary search to find the kth smallest number in the table.  
**Time Complexity:** O(m log(m * n))

```cpp
class Solution {
public:
    int findKthNumber(int m, int n, int k) {
        int left = 1, right = m * n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            int count = 0;
            for (int i = 1; i <= m; i++) {
                count += min(mid / i, n);
            }
            if (count < k) left = mid + 1;
            else right = mid;
        }
        return left;
    }
};
```

---

##24 ****[Problem Link]https://leetcode.com/problems/minimum-cost-to-hire-k-workers****  
**Approach:** Sort by wage-to-quality ratio and use a max heap to select k workers.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

class Solution {
public:
    double mincostToHireWorkers(vector<int>& quality, vector<int>& wage, int k) {
        vector<pair<double, int>> workers;
        for (int i = 0; i < quality.size(); i++) {
            workers.push_back({(double)wage[i] / quality[i], quality[i]});
        }
        sort(workers.begin(), workers.end());

        priority_queue<int> maxHeap;
        int qualitySum = 0;
        double minCost = DBL_MAX;

        for (auto& w : workers) {
            maxHeap.push(w.second);
            qualitySum += w.second;

            if (maxHeap.size() > k) {
                qualitySum -= maxHeap.top();
                maxHeap.pop();
            }
            if (maxHeap.size() == k) {
                minCost = min(minCost, qualitySum * w.first);
            }
        }
        return minCost;
    }
};
```

---

##25 ****[Problem Link]https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended****  
**Approach:** Use a min heap to attend events in increasing order of end date.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxEvents(vector<vector<int>>& events) {
        sort(events.begin(), events.end());
        priority_queue<int, vector<int>, greater<int>> minHeap;
        int day = 0, i = 0, count = 0, n = events.size();

        while (i < n || !minHeap.empty()) {
            if (minHeap.empty()) day = events[i][0];
            
            while (i < n && events[i][0] == day) {
                minHeap.push(events[i][1]);
                i++;
            }

            minHeap.pop();
            count++;
            day++;

            while (!minHeap.empty() && minHeap.top() < day) {
                minHeap.pop();
            }
        }

        return count;
    }
};
```

---

##26-30  
(The remaining problems 26-30 will be added shortly.)


---

##26 ****[Problem Link]https://leetcode.com/problems/uncrossed-lines****  
**Approach:** Use dynamic programming (LCS variation).  
**Time Complexity:** O(m * n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int maxUncrossedLines(vector<int>& A, vector<int>& B) {
        int m = A.size(), n = B.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (A[i - 1] == B[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
};
```

---

##27 ****[Problem Link]https://leetcode.com/problems/reorder-data-in-log-files****  
**Approach:** Sort letter-logs lexicographically and keep digit-logs in order.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<string> reorderLogFiles(vector<string>& logs) {
        auto compare = [](const string& a, const string& b) {
            int posA = a.find(' '), posB = b.find(' ');
            bool isDigitA = isdigit(a[posA + 1]);
            bool isDigitB = isdigit(b[posB + 1]);
            
            if (!isDigitA && !isDigitB) {
                string subA = a.substr(posA + 1);
                string subB = b.substr(posB + 1);
                return subA == subB ? a < b : subA < subB;
            }
            return isDigitA ? false : true;
        };

        stable_sort(logs.begin(), logs.end(), compare);
        return logs;
    }
};
```

---

##28 ****[Problem Link]https://leetcode.com/problems/sum-of-nodes-with-even-valued-grandparent****  
**Approach:** Use DFS to track even-valued grandparents.  
**Time Complexity:** O(n)

```cpp

class Solution {
public:
    int sumEvenGrandparent(TreeNode* root) {
        return dfs(root, nullptr, nullptr);
    }

    int dfs(TreeNode* node, TreeNode* parent, TreeNode* grandparent) {
        if (!node) return 0;
        int sum = (grandparent && grandparent->val % 2 == 0) ? node->val : 0;
        sum += dfs(node->left, node, parent);
        sum += dfs(node->right, node, parent);
        return sum;
    }
};
```

---

##29 ****[Problem Link]https://leetcode.com/problems/total-hamming-distance****  
**Approach:** Count bit differences across all numbers for each bit position.  
**Time Complexity:** O(n log m)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int totalHammingDistance(vector<int>& nums) {
        int total = 0, n = nums.size();
        for (int i = 0; i < 32; i++) {
            int countOnes = 0;
            for (int num : nums) {
                countOnes += (num >> i) & 1;
            }
            total += countOnes * (n - countOnes);
        }
        return total;
    }
};
```

---

##30 ****[Problem Link]https://leetcode.com/problems/relative-sort-array****  
**Approach:** Use a frequency map and sort remaining elements normally.  
**Time Complexity:** O(n log n)


class Solution {
public:
    vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
        unordered_map<int, int> order;
        for (int i = 0; i < arr2.size(); i++) {
            order[arr2[i]] = i;
        }

        auto compare = [&](int a, int b) {
            if (order.count(a) && order.count(b)) return order[a] < order[b];
            if (order.count(a)) return true;
            if (order.count(b)) return false;
            return a < b;
        };

        sort(arr1.begin(), arr1.end(), compare);
        return arr1;
    }
};
```cpp

```

##31 ****[Problem Link]https://leetcode.com/problems/binary-tree-tilt****  
**Approach:** Use DFS to compute tilt for each node and sum it.  
**Time Complexity:** O(n)

```cpp

class Solution {
public:
    int findTilt(TreeNode* root) {
        int tilt = 0;
        dfs(root, tilt);
        return tilt;
    }

    int dfs(TreeNode* node, int& tilt) {
        if (!node) return 0;
        int left = dfs(node->left, tilt);
        int right = dfs(node->right, tilt);
        tilt += abs(left - right);
        return node->val + left + right;
    }
};
```

---

##32 ****[Problem Link]https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays****  
**Approach:** Compute prefix sum and find max L and M sum.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    int maxSumTwoNoOverlap(vector<int>& nums, int L, int M) {
        return max(helper(nums, L, M), helper(nums, M, L));
    }

    int helper(vector<int>& nums, int L, int M) {
        int n = nums.size(), maxL = 0, result = 0;
        vector<int> prefix(n + 1, 0);
        
        for (int i = 1; i <= n; i++) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
        }
        
        for (int i = L + M; i <= n; i++) {
            maxL = max(maxL, prefix[i - M] - prefix[i - M - L]);
            result = max(result, maxL + prefix[i] - prefix[i - M]);
        }
        
        return result;
    }
};
```

---

##33 ****[Problem Link]https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed****  
**Approach:** Use an unordered map for fast insert, delete, and random access.  
**Time Complexity:** O(1)

```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class RandomizedCollection {
    vector<int> nums;
    unordered_map<int, unordered_set<int>> indices;

public:
    bool insert(int val) {
        indices[val].insert(nums.size());
        nums.push_back(val);
        return indices[val].size() == 1;
    }

    bool remove(int val) {
        if (indices[val].empty()) return false;
        int idx = *indices[val].begin();
        indices[val].erase(idx);

        if (idx != nums.size() - 1) {
            int last = nums.back();
            nums[idx] = last;
            indices[last].erase(nums.size() - 1);
            indices[last].insert(idx);
        }
        
        nums.pop_back();
        return true;
    }

    int getRandom() {
        return nums[rand() % nums.size()];
    }
};
```

---

##34 ****[Problem Link]https://leetcode.com/problems/couples-holding-hands****  
**Approach:** Use Union-Find to count the number of swaps needed.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int minSwapsCouples(vector<int>& row) {
        int n = row.size() / 2;
        vector<int> parent(n);
        
        for (int i = 0; i < n; i++) parent[i] = i;
        
        function<int(int)> find = [&]int x {
            return parent[x] == x ? x : parent[x] = find(parent[x]);
        };
        
        int swaps = 0;
        for (int i = 0; i < row.size(); i += 2) {
            int a = row[i] / 2, b = row[i + 1] / 2;
            int pa = find(a), pb = find(b);
            if (pa != pb) {
                parent[pa] = pb;
                swaps++;
            }
        }
        
        return swaps;
    }
};
```

---

##35 ****[Problem Link]https://leetcode.com/problems/n-ary-tree-preorder-traversal****  
**Approach:** Use recursive DFS for traversal.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Node {
public:
    int val;
    vector<Node*> children;
    Node(int _val) { val = _val; }
};

class Solution {
public:
    vector<int> preorder(Node* root) {
        vector<int> result;
        dfs(root, result);
        return result;
    }

    void dfs(Node* node, vector<int>& result) {
        if (!node) return;
        result.push_back(node->val);
        for (Node* child : node->children) {
            dfs(child, result);
        }
    }
};
```

---

(Next problems 36 to 40 will be added shortly.)


---

##36 ****[Problem Link]https://leetcode.com/problems/letter-tile-possibilities****  
**Approach:** Use backtracking with frequency counting.  
**Time Complexity:** O(n!)

```cpp
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    int numTilePossibilities(string tiles) {
        vector<int> freq(26, 0);
        for (char c : tiles) freq[c - 'A']++;
        return backtrack(freq);
    }

    int backtrack(vector<int>& freq) {
        int sum = 0;
        for (int i = 0; i < 26; i++) {
            if (freq[i] > 0) {
                sum++;
                freq[i]--;
                sum += backtrack(freq);
                freq[i]++;
            }
        }
        return sum;
    }
};
```

---

##37 ****[Problem Link]https://leetcode.com/problems/design-circular-queue****  
**Approach:** Use an array with head and tail pointers.  
**Time Complexity:** O(1)

```cpp
#include <vector>

using namespace std;

class MyCircularQueue {
private:
    vector<int> data;
    int head, tail, size, capacity;

public:
    MyCircularQueue(int k) : data(k), head(-1), tail(-1), size(0), capacity(k) {}

    bool enQueue(int value) {
        if (isFull()) return false;
        if (isEmpty()) head = 0;
        tail = (tail + 1) % capacity;
        data[tail] = value;
        size++;
        return true;
    }

    bool deQueue() {
        if (isEmpty()) return false;
        if (head == tail) head = tail = -1;
        else head = (head + 1) % capacity;
        size--;
        return true;
    }

    int Front() { return isEmpty() ? -1 : data[head]; }
    int Rear() { return isEmpty() ? -1 : data[tail]; }
    bool isEmpty() { return size == 0; }
    bool isFull() { return size == capacity; }
};
```

---

##38 ****[Problem Link]https://leetcode.com/problems/number-of-subarrays-with-bounded-maximum****  
**Approach:** Use a sliding window to count valid subarrays.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    int numSubarrayBoundedMax(vector<int>& nums, int left, int right) {
        return count(nums, right) - count(nums, left - 1);
    }

    int count(vector<int>& nums, int bound) {
        int sum = 0, cur = 0;
        for (int num : nums) {
            cur = (num <= bound) ? cur + 1 : 0;
            sum += cur;
        }
        return sum;
    }
};
```

---

##39 ****[Problem Link]https://leetcode.com/problems/matrix-block-sum****  
**Approach:** Use prefix sum for efficient matrix calculations.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> matrixBlockSum(vector<vector<int>>& mat, int K) {
        int m = mat.size(), n = mat[0].size();
        vector<vector<int>> prefix(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                prefix[i][j] = mat[i - 1][j - 1] + prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1];
            }
        }

        vector<vector<int>> result(m, vector<int>(n));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int r1 = max(0, i - K), c1 = max(0, j - K);
                int r2 = min(m - 1, i + K), c2 = min(n - 1, j + K);
                result[i][j] = prefix[r2 + 1][c2 + 1] - prefix[r1][c2 + 1] - prefix[r2 + 1][c1] + prefix[r1][c1];
            }
        }
        return result;
    }
};
```

---

##40 ****[Problem Link]https://leetcode.com/problems/largest-sum-of-averages****  
**Approach:** Use dynamic programming with partitioning.  
**Time Complexity:** O(n^2 * k)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    double largestSumOfAverages(vector<int>& nums, int K) {
        int n = nums.size();
        vector<double> prefix(n + 1, 0);
        vector<vector<double>> dp(n + 1, vector<double>(K + 1, 0));

        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + nums[i];
            dp[i + 1][1] = prefix[i + 1] / (i + 1);
        }

        for (int k = 2; k <= K; k++) {
            for (int i = k; i <= n; i++) {
                for (int j = k - 1; j < i; j++) {
                    dp[i][k] = max(dp[i][k], dp[j][k - 1] + (prefix[i] - prefix[j]) / (i - j));
                }
            }
        }
        return dp[n][K];
    }
};
```

##41 ****[Problem Link]https://leetcode.com/problems/flip-equivalent-binary-trees****  
**Approach:** Use DFS to compare trees recursively, allowing flips.  
**Time Complexity:** O(n)

```cpp

class Solution {
public:
    bool flipEquiv(TreeNode* root1, TreeNode* root2) {
        if (!root1 || !root2) return root1 == root2;
        if (root1->val != root2->val) return false;
        return (flipEquiv(root1->left, root2->left) && flipEquiv(root1->right, root2->right)) ||
               (flipEquiv(root1->left, root2->right) && flipEquiv(root1->right, root2->left));
    }
};
```

---

##42 ****[Problem Link]https://leetcode.com/problems/number-of-submatrices-that-sum-to-target****  
**Approach:** Use a prefix sum approach and hash map for subarray sum calculation.  
**Time Complexity:** O(m * n^2)

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int numSubmatrixSumTarget(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size(), count = 0;
        
        for (int r = 0; r < m; r++) {
            vector<int> sum(n, 0);
            for (int i = r; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    sum[j] += matrix[i][j];
                }
                
                count += subarraySum(sum, target);
            }
        }
        return count;
    }

    int subarraySum(vector<int>& nums, int target) {
        unordered_map<int, int> prefix;
        prefix[0] = 1;
        int sum = 0, count = 0;

        for (int num : nums) {
            sum += num;
            if (prefix.find(sum - target) != prefix.end()) {
                count += prefix[sum - target];
            }
            prefix[sum]++;
        }

        return count;
    }
};
```

---

##43 ****[Problem Link]https://leetcode.com/problems/matchsticks-to-square****  
**Approach:** Use backtracking with pruning to check if matchsticks can form a square.  
**Time Complexity:** O(4^n)

```cpp
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    bool makesquare(vector<int>& matchsticks) {
        int sum = accumulate(matchsticks.begin(), matchsticks.end(), 0);
        if (sum % 4 != 0) return false;
        vector<int> sides(4, 0);
        sort(matchsticks.rbegin(), matchsticks.rend());
        return dfs(matchsticks, sides, 0, sum / 4);
    }

    bool dfs(vector<int>& matchsticks, vector<int>& sides, int index, int target) {
        if (index == matchsticks.size()) {
            return sides[0] == target && sides[1] == target && sides[2] == target;
        }
        
        for (int i = 0; i < 4; i++) {
            if (sides[i] + matchsticks[index] > target) continue;
            sides[i] += matchsticks[index];
            if (dfs(matchsticks, sides, index + 1, target)) return true;
            sides[i] -= matchsticks[index];
            if (sides[i] == 0) break;
        }
        
        return false;
    }
};
```

---

##44 ****[Problem Link]https://leetcode.com/problems/balance-a-binary-search-tree****  
**Approach:** Convert BST to sorted array and reconstruct it using DFS.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;


class Solution {
public:
    TreeNode* balanceBST(TreeNode* root) {
        vector<int> nums;
        inorder(root, nums);
        return buildTree(nums, 0, nums.size() - 1);
    }

    void inorder(TreeNode* node, vector<int>& nums) {
        if (!node) return;
        inorder(node->left, nums);
        nums.push_back(node->val);
        inorder(node->right, nums);
    }

    TreeNode* buildTree(vector<int>& nums, int left, int right) {
        if (left > right) return nullptr;
        int mid = left + (right - left) / 2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = buildTree(nums, left, mid - 1);
        root->right = buildTree(nums, mid + 1, right);
        return root;
    }
};
```

---

##45 ****[Problem Link]https://leetcode.com/problems/third-maximum-number****  
**Approach:** Use a set to maintain the top 3 maximum values.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <set>

using namespace std;

class Solution {
public:
    int thirdMax(vector<int>& nums) {
        set<int> maxSet;
        for (int num : nums) {
            maxSet.insert(num);
            if (maxSet.size() > 3) {
                maxSet.erase(maxSet.begin());
            }
        }
        return maxSet.size() == 3 ? *maxSet.begin() : *maxSet.rbegin();
    }
};
```

---

##46 ****[Problem Link]https://leetcode.com/problems/fraction-to-recurring-decimal****  
**Approach:** Use hash map to detect repeating remainders and determine cycles.  
**Time Complexity:** O(n)

```cpp
#include <unordered_map>
#include <string>

using namespace std;

class Solution {
public:
    string fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        string result;
        if ((numerator < 0) ^ (denominator < 0)) result += "-";

        long long n = abs((long long)numerator);
        long long d = abs((long long)denominator);

        result += to_string(n / d);
        long long remainder = n % d;

        if (remainder == 0) return result;
        result += ".";

        unordered_map<long long, int> seen;
        while (remainder) {
            if (seen.find(remainder) != seen.end()) {
                result.insert(seen[remainder], "(");
                result += ")";
                break;
            }
            seen[remainder] = result.size();
            remainder *= 10;
            result += to_string(remainder / d);
            remainder %= d;
        }
        return result;
    }
};
```

---

##47 ****[Problem Link]https://leetcode.com/problems/stream-of-characters****  
**Approach:** Use a Trie with a reversed word storage and query suffixes efficiently.  
**Time Complexity:** O(n) per query

```cpp
#include <vector>
#include <string>
#include <deque>
#include <unordered_map>
#include <unordered_set>

using namespace std;

class StreamChecker {
private:
    struct TrieNode {
        unordered_map<char, TrieNode*> children;
        bool isEnd = false;
    };

    TrieNode* root;
    deque<char> stream;
    size_t maxWordLen;  // Maximum word length

public:
    StreamChecker(vector<string>& words) {
        root = new TrieNode();
        maxWordLen = 0;

        // Remove duplicate words
        unordered_set<string> uniqueWords(words.begin(), words.end());

        // Insert words into the Trie in reverse order
        for (const string& word : uniqueWords) {
            TrieNode* node = root;
            maxWordLen = max(maxWordLen, word.size());
            for (int i = word.size() - 1; i >= 0; i--) {
                if (!node->children.count(word[i])) {
                    node->children[word[i]] = new TrieNode();
                }
                node = node->children[word[i]];
            }
            node->isEnd = true;
        }
    }

    bool query(char letter) {
        stream.push_front(letter);
        if (stream.size() > maxWordLen) {
            stream.pop_back();  // Keep the stream size within the longest word length
        }

        TrieNode* node = root;
        for (char c : stream) {
            if (!node->children.count(c)) return false;
            node = node->children[c];
            if (node->isEnd) return true;
        }
        return false;
    }
};
```

---

##48 ****[Problem Link]https://leetcode.com/problems/redundant-connection-ii****  
**Approach:** Use Union-Find and check for cycles and double parents.  
**Time Complexity:** O(n)

```cpp
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        vector<int> parent(n + 1, 0), canA, canB;

        for (auto& edge : edges) {
            if (parent[edge[1]] == 0) {
                parent[edge[1]] = edge[0];
            } else {
                canA = {parent[edge[1]], edge[1]};
                canB = edge;
                edge[1] = 0;
            }
        }

        vector<int> root(n + 1);
        for (int i = 1; i <= n; i++) root[i] = i;

        for (auto& edge : edges) {
            if (edge[1] == 0) continue;
            int u = edge[0], v = edge[1], pu = find(root, u);
            if (pu == v) return canA.empty() ? edge : canA;
            root[v] = pu;
        }
        return canB;
    }

    int find(vector<int>& root, int u) {
        return root[u] == u ? u : root[u] = find(root, root[u]);
    }
};
```

---

##49 ****[Problem Link]https://leetcode.com/problems/minimum-distance-between-bst-nodes****  
**Approach:** Use inorder traversal to find minimum absolute difference.  
**Time Complexity:** O(n)

```cpp
#include <climits>

using namespace std;


class Solution {
public:
    int minDiffInBST(TreeNode* root) {
        int minDiff = INT_MAX, prev = -1;
        inorder(root, prev, minDiff);
        return minDiff;
    }

    void inorder(TreeNode* node, int& prev, int& minDiff) {
        if (!node) return;
        inorder(node->left, prev, minDiff);
        if (prev != -1) minDiff = min(minDiff, node->val - prev);
        prev = node->val;
        inorder(node->right, prev, minDiff);
    }
};
```


---

##50 ****[Problem Link]https://leetcode.com/problems/exclusive-time-of-functions****  
**Approach:** Use a stack to track function calls and compute exclusive execution times.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>
#include <stack>

using namespace std;

class Solution {
public:
    vector<int> exclusiveTime(int n, vector<string>& logs) {
        vector<int> result(n, 0);
        stack<int> st;
        int prevTime = 0;

        for (string& log : logs) {
            int pos1 = log.find(":"), pos2 = log.rfind(":");
            int id = stoi(log.substr(0, pos1));
            string type = log.substr(pos1 + 1, pos2 - pos1 - 1);
            int timestamp = stoi(log.substr(pos2 + 1));

            if (!st.empty()) {
                result[st.top()] += timestamp - prevTime;
            }
            prevTime = timestamp;

            if (type == "start") {
                st.push(id);
            } else {
                result[st.top()]++;
                st.pop();
                prevTime++;
            }
        }

        return result;
    }
};
```


##51 ****[Problem Link]https://leetcode.com/problems/long-pressed-name****  
**Approach:** Use two pointers to simulate the typing process and check if it matches the given name.  
**Time Complexity:** O(n)  

```cpp
#include <string>
using namespace std;

class Solution {
public:
    bool isLongPressedName(string name, string typed) {
        int i = 0, j = 0;
        while (j < typed.size()) {
            if (i < name.size() && name[i] == typed[j]) {
                i++;
            } else if (j == 0 || typed[j] != typed[j - 1]) {
                return false;
            }
            j++;
        }
        return i == name.size();
    }
};
```

---

##52 ****[Problem Link]https://leetcode.com/problems/sort-the-matrix-diagonally****  
**Approach:** Use a hashmap to store diagonal elements, sort and write back.  
**Time Complexity:** O(m * n)  

```cpp
class Solution {
public:
    vector<vector<int>> diagonalSort(vector<vector<int>>& mat) {
        unordered_map<int, vector<int>> diagonals;
        int m = mat.size(), n = mat[0].size();
        
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                diagonals[i - j].push_back(mat[i][j]);

        for (auto& [key, vec] : diagonals)
            sort(vec.begin(), vec.end(), greater<int>());

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                mat[i][j] = diagonals[i - j].back();
                diagonals[i - j].pop_back();
            }

        return mat;
    }
};
```

---

##53 ****[Problem Link]https://leetcode.com/problems/duplicate-zeros****  
**Approach:** Use two passes to shift and insert zeros efficiently.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    void duplicateZeros(vector<int>& arr) {
        int n = arr.size(), count = 0;
        for (int i = 0; i < n; i++) {
            count += arr[i] == 0 ? 2 : 1;
        }
        
        for (int i = n - 1, j = count - 1; i >= 0; i--, j--) {
            if (j < n) arr[j] = arr[i];
            if (arr[i] == 0 && --j < n) arr[j] = 0;
        }
    }
};
```

---

##54 ****[Problem Link]https://leetcode.com/problems/count-number-of-nice-subarrays****  
**Approach:** Use prefix sums and hashmap to count subarrays with k odd numbers.  
**Time Complexity:** O(n)  

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int numberOfSubarrays(vector<int>& nums, int k) {
        unordered_map<int, int> count;
        count[0] = 1;
        int oddCount = 0, result = 0;
        
        for (int num : nums) {
            oddCount += num % 2;
            result += count[oddCount - k];
            count[oddCount]++;
        }
        return result;
    }
};
```

---

##55 ****[Problem Link]https://leetcode.com/problems/ransom-note****  
**Approach:** Use a frequency array to track character counts.  
**Time Complexity:** O(n)  

```cpp
#include <string>
using namespace std;

class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        int count[26] = {};
        for (char c : magazine) count[c - 'a']++;
        for (char c : ransomNote) {
            if (--count[c - 'a'] < 0) return false;
        }
        return true;
    }
};
```
##56 ****[Problem Link]https://leetcode.com/problems/minimum-distance-between-bst-nodes****  
**Approach:** Perform an in-order traversal to get a sorted array and find the minimum difference between consecutive elements.  
**Time Complexity:** O(n)

```cpp
#include <iostream>
#include <climits>
using namespace std;


class Solution {
public:
    int prevVal = -1;
    int minDiff = INT_MAX;

    int minDiffInBST(TreeNode* root) {
        if (!root) return minDiff;
        minDiffInBST(root->left);
        if (prevVal != -1) {
            minDiff = min(minDiff, root->val - prevVal);
        }
        prevVal = root->val;
        minDiffInBST(root->right);
        return minDiff;
    }
};
```

---

##57 ****[Problem Link]https://leetcode.com/problems/n-queens-ii****  
**Approach:** Use backtracking to explore all valid board states and count the solutions.  
**Time Complexity:** O(N!) for N-Queens.

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int totalNQueens(int n) {
        vector<int> cols(n, 0), diag1(2 * n, 0), diag2(2 * n, 0);
        return solve(0, n, cols, diag1, diag2);
    }

    int solve(int row, int n, vector<int>& cols, vector<int>& diag1, vector<int>& diag2) {
        if (row == n) return 1;
        int count = 0;
        for (int col = 0; col < n; col++) {
            if (cols[col] || diag1[row - col + n] || diag2[row + col]) continue;
            cols[col] = diag1[row - col + n] = diag2[row + col] = 1;
            count += solve(row + 1, n, cols, diag1, diag2);
            cols[col] = diag1[row - col + n] = diag2[row + col] = 0;
        }
        return count;
    }
};
```

---

##58 ****[Problem Link]https://leetcode.com/problems/longest-harmonious-subsequence****  
**Approach:** Use an unordered map to count frequencies and find max length subsequence with elements differing by 1.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int findLHS(vector<int>& nums) {
        unordered_map<int, int> freq;
        int result = 0;
        for (int num : nums) freq[num]++;
        for (auto& [num, count] : freq) {
            if (freq.count(num + 1)) {
                result = max(result, count + freq[num + 1]);
            }
        }
        return result;
    }
};
```

---

##59 ****[Problem Link]https://leetcode.com/problems/employee-importance****  
**Approach:** Perform BFS using a queue to traverse employees and calculate total importance.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>
#include <queue>
using namespace std;


class Solution {
public:
    int getImportance(vector<Employee*> employees, int id) {
        unordered_map<int, Employee*> empMap;
        for (auto& emp : employees) empMap[emp->id] = emp;

        int totalImportance = 0;
        queue<int> q;
        q.push(id);

        while (!q.empty()) {
            int currId = q.front();
            q.pop();
            Employee* emp = empMap[currId];
            totalImportance += emp->importance;
            for (int subId : emp->subordinates) {
                q.push(subId);
            }
        }
        return totalImportance;
    }
};
```

---

##60 ****[Problem Link]https://leetcode.com/problems/long-pressed-name****  
**Approach:** Use two pointers to check if the name can be formed from typed characters by allowing long presses.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    bool isLongPressedName(string name, string typed) {
        int i = 0, j = 0;
        while (j < typed.size()) {
            if (i < name.size() && name[i] == typed[j]) {
                i++;
                j++;
            } else if (j > 0 && typed[j] == typed[j - 1]) {
                j++;
            } else {
                return false;
            }
        }
        return i == name.size();
    }
};
```
##61 ****[Problem Link]https://leetcode.com/problems/sort-the-matrix-diagonally****  
**Approach:** Use a hashmap to group diagonal elements using (i - j) as the key. Sort and reassign values.  
**Time Complexity:** O(m * n * log(min(m, n)))

```cpp
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> diagonalSort(vector<vector<int>>& mat) {
        unordered_map<int, vector<int>> diagonals;
        int m = mat.size(), n = mat[0].size();
        
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                diagonals[i - j].push_back(mat[i][j]);

        for (auto& [key, vec] : diagonals)
            sort(vec.begin(), vec.end(), greater<int>());

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                mat[i][j] = diagonals[i - j].back();
                diagonals[i - j].pop_back();
            }

        return mat;
    }
};
```

---

##62 ****[Problem Link]https://leetcode.com/problems/duplicate-zeros****  
**Approach:** Use two passes - first to calculate the space needed, second to fill the array from the end.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    void duplicateZeros(vector<int>& arr) {
        int countZeros = 0;
        int n = arr.size();

        for (int i = 0; i < n; i++) {
            if (arr[i] == 0) countZeros++;
        }

        int i = n - 1, j = n + countZeros - 1;
        while (i >= 0) {
            if (j < n) arr[j] = arr[i];
            if (arr[i] == 0) {
                j--;
                if (j < n) arr[j] = 0;
            }
            i--;
            j--;
        }
    }
};
```

---

##63 ****[Problem Link]https://leetcode.com/problems/count-number-of-nice-subarrays****  
**Approach:** Use a sliding window with at most K odd numbers to count nice subarrays.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int atMostK(vector<int>& nums, int k) {
        int i = 0, count = 0;
        for (int j = 0; j < nums.size(); j++) {
            k -= nums[j] % 2;
            while (k < 0) {
                k += nums[i] % 2;
                i++;
            }
            count += j - i + 1;
        }
        return count;
    }

    int numberOfSubarrays(vector<int>& nums, int k) {
        return atMostK(nums, k) - atMostK(nums, k - 1);
    }
};
```

---

##64 ****[Problem Link]https://leetcode.com/problems/ransom-note****  
**Approach:** Use a hashmap to count characters in the magazine and verify if the ransom note can be formed.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>
using namespace std;

class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        unordered_map<char, int> charCount;
        for (char c : magazine) charCount[c]++;
        for (char c : ransomNote) {
            if (charCount[c]-- <= 0) return false;
        }
        return true;
    }
};
```

---

##65 ****[Problem Link]https://leetcode.com/problems/k-th-symbol-in-grammar****  
**Approach:** Use recursion to find the K-th symbol by observing the parent symbol.  
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int kthGrammar(int n, int k) {
        if (n == 1) return 0;
        int parent = kthGrammar(n - 1, (k + 1) / 2);
        return parent == 0 ? (k % 2 == 1 ? 0 : 1) : (k % 2 == 1 ? 1 : 0);
    }
};
```

---

##66 ****[Problem Link]https://leetcode.com/problems/remove-boxes****  
**Approach:** Use dynamic programming with memoization to maximize points by removing boxes strategically.  
**Time Complexity:** O(n^4)

```cpp
class Solution 
{
public:

    int solve(int l , int r , vector < int > & boxes , int k , int dp[101][101][101]){
       
        if(l > r) return 0; 

        int k_here = k; 
        int l_here = l; 

        if(dp[l][r][k_here] >= 0) return dp[l][r][k];  

        while(l+1 <= r && boxes[l] == boxes[l+1]) 
        {
            k++; 
            l++; 
        }

        int ans = (k+1)*(k+1) + solve(l+1 , r , boxes , 0 , dp); 

        for(int m = l+1 ; m <=r ; m++)
        {
            if(boxes[l] == boxes[m]){

                ans = max(ans , solve(m , r, boxes , k+1 , dp) + solve(l+1 , m-1 , boxes , 0 , dp)); 

            }
        }

        return dp[l_here][r][k_here] = ans; 
    }
    int removeBoxes(vector<int>& boxes) 
    {
        int dp[101][101][101]; 
        memset(dp , -1 , sizeof(dp)); 
        int n = boxes.size(); 
        return solve(0 , n-1 , boxes , 0 , dp);     
    }
};
```

##67 ****[Problem Link]https://leetcode.com/problems/n-ary-tree-postorder-traversal****  
**Approach:** Use recursion or an iterative approach with a stack to perform postorder traversal of an N-ary tree.  
**Time Complexity:** O(n)

```cpp
class Pair
{
public:
    Node* node;
    int childrenIndex;
    Pair(Node* _node, int _childrenIndex)
    {
        node = _node;
        childrenIndex = _childrenIndex;
    }
};

class Solution {
public:
    vector<int> postorder(Node* root)
    {
        int currentRootIndex = 0;
        stack<Pair*> st;
        vector<int> postorderTraversal;
        while (root != NULL || st.size() > 0)
        {
            if (root != NULL)
            {
                st.push(new Pair(root, currentRootIndex));
                currentRootIndex = 0;
                if (root->children.size() >= 1)
                {
                    root = root->children[0];
                }
                else
                {
                    root = NULL;
                }
                continue;
            }
            Pair* temp = st.top();
            st.pop();
            postorderTraversal.push_back(temp->node->val);
            while (st.size() > 0 && temp->childrenIndex == st.top()->node->children.size() - 1)
            {
                temp = st.top();
                st.pop();
                postorderTraversal.push_back(temp->node->val);
            }
            if (st.size() > 0)
            {
                root = st.top()->node->children[temp->childrenIndex + 1];
                currentRootIndex = temp->childrenIndex + 1;
            }
    }
    return postorderTraversal;    
    }
};
```

---

##68 ****[Problem Link]https://leetcode.com/problems/minimum-cost-to-merge-stones****  
**Approach:** Use dynamic programming to minimize the cost of merging stones.  
**Time Complexity:** O(n^3)

```cpp
#include <vector>
#include <climits>
using namespace std;

class Solution {
public:
    int mergeStones(vector<int>& stones, int k) {
        int n = stones.size();
        if ((n - 1) % (k - 1) != 0) return -1;

        vector<int> prefix(n + 1, 0);
        for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + stones[i];

        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int length = k; length <= n; length++) {
            for (int i = 0; i + length <= n; i++) {
                int j = i + length - 1;
                dp[i][j] = INT_MAX;
                for (int m = i; m < j; m += k - 1) {
                    dp[i][j] = min(dp[i][j], dp[i][m] + dp[m + 1][j]);
                }
                if ((j - i) % (k - 1) == 0) {
                    dp[i][j] += prefix[j + 1] - prefix[i];
                }
            }
        }
        return dp[0][n - 1];
    }
};
```

---

##69 ****[Problem Link]https://leetcode.com/problems/furthest-building-you-can-reach****  
**Approach:** Use a max heap (priority queue) to keep track of the highest jumps.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int furthestBuilding(vector<int>& heights, int bricks, int ladders) {
        priority_queue<int> maxHeap;
        for (int i = 0; i < heights.size() - 1; i++) {
            int diff = heights[i + 1] - heights[i];
            if (diff > 0) {
                maxHeap.push(diff);
                bricks -= diff;
                if (bricks < 0) {
                    if (ladders == 0) return i;
                    bricks += maxHeap.top();
                    maxHeap.pop();
                    ladders--;
                }
            }
        }
        return heights.size() - 1;
    }
};
```

---

##70 ****[Problem Link]https://leetcode.com/problems/count-of-range-sum****  
**Approach:** Use a merge sort based algorithm to count range sums.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int countRangeSum(vector<int>& nums, int lower, int upper) {
        vector<long> prefixSum(1, 0);
        for (int num : nums) {
            prefixSum.push_back(prefixSum.back() + num);
        }
        return mergeSort(prefixSum, 0, prefixSum.size() - 1, lower, upper);
    }

    int mergeSort(vector<long>& sum, int left, int right, int lower, int upper) {
        if (left >= right) return 0;
        int mid = left + (right - left) / 2;
        int count = mergeSort(sum, left, mid, lower, upper) + mergeSort(sum, mid + 1, right, lower, upper);

        int j = mid + 1, k = mid + 1, t = mid + 1;
        vector<long> temp;

        for (int i = left; i <= mid; i++) {
            while (k <= right && sum[k] - sum[i] < lower) k++;
            while (j <= right && sum[j] - sum[i] <= upper) j++;
            while (t <= right && sum[t] < sum[i]) temp.push_back(sum[t++]);
            temp.push_back(sum[i]);
            count += j - k;
        }

        while (t <= right) temp.push_back(sum[t++]);
        for (int i = left; i <= right; i++) sum[i] = temp[i - left];
        return count;
    }
};
```

---

##71 ****[Problem Link]https://leetcode.com/problems/number-of-closed-islands****  
**Approach:** Use depth-first search (DFS) or breadth-first search (BFS) to find and count closed islands.  
**Time Complexity:** O(m * n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int closedIsland(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int count = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    count += dfs(grid, i, j);
                }
            }
        }
        return count;
    }

    int dfs(vector<vector<int>>& grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size()) return 0;
        if (grid[i][j] == 1) return 1;
        grid[i][j] = 1;
        int up = dfs(grid, i - 1, j);
        int down = dfs(grid, i + 1, j);
        int left = dfs(grid, i, j - 1);
        int right = dfs(grid, i, j + 1);
        return up & down & left & right;
    }
};
```

##72 ****[Problem Link]https://leetcode.com/problems/reverse-vowels-of-a-string****  
**Approach:** Use two pointers to swap vowels in the string from both ends.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_set>
using namespace std;

class Solution {
public:
    string reverseVowels(string s) {
        unordered_set<char> vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'};
        int left = 0, right = s.size() - 1;
        
        while (left < right) {
            while (left < right && vowels.find(s[left]) == vowels.end()) left++;
            while (left < right && vowels.find(s[right]) == vowels.end()) right--;
            swap(s[left++], s[right--]);
        }
        return s;
    }
};
```

---

##73 ****[Problem Link]https://leetcode.com/problems/out-of-boundary-paths****  
**Approach:** Use dynamic programming with memoization to count paths.  
**Time Complexity:** O(N * m * n)

```cpp
class Solution {
public:
    int dp[55][55][55];
    long long mod = 1e9 + 7;
    
    int dfs(int i, int j, int n, int m, int moves)
    {
        
        if(i < 0 || i >= n || j < 0 || j >= m)
            return 1;
        
        if(moves <= 0)
            return 0;
        
        if(dp[i][j][moves] != -1)
            return dp[i][j][moves];
        
        int up = dfs(i - 1, j, n, m, moves - 1);
        int down = dfs(i + 1, j, n, m, moves - 1);
        int left = dfs(i, j - 1, n, m, moves - 1);
        int right = dfs(i, j + 1, n, m, moves - 1);
        
        return dp[i][j][moves] = (up % mod + down % mod + left % mod + right % mod) % mod;
    }
    
    int findPaths(int n, int m, int maxMove, int startRow, int startColumn) {
        memset(dp, -1, sizeof(dp));
        return dfs(startRow, startColumn, n, m, maxMove);
    }
};
```

---

##74 ****[Problem Link]https://leetcode.com/problems/all-elements-in-two-binary-search-trees****  
**Approach:** Perform an inorder traversal of both trees and merge the results using a two-pointer technique.  
**Time Complexity:** O(n + m)

```cpp
#include <vector>
using namespace std;


class Solution {
public:
    vector<int> getAllElements(TreeNode* root1, TreeNode* root2) {
        vector<int> result, list1, list2;
        inorder(root1, list1);
        inorder(root2, list2);
        
        int i = 0, j = 0;
        while (i < list1.size() && j < list2.size()) {
            if (list1[i] < list2[j]) result.push_back(list1[i++]);
            else result.push_back(list2[j++]);
        }
        while (i < list1.size()) result.push_back(list1[i++]);
        while (j < list2.size()) result.push_back(list2[j++]);
        return result;
    }

    void inorder(TreeNode* root, vector<int>& result) {
        if (!root) return;
        inorder(root->left, result);
        result.push_back(root->val);
        inorder(root->right, result);
    }
};
```

---

##75 ****[Problem Link]https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome****  
**Approach:** Use dynamic programming to calculate the minimum insertions needed.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    int minInsertions(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1];
                } else {
                    dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[0][n - 1];
    }
};
```

---

##76 ****[Problem Link]https://leetcode.com/problems/maximum-average-subarray-i****  
**Approach:** Use a sliding window of size k to find the maximum average.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        double sum = 0;
        for (int i = 0; i < k; i++) {
            sum += nums[i];
        }
        double maxSum = sum;

        for (int i = k; i < nums.size(); i++) {
            sum += nums[i] - nums[i - k];
            maxSum = max(maxSum, sum);
        }
        return maxSum / k;
    }
};
```

##77 ****[Problem Link]https://leetcode.com/problems/number-of-steps-to-reduce-a-number-to-zero****  
**Approach:** Use a simple loop to reduce the number using the given rules.  
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int numberOfSteps(int num) {
        int steps = 0;
        while (num > 0) {
            num = (num % 2 == 0) ? num / 2 : num - 1;
            steps++;
        }
        return steps;
    }
};
```

---

##78 ****[Problem Link]https://leetcode.com/problems/n-th-tribonacci-number****  
**Approach:** Use dynamic programming to calculate the nth Tribonacci number.  
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int tribonacci(int n) {
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;
        
        int a = 0, b = 1, c = 1;
        for (int i = 3; i <= n; i++) {
            int d = a + b + c;
            a = b;
            b = c;
            c = d;
        }
        return c;
    }
};
```

---

##79 ****[Problem Link]https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix****  
**Approach:** Use a priority queue to find the k weakest rows.  
**Time Complexity:** O(m * n + m log k)

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    vector<int> kWeakestRows(vector<vector<int>>& mat, int k) {
        priority_queue<pair<int, int>> pq;
        for (int i = 0; i < mat.size(); i++) {
            int soldiers = count(mat[i].begin(), mat[i].end(), 1);
            pq.push({soldiers, i});
            if (pq.size() > k) pq.pop();
        }
        vector<int> result(k);
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pq.top().second;
            pq.pop();
        }
        return result;
    }
};
```

---

##80 ****[Problem Link]https://leetcode.com/problems/valid-mountain-array****  
**Approach:** Check the increasing and decreasing sequence to validate a mountain array.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool validMountainArray(vector<int>& arr) {
        int n = arr.size(), i = 0, j = n - 1;
        
        while (i + 1 < n && arr[i] < arr[i + 1]) i++;
        while (j > 0 && arr[j] < arr[j - 1]) j--;
        
        return i > 0 && i == j && j < n - 1;
    }
};
```

---

##81 ****[Problem Link]https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets****  
**Approach:** Use binary search to minimize the number of days.  
**Time Complexity:** O(n log maxDay)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool canMake(vector<int>& bloomDay, int m, int k, int day) {
        int count = 0, bouquets = 0;
        for (int bloom : bloomDay) {
            if (bloom <= day) {
                count++;
                if (count == k) {
                    bouquets++;
                    count = 0;
                }
            } else {
                count = 0;
            }
        }
        return bouquets >= m;
    }
    
    int minDays(vector<int>& bloomDay, int m, int k) {
        long long totalFlowers = (long long)m * k;
        if (totalFlowers > bloomDay.size()) return -1;
        
        int low = *min_element(bloomDay.begin(), bloomDay.end());
        int high = *max_element(bloomDay.begin(), bloomDay.end());
        
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (canMake(bloomDay, m, k, mid)) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }
};
```

##82 ****[Problem Link]https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero****  
**Approach:** Use a sliding window to find the longest subarray with a sum equal to `sum(nums) - x`.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int minOperations(vector<int>& nums, int x) {
        int totalSum = accumulate(nums.begin(), nums.end(), 0);
        int target = totalSum - x;
        if (target < 0) return -1;

        int maxLength = -1, currentSum = 0, left = 0;
        for (int right = 0; right < nums.size(); right++) {
            currentSum += nums[right];
            while (currentSum > target) {
                currentSum -= nums[left++];
            }
            if (currentSum == target) {
                maxLength = max(maxLength, right - left + 1);
            }
        }
        return maxLength == -1 ? -1 : nums.size() - maxLength;
    }
};
```

---

##83 ****[Problem Link]https://leetcode.com/problems/count-submatrices-with-all-ones****  
**Approach:** Use a dynamic programming matrix to count submatrices with all ones.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int numSubmat(vector<vector<int>>& mat) {
        int rows = mat.size(), cols = mat[0].size();
        vector<vector<int>> dp(rows, vector<int>(cols, 0));
        int result = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mat[i][j] == 1) {
                    dp[i][j] = (j == 0) ? 1 : dp[i][j-1] + 1;
                    int minWidth = dp[i][j];
                    for (int k = i; k >= 0; k--) {
                        minWidth = min(minWidth, dp[k][j]);
                        result += minWidth;
                    }
                }
            }
        }
        return result;
    }
};
```

---

##84 ****[Problem Link]https://leetcode.com/problems/guess-number-higher-or-lower-ii****  
**Approach:** Use dynamic programming to minimize the cost of guessing.  
**Time Complexity:** O(n^3)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int getMoneyAmount(int n) {
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
        for (int len = 2; len <= n; len++) {
            for (int i = 1; i <= n - len + 1; i++) {
                int j = i + len - 1;
                dp[i][j] = INT_MAX;
                for (int k = i; k <= j; k++) {
                    int cost = k + max(k > i ? dp[i][k-1] : 0, k < j ? dp[k+1][j] : 0);
                    dp[i][j] = min(dp[i][j], cost);
                }
            }
        }
        return dp[1][n];
    }
};
```

---

##85 ****[Problem Link]https://leetcode.com/problems/as-far-from-land-as-possible****  
**Approach:** Use a multi-source BFS from all land cells.  
**Time Complexity:** O(n^2)

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int maxDistance(vector<vector<int>>& grid) {
        int n = grid.size();
        queue<pair<int, int>> q;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    q.push({i, j});
                }
            }
        }
        
        if (q.empty() || q.size() == n * n) return -1;
        
        int distance = -1;
        vector<pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        while (!q.empty()) {
            int size = q.size();
            distance++;
            for (int i = 0; i < size; i++) {
                auto [x, y] = q.front();
                q.pop();
                for (auto [dx, dy] : directions) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && ny >= 0 && nx < n && ny < n && grid[nx][ny] == 0) {
                        grid[nx][ny] = 1;
                        q.push({nx, ny});
                    }
                }
            }
        }
        return distance;
    }
};
```

---

##86 ****[Problem Link]https://leetcode.com/problems/time-needed-to-inform-all-employees****  
**Approach:** Use a BFS or DFS to find the time needed to inform all employees.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int numOfMinutes(int n, int headID, vector<int>& manager, vector<int>& informTime) {
        vector<vector<int>> tree(n);
        for (int i = 0; i < n; i++) {
            if (manager[i] != -1) {
                tree[manager[i]].push_back(i);
            }
        }
        
        queue<pair<int, int>> q;
        q.push({headID, 0});
        int maxTime = 0;

        while (!q.empty()) {
            auto [id, time] = q.front();
            q.pop();
            maxTime = max(maxTime, time);
            for (int sub : tree[id]) {
                q.push({sub, time + informTime[id]});
            }
        }
        return maxTime;
    }
};
```

##87 ****[Problem Link]https://leetcode.com/problems/smallest-subsequence-of-distinct-characters****  
**Approach:** Use a stack and a frequency map to construct the smallest subsequence.  
**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <stack>
using namespace std;

class Solution {
public:
    string smallestSubsequence(string s) {
        unordered_map<char, int> freq;
        unordered_set<char> seen;
        stack<char> st;

        for (char c : s) freq[c]++;

        for (char c : s) {
            freq[c]--;
            if (seen.find(c) != seen.end()) continue;

            while (!st.empty() && st.top() > c && freq[st.top()] > 0) {
                seen.erase(st.top());
                st.pop();
            }

            st.push(c);
            seen.insert(c);
        }

        string result;
        while (!st.empty()) {
            result = st.top() + result;
            st.pop();
        }
        return result;
    }
};
```

---

##88 ****[Problem Link]https://leetcode.com/problems/replace-words****  
**Approach:** Use a trie to efficiently replace words with their roots.  
**Time Complexity:** O(n * m)

```cpp
#include <vector>
#include <string>
#include <unordered_map>
using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    string word;
};

class Solution {
public:
    void insert(TrieNode* root, string& word) {
        TrieNode* node = root;
        for (char c : word) {
            if (!node->children.count(c)) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->word = word;
    }

    string replaceWords(vector<string>& dictionary, string sentence) {
        TrieNode* root = new TrieNode();
        for (string& word : dictionary) {
            insert(root, word);
        }

        stringstream ss(sentence);
        string word, result = "";
        while (ss >> word) {
            TrieNode* node = root;
            for (char c : word) {
                if (!node->children.count(c) || !node->word.empty()) break;
                node = node->children[c];
            }
            result += (node->word.empty() ? word : node->word) + " ";
        }
        result.pop_back();
        return result;
    }
};
```

---

##89 ****[Problem Link]https://leetcode.com/problems/sliding-puzzle****  
**Approach:** Use BFS for state exploration to find the minimum moves.  
**Time Complexity:** O(n!)  

```cpp
#include <vector>
#include <queue>
#include <string>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int slidingPuzzle(vector<vector<int>>& board) {
        string goal = "123450";
        string start = "";
        for (auto& row : board) {
            for (int num : row) {
                start += to_string(num);
            }
        }

        vector<vector<int>> neighbors = {{1, 3}, {0, 2, 4}, {1, 5}, {0, 4}, {1, 3, 5}, {2, 4}};
        queue<pair<string, int>> q;
        unordered_set<string> visited;
        q.push({start, 0});
        visited.insert(start);

        while (!q.empty()) {
            auto [curr, moves] = q.front();
            q.pop();

            if (curr == goal) return moves;

            int zeroPos = curr.find('0');
            for (int n : neighbors[zeroPos]) {
                string newState = curr;
                swap(newState[zeroPos], newState[n]);
                if (!visited.count(newState)) {
                    visited.insert(newState);
                    q.push({newState, moves + 1});
                }
            }
        }
        return -1;
    }
};
```

---

##90 ****[Problem Link]https://leetcode.com/problems/reverse-only-letters****  
**Approach:** Use two pointers to reverse only letters.  
**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

class Solution {
public:
    string reverseOnlyLetters(string s) {
        int left = 0, right = s.size() - 1;

        while (left < right) {
            if (!isalpha(s[left])) {
                left++;
            } else if (!isalpha(s[right])) {
                right--;
            } else {
                swap(s[left++], s[right--]);
            }
        }
        return s;
    }
};
```

---

##91 ****[Problem Link]https://leetcode.com/problems/increasing-subsequences****  
**Approach:** Use backtracking to generate increasing subsequences.  
**Time Complexity:** O(2^n)

```cpp
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
public:
    void backtrack(vector<int>& nums, int index, vector<int>& current, vector<vector<int>>& result) {
        if (current.size() >= 2) {
            result.push_back(current);
        }
        unordered_set<int> used;

        for (int i = index; i < nums.size(); i++) {
            if ((current.empty() || nums[i] >= current.back()) && used.find(nums[i]) == used.end()) {
                used.insert(nums[i]);
                current.push_back(nums[i]);
                backtrack(nums, i + 1, current, result);
                current.pop_back();
            }
        }
    }

    vector<vector<int>> findSubsequences(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> current;
        backtrack(nums, 0, current, result);
        return result;
    }
};
```

##92 ****[Problem Link]https://leetcode.com/problems/monotonic-array****  
**Approach:** Check if the array is monotonic (either non-increasing or non-decreasing).  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    bool isMonotonic(vector<int>& nums) {
        bool increasing = true, decreasing = true;
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] > nums[i-1]) decreasing = false;
            if (nums[i] < nums[i-1]) increasing = false;
        }
        return increasing || decreasing;
    }
};
```

---

##93 ****[Problem Link]https://leetcode.com/problems/snapshot-array****  
**Approach:** Use a hashmap with versioning to store snapshots efficiently.  
**Time Complexity:** O(log n) for get and O(1) for set.

```cpp
#include <unordered_map>
#include <vector>
using namespace std;

class SnapshotArray {
private:
    vector<unordered_map<int, int>> snaps;
    unordered_map<int, int> current;
    int snapId = 0;

public:
    SnapshotArray(int length) {
        snaps.resize(length);
    }

    void set(int index, int val) {
        current[index] = val;
    }

    int snap() {
        for (auto& [key, val] : current) {
            snaps[key][snapId] = val;
        }
        return snapId++;
    }

    int get(int index, int snap_id) {
        auto it = snaps[index].upper_bound(snap_id);
        if (it == snaps[index].begin()) return 0;
        return prev(it)->second;
    }
};
```

---

##94 ****[Problem Link]https://leetcode.com/problems/minimum-absolute-difference****  
**Approach:** Sort the array and find the minimum absolute difference.  
**Time Complexity:** O(n log n)

```cpp
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<vector<int>> minimumAbsDifference(vector<int>& arr) {
        sort(arr.begin(), arr.end());
        vector<vector<int>> result;
        int minDiff = INT_MAX;

        for (int i = 1; i < arr.size(); i++) {
            int diff = arr[i] - arr[i-1];
            if (diff < minDiff) {
                minDiff = diff;
                result.clear();
            }
            if (diff == minDiff) {
                result.push_back({arr[i-1], arr[i]});
            }
        }
        return result;
    }
};
```

---

##95 ****[Problem Link]https://leetcode.com/problems/summary-ranges****  
**Approach:** Traverse the array to identify continuous ranges.  
**Time Complexity:** O(n)

```cpp
#include <vector>
#include <string>
using namespace std;

class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums) {
        vector<string> result;
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            int start = nums[i];
            while (i + 1 < n && nums[i] + 1 == nums[i + 1]) {
                i++;
            }
            if (start != nums[i]) {
                result.push_back(to_string(start) + "->" + to_string(nums[i]));
            } else {
                result.push_back(to_string(start));
            }
        }
        return result;
    }
};
```

---

##96 ****[Problem Link]https://leetcode.com/problems/domino-and-tromino-tiling****  
**Approach:** Use dynamic programming to calculate possible ways of tiling.  
**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    int numTilings(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;
        if (n == 3) return 5;
        
        vector<long long> dp(n + 1);
        dp[1] = 1; dp[2] = 2; dp[3] = 5;
        int MOD = 1e9 + 7;
        
        for (int i = 4; i <= n; i++) {
            dp[i] = (2 * dp[i - 1] + dp[i - 3]) % MOD;
        }
        return dp[n];
    }
};
```


##97 ****[Problem Link]https://leetcode.com/problems/length-of-longest-fibonacci-subsequence****  
**Approach:** Use dynamic programming with a hashmap to store pairs and their longest Fibonacci-like subsequence length.  
**Time Complexity:** O(n²)  

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int lenLongestFibSubseq(vector<int>& arr) {
        int n = arr.size();
        unordered_map<int, int> index;
        for (int i = 0; i < n; i++) index[arr[i]] = i;

        unordered_map<int, int> dp;
        int maxLen = 0;

        for (int k = 0; k < n; k++) {
            for (int j = 0; j < k; j++) {
                int i = index.count(arr[k] - arr[j]) ? index[arr[k] - arr[j]] : -1;
                if (i >= 0 && i < j) {
                    dp[j * n + k] = (dp[i * n + j] + 1);
                    maxLen = max(maxLen, dp[j * n + k]);
                }
            }
        }
        return maxLen >= 3 ? maxLen : 0;
    }
};
```

---

##98 ****[Problem Link]https://leetcode.com/problems/swapping-nodes-in-a-linked-list****  
**Approach:** Use two-pointer technique to locate the k-th node from the beginning and end, then swap their values.  
**Time Complexity:** O(n)  

```cpp
#include <iostream>
using namespace std;


class Solution {
public:
    ListNode* swapNodes(ListNode* head, int k) {
        ListNode* first = head, * second = head, * curr = head;
        int count = 1;

        while (curr) {
            if (count < k) first = first->next;
            if (count > k) second = second->next;
            curr = curr->next;
            count++;
        }

        swap(first->val, second->val);
        return head;
    }
};
```

---

##99 ****[Problem Link]https://leetcode.com/problems/gray-code****  
**Approach:** Use bitwise manipulation to generate Gray code sequence.  
**Time Complexity:** O(2^n)  

```cpp
#include <vector>
using namespace std;

class Solution {
public:
    vector<int> grayCode(int n) {
        vector<int> result(1 << n);
        for (int i = 0; i < (1 << n); i++) {
            result[i] = i ^ (i >> 1);
        }
        return result;
    }
};
```

---

##100 ****[Problem Link]https://leetcode.com/problems/minimum-area-rectangle****  
**Approach:** Use hash set to store coordinates and check pairs to form rectangles.  
**Time Complexity:** O(n²)  

```cpp
#include <vector>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int minAreaRect(vector<vector<int>>& points) {
        unordered_set<string> pointSet;
        for (auto& p : points) {
            pointSet.insert(to_string(p[0]) + "," + to_string(p[1]));
        }

        int minArea = INT_MAX;
        for (int i = 0; i < points.size(); i++) {
            for (int j = i + 1; j < points.size(); j++) {
                auto& p1 = points[i];
                auto& p2 = points[j];
                if (p1[0] != p2[0] && p1[1] != p2[1] &&
                    pointSet.count(to_string(p1[0]) + "," + to_string(p2[1])) &&
                    pointSet.count(to_string(p2[0]) + "," + to_string(p1[1]))) {
                    int area = abs(p1[0] - p2[0]) * abs(p1[1] - p2[1]);
                    minArea = min(minArea, area);
                }
            }
        }
        return minArea == INT_MAX ? 0 : minArea;
    }
};
```

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
