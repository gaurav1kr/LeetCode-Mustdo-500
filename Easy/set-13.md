# Optimized C++ Solutions for LeetCode Problems

## 1. Count Negative Numbers in a Sorted Matrix

**Approach:**
- Use binary search in each row since they are sorted.
- Count the number of negatives efficiently.

**Time Complexity:** O(m log n)

```cpp
class Solution {
public:
    int countNegatives(vector<vector<int>>& grid) {
        int count = 0;
        for (auto &row : grid) {
            count += row.end() - upper_bound(row.begin(), row.end(), -1, greater<int>());
        }
        return count;
    }
};
```

---
## 2. Reshape the Matrix

**Approach:**
- Flatten the matrix and reshape it using indexing.

**Time Complexity:** O(m*n)

```cpp
class Solution {
public:
    vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
        int m = mat.size(), n = mat[0].size();
        if (m * n != r * c) return mat;
        vector<vector<int>> res(r, vector<int>(c));
        for (int i = 0; i < m * n; ++i)
            res[i / c][i % c] = mat[i / n][i % n];
        return res;
    }
};
```

---
## 3. Toeplitz Matrix

**Approach:**
- Check if each diagonal contains the same element.

**Time Complexity:** O(m*n)

```cpp
class Solution {
public:
    bool isToeplitzMatrix(vector<vector<int>>& matrix) {
        for (int i = 1; i < matrix.size(); i++) {
            for (int j = 1; j < matrix[0].size(); j++) {
                if (matrix[i][j] != matrix[i - 1][j - 1])
                    return false;
            }
        }
        return true;
    }
};
```

---
## 4. Range Sum Query - Immutable

**Approach:**
- Use prefix sum array.

**Time Complexity:** O(1) per query

```cpp
class NumArray {
    vector<int> prefixSum;
public:
    NumArray(vector<int>& nums) {
        prefixSum.push_back(0);
        for (int num : nums)
            prefixSum.push_back(prefixSum.back() + num);
    }
    int sumRange(int left, int right) {
        return prefixSum[right + 1] - prefixSum[left];
    }
};
```

---
## 5. Implement Stack using Queues

**Approach:**
- Use two queues to simulate stack behavior.

**Time Complexity:** O(1) for push, O(n) for pop

```cpp
class MyStack {
    queue<int> q;
public:
    void push(int x) {
        q.push(x);
        for (int i = 0; i < q.size() - 1; ++i) {
            q.push(q.front());
            q.pop();
        }
    }
    int pop() {
        int top = q.front();
        q.pop();
        return top;
    }
    int top() { return q.front(); }
    bool empty() { return q.empty(); }
};
```

---
## 6. Next Greater Element I

**Approach:**
- Use a stack to maintain the next greater element.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> nge;
        stack<int> s;
        for (int num : nums2) {
            while (!s.empty() && s.top() < num) {
                nge[s.top()] = num;
                s.pop();
            }
            s.push(num);
        }
        vector<int> res;
        for (int num : nums1)
            res.push_back(nge.count(num) ? nge[num] : -1);
        return res;
    }
};
```

---
## 7. Sum of Root to Leaf Binary Numbers

**Approach:**
- Use DFS to traverse the tree and calculate sum.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int sumRootToLeaf(TreeNode* root, int sum = 0) {
        if (!root) return 0;
        sum = sum * 2 + root->val;
        return root->left == root->right ? sum : sumRootToLeaf(root->left, sum) + sumRootToLeaf(root->right, sum);
    }
};
```

---
## 8. Set Mismatch

**Approach:**
- Use frequency count to find missing and duplicate numbers.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    vector<int> findErrorNums(vector<int>& nums) {
        vector<int> count(nums.size() + 1, 0);
        for (int num : nums) count[num]++;
        int dup = -1, missing = -1;
        for (int i = 1; i < count.size(); i++) {
            if (count[i] == 2) dup = i;
            if (count[i] == 0) missing = i;
        }
        return {dup, missing};
    }
};
```

---
## 9. Rotate String

**Approach:**
- Check if `goal` is a substring of `s + s`.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool rotateString(string s, string goal) {
        return s.size() == goal.size() && (s + s).find(goal) != string::npos;
    }
};
```

---
## 10. Unique Email Addresses

**Approach:**
- Normalize emails and use a set to count unique ones.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int numUniqueEmails(vector<string>& emails) {
        unordered_set<string> uniqueEmails;
        for (string email : emails) {
            string local, domain;
            int i = 0;
            while (i < email.size() && email[i] != '@' && email[i] != '+') {
                if (email[i] != '.') local += email[i];
                i++;
            }
            while (email[i] != '@') i++;
            domain = email.substr(i);
            uniqueEmails.insert(local + domain);
        }
        return uniqueEmails.size();
    }
};
```