# Optimized C++ Solutions for LeetCode Problems

## 1. Binary Tree Postorder Traversal
### Approach
We use an iterative approach with a stack to efficiently perform postorder traversal (Left, Right, Root).

### Time Complexity
**O(n)** - We visit each node exactly once.

### C++ Code
```cpp
#include <vector>
#include <stack>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

vector<int> postorderTraversal(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    stack<TreeNode*> s;
    TreeNode* last = nullptr;
    while (!s.empty() || root) {
        if (root) {
            s.push(root);
            root = root->left;
        } else {
            TreeNode* node = s.top();
            if (node->right && last != node->right)
                root = node->right;
            else {
                result.push_back(node->val);
                last = node;
                s.pop();
            }
        }
    }
    return result;
}
```

---
## 2. Is Subsequence
### Approach
Use two pointers to check if `s` is a subsequence of `t`.

### Time Complexity
**O(n)** - We traverse `t` at most once.

### C++ Code
```cpp
#include <string>
using namespace std;

bool isSubsequence(string s, string t) {
    int i = 0, j = 0;
    while (i < s.size() && j < t.size()) {
        if (s[i] == t[j]) i++;
        j++;
    }
    return i == s.size();
}
```

---
## 3. Binary Tree Paths
### Approach
Use DFS with backtracking to generate all root-to-leaf paths.

### Time Complexity
**O(n)** - We visit each node once.

### C++ Code
```cpp
#include <vector>
#include <string>
using namespace std;

void dfs(TreeNode* root, string path, vector<string>& paths) {
    if (!root) return;
    path += to_string(root->val);
    if (!root->left && !root->right) {
        paths.push_back(path);
        return;
    }
    path += "->";
    dfs(root->left, path, paths);
    dfs(root->right, path, paths);
}

vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> paths;
    dfs(root, "", paths);
    return paths;
}
```

---
## 4. Minimum Depth of Binary Tree
### Approach
Use BFS to find the first leaf node.

### Time Complexity
**O(n)** - We visit each node at most once.

### C++ Code
```cpp
#include <queue>
using namespace std;

int minDepth(TreeNode* root) {
    if (!root) return 0;
    queue<pair<TreeNode*, int>> q;
    q.push({root, 1});
    while (!q.empty()) {
        auto [node, depth] = q.front(); q.pop();
        if (!node->left && !node->right) return depth;
        if (node->left) q.push({node->left, depth + 1});
        if (node->right) q.push({node->right, depth + 1});
    }
    return 0;
}
```

---
## 5. Binary Tree Preorder Traversal
### Approach
Use an iterative approach with a stack.

### Time Complexity
**O(n)**

### C++ Code
```cpp
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    stack<TreeNode*> s;
    s.push(root);
    while (!s.empty()) {
        TreeNode* node = s.top(); s.pop();
        result.push_back(node->val);
        if (node->right) s.push(node->right);
        if (node->left) s.push(node->left);
    }
    return result;
}
```

---
## 6. Plus One
### Approach
Simulate addition from the least significant digit.

### Time Complexity
**O(n)**

### C++ Code
```cpp
#include <vector>
using namespace std;

vector<int> plusOne(vector<int>& digits) {
    int n = digits.size();
    for (int i = n - 1; i >= 0; i--) {
        if (digits[i] < 9) {
            digits[i]++;
            return digits;
        }
        digits[i] = 0;
    }
    digits.insert(digits.begin(), 1);
    return digits;
}
```

---
## 7. Backspace String Compare
### Approach
Use a stack to simulate typing.

### Time Complexity
**O(n)**

### C++ Code
```cpp
#include <string>
using namespace std;

string buildString(string s) {
    string result;
    for (char c : s) {
        if (c == '#' && !result.empty()) result.pop_back();
        else if (c != '#') result.push_back(c);
    }
    return result;
}

bool backspaceCompare(string s, string t) {
    return buildString(s) == buildString(t);
}
```

---
## 8. Implement strStr()
### Approach
Use the KMP algorithm for efficient substring search.

### Time Complexity
**O(n + m)**

### C++ Code
```cpp
#include <vector>
using namespace std;

int strStr(string haystack, string needle) {
    return haystack.find(needle);
}
```

---
## 9. Contains Duplicate
### Approach
Use an unordered set to check for duplicates.

### Time Complexity
**O(n)**

### C++ Code
```cpp
#include <unordered_set>
using namespace std;

bool containsDuplicate(vector<int>& nums) {
    unordered_set<int> s;
    for (int num : nums) {
        if (s.count(num)) return true;
        s.insert(num);
    }
    return false;
}
```

---
## 10. Jewels and Stones
### Approach
Use an unordered set for quick lookups.

### Time Complexity
**O(n)**

### C++ Code
```cpp
#include <unordered_set>
using namespace std;

int numJewelsInStones(string jewels, string stones) {
    unordered_set<char> jewelSet(jewels.begin(), jewels.end());
    int count = 0;
    for (char stone : stones) {
        if (jewelSet.count(stone)) count++;
    }
    return count;
}
```
