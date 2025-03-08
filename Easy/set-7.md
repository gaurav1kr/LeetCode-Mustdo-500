# Optimized C++ Solutions for LeetCode Problems

## 1. Flood Fill
**Approach:** Use DFS or BFS to traverse and fill connected pixels.
**Time Complexity:** O(m * n)

```cpp
class Solution {
public:
    void dfs(vector<vector<int>>& image, int sr, int sc, int color, int newColor) {
        if (sr < 0 || sc < 0 || sr >= image.size() || sc >= image[0].size() || image[sr][sc] != color) return;
        image[sr][sc] = newColor;
        dfs(image, sr + 1, sc, color, newColor);
        dfs(image, sr - 1, sc, color, newColor);
        dfs(image, sr, sc + 1, color, newColor);
        dfs(image, sr, sc - 1, color, newColor);
    }
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        if (image[sr][sc] != newColor) dfs(image, sr, sc, image[sr][sc], newColor);
        return image;
    }
};
```

## 2. Two Sum IV - Input is a BST
**Approach:** Use inorder traversal with a hash set to check for sum.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool findTarget(TreeNode* root, int k) {
        unordered_set<int> s;
        return inorder(root, k, s);
    }
    bool inorder(TreeNode* node, int k, unordered_set<int>& s) {
        if (!node) return false;
        if (s.count(k - node->val)) return true;
        s.insert(node->val);
        return inorder(node->left, k, s) || inorder(node->right, k, s);
    }
};
```

## 3. Sqrt(x)
**Approach:** Use binary search to find the square root.
**Time Complexity:** O(log x)

```cpp
class Solution {
public:
    int mySqrt(int x) {
        if (x == 0) return 0;
        int left = 1, right = x, ans;
        while (left <= right) {
            long mid = left + (right - left) / 2;
            if (mid * mid <= x) {
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;
    }
};
```

## 4. Isomorphic Strings
**Approach:** Use two maps to track character mappings.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        unordered_map<char, char> s_map, t_map;
        for (int i = 0; i < s.size(); i++) {
            if (s_map[s[i]] && s_map[s[i]] != t[i]) return false;
            if (t_map[t[i]] && t_map[t[i]] != s[i]) return false;
            s_map[s[i]] = t[i];
            t_map[t[i]] = s[i];
        }
        return true;
    }
};
```

## 5. Binary Search
**Approach:** Classic binary search.
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
};
```

## 6. Repeated Substring Pattern
**Approach:** Check string concatenation without the first and last character.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        string doubled = s + s;
        return doubled.substr(1, doubled.size() - 2).find(s) != string::npos;
    }
};
```

## 7. Remove Element
**Approach:** Use two pointers.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int j = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != val) nums[j++] = nums[i];
        }
        return j;
    }
};
```

## 8. Sum of Left Leaves
**Approach:** Use recursion to sum left leaves.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        if (!root) return 0;
        int sum = 0;
        if (root->left && !root->left->left && !root->left->right) sum += root->left->val;
        return sum + sumOfLeftLeaves(root->left) + sumOfLeftLeaves(root->right);
    }
};
```

## 9. Hamming Distance
**Approach:** Use XOR and count bits.
**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int hammingDistance(int x, int y) {
        int xor_val = x ^ y, count = 0;
        while (xor_val) {
            count += xor_val & 1;
            xor_val >>= 1;
        }
        return count;
    }
};
```

## 10. Valid Palindrome
**Approach:** Two pointers, ignoring non-alphanumeric characters.
**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isPalindrome(string s) {
        int left = 0, right = s.size() - 1;
        while (left < right) {
            while (left < right && !isalnum(s[left])) left++;
            while (left < right && !isalnum(s[right])) right--;
            if (tolower(s[left]) != tolower(s[right])) return false;
            left++, right--;
        }
        return true;
    }
};
```

---

Each solution is optimized for time and space efficiency. Let me know if you need any modifications!
