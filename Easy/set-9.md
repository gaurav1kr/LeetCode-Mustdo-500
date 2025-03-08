# Optimized C++ Solutions for LeetCode Problems

## 1. Fibonacci Number
**Approach:** Use dynamic programming with memoization to achieve O(n) time complexity.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int fib(int n) {
        if (n <= 1) return n;
        int a = 0, b = 1, c;
        for (int i = 2; i <= n; ++i) {
            c = a + b;
            a = b;
            b = c;
        }
        return b;
    }
};
```

---

## 2. Reverse Bits
**Approach:** Iterate over 32 bits, shift and set them in reverse order.

**Time Complexity:** O(1)

```cpp
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t result = 0;
        for (int i = 0; i < 32; ++i) {
            result = (result << 1) | (n & 1);
            n >>= 1;
        }
        return result;
    }
};
```

---

## 3. Word Pattern
**Approach:** Use two hash maps to maintain mappings between words and characters.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool wordPattern(string pattern, string s) {
        unordered_map<char, string> pToS;
        unordered_map<string, char> sToP;
        istringstream in(s);
        string word;
        vector<string> words;
        while (in >> word) words.push_back(word);
        if (words.size() != pattern.size()) return false;
        
        for (int i = 0; i < pattern.size(); ++i) {
            if (pToS.count(pattern[i]) && pToS[pattern[i]] != words[i]) return false;
            if (sToP.count(words[i]) && sToP[words[i]] != pattern[i]) return false;
            pToS[pattern[i]] = words[i];
            sToP[words[i]] = pattern[i];
        }
        return true;
    }
};
```

---

## 4. Verifying an Alien Dictionary
**Approach:** Use a map to store the order of characters and compare words accordingly.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isAlienSorted(vector<string>& words, string order) {
        unordered_map<char, int> dict;
        for (int i = 0; i < order.size(); ++i) dict[order[i]] = i;
        for (int i = 0; i < words.size() - 1; ++i) {
            for (int j = 0; j < words[i].size(); ++j) {
                if (j >= words[i+1].size()) return false;
                if (dict[words[i][j]] < dict[words[i+1][j]]) break;
                if (dict[words[i][j]] > dict[words[i+1][j]]) return false;
            }
        }
        return true;
    }
};
```

---

## 5. Cousins in Binary Tree
**Approach:** Use BFS to find depth and parent of the given nodes.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isCousins(TreeNode* root, int x, int y) {
        queue<pair<TreeNode*, TreeNode*>> q;
        q.push({root, nullptr});
        while (!q.empty()) {
            int size = q.size();
            TreeNode *parentX = nullptr, *parentY = nullptr;
            for (int i = 0; i < size; ++i) {
                auto [node, parent] = q.front(); q.pop();
                if (node->val == x) parentX = parent;
                if (node->val == y) parentY = parent;
                if (node->left) q.push({node->left, node});
                if (node->right) q.push({node->right, node});
            }
            if (parentX && parentY) return parentX != parentY;
            if (parentX || parentY) return false;
        }
        return false;
    }
};
```

---

## 6. Count Binary Substrings
**Approach:** Use a single pass to count consecutive groups of 0s and 1s.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int countBinarySubstrings(string s) {
        int prev = 0, cur = 1, count = 0;
        for (int i = 1; i < s.size(); ++i) {
            if (s[i] == s[i - 1])
                cur++;
            else {
                count += min(prev, cur);
                prev = cur;
                cur = 1;
            }
        }
        return count + min(prev, cur);
    }
};
```

---

## 7. Sort Array by Parity
**Approach:** Use two-pointer technique.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    vector<int> sortArrayByParity(vector<int>& nums) {
        int i = 0, j = nums.size() - 1;
        while (i < j) {
            if (nums[i] % 2 > nums[j] % 2) swap(nums[i], nums[j]);
            if (nums[i] % 2 == 0) ++i;
            if (nums[j] % 2 == 1) --j;
        }
        return nums;
    }
};
```

---

## 8. Number of 1 Bits
**Approach:** Use bitwise AND and shift operations.

**Time Complexity:** O(1)

```cpp
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count = 0;
        while (n) {
            count += n & 1;
            n >>= 1;
        }
        return count;
    }
};
```

---

## 9. Maximum Product of Three Numbers
**Approach:** Find the top three largest and two smallest numbers.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int maximumProduct(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        return max(nums[0] * nums[1] * nums[n - 1],
                   nums[n - 1] * nums[n - 2] * nums[n - 3]);
    }
};
```

---

## 10. Excel Sheet Column Title
**Approach:** Convert number to base-26 format with character mapping.

**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    string convertToTitle(int columnNumber) {
        string result = "";
        while (columnNumber) {
            columnNumber--;
            result = char('A' + columnNumber % 26) + result;
            columnNumber /= 26;
        }
        return result;
    }
};
```
