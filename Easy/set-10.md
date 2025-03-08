# LeetCode Solutions in C++

This document provides optimized C++ solutions for a set of LeetCode problems. Each problem includes an approach explanation, time complexity analysis, and the corresponding C++ code.

## 1. Intersection of Two Arrays

**Approach:** Use an unordered set for quick lookup and another set to store the intersection.

**Time Complexity:** O(m + n)

```cpp
#include <vector>
#include <unordered_set>
using namespace std;

vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
    unordered_set<int> set1(nums1.begin(), nums1.end()), result;
    for (int num : nums2) {
        if (set1.count(num)) result.insert(num);
    }
    return vector<int>(result.begin(), result.end());
}
```

---

## 2. Longest Palindrome

**Approach:** Use an unordered map to count character frequencies.

**Time Complexity:** O(n)

```cpp
#include <string>
#include <unordered_map>
using namespace std;

int longestPalindrome(string s) {
    unordered_map<char, int> count;
    for (char c : s) count[c]++;
    int length = 0, odd = 0;
    for (auto& p : count) {
        length += p.second / 2 * 2;
        if (p.second % 2) odd = 1;
    }
    return length + odd;
}
```

---

## 3. Search in a Binary Search Tree

**Approach:** Recursively traverse the BST to find the target value.

**Time Complexity:** O(h) (where h is the height of the tree)

```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* searchBST(TreeNode* root, int val) {
    if (!root || root->val == val) return root;
    return val < root->val ? searchBST(root->left, val) : searchBST(root->right, val);
}
```

---

## 4. Reverse Words in a String III

**Approach:** Reverse each word in place.

**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

string reverseWords(string s) {
    int i = 0, n = s.size();
    for (int j = 0; j <= n; j++) {
        if (j == n || s[j] == ' ') {
            reverse(s.begin() + i, s.begin() + j);
            i = j + 1;
        }
    }
    return s;
}
```

---

## 5. Excel Sheet Column Number

**Approach:** Convert from base-26 notation to decimal.

**Time Complexity:** O(n)

```cpp
#include <string>
using namespace std;

int titleToNumber(string columnTitle) {
    int result = 0;
    for (char c : columnTitle) {
        result = result * 26 + (c - 'A' + 1);
    }
    return result;
}
```

---

## 6. Design HashMap

**Approach:** Use an array of linked lists for collision handling.

**Time Complexity:** O(1) (average case)

```cpp
#include <vector>
using namespace std;

class MyHashMap {
private:
    vector<int> map;
public:
    MyHashMap() : map(1e6 + 1, -1) {}
    void put(int key, int value) { map[key] = value; }
    int get(int key) { return map[key]; }
    void remove(int key) { map[key] = -1; }
};
```

---

## 7. Convert Binary Number in a Linked List to Integer

**Approach:** Use bitwise operations to compute the integer value.

**Time Complexity:** O(n)

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

int getDecimalValue(ListNode* head) {
    int num = 0;
    while (head) {
        num = (num << 1) | head->val;
        head = head->next;
    }
    return num;
}
```

---

## 8. Max Consecutive Ones

**Approach:** Iterate through the array while keeping track of max consecutive ones.

**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

int findMaxConsecutiveOnes(vector<int>& nums) {
    int maxCount = 0, count = 0;
    for (int num : nums) {
        count = (num == 1) ? count + 1 : 0;
        maxCount = max(maxCount, count);
    }
    return maxCount;
}
```

---

## 9. Running Sum of 1D Array

**Approach:** Maintain a running sum while iterating through the array.

**Time Complexity:** O(n)

```cpp
#include <vector>
using namespace std;

vector<int> runningSum(vector<int>& nums) {
    for (int i = 1; i < nums.size(); i++) {
        nums[i] += nums[i - 1];
    }
    return nums;
}
```

---

## 10. Pascal's Triangle II

**Approach:** Use dynamic programming with a single vector.

**Time Complexity:** O(k^2)

```cpp
#include <vector>
using namespace std;

vector<int> getRow(int rowIndex) {
    vector<int> row(rowIndex + 1, 1);
    for (int i = 1; i < rowIndex; i++) {
        for (int j = i; j > 0; j--) {
            row[j] += row[j - 1];
        }
    }
    return row;
}
```
