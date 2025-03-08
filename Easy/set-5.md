# LeetCode Solutions in C++

## 1. Island Perimeter
**Approach:**  
Iterate through the grid and count the number of land cells. For each land cell, add 4 to the perimeter and subtract 2 for each adjacent land cell.

**Time Complexity:** O(m * n)

```cpp
class Solution {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int perimeter = 0;
        int rows = grid.size(), cols = grid[0].size();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    perimeter += 4;
                    if (i > 0 && grid[i-1][j] == 1) perimeter -= 2;
                    if (j > 0 && grid[i][j-1] == 1) perimeter -= 2;
                }
            }
        }
        return perimeter;
    }
};
```

## 2. Remove Duplicates from Sorted List
**Approach:**  
Use a single pass through the linked list to remove consecutive duplicate elements.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* current = head;
        while (current && current->next) {
            if (current->val == current->next->val) {
                current->next = current->next->next;
            } else {
                current = current->next;
            }
        }
        return head;
    }
};
```

## 3. Add Binary
**Approach:**  
Traverse both strings from end to start, add digits with carry and construct the result.

**Time Complexity:** O(max(n, m))

```cpp
class Solution {
public:
    string addBinary(string a, string b) {
        string result = "";
        int carry = 0, i = a.size() - 1, j = b.size() - 1;
        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) sum += a[i--] - '0';
            if (j >= 0) sum += b[j--] - '0';
            result += (sum % 2) + '0';
            carry = sum / 2;
        }
        reverse(result.begin(), result.end());
        return result;
    }
};
```

## 4. Valid Anagram
**Approach:**  
Count the frequency of characters in both strings and compare.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        if (s.size() != t.size()) return false;
        vector<int> count(26, 0);
        for (char c : s) count[c - 'a']++;
        for (char c : t) {
            if (--count[c - 'a'] < 0) return false;
        }
        return true;
    }
};
```

## 5. First Bad Version
**Approach:**  
Use binary search to minimize API calls.

**Time Complexity:** O(log n)

```cpp
class Solution {
public:
    int firstBadVersion(int n) {
        int left = 1, right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (isBadVersion(mid)) right = mid;
            else left = mid + 1;
        }
        return left;
    }
};
```

## 6. Valid Palindrome II
**Approach:**  
Use two pointers and allow at most one deletion.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    bool isPalindrome(string s, int left, int right) {
        while (left < right) {
            if (s[left++] != s[right--]) return false;
        }
        return true;
    }
    
    bool validPalindrome(string s) {
        int left = 0, right = s.size() - 1;
        while (left < right) {
            if (s[left] != s[right]) {
                return isPalindrome(s, left + 1, right) || isPalindrome(s, left, right - 1);
            }
            left++; right--;
        }
        return true;
    }
};
```

## 7. Delete Node in a Linked List
**Approach:**  
Copy the next node's value and delete it.

**Time Complexity:** O(1)

```cpp
class Solution {
public:
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }
};
```

## 8. Intersection of Two Arrays II
**Approach:**  
Use a hashmap to count occurrences in one array and match with the other.

**Time Complexity:** O(n + m)

```cpp
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> count;
        vector<int> result;
        for (int num : nums1) count[num]++;
        for (int num : nums2) {
            if (count[num] > 0) {
                result.push_back(num);
                count[num]--;
            }
        }
        return result;
    }
};
```

## 9. Reverse String
**Approach:**  
Use two-pointer technique.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    void reverseString(vector<char>& s) {
        int left = 0, right = s.size() - 1;
        while (left < right) {
            swap(s[left++], s[right--]);
        }
    }
};
```

## 10. Range Sum of BST
**Approach:**  
Use DFS to sum nodes within range.

**Time Complexity:** O(n)

```cpp
class Solution {
public:
    int rangeSumBST(TreeNode* root, int low, int high) {
        if (!root) return 0;
        int sum = 0;
        if (root->val >= low && root->val <= high) sum += root->val;
        if (root->val > low) sum += rangeSumBST(root->left, low, high);
        if (root->val < high) sum += rangeSumBST(root->right, low, high);
        return sum;
    }
};
```