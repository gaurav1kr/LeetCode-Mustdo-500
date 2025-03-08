# Optimized C++ Solutions for LeetCode Problems

## 1. Increasing Order Search Tree
**Approach:** Perform an in-order traversal and store nodes in a new tree.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    TreeNode* increasingBST(TreeNode* root, TreeNode* tail = nullptr) {
        if (!root) return tail;
        TreeNode* res = increasingBST(root->left, root);
        root->left = nullptr;
        root->right = increasingBST(root->right, tail);
        return res;
    }
};
```

## 2. Find Common Characters
**Approach:** Use frequency arrays to track character counts.
**Time Complexity:** O(N*M)
```cpp
class Solution {
public:
    vector<string> commonChars(vector<string>& words) {
        vector<int> common(26, INT_MAX);
        for (const string& word : words) {
            vector<int> freq(26, 0);
            for (char c : word) freq[c - 'a']++;
            for (int i = 0; i < 26; ++i)
                common[i] = min(common[i], freq[i]);
        }
        vector<string> result;
        for (int i = 0; i < 26; ++i)
            while (common[i]-- > 0)
                result.push_back(string(1, 'a' + i));
        return result;
    }
};
```

## 3. Shortest Distance to a Character
**Approach:** Use two passes (left-to-right and right-to-left) to track distances.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    vector<int> shortestToChar(string s, char c) {
        int n = s.size();
        vector<int> res(n, n);
        int pos = -n;
        for (int i = 0; i < n; ++i) {
            if (s[i] == c) pos = i;
            res[i] = i - pos;
        }
        for (int i = n - 1; i >= 0; --i) {
            if (s[i] == c) pos = i;
            res[i] = min(res[i], abs(i - pos));
        }
        return res;
    }
};
```

## 4. Can Place Flowers
**Approach:** Iterate and check adjacent spots for availability.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int count = 0;
        for (int i = 0; i < flowerbed.size(); ++i) {
            if (flowerbed[i] == 0 && (i == 0 || flowerbed[i - 1] == 0) && (i == flowerbed.size() - 1 || flowerbed[i + 1] == 0)) {
                flowerbed[i] = 1;
                count++;
                if (count >= n) return true;
            }
        }
        return count >= n;
    }
};
```

## 5. Contains Duplicate II
**Approach:** Use an unordered_map to track indices.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        for (int i = 0; i < nums.size(); ++i) {
            if (mp.count(nums[i]) && i - mp[nums[i]] <= k)
                return true;
            mp[nums[i]] = i;
        }
        return false;
    }
};
```

## 6. Number of Good Pairs
**Approach:** Use a frequency map to count occurrences.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    int numIdenticalPairs(vector<int>& nums) {
        unordered_map<int, int> freq;
        int count = 0;
        for (int num : nums) {
            count += freq[num];
            freq[num]++;
        }
        return count;
    }
};
```

## 7. Shuffle the Array
**Approach:** Reconstruct array using interleaving.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    vector<int> shuffle(vector<int>& nums, int n) {
        vector<int> res;
        for (int i = 0; i < n; ++i) {
            res.push_back(nums[i]);
            res.push_back(nums[i + n]);
        }
        return res;
    }
};
```

## 8. Kth Missing Positive Number
**Approach:** Use binary search for optimized lookup.
**Time Complexity:** O(log N)
```cpp
class Solution {
public:
    int findKthPositive(vector<int>& arr, int k) {
        int left = 0, right = arr.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] - (mid + 1) < k)
                left = mid + 1;
            else
                right = mid;
        }
        return left + k;
    }
};
```

## 9. Peak Index in a Mountain Array
**Approach:** Use binary search for efficient peak finding.
**Time Complexity:** O(log N)
```cpp
class Solution {
public:
    int peakIndexInMountainArray(vector<int>& arr) {
        int left = 0, right = arr.size() - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] < arr[mid + 1])
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }
};
```

## 10. Find Mode in Binary Search Tree
**Approach:** Perform in-order traversal and track frequency.
**Time Complexity:** O(N)
```cpp
class Solution {
public:
    vector<int> findMode(TreeNode* root) {
        unordered_map<int, int> freq;
        int maxFreq = 0;
        vector<int> result;
        function<void(TreeNode*)> inorder = [&](TreeNode* node) {
            if (!node) return;
            inorder(node->left);
            maxFreq = max(maxFreq, ++freq[node->val]);
            inorder(node->right);
        };
        inorder(root);
        for (const auto& p : freq)
            if (p.second == maxFreq)
                result.push_back(p.first);
        return result;
    }
};
```