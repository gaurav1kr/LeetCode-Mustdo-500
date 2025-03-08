# LeetCode Solutions in C++

## 1. Add Strings ([Problem Link](https://leetcode.com/problems/add-strings))

### Approach:
- Use two pointers to traverse the strings from the end.
- Perform addition digit by digit, keeping track of carry.
- Append the result to a string and reverse it.

### Time Complexity: O(max(N, M))

```cpp
class Solution {
public:
    string addStrings(string num1, string num2) {
        int i = num1.size() - 1, j = num2.size() - 1, carry = 0;
        string result = "";
        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) sum += num1[i--] - '0';
            if (j >= 0) sum += num2[j--] - '0';
            result += (sum % 10) + '0';
            carry = sum / 10;
        }
        reverse(result.begin(), result.end());
        return result;
    }
};
```

## 2. Implement Queue using Stacks ([Problem Link](https://leetcode.com/problems/implement-queue-using-stacks))

### Approach:
- Use two stacks: one for enqueue and one for dequeue.
- Transfer elements from the input stack to the output stack when needed.

### Time Complexity: O(1) amortized per operation

```cpp
class MyQueue {
    stack<int> input, output;
public:
    void push(int x) {
        input.push(x);
    }
    int pop() {
        peek();
        int val = output.top(); output.pop();
        return val;
    }
    int peek() {
        if (output.empty()) {
            while (!input.empty()) {
                output.push(input.top());
                input.pop();
            }
        }
        return output.top();
    }
    bool empty() {
        return input.empty() && output.empty();
    }
};
```

## 3. Roman to Integer ([Problem Link](https://leetcode.com/problems/roman-to-integer))

### Approach:
- Use a hashmap to store values.
- Traverse from left to right, subtracting when a smaller value precedes a larger one.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    int romanToInt(string s) {
        unordered_map<char, int> values = {{'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}};
        int result = 0, prev = 0;
        for (char c : s) {
            int curr = values[c];
            result += (curr > prev) ? (curr - 2 * prev) : curr;
            prev = curr;
        }
        return result;
    }
};
```

## 4. Find the Town Judge ([Problem Link](https://leetcode.com/problems/find-the-town-judge))

### Approach:
- Use an array to track trust scores.
- Judge should have N-1 trust and trust no one.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    int findJudge(int n, vector<vector<int>>& trust) {
        vector<int> trustScore(n + 1, 0);
        for (auto& t : trust) {
            trustScore[t[0]]--;
            trustScore[t[1]]++;
        }
        for (int i = 1; i <= n; ++i) {
            if (trustScore[i] == n - 1) return i;
        }
        return -1;
    }
};
```

## 5. Merge Sorted Array ([Problem Link](https://leetcode.com/problems/merge-sorted-array))

### Approach:
- Merge from the end to avoid extra space.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while (j >= 0) {
            nums1[k--] = (i >= 0 && nums1[i] > nums2[j]) ? nums1[i--] : nums2[j--];
        }
    }
};
```

## 6. Remove All Adjacent Duplicates in String ([Problem Link](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string))

### Approach:
- Use a stack-based approach.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    string removeDuplicates(string s) {
        string result;
        for (char c : s) {
            if (!result.empty() && result.back() == c)
                result.pop_back();
            else
                result.push_back(c);
        }
        return result;
    }
};
```

## 7. Average of Levels in Binary Tree ([Problem Link](https://leetcode.com/problems/average-of-levels-in-binary-tree))

### Approach:
- Use BFS to compute level-wise sum.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    vector<double> averageOfLevels(TreeNode* root) {
        vector<double> result;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            long sum = 0, count = q.size();
            for (int i = 0; i < count; i++) {
                TreeNode* node = q.front(); q.pop();
                sum += node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            result.push_back((double)sum / count);
        }
        return result;
    }
};
```

## 8. Find Pivot Index ([Problem Link](https://leetcode.com/problems/find-pivot-index))

### Approach:
- Calculate prefix sums and compare.

### Time Complexity: O(N)

```cpp
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        int totalSum = accumulate(nums.begin(), nums.end(), 0), leftSum = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (leftSum == totalSum - leftSum - nums[i]) return i;
            leftSum += nums[i];
        }
        return -1;
    }
};
```

## 9. How Many Numbers Are Smaller Than the Current Number ([Problem Link](https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number))

### Time Complexity: O(N)

```cpp
class Solution {
public:
    vector<int> smallerNumbersThanCurrent(vector<int>& nums) {
        vector<int> count(101, 0);
        for (int num : nums) count[num]++;
        for (int i = 1; i < 101; i++) count[i] += count[i - 1];
        vector<int> result;
        for (int num : nums) result.push_back(num ? count[num - 1] : 0);
        return result;
    }
};
```

## 10. Power of Two ([Problem Link](https://leetcode.com/problems/power-of-two))

### Time Complexity: O(1)

```cpp
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
};
```