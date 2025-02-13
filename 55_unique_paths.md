```cpp
#include <iostream>

class Solution {
public:
    int uniquePaths(int m, int n) {
        long long res = 1;
        int N = m + n - 2; // Total steps (right + down)
        int k = std::min(m - 1, n - 1); // Choose the smaller set (right or down)
        
        for (int i = 1; i <= k; i++) {
            res = res * (N - i + 1) / i; // Compute binomial coefficient (N choose k)
        }
        
        return res;
    }
};

int main() {
    Solution sol;
    int m = 3, n = 7;
    std::cout << "Unique Paths: " << sol.uniquePaths(m, n) << std::endl;
    return 0;
}
```