# Count Primes - Optimized C++ Solution

## Problem Description
LeetCode Problem: [Count Primes](https://leetcode.com/problems/count-primes/description/)

Given an integer `n`, return the number of prime numbers that are strictly less than `n`.

---

## Optimized Solution
The most efficient solution for this problem is to use the **Sieve of Eratosthenes** algorithm. Below is the C++ implementation:

### Code
```cpp
class Solution {
public:
    int countPrimes(int n) {
        if (n <= 2) return 0; // No primes below 2
        
        vector<bool> isPrime(n, true);
        isPrime[0] = isPrime[1] = false; // 0 and 1 are not primes
        
        for (int i = 2; i * i < n; ++i) {
            if (isPrime[i]) {
                for (int j = i * i; j < n; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        
        return count(isPrime.begin(), isPrime.end(), true);
    }
};
```

---

## Explanation
### Key Steps:
1. **Edge Case Handling**:
   - If `n <= 2`, return `0` because there are no primes below `2`.
2. **Initialization**:
   - Use a `vector<bool>` called `isPrime` to track whether each number less than `n` is prime.
   - Set `isPrime[0]` and `isPrime[1]` to `false` since `0` and `1` are not primes.
3. **Sieve of Eratosthenes**:
   - Iterate through numbers starting from `2`.
   - For each number `i`, if it is prime (`isPrime[i] == true`), mark all multiples of `i` starting from `i * i` as `false`.
   - Stop iterating when `i * i >= n` as smaller factors would have already been marked.
4. **Count Primes**:
   - Use `std::count` from the STL to count the number of `true` values in `isPrime`.

---

## Complexity
- **Time Complexity**: ⩽ \(O(n \log \log n)\)
  - This is the time complexity of the Sieve of Eratosthenes.
- **Space Complexity**: \(O(n)\)
  - Due to the storage of the `isPrime` vector.

---

## Example Usage
```cpp
#include <iostream>
using namespace std;

int main() {
    Solution solution;
    cout << solution.countPrimes(10) << endl; // Output: 4 (Primes: 2, 3, 5, 7)
    cout << solution.countPrimes(0) << endl;  // Output: 0
    cout << solution.countPrimes(1) << endl;  // Output: 0
    return 0;
}
```

---

## Advantages of This Solution
- **Efficient for Large `n`**: The Sieve of Eratosthenes algorithm ensures optimal performance for counting primes up to large values of `n`.
- **Clean and Compact**: The code is concise and leverages STL functions like `std::count` for simplicity.

---

## Additional Notes
- This solution is well-suited for competitive programming due to its efficiency.
- Make sure the input size `n` fits within memory constraints, as the space complexity is \(O(n)\).
