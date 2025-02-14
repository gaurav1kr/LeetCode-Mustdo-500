## Manacher's Algorithm - Longest Palindromic Substring

### **Flow of the Code**
The given C++ code implements **Manacher’s Algorithm**, which efficiently finds the **longest palindromic substring** in \( O(n) \) time complexity.

---

### **1. Preprocessing (`preprocess` function)**
- Transforms the input string by inserting special characters (`#`) between each character and adding sentinels (`^` and `$` at the start and end).  
- This ensures that even-length palindromes are handled uniformly with odd-length palindromes.

For example, if `s = "babad"`, the transformed string becomes:  
**`^#b#a#b#a#d#$`**

---

### **2. Main Logic (`longestPalindrome` function)**
- Uses **Manacher’s Algorithm** to find the longest palindromic substring in linear time.
- Initializes:
  - `p[n]`: An array where `p[i]` stores the radius of the palindrome centered at index `i`.
  - `c, r`: The current palindrome's center and right boundary.
  - `maxLen, centerIndex`: Track the longest palindrome found.

- **Iterating through the transformed string (`t`)**
  - **Mirror Property:** If `i` is within the current boundary (`r`), set `p[i] = min(r - i, p[mirror])`, where `mirror = 2c - i`.
  - **Expand Around Center:** Attempt to expand the palindrome centered at `i`.
  - **Update Center & Right Boundary:** If `i + p[i] > r`, update `c = i` and `r = i + p[i]`.
  - **Track Maximum Palindrome:** Update `maxLen` and `centerIndex` whenever a longer palindrome is found.

- **Extract the longest palindromic substring**  
  - Since the transformed string has additional characters (`#`), we convert the `centerIndex` and `maxLen` back to the original string indices.

---

### **Time and Space Complexity**
#### **1. Time Complexity: \( O(n) \)**
- **Preprocessing:** \( O(n) \) (constructing the modified string)
- **Manacher’s Algorithm:** \( O(n) \) (each character is processed at most once)
- **Substring Extraction:** \( O(n) \) (finding the result)
- **Overall:** **\( O(n) \)**

#### **2. Space Complexity: \( O(n) \)**
- **Extra space for transformed string (`t`)** → \( O(n) \)
- **Extra array `p` of size \( O(n) \)** → \( O(n) \)
- **Overall:** \( O(n) \) (ignoring the output string)

---

### **Summary**
- **Efficient:** Runs in **linear time** (\( O(n) \)) compared to the brute-force approach (\( O(n^3) \)) and the dynamic programming approach (\( O(n^2) \)).
- **Handles odd and even-length palindromes uniformly** via transformation.
- **Space Complexity is also \( O(n) \)** due to the additional storage required.

This implementation of **Manacher's Algorithm** is one of the fastest ways to find the longest palindromic substring. 🚀
