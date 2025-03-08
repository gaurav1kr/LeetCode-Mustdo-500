class Solution 
{
public:
     int characterReplacement(string s, int k) 
	{
        unordered_map<char, int> alphabets;
        int ans = 0;
        int left = 0;
        int right = 0;
        int maxf = 0;

        for (right = 0; right < s.size(); right++) 
	{
            alphabets[s[right]] = 1 + alphabets[s[right]];
            maxf = max(maxf, alphabets[s[right]]);

            if ((right - left + 1) - maxf > k) 
	    {
                alphabets[s[left]] -= 1;
                left++;
            } 
	    else 
	    {
                ans = max(ans, (right - left + 1));
            }
        }

        return ans;
    }
};

//Approach - 
# **Longest Substring with At Most K Changes**

## **Intuition**  
The problem requires finding the length of the longest substring containing the same letter after at most `k` changes to any character in the string. To solve this efficiently, we use a **sliding window** approach to process the string and keep track of the frequency of characters within the window.

---

## **Approach**  

1. **Initialization**  
   - Use two pointers (`left` and `right`) to define a sliding window.  
   - Maintain an unordered map (or array) called `alphabets` to store the frequency of characters in the current window.

   **Reason:**  
   The sliding window approach efficiently processes a specific segment of the string without redundant computations, while the frequency map keeps track of character occurrences.

2. **Frequency Count and Maximum Frequency**  
   - As the `right` pointer expands the window, update the frequency of the characters encountered in the `alphabets` map.  
   - Keep track of the maximum frequency (`maxf`) of any character in the current window.  

   **Reason:**  
   Maintaining the frequency count and the maximum frequency allows us to efficiently identify the most frequent character in the current window.

3. **Window Length Check**  
   - Check if the length of the current window minus `maxf` exceeds `k`.  
   - Condition: `(right - left + 1) - maxf > k`.  

   **Reason:**  
   If this condition is met, it means the window is invalid as more than `k` changes are needed. Adjust the window to maintain validity.

4. **Adjusting the Window**  
   - If the condition in step 3 is met, move the `left` pointer to shrink the window and update the frequency count accordingly until the condition is satisfied.  

   **Reason:**  
   Adjusting the window ensures that at most `k` changes are allowed, keeping the window valid.

5. **Update Maximum Length**  
   - During each iteration, update the result (`ans`) with the maximum length of the valid window:  
     `ans = max(ans, right - left + 1)`.  

   **Reason:**  
   This keeps track of the maximum length of any valid window seen so far.

---

## **Complexity**  

- **Time Complexity:** O(n)  
  - Each character is processed once during the sliding window traversal.  

- **Space Complexity:** O(1)  
  - The `alphabets` map has at most 26 entries (uppercase English letters), making the space usage constant.

---

This approach ensures an efficient and clear solution to the problem while adhering to the constraints of `k` changes.

