class Solution 
{
public:
    string shortestPalindrome(string s) 
	{
        const int n = s.size();
        int i = 0;
        for (int j=n-1; j>= 0; j--) 
		{
            while (j>=0 && s[i] == s[j])
                i++, j--;
        }
        if (i==n) 
            return s;
        string sub= s.substr(i), remain_rev=sub;
        reverse(remain_rev.begin(), remain_rev.end());
        return remain_rev + shortestPalindrome(s.substr(0, i)) + sub;
    }
};

// The function aims to find the shortest palindrome by adding the fewest characters to the start of the input string s. To achieve this, the algorithm:

// Identifies the largest palindrome starting from the first character.
// Reverses the remaining substring and prepends it to the original string to form a palindrome.
// Detailed Steps:
// Initialization:

// The string length is stored in n.
// Two indices, i and j, are initialized. i starts from the beginning (index 0), while j starts from the end of the string (index n-1).
// Finding the Largest Palindrome from the Start:

// A for loop iterates from the end of the string (j = n - 1) to the start.
// Inside the loop, while the characters at indices i and j are equal (s[i] == s[j]), both i and j are decremented and incremented, respectively.
// The loop aims to find the longest palindromic prefix. When the characters differ, the loop exits.
// Base Case Check:

// If i reaches n (meaning the entire string is already a palindrome), the original string s is returned as no additional characters are needed.
// Handling Non-palindrome:

// If the string is not a palindrome, the remainder of the string (from index i to the end) is stored in sub.
// The reverse of this substring (remain_rev) is created and prepended to form the palindrome.
// Recursive Step:

// The function then calls itself recursively on the palindrome prefix (s.substr(0, i)) and adds the reversed substring before it and the remaining substring (sub) after it.
// Complexity:
// Time complexity: The time complexity of the algorithm is approximately 
// 𝑂(𝑛^2)
// This is because in the worst case, the recursive function could be called up to 
// n times (though less often in practice), and each call performs operations like reversing a substring and recursion on a smaller part of the string.

// Space complexity: The space complexity is 
// 𝑂(𝑛)
