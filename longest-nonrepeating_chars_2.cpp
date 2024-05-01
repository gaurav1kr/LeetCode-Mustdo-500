#define NO_OF_CHARS 130
class Solution {
public:
    int lengthOfLongestSubstring(string str) 
    {
        int n = str.size();
        if(n==0)
            return 0 ;
        int cur_len = 1; 
        int max_len = 1; 
        int prev_index; 

        int* visited = new int[NO_OF_CHARS];
        for (int i = 0; i < NO_OF_CHARS; i++)
            visited[i] = -1;

        visited[str[0]] = 0;
        for (int i = 1; i < n; i++)
        {
            prev_index = visited[str[i]];
            if (prev_index == -1 || i - cur_len > prev_index)
                cur_len++;
            else 
            { 
                if (cur_len > max_len)
                    max_len = cur_len;
                cur_len = i - prev_index;
            }
            visited[str[i]] = i;
        }
        if (cur_len > max_len)
            max_len = cur_len;
        delete []visited; 
        return max_len;    
    }
};

/* How exactly it is working :-
This C++ code defines a class called Solution with a member function lengthOfLongestSubstring, which takes a string str as input and returns an integer representing the length of the longest substring without repeating characters.

Here's a breakdown of what the code does:

1. Preprocessor Directive:
#define NO_OF_CHARS 130: This line defines a macro NO_OF_CHARS with the value 130. It seems like this macro is used to define the size of an array.

2.Class Definition:
class Solution { ... };: Defines a class named Solution which encapsulates the logic for finding the length of the longest substring without repeating characters.

3.Member Function lengthOfLongestSubstring:
Signature: int lengthOfLongestSubstring(string str)
Purpose: This function calculates the length of the longest substring in the input string str without repeating characters.

Steps:
Initialize variables:
n: Stores the length of the input string str.

cur_len: Stores the length of the current substring being checked.

max_len: Stores the length of the longest substring without repeating characters found so far.

prev_index: Stores the index of the previous occurrence of the current character.

visited: Dynamic array of integers used to keep track of the most recent index where each character was seen.

Loop through each character in the string:
Initialize the visited array with -1.
Update the visited array with the index of the current character.
Check if the current character has been visited before:
If not visited (prev_index == -1) or if the previous occurrence is not within the current substring (i - cur_len > prev_index), increment cur_len.
Otherwise, update max_len if necessary and reset cur_len to the length between the current index and the previous occurrence of the character.
Update the visited array with the current index.
Return max_len, which represents the length of the longest substring without repeating characters.
Memory Management:
delete []visited;: Deallocates the memory allocated for the visited array to prevent memory leaks.
*/
