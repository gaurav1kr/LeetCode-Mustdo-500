class Solution 
{
public:
  string frequencySort(string str) 
  {
    vector<int> freqs(256);
    for (const char& ch : str) ++freqs[ch];

    vector<vector<char>> bucket(str.length() + 1);
    for (int i = 255; i >= 0; --i)
      if (freqs[i]) 
        bucket[freqs[i]].push_back(static_cast<char>(i));

    string sorted;

    for (int freq = str.length(); freq >= 0; --freq) 
      for (const char& ch : bucket[freq])
        sorted += string(freq, ch);

    return sorted;
  }
};

//Explanation
```
The idea in this solution is to avoid the logarithmic time complexity that sorting/heaps bring by using an important fact: we know the maximum and minimum number of times a character an occur in a string. The maximum being n times (where n is the length of the string) and 1 being the minimum.

This fact helps, since if we know the upper and lower limit for a character's frequency, then we can create a container of size n to store a characteer with a frequency i in the ith index.

Once we have every character's frequency stored in an array from 0..n, then creating a string sorted by frequency in decreasing-order is trivial by just looping backwards through the array.

This technique is known as bucket sorting, which is a variant of count sorting.

Time Complexity
Counting the frequency of each character contributes O(n) time complexity, since we're looping through the entire string.

Storing each characters' frequency in the bucket array contributes a constant O(256) time complexity.

Adding character frequencies in descending order takes O(n) time complexity, since we're just looping backwards in the bucket array which is of size n + 1.

Overall time complexity is O(n)

Space Complexity
The frequency array has a fixed size of 256, which is independent of the input size and thus contributes O(1) space.

The bucket vector has a length equal to the input string length n + 1. Each bucket potentially holds multiple characters, but the total number of character frequencies across all buckets cannot exceed n. Thus, the space complexity for the bucket vector is O(n).

The output string sorted will have the same length as the input string str, also contributing O(n) space.

Overall space complexity is O(n)
```
