class Solution 
{
public:
    int CeilIndex(std::vector<int>& v, int l, int r, int key) 
    { 
        while (r - l > 1) 
        { 
            int m = l + (r - l) / 2; 
            if (v[m] >= key) 
                r = m; 
            else
                l = m; 
        } 

        return r; 
    } 
    int lengthOfLIS(vector<int>& v) 
    {
        if (v.size() == 0) 
        return 0; 
  
        std::vector<int> tail(v.size(), 0); 
        int length = 1; // always points empty slot in tail 

        tail[0] = v[0]; 
        for (size_t i = 1; i < v.size(); i++) 
        { 
            if (v[i] < tail[0]) 
                tail[0] = v[i]; 

            else if (v[i] > tail[length - 1]) 
                tail[length++] = v[i]; 

            else
                tail[CeilIndex(tail, -1, length - 1, v[i])] = v[i]; 
        } 

        return length; 
    }
};

//Approach explanation :- 

This code implement the "Longest Increasing Subsequence" problem using the patience sorting algorithm. The problem asks for finding the length of the longest subsequence in a given array of integers such that all elements of the subsequence are sorted in increasing order.

Here's how the code works:

The CeilIndex function finds the index of the smallest element in v that is greater than or equal to the given key using binary search. It maintains a window [l, r] where l is the lower bound and r is the upper bound of the search space.

The lengthOfLIS function initializes a vector tail to store the last element of each increasing subsequence found so far. The length variable keeps track of the length of the current longest increasing subsequence.

It iterates through the elements of the input vector v.

For each element v[i]:

If v[i] is smaller than the smallest element in tail, it updates the smallest element in tail to v[i].
If v[i] is greater than the largest element in tail, it appends v[i] to tail and increments length.
If v[i] is in between the elements in tail, it finds the index where v[i] can be inserted in tail using CeilIndex and updates tail.
Finally, it returns the length of the longest increasing subsequence stored in length.
