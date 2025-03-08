class Solution 
{
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) 
    {
            size_t size_n1 = nums1.size() ;
			size_t size_n2 = nums2.size() ;
			int total = size_n1 + size_n2 ;
			vector<int> pair(2,0) ;
			int mid_n = total/2 + 1 ;
			int ptr1 = 0 ;
			int ptr2 = 0 ;
			
			while(ptr1+ ptr2 != mid_n)
			{
				pair[0] = pair[1] ;
				if( (ptr1 != size_n1) && (ptr2 != size_n2) )
				{
					pair[1] = (nums1[ptr1] < nums2[ptr2])?nums1[ptr1++]:nums2[ptr2++] ;
				}
				else
				{
					if(ptr1 != size_n1)
					{
						pair[1] = nums1[ptr1++] ;
					}
					else
					{
						pair[1] = nums2[ptr2++] ;
					}
				}
			}
			if(total & 1)
			{
				return pair[1] ;
			}
			return (double)(pair[0] + pair[1])/2.0 ;
    }
};
/*
size_n1 and size_n2 store the sizes of nums1 and nums2.

total stores the total number of elements in both arrays.

pair is a vector of size 2 initialized with zeros. It will store the two elements needed to calculate the median.

mid_n stores the index of the middle element (or the middle two elements if total is even).

ptr1 and ptr2 are pointers to track the current positions in nums1 and nums2, respectively.

This while loop iterates until ptr1 + ptr2 reaches the index of the middle element(s).

At each iteration, it updates pair[0] with the previous value of pair[1] and then updates pair[1] with the smaller of the current elements pointed by ptr1 and ptr2.

The pointers ptr1 and ptr2 are incremented accordingly.

If the total number of elements is odd (total & 1 checks for oddness), the median is the second element in pair.

If the total number of elements is even, the median is the average of the two elements in pair.
*/
