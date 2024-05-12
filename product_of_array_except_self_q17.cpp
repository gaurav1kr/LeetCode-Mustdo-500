class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) 
    { 
        int n = nums.size() ;
        vector<int> int_vec(nums.size() , 1) ;
        if (n == 1) 
        {
            int_vec.push_back(0) ;
            return int_vec ;
        } 
  
        int i, temp = 1; 
        for (i = 0; i < n; i++) 
        { 
            int_vec[i] = temp; 
            temp *= nums[i]; 
        } 
        temp = 1; 
  
        for (i = n - 1; i >= 0; i--) 
        {    
            int_vec[i] *= temp; 
            temp *= nums[i]; 
        }  
       return int_vec ;  
    }                         
};
//Approach
It first calculates the product of all elements to the left of each element in the input array and stores it in a vector called int_vec.

Then, it calculates the product of all elements to the right of each element in the input array, multiplying it with the corresponding element in int_vec.

Finally, it returns int_vec, which contains the product of all elements except the current element for each position in the original array.
