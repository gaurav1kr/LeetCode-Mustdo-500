class Solution {
    vector<vector<int>> final_vec ;
public:
    void Get_permute(vector<int> a , int l , int r)
    {
        if(l==r)
        {
            final_vec.push_back(a) ;
        }
        else
        {
            // Permutations made  
            for (int i = l; i <= r; i++)  
            {  

                // Swapping done  
                swap(a[l], a[i]);    
                Get_permute(a, l+1, r);  
                swap(a[l], a[i]);  
            }  
        }  
    }
    vector<vector<int>> permute(vector<int>& nums) 
    {
        int lower = 0 ;
        int upper = nums.size() ;
        Get_permute(nums, lower, upper-1);
        return final_vec ;
    }
};

// Time complexiety :- o(n*n!)
