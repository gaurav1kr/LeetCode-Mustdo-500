class Solution 
{
public:
    int getSum(int a, int b) 
    {
        if(b == 0)
        {
            return a ; 
        }
        return getSum(a ^ b , (a & b) << 1) ;
    }
};

// The problem can be solved using bitwise XOR and AND operations:

// XOR Operation: The XOR operation (a ^ b) will give us the sum of a and b without considering the carry.

// AND Operation: The AND operation (a & b) finds the carry bits. 
// The carry needs to be added to the result,
// and this can be done by left-shifting the carry bits by one and recursively calling the function.

// Base Case: When there is no carry left (b == 0), the addition is complete, and we can return a as the final result.
