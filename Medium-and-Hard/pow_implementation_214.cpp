class Solution 
{
public:
    double myPow(double x, int n)
    {
        if (n == 0) return 1;
        if (n < 0) return (double)1.0/x * myPow((double)1.0/x, -(++n));
        if (n % 2 == 1) return x * myPow(x*x, n/2);
        else return myPow(x*x, n/2);
    }
};

//TC and SC - O(logn)
