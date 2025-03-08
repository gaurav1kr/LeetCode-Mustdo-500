class NumArray {
int seg[100000];
vector<int> arr;
int n;
void build(int ind, int low, int high, vector<int>& arr)
{
   if(low == high) //base case
   {
       seg[ind] = arr[low];
       return;
   }
   int mid = (low + high)/2;
   build(2*ind+1,low,mid,arr);             //call for left sub-tree
   build(2*ind+2,mid+1,high,arr);          //call for right sub-tree
   seg[ind] = seg[2*ind+1] + seg[2*ind+2]; //store total sum
}

int query(int ind, int low, int high, int l, int r)
{
    if(low > r || high < l) //CASE 1: out of range
     return 0;
                                
    if(l <= low && high <= r) //CASE 2: overlapping range
     return seg[ind];
     
    //CASE 3: partial overlapping range
    int mid = (low + high)/2;
    int left = query(2*ind+1,low,mid,l,r);
    int right = query(2*ind+2,mid+1,high,l,r);
    return left + right;
}
public:
    NumArray(vector<int>& nums) {
        arr = nums;
        n = arr.size();
        build(0,0,n-1,arr); //build segment tree
    }
    
    int sumRange(int left, int right) {
        return query(0,0,n-1,left,right);
    }
};class NumArray {
int seg[100000];
vector<int> arr;
int n;
void build(int ind, int low, int high, vector<int>& arr)
{
   if(low == high) //base case
   {
       seg[ind] = arr[low];
       return;
   }
   int mid = (low + high)/2;
   build(2*ind+1,low,mid,arr);             //call for left sub-tree
   build(2*ind+2,mid+1,high,arr);          //call for right sub-tree
   seg[ind] = seg[2*ind+1] + seg[2*ind+2]; //store total sum
}

int query(int ind, int low, int high, int l, int r)
{
    if(low > r || high < l) //CASE 1: out of range
     return 0;
                                
    if(l <= low && high <= r) //CASE 2: overlapping range
     return seg[ind];
     
    //CASE 3: partial overlapping range
    int mid = (low + high)/2;
    int left = query(2*ind+1,low,mid,l,r);
    int right = query(2*ind+2,mid+1,high,l,r);
    return left + right;
}
public:
    NumArray(vector<int>& nums) {
        arr = nums;
        n = arr.size();
        build(0,0,n-1,arr); //build segment tree
    }
    
    int sumRange(int left, int right) {
        return query(0,0,n-1,left,right);
    }
};
