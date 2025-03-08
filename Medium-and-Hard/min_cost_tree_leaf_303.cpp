int mctFromLeafValues(vector<int>& arr) 
{
        int sum=0;
        vector<int> v;
        for(int i=0; i<arr.size(); i++)
        {
            while(v.size()!=0 && v.back()<=arr[i])
            {
                if(v.size()>1 && arr[i]>v[v.size()-2])
                    sum+=v.back()*v[v.size()-2];
                else
                    sum+=v.back()*arr[i];
                v.pop_back();
            }
            v.push_back(arr[i]);
        }
        for(int i=v.size()-2; i>=0; i--)
		{
            sum+=v[i]*v[i+1];
		}
        return sum;
}