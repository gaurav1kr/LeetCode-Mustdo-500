class Solution
 {
public:
    string customSortString(string order, string s) 
    {
        unordered_map<int,int> umap ;
        for(auto & i:s)
        {
            umap[i]++ ;
        }   
        s.clear() ;

        for(auto & i:order)
        {
            while(umap[i])
            {
                umap[i]-- ;
                s.push_back(i);
            }
           
        }

        for(auto & i : umap)
        {
            while(i.second)
            {
                s.push_back(i.first);
                i.second-- ;
            }
        }
        return s ;
    }
};
