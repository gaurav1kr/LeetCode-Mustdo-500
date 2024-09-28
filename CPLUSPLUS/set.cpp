#include<iostream>
#include<set>
using namespace std ;
int main()
{
	int arr[] = {21,12,32,23,43,34,54,56};
	set<int> s1 ;
	for(auto & i:arr)
	{
		s1.insert(i) ;
	}

	for(auto & i:s1)
	{
		cout<<i<<"\n" ;	
	}
	return 0 ;
}
