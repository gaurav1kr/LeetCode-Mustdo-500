Structure padding and packing - 
Compiler adds extra memory for the data types as per the sequence of variables which is mentioned
in the structure or classes in C++. This is called structure padding.
The allocation is inlined with whether it's a 32 bit or 64 bit machine

Consider below 2 cases for a better understanding :- 

Case-1
```
struct Base
{
	char a ;
	char b ;
	int i ;
	char c ;
}

int main()
{
	cout<<sizeof(Base)<<"\n" ;
	return 0 ;
}

# The expected output is '7' (Assuming that char size is '1' byte and int size is '4' byte)
# Indeed, the output will be '12' because of structure padding. let's see how it works.
# The compiler will allocate the memory on blocks of highest memory size of the data inside the structure.
# In the above example , 4 byte chunk will be allocated for the data. So it will look like this - 

---- ---- ----
ab    i   c

# First 4 bytes will be allocated and since 'a' and 'b' are taking 1 bytes they will be adjusted
  in first 2 bytes while rest 2 bytes will be blank.

# Second 4 byte will be allocated for the variable 'i' and that will consume whole '4' bytes.
# Third 4 byte will be allocated for the variable 'c'.
```

Case-2 :-
```
# Consider a case where we will change the sequence of variables as below :- 

struct Base
{
	char a ;
	char b ;
	char c ;
	int i ;
}

int main()
{
	cout<<sizeof(Base)<<"\n" ;
	return 0 ;
}

# In this case, only 2 blocks of '4' bytes are suffice to allocate it. so the size will be 8 bytes

---- ----
abc    i

# In case, if we want to stop this padding , we need to add an additional preprocessor directive -
#pragma pack(1)
```
