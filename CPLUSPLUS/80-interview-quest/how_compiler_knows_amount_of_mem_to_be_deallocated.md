** We will discuss how "delete [] p ;" knows how much memory we need to deallocate.
    Basically it is done via overallocation technique which compiler performs internally when new is called. An additional value 'n'
    is saved on the begining of the array. Further the actual values of the array is saved afterwards. While calling delete , that 'n' is being picked and that many times ,destructor is called which destroys all of the array allocation one by one.
    An illustration is being shown below :- 
        int const n = 10 ;
        class Base
        {
            int b_var ;
        };

        int main()
        {
            Base *bp = new Base[n] ;
            //The compiler replaces below lines when it sees the above array allocation with new
            // char *tmp = (char *) operator new[](WORDSIZE + n*sizeof(Base)) ; (WORDSIZE is the no of bytes which the respective data type holds. )
                //Base *bp = (Base *) (tmp + WORDSIZE) ;
                // *(size_t*) tmp = n ;
                // for(int i=0;i<n;++i)
                //      new(bp + i) Base(); (This technique is called placement new)

            delete [] bp ;
            //The compiler replaces below lines when it sees the above array deallocation with delete
                // size_t n = * (size_t*) ((char *)bp - WORDSIZE);
                // while(n-- != 0)
                //      (bp+n) -> ~Base() ; (calling the destructor in a loop of 'n' which was saved at the time of allocation)
                // operator delete[] ((char *)bp - WORDSIZE) ; (Finally we are destroying the array)
        }
    
    Another mechanism is there which is called Associative array through which compiler decides how much
    memory needs to be deallocated. The approach is almost same. Here we are creating an associative array and saving the 'n' on it. The internal code looks like below :- 
            Base *bp = new Base[n];
            //The compiler replaces it with below set of lines
                // Base *bp = (Base *) operator new[] (n * sizeof(Base));
                // size_t i ;
                // for(i=0;i<n;++i)
                // {
                //     new(p+i) Base();  (This is placement new)
                // }
                // associationArray.insert(bp,n) ;


