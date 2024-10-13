1. It is a class template.
2. Provided by C++11 to avoid memory leak.
3. It shares ownership with only one pointer at a time. Ownership can't be copied but can be moved.
4. In case of exception occurrence , unique poiner will deallocate the memory so there is no change of memory leak.
