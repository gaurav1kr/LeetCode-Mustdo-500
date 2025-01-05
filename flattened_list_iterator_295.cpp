/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * class NestedInteger {
 *   public:
 *     // Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool isInteger() const;
 *
 *     // Return the single integer that this NestedInteger holds, if it holds a single integer
 *     // The result is undefined if this NestedInteger holds a nested list
 *     int getInteger() const;
 *
 *     // Return the nested list that this NestedInteger holds, if it holds a nested list
 *     // The result is undefined if this NestedInteger holds a single integer
 *     const vector<NestedInteger> &getList() const;
 * };
 */
class NestedIterator 
{
private:
    
    vector<NestedInteger> nestedList;
    vector<int> flattenedList;
    int currentIndex = 0;

    void flatten(vector<NestedInteger>& currentList) 
	{
        for (int i = 0; i < currentList.size(); i++) 
		{
            if (currentList[i].isInteger()) 
			{
                flattenedList.push_back(currentList[i].getInteger());
            } 
			else 
			{
                flatten(currentList[i].getList());
            }
        }
    }

public:
    NestedIterator(vector<NestedInteger> &nestedList) 
	{
        this->nestedList = nestedList;
        flatten(nestedList);
    }

    
    int next() 
	{
        int number = flattenedList[currentIndex];
        currentIndex++;
        return number;
    }

    bool hasNext()
	{
        return currentIndex < flattenedList.size();
    }
};