struct ListNode 
{
     int val;
     ListNode *next;
     ListNode() : val(0), next(nullptr) {}
     ListNode(int x) : val(x), next(nullptr) {}
     ListNode(int x, ListNode *next) : val(x), next(next) {}
};

struct compare
{
		bool operator()(ListNode *const & a , ListNode *const & b)
		{
			return (a-> val > b->val) ;
		}
};

class Solution 
{
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) 
    {
        if(lists.empty())
            return NULL ;
        priority_queue<ListNode* , vector<ListNode *> , compare> pq ;
			
			for(auto & ll : lists)
			{
                if(ll)
				pq.push(ll) ;
			}
			ListNode *dummy_node = new ListNode(0) ;
			ListNode *tail = dummy_node ;
			
			while(!pq.empty())
			{
				tail->next = pq.top() ;
				pq.pop() ;
				tail = tail->next ;
				if(tail && tail->next != nullptr)
				{
					pq.emplace(tail->next) ;
				}
			}
			return dummy_node->next ;    
    }
};

// Time complexiety :- O(n*log(k)).

//Logic :- 
1. The compare struct defines a functor to compare two ListNode pointers. It compares the val member of the ListNode objects pointed to by a and b.

2. The Solution class contains the mergeKLists function, which takes a vector of pointers to ListNode objects representing k sorted linked lists.

3. It first checks if the input vector lists is empty. If so, it returns NULL.

4. It initializes a priority queue pq of ListNode pointers. The priority queue is configured to use the custom comparison operator defined in the compare struct.

5. It iterates through each linked list in the input vector lists. If the linked list is not empty (ll), it pushes the head of the linked list onto the priority queue.

6. It creates a dummy node and initializes tail to point to it.

7. It enters a loop that continues until the priority queue is empty.

8. Inside the loop, it pops the top element (which has the smallest value according to the custom comparison function) from the priority queue and appends it to the end of the merged list.

9. If the popped node has a next node, it pushes the next node onto the priority queue.

10. Finally, it returns the next node of the dummy node, which is the head of the merged linked list.
