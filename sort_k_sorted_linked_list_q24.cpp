/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
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
