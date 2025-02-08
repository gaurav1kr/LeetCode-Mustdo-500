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
class Solution 
{
	public:
        ListNode* rotateRight(ListNode* head, int k) 
        {
        	if(!head || !head->next) return head;
        	int len = 1;
        	ListNode* curr = head;
        	while(curr->next)
		{
            		curr = curr->next;
            		len++;
        	}
        	k = len - (k%len);
        	if(k==len) return head;

        	ListNode* last = head;
        	while(--k)
		{
            		last = last->next;
       	 	}

        	ListNode* ans = last->next;
        	curr->next = head;
        	last->next = NULL;

        	return ans;
    	}
};
