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
    ListNode* partition(ListNode* head, int x) 
	{
        ListNode *d1 = new ListNode(0), *d2 = new ListNode(0), *n1, *n2;
        n1 = d1;
        n2=d2;
        while(head!=NULL)
		{
            if(head->val < x)
			{
                n1->next=head;
                n1=n1->next;
            } 
			else
			{
                n2->next=head;
                n2=n2->next;
            }
            head=head->next;
        }

        n2->next=NULL;
        n1->next=d2->next;
        return d1->next;
    }
};