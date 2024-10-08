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
    ListNode* insertionSortList(ListNode* head) 
    {
        if (head == nullptr) 
        {
            return head;
        }
 
        ListNode* helper = new ListNode(0); 
        ListNode* cur = head; 
        ListNode* pre = helper; 
        ListNode* next = nullptr; 
 
 
        while (cur != nullptr) 
        {
            next = cur->next;
 
            while (pre->next != nullptr && pre->next->val < cur->val) 
            {
                pre = pre->next;
            }
 
            cur->next = pre->next;
            pre->next = cur;
            pre = helper;
            cur = next;
        }
 
        return helper->next;
    }
};
