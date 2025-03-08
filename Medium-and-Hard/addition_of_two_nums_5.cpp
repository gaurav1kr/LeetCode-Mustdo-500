class Solution {
public:
    ListNode *newNode(int data)  
    {  
        ListNode *new_node = new ListNode(); 
        new_node->val = data;  
        new_node->next = NULL;  
        return new_node;  
    }  
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) 
    {
        int sum = 0 , carry = 0 , n1 = 0 ;
        ListNode *temp = NULL , *prev = NULL , *result = NULL;
        while(l1 || l2)
        {
            sum = carry + (l1 ? l1->val:0) + (l2 ? l2->val:0) ;
            carry = sum/10 ;
            n1 = sum % 10 ;
            temp = newNode(n1) ;
            if(!result)
                result = temp ;
            else
                prev->next = temp ;
            prev = temp ;
            if(l1)
                l1 = l1->next ;
            if(l2)
                l2 = l2->next ;
        }
        if(carry)
            temp->next = newNode(carry) ;
        return result ;
    }
};
