ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) 
{
        stack<int> s1, s2;
        ListNode* head = NULL;  // new head of result LL
        int carry = 0;
        
		// insert elements of both linked lists into stacks
        ListNode* h1 = l1;
        while(h1) 
		{
            s1.push(h1->val);
            h1 = h1->next;
        }
        
        ListNode* h2 = l2;
        while(h2) 
		{
            s2.push(h2->val);
            h2 = h2->next;
        }
        
        while(!s1.empty() || !s2.empty())
		{
            int sum = (s1.empty() ? 0 : s1.top()) + (s2.empty() ? 0 : s2.top()) + carry;
            carry = sum>=10 ? 1 : 0;
            ListNode * temp = new ListNode(sum%10);
            
            if(head==NULL)
                head = temp;
            else
			{
                temp->next = head;
                head = temp;
            }
            
            if(!s1.empty()) s1.pop();
            if(!s2.empty()) s2.pop();
            
        }
        // 9->9 + 9->9 = 1->9->8 type of case
        if(carry == 1)
		{
            ListNode * temp = new ListNode(1);
            temp->next = head;
            head = temp;
        }
        
        return head;
  }