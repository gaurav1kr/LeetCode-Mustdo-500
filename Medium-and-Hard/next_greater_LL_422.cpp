class Solution
{
public:
    vector<int> nextLargerNodes(ListNode* head)
    {
        vector<int> result;
        stack<int> stk1;
        multimap<int, int> umap;
        ListNode* temp = head;
        while (head)
        {
            if (stk1.empty() || head->val < stk1.top())
            {
                stk1.push(head->val);
            }
            else
            {
                while (!stk1.empty() && head->val > stk1.top())
                {
                    umap.insert({stk1.top() ,head->val });
                    stk1.pop();
                }
                stk1.push(head->val);
            }
            head = head->next;
        }
        while (!stk1.empty())
        {
            umap.insert({ stk1.top() ,0 });
            stk1.pop();
        }

        while (temp)
        {
            auto range = umap.equal_range(temp->val);
            auto it = range.first;
            result.push_back(it->second);
            auto key = umap.find(temp->val);
            if (it != umap.end())
            {
                umap.erase(it);
            }
            temp = temp->next;
        }
        return result;
    }
};