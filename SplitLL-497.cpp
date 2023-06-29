#include<iostream>
#include<vector>
using namespace std;

struct ListNode
{
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

void InsertList(ListNode*& root, int x)
{
    ListNode* temp = new ListNode(x);
    temp->next = root;
    root = temp;
}

void PrintList(ListNode* root)
{
    while (root)
    {
        cout << root->val << "\n";
        root = root->next;
    }
}

class Solution {
public:
    vector<ListNode*> splitListToParts(ListNode* root, int k) {
        //first calculate the length
        int len = 0;
        ListNode* temp = root;
        while (temp)
        {
            len++;
            temp = temp->next;
        }

        int numNodes = len / k; 
        int extra = len % k;  
        int i = 0;
        vector<ListNode*> res;
        temp = root;
        while (temp)
        {
            res.push_back(temp);
            int currLen = 1;
            while (currLen < numNodes)
            {
                temp = temp->next;
                currLen++;
            }
            if (extra > 0 && len > k)
            {
                temp = temp->next;
                extra--;
            }
            ListNode* x = temp->next;
            temp->next = NULL;
            temp = x;
        }
        //if the number of nodes are less than k we add NULL
        while (len < k)
        {
            res.push_back(NULL);
            len++;
        }
        return res;
    }
};

int main()
{
    Solution sol;
    int arr[] = { 4,5,3,2,1 };
    int size_of_arr = sizeof(arr) / sizeof(arr[0]);
    ListNode *list1 = NULL;
    for(int i=0;i<size_of_arr;i++)
    {
        InsertList(list1 , arr[i]);
    }
    PrintList(list1);

    vector<ListNode*> lst1 = sol.splitListToParts(list1, 3);
   
    return 0;
}
