#define INDEX(c) (int)c - (int)'a'
struct Node 
{
    Node* child[26];
    int end;
    vector<int> palindromes_below;
    
    Node() 
	{
        for (int i = 0; i < 26; i++) 
		{
            child[i] = NULL;
        }
        end = -1;
    }
};

class Solution 
{
    Node* root = new Node();
    
	public:
		vector<vector<int>> ans;
		bool check(const string& s, int i, int j) 
		{
			while (i < j) 
			{
				if (s[i++] != s[j--]) return false;
			}
			return true;
		}

		void insert(string& s, int ind) 
		{
			Node* temp = root;
			for (int i = s.size() - 1; i >= 0; i--) 
			{
				if (check(s, 0, i)) 
				{
					temp->palindromes_below.push_back(ind);
				}
				if (temp->child[INDEX(s[i])] == NULL) 
				{
					temp->child[INDEX(s[i])] = new Node;
				}
				temp = temp->child[INDEX(s[i])];
			}
			temp->end = ind;
			temp->palindromes_below.push_back(ind);
		}

		void search(string& s, int ind) 
		{
			Node* temp = root;
			for (int i = 0; i < s.size(); i++) 
			{
				if (temp->end != -1 && temp->end != ind && check(s, i, s.size() - 1)) 
				{
					ans.push_back({ind, temp->end});
				}
				
				if (temp->child[INDEX(s[i])] == NULL) 
					return;
				
				temp = temp->child[INDEX(s[i])];
			}
			for (int idx : temp->palindromes_below) 
			{
				if (ind != idx) ans.push_back({ind, idx});
			}
		}

		vector<vector<int>> palindromePairs(vector<string>& words) 
		{
			for (int i = 0; i < words.size(); i++) 
			{
				insert(words[i], i);
			}
			for (int i = 0; i < words.size(); i++) 
			{
				search(words[i], i);
			}
			return ans;
		}
};