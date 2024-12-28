class Solution 
{
public:
    struct trieNode 
    {
        trieNode* left = nullptr; // 0
        trieNode* right = nullptr; // 1
    };

    void insert(trieNode* root, int num) 
    {
        trieNode* pcrawl = root;

        for (int i = 31; i >= 0; i--) 
        {
            int ith_bit = (num >> i) & 1;
            if (ith_bit == 0) 
            {
                if (pcrawl->left == NULL) 
                {
                    pcrawl->left = new trieNode();
                }
                pcrawl = pcrawl->left;
            } 
			else 
            {
                if (pcrawl->right == NULL) 
                {
                    pcrawl->right = new trieNode();
                }
                pcrawl = pcrawl->right;
            }
        }
    }

    int findMaxXor(trieNode* root, int num)
    {
        int maxi = 0;
        trieNode* pcrawl = root;

        for (int i = 31; i >= 0; i--) 
        {
            int ith_bit = (num >> i) & 1;
            if (ith_bit == 1) 
            {
                if (pcrawl->left != NULL) 
                {
                    maxi += pow(2, i);
                    pcrawl = pcrawl->left;
                } 
                else 
                {
                    pcrawl = pcrawl->right;
                }
            } 
            else 
            {
                if (pcrawl->right != NULL) 
                {
                    maxi += pow(2, i);
                    pcrawl = pcrawl->right;
                } 
                else 
                {
                    pcrawl = pcrawl->left;
                }
            }
        }
        return maxi;
    }

    int findMaximumXOR(std::vector<int>& nums) 
    {
        trieNode* root = new trieNode();
        for (int& num : nums) 
        {
            insert(root, num);
        }

        int maxResult = 0;
        for (int i = 0; i < nums.size(); i++) 
        {
            int temp = findMaxXor(root, nums[i]);
            maxResult = std::max(maxResult, temp);
        }
        return maxResult;
    }
};