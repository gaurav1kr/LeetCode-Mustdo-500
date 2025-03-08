class Solution {
public:
  bool isCompleteTree(TreeNode *root) {
    queue<TreeNode *> q;
    int n;
    bool a = false;
    TreeNode *node;
    q.push(root);
    while (!q.empty()) {
      node = q.front();
      q.pop();
      if (a && (node->left || node->right))
        return false;
      if (node->left) {
        q.push(node->left);
      } else {
        a = true;
      }
      if (a && node->right)
        return false;
      if (node->right) {
        q.push(node->right);
      } else {
        a = true;
      }
    }
    return true;
  }
};
