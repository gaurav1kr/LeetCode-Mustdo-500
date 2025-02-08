class Solution {
public:
    int calculate(string s) {
        int num = 0, n = s.size(), result = 0, last = 0;
        char op = '+';
        for (int i = 0; i < n; ++i) {
            if (isdigit(s[i])) {
                num = num * 10 + (s[i] - '0');
            }
            if (!isdigit(s[i]) && !isspace(s[i]) || i == n - 1) {
                if (op == '+') { result += last; last = num; }
                else if (op == '-') { result += last; last = -num; }
                else if (op == '*') { last *= num; }
                else if (op == '/') { last /= num; }
                op = s[i];
                num = 0;
            }
        }
        return result + last;
    }
};
