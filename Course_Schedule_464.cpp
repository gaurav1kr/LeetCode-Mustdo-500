class Solution 
{
public: 
    static bool comp(vector<int>& a, vector<int>& b)
    {
        return a[1] < b[1];
    }
    int scheduleCourse(vector<vector<int>>& courses)
    {
        priority_queue<int>pq;  
        sort(courses.begin(), courses.end(), comp);
        int time = 0;
        for (auto course : courses) 
        {
            time += course[0];
            pq.push(course[0]);

            if (time > course[1] )

            {
                time -= pq.top();
                pq.pop();
            }
        } 
        return pq.size();
    }
};
