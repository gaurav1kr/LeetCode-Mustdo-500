#include<vector>
#include<unordered_map>
#include<unordered_set>
using namespace std ;
class Twitter 
{
public:
    Twitter() 
    {
 
    }
 
    void postTweet(int userId, int tweetId) 
    {
        feed.push_back({ userId, tweetId });
    }
 
    vector<int> getNewsFeed(int userId) 
    {
        vector<int> ret;
        for (int i = feed.size() - 1; i >= 0; --i) 
        {
            auto& [user, tweet] = feed[i];
            if (userId == user || (following[userId].contains(user))) 
            {
                ret.push_back(tweet);
                if (ret.size() == 10) break;
            }
        }
        return ret;
    }
 
    void follow(int followerId, int followeeId) {
        following[followerId].insert(followeeId);
    }
 
    void unfollow(int followerId, int followeeId) {
        following[followerId].erase(followeeId);
    }
 
    std::unordered_map<int, std::unordered_set<int>> following;
    std::vector<std::pair<int, int>> feed;
};
 
/**
 * Your Twitter object will be instantiated and called as such:
 * Twitter* obj = new Twitter();
 * obj->postTweet(userId,tweetId);
 * vector<int> param_2 = obj->getNewsFeed(userId);
 * obj->follow(followerId,followeeId);
 * obj->unfollow(followerId,followeeId);
 */
