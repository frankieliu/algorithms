##### Design Facebook Privacy

by ZenP• Mid-level• 16 days ago

3

1

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 15 days ago](https://www.hellointerview.com/community/submissions/cme6j153d00d1ad08k1njax71#comment-cme6j1v2w002iad3arv20rgg4)

Nice design! You've created a thoughtful approach to Facebook's privacy system with good separation of concerns between profile privacy, post privacy, and friend relationships. Your use of Redis for caching friend/FoF relationships and Kafka for asynchronous processing shows solid understanding of scalability patterns, though there are some critical consistency issues that need addressing.

1.  **Positives**
    
    -   Smart use of Redis to cache friend and FoF relationships, which will significantly reduce database queries for the most common privacy checks
    -   Good separation between profile privacy and post privacy settings, allowing users granular control over their data
    -   Appropriate use of Kafka for asynchronous processing of friend graph updates, which prevents blocking operations during friend additions
2.  **Potential Issues** **Privacy Check Race Conditions**
    
    -   No distributed locking mechanism when updating privacy settings means concurrent updates could result in inconsistent privacy states
    -   A user could temporarily see content they shouldn't access if privacy settings are updated while they're viewing a profile
    -   This violates the "high consistency for privacy" requirement and could expose sensitive user data
    
    **Missing Privacy Enforcement Layer**
    
    -   The design shows data flow but doesn't specify where privacy rules are actually enforced
    -   Without a centralized privacy service, each component (feed, profile view) must implement its own privacy logic
    -   This creates multiple points of failure where privacy bugs could expose user data
    
    **Cache Invalidation Strategy**
    
    -   No clear mechanism for invalidating Redis caches when friendships change or privacy settings update
    -   Stale cache data could allow users to see posts/profiles after being unfriended
    -   This directly violates the high consistency requirement for privacy settings
3.  **Follow-up Questions**
    
    -   How do you handle privacy checks for custom lists - where is the membership data stored and how is it queried efficiently?
    -   What happens when a user changes their privacy settings while millions of users might be viewing their content?
    -   How do you ensure the Redis cache stays consistent with the source of truth in DynamoDB/Cassandra?
    -   How does the system handle privacy for shared content like photos where multiple users might have different privacy preferences?
    -   What's your strategy for handling the "friends of friends" calculation at scale - this could be computationally expensive for users with many friends?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply