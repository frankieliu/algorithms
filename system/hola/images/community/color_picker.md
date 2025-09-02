##### Design a Color Picker

by Mynameisdon• Senior Manager• 22 days ago

Building as part of hackathon , only some users are facing latency

1

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

[• 22 days ago](https://www.hellointerview.com/community/submissions/cmdwjqz9u0026ad0920lo4wfx#comment-cmdwjrfiw0001ad3bv691192y)

Nice design! You've built a solid foundation for a color picker system with good architectural choices like CDN integration, caching layers, and database sharding. Your strongest insight is recognizing the need for multi-layer caching to address latency issues, though the design has some critical gaps around data consistency and the sharing mechanism that need attention.

1.  **Positives**
    
    -   Smart use of CDN and geographic distribution to address latency for some users - this is a thoughtful approach to solving the stated problem
    -   Well-structured caching hierarchy with separate caches for user info, permissions, favorite lists, and color metadata reduces database load effectively
    -   Good choice of consistent hashing for sharding by user\_id which will distribute load evenly and make scaling predictable
2.  **Potential Issues** **Missing Share Implementation**
    
    -   The API defines sharing functionality but the architecture diagram shows no components for handling shared lists or email notifications
    -   Users won't be able to view colors shared with them as required since there's no shared list retrieval mechanism
    -   This violates the functional requirement that users can view colors shared with them
    
    **Undefined Permission Model**
    
    -   The API mentions "viewer|editor" roles but there's no data model for permissions or sharing relationships
    -   Without a proper ACL system, users could access or modify lists they shouldn't have access to
    -   This creates security vulnerabilities and breaks the sharing feature's access control
    
    **Cache Consistency Strategy Missing**
    
    -   Multiple cache layers (user info, permissions, favorite lists, colors) with no defined invalidation strategy
    -   When a user updates their favorite list, stale data could be served from various cache layers
    -   This violates the <100ms operation requirement as cache misses cascade through layers
3.  **Follow-up Questions**
    
    -   How does the sharing mechanism work - where are shared list relationships stored and how are email notifications sent?
    -   What happens when a user updates a favorite list that's been shared with others - how do viewers see the updates?
    -   How do you handle cache invalidation across your multiple cache layers when data changes?
    -   Why do you need a separate colors table and cache when colors are just hex values - isn't this overengineering?
    -   How does your permission cache get populated and stay in sync with sharing changes?
    -   What's your strategy for handling the "some users facing latency" beyond CDN - have you identified if it's geographic, device-specific, or load-related?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply