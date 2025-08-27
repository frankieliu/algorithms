##### Design an Ads Campaign for a 5-Minute Availability with Coupon

by RemainingAmaranthBlackbird581• Mid-level• 13 days ago

Few assumptions - User wont be able to search by some keywords (as text-based search and bifurcating on text in 5 mins) will be quite slow.

1

2

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 13 days ago](https://www.hellointerview.com/community/submissions/cmea7qh2c00vaad07wa2evnsq#comment-cmea7ruwl002sad3agx76dw1r)

Nice design! You've identified the key challenge of the 5-minute availability window and made smart choices around caching and TTL-based storage. Your approach of using in-memory caches with horizontal scaling shows good understanding of how to achieve the 50ms latency requirement for this read-heavy system.

1.  **Positives**
    
    -   Using TTL-based storage for campaigns is perfect for the 5-minute expiration requirement - this ensures expired coupons are automatically removed without manual cleanup
    -   Implementing both local server cache and shared cache is a sophisticated approach that reduces load on any single cache instance while maintaining low latency
    -   Horizontal scaling of the campaign feed service recognizes the read-heavy nature of the system and allows handling the high concurrent read operations
2.  **Potential Issues** **Cache Consistency Problem**
    
    -   You have multiple cache layers (local + shared) but no clear invalidation strategy when new campaigns are posted
    -   This means users might not see newly posted campaigns immediately, violating the consistency requirement
    -   The functional requirement states coupons should be available "as soon as they are available" but stale caches would prevent this
    
    **Missing Write Path Details**
    
    -   The campaign service writes to a TTL database but there's no clear mechanism to propagate these writes to all cache instances
    -   Without proper cache warming or invalidation, new campaigns won't appear in user feeds until caches expire
    -   This creates a window where campaigns exist in the database but are invisible to users
    
    **Unclear Data Model**
    
    -   The schema shows campaignAdId, description, couponId, createdAt but no product categorization or targeting information
    -   Without product/category fields, the "view by product query" requirement cannot be fulfilled
    -   The simple SELECT \* LIMIT 50 query doesn't support filtering by product or any targeting logic
3.  **Follow-up Questions**
    
    -   How does the system propagate new campaign writes from the Campaign service to all the distributed cache instances?
    -   What happens when multiple businesses post campaigns for the same coupon simultaneously - how do you handle deduplication?
    -   How do you implement the "product query" filtering when your schema doesn't include product categories or tags?
    -   What's your strategy for cache invalidation when a campaign is posted to ensure immediate visibility?
    -   How do you prevent showing expired campaigns if the TTL cleanup hasn't run yet but the 5-minute window has passed?
    -   What's the purpose of storing campaignAdId as both key and value in your cache design?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply