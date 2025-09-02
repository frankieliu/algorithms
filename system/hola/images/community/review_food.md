##### Design a Review App for Food Items

by Akshay• Senior• 16 days ago

Design a scalable system that enables customers of a food-delivery platform to create and consume high-quality reviews for delivered food items, while fostering community engagement and rewarding valuable contributions. Goals • Let users review and rate food items from their completed orders. • Enable social feedback (like/dislike) on reviews to surface quality content. • Provide review discovery per food item (sorted by relevance/recency). • Issue monthly rewards to reviewers based on transparent criteria (e.g., number of likes).

Collapse Description

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

[• 16 days ago](https://www.hellointerview.com/community/submissions/cme69192e0qldad086wsou072#comment-cme692f3j002jad3b4paspacr)

Nice design! You've created a comprehensive system for food item reviews with thoughtful consideration of scalability, data consistency, and user engagement. Your use of optimistic concurrency control for rating updates and the unique database constraint for enforcing the one-review-per-month limit shows strong technical insight. However, there are some critical issues around data consistency and system reliability that need addressing, particularly with how likes are tracked and the potential for lost updates in your current architecture.

1.  **Positives**
    
    -   Excellent use of database constraints with the unique index on (userId, foodItemId, date\_trunc("month", timestamp)) to enforce the one-review-per-month requirement at the database level - this is much more reliable than application-level checks
    -   Smart application of optimistic concurrency control for updating average ratings, which helps maintain consistency while avoiding lock contention during concurrent review submissions
    -   Good separation of read and write services with appropriate use of read replicas to handle the 10M DAU read load, recognizing that eventual consistency is acceptable for reviews
2.  **Potential Issues** **Data Inconsistency Between Likes Table and Review Likes Count**
    
    -   You're storing likes in a separate Likes table but also maintaining a likes count on the Review table, creating two sources of truth
    -   This will lead to inconsistencies when the atomic update on the review succeeds but the insert into the Likes table fails (or vice versa)
    -   The system violates the requirement for accurate reward calculation since the likes count used for monthly rewards may not match the actual likes in the database
    
    **Missing Idempotency for Like Operations**
    
    -   The POST /fooditems/:id/reviews/like endpoint lacks idempotency handling, allowing users to potentially like the same review multiple times
    -   Network retries or double-clicks could result in inflated like counts, unfairly affecting the monthly reward system
    -   This breaks the fairness requirement for transparent reward criteria since like counts could be artificially inflated
    
    **Single Point of Failure for Reward Processing**
    
    -   The reward worker appears to be a single component with no redundancy or failure handling mentioned
    -   If the cron job or worker fails at month-end, no rewards will be distributed, potentially losing user trust
    -   There's no mention of how to handle partial failures or ensure exactly-once reward distribution
3.  **Follow-up Questions**
    
    -   How do you handle the race condition when multiple users simultaneously like/unlike the same review, especially given your atomic update approach?
    -   What happens if the reward worker crashes midway through processing monthly rewards - how do you ensure some users don't get double rewards on retry?
    -   How do you plan to handle the eventual consistency lag between primary and replica databases when users expect to see their review immediately after posting?
    -   Given your 100:1 read/write ratio assumption, have you considered using a dedicated search service like Elasticsearch for review discovery instead of hitting the database directly?
    -   How would you handle the scenario where a food item is removed from the platform but still has associated reviews that users want to access?
    -   What's your strategy for handling review spam or abuse beyond the one-per-month limit, especially since likes directly impact monetary rewards?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply