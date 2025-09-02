##### Design a Review and Reward System for Food Delivery

by ZealousSalmonFrog165• Mid-level• 25 days ago

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

[• 24 days ago](https://www.hellointerview.com/community/submissions/cmdtpz2xd030gad08ybmzob4d#comment-cmdtpzgzz003xad3becfqykqi)

Nice design! You've created a thoughtful architecture for a review and reward system that handles the core requirements well. Your use of DynamoDB with appropriate partition keys for different access patterns shows good understanding of NoSQL design, and the CDC-based approach for reward processing is an elegant solution that avoids the complexity of distributed transactions. However, there are some critical issues around the 3 upvotes per day limit enforcement and potential race conditions in the voting system that need to be addressed.

1.  **Positives**
    
    -   Smart use of CDC (Change Data Capture) on the vote table to trigger reward eligibility checks - this ensures eventual consistency without complex distributed transactions
    -   Good DynamoDB schema design with itemId as partition key and createdAt as sort key for reviews, enabling efficient queries for recent reviews
    -   Thoughtful separation of concerns with dedicated Review and Vote services, making the system more maintainable and scalable
2.  **Potential Issues** **Missing Daily Vote Limit Enforcement**
    
    -   The current design doesn't track or enforce the 3 upvotes per user per day requirement
    -   Users could upvote unlimited times per day, violating a core functional requirement
    -   This breaks the system's ability to prevent vote manipulation and maintain fairness
    
    **Race Conditions in Vote Counting**
    
    -   Using DynamoDB ADD expressions without conditional checks can lead to incorrect counts
    -   Concurrent votes might overwrite each other or create inconsistent states between the Votes table and review counts
    -   This could result in users receiving rewards based on incorrect vote counts
    
    **Inefficient Monthly Reward Calculation**
    
    -   The cron job approach requires scanning all reviews with >50 votes every month
    -   This becomes increasingly expensive as the system grows, potentially taking hours for 100M DAU
    -   The design acknowledges users could lose eligibility between checks but doesn't address this properly
3.  **Follow-up Questions**
    
    -   How does the system track and enforce the 3 upvotes per user per day limit?
    -   What happens when multiple users vote on the same review simultaneously - how do you prevent lost updates?
    -   How do you handle the scenario where a user's review drops below 80% positive votes after they've already received a reward?
    -   What's the expected latency for the monthly cron job with millions of reviews, and how does this scale?
    -   How does the system handle CDC failures or delays that might cause missed reward eligibility events?
    -   What prevents a user from voting multiple times on the same item by calling the API repeatedly?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply