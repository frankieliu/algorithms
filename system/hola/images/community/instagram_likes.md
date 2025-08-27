##### Design Likes Feature For Instagram/Twitter

by Bhanu Prakash Thipparthi• Mid-level• 17 days ago

Design Overview :

User sends like/unlike → Goes through API Gateway (auth, rate limit, routing).

Like Service checks Redis Sorted Set → avoids duplicate likes from same user.

Update Redis counter → real-time like count (fast, hot storage).

Send event to Kafka → for processing in background.

Flink reads from Kafka → aggregates likes every 1 min → writes to Cassandra (cold storage).

Count Service reads like count → gets base count from Cassandra + delta from Redis → returns total.

S3 stores periodic snapshots → MapReduce checks for mismatches.

Reconciliation Worker fixes issues → corrects counts & removes duplicates.

Cassandra handles large writes → stores permanent data with horizontal scaling.

Collapse Description

1

3

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 17 days ago](https://www.hellointerview.com/community/submissions/cme4nuu5z0anwad07be2e8d4a#comment-cme4nvj2f000lad3b44dklf9x)

Nice design! You've created a thoughtful architecture that separates hot and cold storage effectively, using Redis for real-time counts and Cassandra for durable storage. The use of Kafka for asynchronous processing and Flink for aggregation shows good understanding of stream processing patterns, and your reconciliation worker addresses the important concern of data consistency in an eventually consistent system.

1.  **Positives**
    
    -   Smart use of Redis Sorted Set for idempotency checking prevents duplicate likes from the same user, which is crucial for data integrity
    -   Excellent hot/cold storage pattern with Redis handling real-time counts and Cassandra storing aggregated data, optimizing for both latency and durability
    -   Good choice of Flink for stream processing with 1-minute aggregation windows, reducing write pressure on Cassandra while maintaining reasonable freshness
2.  **Potential Issues** **Race Condition in Like/Unlike Operations**
    
    -   Your design checks Redis Sorted Set for duplicates but doesn't show atomic operations for the check-and-update flow
    -   This allows two concurrent like requests from the same user to both pass the duplicate check before either updates the sorted set
    -   This violates your "one like per post per user" requirement and could lead to inflated like counts
    
    **Missing Failure Handling for Kafka**
    
    -   No mention of what happens when Kafka is unavailable or messages fail to publish after Redis is updated
    -   This creates a split-brain scenario where Redis shows the like but it never reaches Cassandra
    -   Your reconciliation worker might not catch this if it only compares S3 snapshots with Cassandra, missing the Redis-only likes
    
    **Unclear CDC Implementation**
    
    -   You mention CDC (Change Data Capture) from Redis to Cassandra but Redis doesn't natively support CDC
    -   Without proper implementation details, there's no clear path for how changes flow from hot to cold storage
    -   This could lead to data loss if Redis fails before data is persisted to Cassandra
3.  **Follow-up Questions**
    
    -   How do you handle the scenario where a user unlikes immediately after liking, but the like event is still in Kafka queue?
    -   What happens to the Redis delta counts after they're aggregated to Cassandra - are they reset, and how do you prevent count loss during this transition?
    -   How does your reconciliation worker handle conflicts when it finds mismatches - which source of truth takes precedence?
    -   What's your strategy for handling Redis memory limits when storing sorted sets for millions of posts?
    -   How do you ensure the Count Service doesn't return stale data when reading from both Redis and Cassandra during the aggregation window?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

Z

ZoophagousPlumOtter235

[• 16 days ago](https://www.hellointerview.com/community/submissions/cme4nuu5z0anwad07be2e8d4a#comment-cme5awms40gdnad07yo7h5ucx)

Hi, could you please walk me through workflow ?

Show more

0

Reply

![Pranny](https://lh3.googleusercontent.com/a/ACg8ocKR7mU4GBFKx_Iml338CY2V0e1UtPvMVTTTllrbr0k4Xpxm=s96-c)

Pranny

[• 13 days ago](https://www.hellointerview.com/community/submissions/cme4nuu5z0anwad07be2e8d4a#comment-cmeaejmrw0185ad07n5g31ito)

How would you define the TTL on the post\_id,like \_count redis cache ? When summing the base+ delta counts in the count service, how do you know that the delta is not part of the base counts ? What if the delta counts have made its way to Cassandra through Flink?

Show more

0

Reply