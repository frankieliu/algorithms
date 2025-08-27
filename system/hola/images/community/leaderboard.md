##### Design an Online Game Leaderboard

by Manasi• Manager• 3 days ago

Build a system for a game played N billion times per day that retrieves a user’s score, the K global scores surrounding it, and the K scores around the user within their friends list. The system must handle requests in near real-time.

3

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

[• 3 days ago](https://www.hellointerview.com/community/submissions/cmeolbmm90531ad08nredgzyq#comment-cmeolcarh003had3abien2vhm)

Nice design! You've created a sophisticated architecture that cleverly uses rank-based sharding and prefix counts for efficient global rank computation. The in-memory RocksDB approach with block-based distribution shows good understanding of how to handle high-throughput ranking operations, and your tie-breaker strategy (score DESC, lastUpdated ASC, userId ASC) properly handles edge cases.

1.  **Positives**
    
    -   Smart use of rank-based sharding with prefix counts enables O(log B + K) lookups instead of scanning millions of records - this is an elegant solution to the global ranking problem
    -   Separating write path through Kafka from read path allows for better scale and reliability, preventing score updates from blocking leaderboard queries
    -   Using in-memory RocksDB instances provides the right balance of performance and durability for this high-throughput use case
2.  **Potential Issues** **Friends Ranking Computation Complexity**
    
    -   Your design mentions "notifies friends-window caches with bounded fanout" but doesn't explain how friend rankings are efficiently computed
    -   When a user's score changes, you'd need to update rankings for potentially thousands of friends who have that user in their friend list
    -   This creates a write amplification problem that could violate your 200ms write latency requirement
    
    **Memory Scaling for Billions of Users**
    
    -   Storing billions of user scores in memory (even with RocksDB) requires enormous memory resources
    -   With 10 billion users × ~100 bytes per entry, you'd need ~1TB just for user data, not counting indexes
    -   This exceeds practical memory limits and would force frequent disk I/O, breaking your 100ms read latency requirement
    
    **Concurrent Updates to Same Rank Range**
    
    -   When multiple users achieve similar scores simultaneously, they'll all try to update the same rank block
    -   This creates contention on specific shards that handle popular score ranges
    -   Your "hotspot defense" mentions range-aware admission but doesn't explain how to prevent write storms from overwhelming a single shard
3.  **Follow-up Questions**
    
    -   How does your system efficiently compute friend rankings when a user has 5,000+ friends scattered across different shards?
    -   What happens to in-flight requests when you need to rebalance rank blocks during peak traffic?
    -   How do you ensure consistent tie-breaker ordering when users with identical scores are on different shards?
    -   How much memory does each shard actually need to handle its portion of billions of users?
    -   What's your strategy for handling the "new period" cutover (e.g., daily to weekly) when all ranks reset?

_Note: This feedback is based solely on the technical architecture provided and aims to help identify areas for improvement in the system design._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply