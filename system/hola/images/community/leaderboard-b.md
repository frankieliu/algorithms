##### Design an Online Game Leaderboard

by Noam• Staff• 21 days ago

_Design a real-time Top-k leaderboard for a gaming application, where the system continuously receives users’ scores along with the points they earn._

### Deep Dive 1 – Scaling to 100M Users

1.  **Event flow:**
    
    -   Every game result is pushed into Kafka. Kafka partitions by user ID, so ingestion workers consume user events sequentially.
        
    -   That eliminates the need for distributed locking and prevents race conditions at the ingestion layer.
        
    
2.  **Leaderboard storage:**
    
    -   Redis sorted sets are the core because they give O(log n) updates and O(log n) rank queries.
        
    -   We shard Redis by user ID to spread load and memory. Each ingestion worker writes only to its corresponding shard.
        
    
3.  **Top-N aggregation:**
    
    -   For global top-10, we periodically pull top-N from each shard and merge in memory. That’s cheap because we only merge a handful of small lists, not 100M entries.
        
    
4.  **Latency considerations:**
    
    -   We accept a small processing delay because ingestion happens asynchronously through Kafka.
        
    -   To keep the UX smooth, the client updates scores optimistically right after a win, and then reconciles with the server’s authoritative score.
        
    

### Deep Dive 2 – Durability & Correctness

1.  **Duplicate and missing events:**
    
    -   Kafka is at-least-once, so duplicates can happen.
        
    -   On ingestion, I check a processed set in Redis. If the event ID is new, I update the score and mark it processed in a **single Redis transaction**.
        
    -   That prevents the “marked but not updated” gap and guarantees atomicity.
        
    
2.  **Out-of-order events:**
    
    -   We don’t care about strict ordering because events are discrete game results, not deltas. The latest score just accumulates.
        
    
3.  **Crash recovery:**
    
    -   Ingestion commits Kafka offsets only after successful processing.
        
    -   If a worker crashes before commit, Kafka will replay the event. The deduplication check ensures idempotency.
        
    
4.  **Concurrency safety:**
    
    -   Partitioning by user ID means only one ingestion worker updates a user’s score. Redis being single-threaded per shard further enforces serialized writes.
        
    
5.  **Durability:**
    
    -   Kafka retains events until successfully processed.
        
    -   Redis persistence (AOF or RDB) plus replication protects against node failures.
        
    

### Deep Dive 3 – Scaling Friends Leaderboard

1.  **Small friend lists:**
    
    -   If a user only has tens of friends, I fetch scores on demand.
        
    -   I get the friend IDs, pipeline the score lookups in Redis, and sort the results in memory.
        
    -   The tradeoff here is slightly higher read cost, but it’s still cheap enough, and it avoids any fan-out writes or extra caching complexity.
        
    
2.  **Large friend lists:**
    
    -   When users have hundreds or thousands of friends, on-demand lookups become too slow.
        
    -   In that case, I precompute a cached friends leaderboard.
        
    -   Every time a user’s score changes, we fan out that update to their friends’ cached leaderboards.
        
    -   Now, writes are more expensive, but reads become nearly free—you just fetch the leaderboard directly.
        
    

Collapse Description

5

12

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 21 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cmdyfcg3g0063ad39war70ntv)

Nice design! You've built a thoughtful architecture for a gaming leaderboard that handles the core requirements well. Your use of Kafka for ingestion with Redis sharding by user ID is a solid approach, and I particularly appreciate your attention to durability and correctness with deduplication mechanisms. However, there are some critical issues around global consistency and the "20 surrounding players" feature that need addressing to meet the stated requirements.

1.  **Positives**
    
    -   Smart sharding strategy by user ID instead of score - this avoids expensive rebalancing operations when scores change and provides predictable performance for user lookups
    -   Well-thought-out durability approach using Kafka as source of truth with Redis persistence (AOF + RDB) and replay mechanisms for recovery
    -   Good handling of at-least-once delivery with deduplication using Redis processed sets and transactional updates to prevent double-counting
2.  **Potential Issues** **Global Top 10 Consistency Violation**
    
    -   You're caching the global top 10 with a 500ms TTL, but requirements specify "global leader should be strongly consistent"
    -   This creates a window where the displayed top 10 could be stale, showing incorrect leaders
    -   Strong consistency requires real-time accuracy for global leaders, not eventual consistency with caching
    
    **Broken "20 Surrounding Players" Feature**
    
    -   Your design admits this is "per-shard, not globally accurate" which fundamentally breaks the requirement
    -   Users expect to see their true global neighbors, not just whoever happens to be in their shard
    -   This makes the feature essentially useless as users in different shards can't compare positions accurately
    
    **Missing Friends Data Flow**
    
    -   The friends leaderboard API is defined but there's no clear data flow showing how friend scores are aggregated
    -   With user-based sharding, friends will be distributed across different shards requiring complex fan-out
    -   No explanation of how this cross-shard query performs within the 500ms latency requirement
3.  **Follow-up Questions**
    
    -   How do you efficiently find the 20 globally surrounding players when users are sharded by ID rather than score?
    -   What happens when a single Redis shard fails - how do you maintain availability for those 100K users?
    -   How does the friends leaderboard query perform when a user has 1000+ friends distributed across all shards?
    -   Why use Kafka for 1-5K events/sec when Redis can handle this directly with much lower latency?
    -   How do you handle the race condition between score updates and leaderboard queries during the Kafka processing delay?

_Warning: This feedback was AI generated and may contain errors._

Show more

2

Reply

![Priyankar Raj gupta](https://lh3.googleusercontent.com/a/ACg8ocLj4znexnJYoaFwdkTmM26gju9vXeJeZHeGkBO0YPITob8d3Rsl=s96-c)

Priyankar Raj gupta

[• 20 days ago• edited 20 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cmdyhg6z206r4ad08sar4ujbz)

how will a user gets his rank ? would not it require scatter and gather ?

Show more

0

Reply

N

Noam

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cmdyi7cez06wrad08qd7cywa0)

Right. Getting a user's global rank of a wasn't part of the requirements, but you could roughly estimate it by taking it's place in that specific shard X number of shards :) For an accurate results you'd need a different set of tradeoffs I think.

Show more

1

Reply

![Shanc](https://lh3.googleusercontent.com/a/ACg8ocJBzCqumXHIHmw1uJgZu9b-hYbCn45YlDlhiveYiE_dMIG7DA=s96-c)

Shanc

[• 19 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cme0me84s0150ad08ij68n2z0)

Nice design and well thought out. I had a similar question in my interview and I used an Amazon SQS instead of Kafka with a redis sorted set for top 10 leaderboard calculation(and a fan out approach for friends relative to user). The interview went through, but I had my doubts on scaling the fan out approach. Your solution is very similar. Thanks for sharing.

Show more

0

Reply

N

Noam

[• 19 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cme0z7bs005ioad0825xc1cj6)

Thank you. I hope your interview went well! Anything you'd change in this design based on your interview?

Show more

1

Reply

![Shanc](https://lh3.googleusercontent.com/a/ACg8ocJBzCqumXHIHmw1uJgZu9b-hYbCn45YlDlhiveYiE_dMIG7DA=s96-c)

Shanc

[• 19 days ago• edited 19 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cme0zfrr005l1ad08wxy0db52)

I would not change much, but would elaborate more on the Friends top list one and how fan out is required to get the computation done. Also, I did add API's to get realtime info. That was a sticky point with the interviewer and how frequently the client (devices would receive real-time updates). It felt like the interviewer wanted me to use the WS route for having real-time dashboards as tabbed options with Top 10 (global) and 20 ( with friends above and below) showing their rank for connected devices.

Show more

0

Reply

N

Noam

[• 19 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cme10og3f05wtad089kofv2i2)

Yeah, it's missing some info there. Seems like SSE wouldn't have better. Since you don't need bi-directional communication.

Show more

0

Reply

S

slowtimetraveler

[• 15 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cme665arp0p9uad074uepior2)

also if when the user shares their location when playing the game, you can make this info be a part of their 'heartbeat' record as well. then with Redis' geospatial index capabilities you can run a query to find people in the same area and find 10 better than given user and 10 worse (by top score). yeah?

Show more

0

Reply

S

slowtimetraveler

[• 15 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cme64zrur0om7ad08vy78xpll)

Would it be feasible, you think, to maintain a pre-calculated list of friends tops scores in the distributed cache for each user? If the social aspect is one of the key aspects of the game, then maintaining a 'heartbeat' record for each user with 'last time online, current total hours, etc.' is maybe beneficial. then assuming that gamers have a reasonable number of friends (<128) we can maintain the list of friends top scores for each user in that 'heartbeat record'. Use graph DB for user connections to identify friends quickly and update friends on 'my new high score'. What do you think?

Show more

0

Reply

![Niilesh Raaje](https://lh3.googleusercontent.com/a/ACg8ocLpFPjUAiYziqFLvF41hrg4lc2anmBMP6kuv2H2_3-7szjt127u=s96-c)

Niilesh Raaje

[• 6 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cmej3r8u001klad08j8k5yjee)

Was this question asked for Infra or product track?

Show more

0

Reply

![Shanc](https://lh3.googleusercontent.com/a/ACg8ocJBzCqumXHIHmw1uJgZu9b-hYbCn45YlDlhiveYiE_dMIG7DA=s96-c)

Shanc

[• 6 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cmejnbncs04lpad07mvzylxcc)

For me, it was an infra track question and the scatter/gather approach for neighbor dashboard (for friends) was tricky and definitely was not convincing. I am not sure about @Noam.

Show more

0

Reply

![Niilesh Raaje](https://lh3.googleusercontent.com/a/ACg8ocLpFPjUAiYziqFLvF41hrg4lc2anmBMP6kuv2H2_3-7szjt127u=s96-c)

Niilesh Raaje

[• 5 days ago](https://www.hellointerview.com/community/submissions/cmdyfc18q06fiad08o0wnw97a#comment-cmek635di01wuad08jek5z5ik)

Did you get the offer?

Show more

0

Reply