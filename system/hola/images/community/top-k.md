##### Design a Top-K System

by Noam• Staff• 27 days ago

This takes a different approach then Stefan's solution. Sometimes called Lambda Architecture, it has fast path to quickly estimates views for the 1-minute, 5-minute, and 1-hour windows. The slow path uses Spark to compute accurate daily aggregates.

_EDIT: This needs to be rewritten to align with the typical interview question format. The current design targets high-throughput, trending scenarios—such as finding the top K trending hashtags. It still works for top K videos, but interview problems about videos are usually framed in a per-user context._

Collapse Description

13

17

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdotagky01msad08r7xepont)

I like it! This problem is interesting because there are _lots_ of potential solutions and interviewers are (if they're not bad!) more interested in your thought process than one specific solution.

I like the solutions on the right, that's a cool way to show (in this medium) the tradeoff discussion! One quick comment on Solution 3: If you're merging N shards each with the top K elements in each, you'll _always_ capture the global top K.

1.  How are you choosing between the fast and slow paths?
2.  Is the fast path being aggregated into the DB? How does that process work?

Show more

2

Reply

N

Noam

[• 27 days ago• edited 27 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdovtacn02rwad07n6k76web)

Thanks Stefan! Reading your questions, I realize the requirements need some clarifications.

There seem to be at least three types of top‑k systems:

-   **Trends**: Top songs or hashtags; predefined time buckets (hour/day/week/month/all‑time), near‑real‑time but not perfectly accurate.
-   **Competition (Leaderboards)**: Exact, always‑updated ranking for global or friend-based scores. Hoping to do that as soon as I get the chance!
-   **Analytics**: Any time range, fast approximate results with an optional slow exact path.

The fast/slow split applies to analytics, but I should also add more detail on how querying works (which I think is what you asked in 2).

Show more

0

Reply

N

Noam

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdpk3qnw004sad084owxp32y)

I've updated the design to answer your questions. Fast is for rolling windows of estimates (1m/5m/1h), which is ideal for showing trending videos. It is aggregated by a global worker that takes each worker's CMS and merges them into a single global count every 10 seconds. In contrast, slow is for flows that require accuracy, such as analytics or royalties and saved to s3.

Show more

0

Reply

![Sandesh Shivaram](https://lh3.googleusercontent.com/a/ACg8ocIPzfkuENRfBnHT6ocXkf4e0OevhYluQDxpMoJYdysfsl3Q1O43=s96-c)

Sandesh Shivaram

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdpzh24902r9ad0723zu6p09)

Thanks Noam for the solution! Quick question on your reasoning with not using Flink. Can we achieve the same with Flink by using secondary pipeline for aggregation or just directly emitting top 5000 to redis ZSET by continuosly pruning entries outside top 5000 in ZSET?

Show more

0

Reply

N

Noam

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdpzz84o02x7ad070zgqguhs)

Happy to help! What do you mean "secondary"? Flink uses RocksDB as a local db. These custom workers aggregate in-memory which is much more performant. Of course the tradeoff is having to implement checkpoints and event time processing (watermark), but to my understanding Flink is not used at google/meta/netflix in production for these custom aggregation. Another benefit that it's an interesting discussion to have during an interview, but I'm sure you could justify Flink if you come up with sound arguments.

Show more

0

Reply

![Sandesh Shivaram](https://lh3.googleusercontent.com/a/ACg8ocIPzfkuENRfBnHT6ocXkf4e0OevhYluQDxpMoJYdysfsl3Q1O43=s96-c)

Sandesh Shivaram

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdq33moh03stad07brj4whiv)

thanks, I meant another operator in the flink pipeline for aggregation (which takes previous topK as source). And we dont need rcoksdb here since our state just has CMS and top 5000 which fits into memory, we can just use external db (s3 )for checkpoint.

Also is redis/s3 okay to store all aggregated data and query from topKsvc? I think we might need ClickHouse(OLAP) ?

Show more

0

Reply

N

Noam

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdqxjmoy02m0ad0893gczwym)

Taking previous topK for redis you mean? I don't think it's needed because this is not a stream, just a simple aggregation. OLAP would be great if you need complex aggregation across dimensions and rollups. If we simply serving simple topK - seems like overkill.

Show more

0

Reply

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdqn9joh000fad3affw5leeh)

Nice design! You've created a thoughtful Lambda Architecture that cleverly separates real-time approximations from batch-accurate processing. The use of Count-Min Sketch for memory-efficient counting at scale shows strong understanding of probabilistic data structures, and your consideration of multiple worker alternatives demonstrates good architectural thinking. However, there are some critical consistency issues between the fast and slow paths that could lead to confusing user experiences, and the complexity of maintaining multiple aggregation systems may introduce operational challenges.

1.  **Positives**
    
    -   Using Count-Min Sketch with a min-heap is an elegant solution for memory-efficient top-K tracking at scale, allowing single machines to handle billions of videos with only ~1% overestimation
    -   The Lambda Architecture approach smartly balances real-time freshness with batch accuracy, giving users quick results while ensuring exact counts for critical use cases like licensing
    -   Keeping top 5000 videos per minute before narrowing to top 1000 prevents boundary videos from being lost due to Count-Min Sketch estimation errors
2.  **Potential Issues** **Inconsistent Results Between Time Windows**
    
    -   The fast path provides approximate counts for 1-minute, 5-minute, and 1-hour windows while the slow path provides exact daily counts
    -   Users could see a video with 1M views in the 1-hour window but 950K views in the daily window due to CMS overestimation
    -   This violates the "exact, not estimation" requirement and creates a confusing user experience where shorter windows show higher counts than longer ones
    
    **Missing Coordination Between Workers**
    
    -   Multiple workers independently push their Count-Min Sketch estimations to Redis every 5 seconds without coordination
    -   This can cause temporary inconsistencies where different workers have different top-K lists at the same moment
    -   The global aggregator might see incomplete or conflicting data when merging worker results, leading to videos flickering in and out of top-K lists
    
    **Undefined Behavior for Month and All-Time Windows**
    
    -   The design only addresses minute/hour (fast path) and daily (slow path) aggregations
    -   There's no clear strategy for computing monthly or all-time top-K, which are listed as functional requirements
    -   Aggregating 30+ daily Spark outputs for monthly views would be extremely slow and wouldn't meet the 100ms latency requirement
3.  **Follow-up Questions**
    
    -   How do you handle the transition when a day ends and the exact Spark counts replace the approximate Redis counts? Won't users see sudden jumps in view counts?
    -   What happens when Redis fails or needs maintenance? Do all real-time queries fail, or is there a fallback mechanism?
    -   How do you ensure the 100ms latency requirement for monthly and all-time queries when these would require aggregating many daily results from S3?
    -   Why maintain both exact and approximate systems when the requirement explicitly states "exact, not estimation"? Wouldn't a single accurate system be simpler?
    -   How do you prevent the same view event from being double-counted if it arrives at multiple workers due to Kafka partition rebalancing?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

M

ModerateMoccasinHawk364

[• 26 days ago• edited 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdqukqgp01rsad09599gftp3)

I don't think sliding time windows can be handled using count-min sketch because the data structure does not allow decrementing counts (it only supports the increment operation).

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdqunqrj01plad082srw9cnt)

It works for turnstile operations where you only decrement your increments (which is what we have here!).

Show more

1

Reply

M

ModerateMoccasinHawk364

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdquye9w01whad09zyoethup)

That makes sense, thinking about it more, we could have a sketch for the decrement operation, and use difference from increment-sketch and decrement-sketch as the final count (I am assuming this is what you mean, or something similar).

Show more

0

Reply

M

ModerateMoccasinHawk364

[• 26 days ago• edited 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdqv6nip01vhad08szhnvkio)

> This design allows a single machine to handle all the counting.

By this statement, I believe you are talking about the memory efficiency gained through the count min sketch data structure.

However, based on the constraint (70 billion views per day = ~700k TPS)- the bottleneck is not the memory, but the traffic TPS. A single machine probably won't be able to handle such high TPS.

If memory is the bottleneck, then count-min-sketch does provide more value in avoiding the partitioning/sharding.

However, for our given constraint, we might still need to partition (unless we want to reduce TPS by rate-limiting consumption from Kafka, which risks building consumer lag and losing ability to show updates in real-time).

Please let me know what you think.

Show more

0

Reply

N

Noam

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdqxbpq102svad08eoqkzupf)

Right. You are correct. The description below was out-of-date with the graphics. We do have multiple workers writing to Redis, then a single aggregator will merge the sketches. I've updated the text. Thanks!

Show more

0

Reply

P

PreciseBlackPanther893

[• 23 days ago• edited 23 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdvdnt3m0iphad08vc7ghwj9)

Can you elaborate how many keys will be stored in Redis (how many 1m,5m,1h) and can we improve by keeping just last 10(1 min) windows and combine into 5 min and combine 5 min into hourly by the aggregator anyways the fast path is approximate?

Show more

0

Reply

E

EconomicOrangeCicada244

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cmdzux3pi0ar0ad070l31v0yn)

Why not save the logs in cassandra and then use cdc or kafka with apache flink and sink it to redis for topK entries ?

Show more

0

Reply

N

Noam

[• 18 days ago](https://www.hellointerview.com/community/submissions/cmdoh8yot08l8ad08wuw5tfjb#comment-cme1flaub08tqad08ygd9bzo4)

I didn't want to use apache flink because we are doing custom CMS calculation, I need the overhead of flink here (there's enough explaining as it is). Also to my understanding in the real world (google/meta...) flink is not used for these trending calculations. But you can absolutely use it.

Show more

0

Reply