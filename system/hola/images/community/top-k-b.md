##### Design a Top-K System

by VariousCopperSturgeon771• Senior• 2 days ago

The approach is similar to the one explain in Hello interview write up but for a random window query instead of last 1min or 1hr window. I have written my whole thought process while designing the system.

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

[• 2 days ago](https://www.hellointerview.com/community/submissions/cmepzmdc400kvad07pgmhw4u0#comment-cmepzmz48002xad3b32d7y47b)

Nice design! You've shown good understanding of approximation algorithms and distributed systems concepts, particularly with the Count Min Sketch optimization and multi-granularity aggregation strategy. However, there are some fundamental issues with message handling and the design is significantly over-engineered for the stated scale of 10k messages per hour (only ~3 messages per second).

1.  **Positives**
    
    -   Using Count Min Sketch to reduce memory from 3TB to 250GB is a clever optimization that trades accuracy for space efficiency
    -   Multi-granularity aggregations (1min, 1hr, 1day, 1week) to reduce query-time computation from 50k to ~100 aggregations is smart
    -   Checkpointing to S3 with Kafka replay provides good fault tolerance and recovery capabilities
2.  **Potential Issues** **Lost Error Message Content**
    
    -   Hashing exception messages to messageId means you can't retrieve the actual error text in query results
    -   The API returns messages\[\] but you only store messageIds and counts in your system
    -   Users need to see actual error messages like "NullPointerException at line 42" not just hash values
    
    **Over-Engineering for Scale**
    
    -   10k messages/hour is only ~3 messages/second - a single server with an in-memory map could handle this
    -   Modern servers can easily handle 100k+ operations/second, so sharding across multiple servers is unnecessary
    -   The complexity of consistent hashing, Kafka partitions, and distributed aggregation isn't justified at this scale
    
    **Complex Time Window Management**
    
    -   Maintaining 50k separate 1-minute intervals (30 days worth) creates unnecessary complexity
    -   Rolling up older data (e.g., keep 1-min for 1 day, 1-hour for 1 week, 1-day for remainder) would be simpler
    -   Each query potentially touches thousands of data structures even with multi-granularity optimization
3.  **Follow-up Questions**
    
    -   How do you map messageIds back to actual error message text for the query response?
    -   What happens when clocks are skewed between different shards during time-based aggregation?
    -   How do you handle schema evolution during rolling deployments when shards have different versions?
    -   Why maintain individual 1-minute granularity for the full 30 days instead of progressively coarser granularity?
    -   How do you ensure consistent minute boundaries across distributed shards?
    -   What's the memory overhead of maintaining 50k Count Min Sketch instances per shard?

_Remember to validate scale requirements against modern hardware capabilities - a single server can often handle more than expected!_

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply