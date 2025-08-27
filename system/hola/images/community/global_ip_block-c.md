##### Design a Global IP Address Blocking System

by OperationalSalmonRoadrunner706• Staff• 1 day ago

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

[• 1 day ago](https://www.hellointerview.com/community/submissions/cmeqkzdlk007qad08zmg0r07p#comment-cmeqkzysv000xad3aka0fy02t)

Nice design! You've identified the key challenge of distributing IP block lists globally and included smart components like CDN and notification queues for real-time updates. However, the current architecture has a critical bottleneck - checking every request against a centralized database will add significant latency and create a massive scaling challenge when handling millions of requests per second across the globe.

1.  **Positives**
    
    -   Using CDN for global distribution is smart - this helps get block lists closer to users worldwide
    -   Horizontal scaling of the API Gateway GIPB shows good understanding of load distribution needs
    -   Notification queue from government sources enables real-time updates beyond just cron job syncing
2.  **Potential Issues** **Inefficient Blocking Architecture**
    
    -   Every request goes through API Gateway to check against Block List database
    -   This adds 5-30ms latency to every single request and creates a massive bottleneck
    -   IP blocking should happen at the edge (CDN/firewall) not at application layer for performance
    
    **Block List Database as Bottleneck**
    
    -   All API gateways query the same Block List database for every request
    -   Even with caching, this database will hit limits at ~50k queries/second
    -   A global system handling millions of requests/second will overwhelm any centralized database
    
    **Unclear CDN Implementation**
    
    -   CDN is shown but requests still flow through API Gateway for blocking checks
    -   This defeats the purpose of edge blocking and adds unnecessary hops
    -   CDN should be enforcing blocks directly, not just serving content
3.  **Follow-up Questions**
    
    -   How do you efficiently store and query IP ranges/CIDR blocks (e.g., 192.168.0.0/16)?
    -   What happens to request latency when checking against millions of blocked IPs?
    -   How quickly can a newly blocked IP propagate to all global edge locations?
    -   How do you handle the Block List database being temporarily unavailable?
    -   What data structure would you use for fast IP lookups (Patricia trie, bloom filter)?
    -   How do you ensure consistency when different edge locations have different versions of the block list?

_Note: Consider moving IP blocking to the CDN/edge layer using efficient data structures, with the API Gateway only handling application logic. This would dramatically improve performance and scalability._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply