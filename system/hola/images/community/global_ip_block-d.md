##### Design a Global IP Address Blocking System

by BalancedScarletKiwi244• Manager• 14 days ago

Assumption: We can not receive a block of banned IPs, we must reach out to a service for each IP and check with the service if the IP is banned or permitted

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

[• 14 days ago](https://www.hellointerview.com/community/submissions/cme938k3d044dad08t84493yb#comment-cme939m44004wad3923dh1dxh)

Nice design! You've created a well-structured IP blocking system with good separation of concerns and appropriate caching strategy. The decision to prioritize availability over consistency with a 5-minute staleness tolerance shows good understanding of the trade-offs, and your audit logging approach through Elastic provides the required traceability.

1.  **Positives**
    
    -   Using Redis for caching IP decisions with a 5-minute TTL is a smart choice that balances performance with freshness requirements
    -   Separating the Allow/Block Service from the main application creates good modularity and allows independent scaling
    -   Implementing comprehensive audit logging through Elastic ensures full traceability of all IP decisions
2.  **Potential Issues** **Cache Stampede Risk**
    
    -   When a popular IP's cache entry expires, multiple concurrent requests will all miss the cache simultaneously
    -   This causes a thundering herd of requests to the IP Permission Provider, potentially overwhelming it
    -   The 50ms p95 latency requirement could be violated during these stampedes
    
    **Single Point of Failure in Redis**
    
    -   The design shows a single Redis instance handling all cache operations
    -   If Redis fails, every request must hit the IP Permission Provider, likely causing cascading failures
    -   This violates the high availability requirement as the system degrades significantly without cache
    
    **Synchronous Blocking on IP Checks**
    
    -   The API Gateway appears to synchronously wait for the Allow/Block Service response before proceeding
    -   This adds latency to every request path, even for allowed IPs
    -   The blocking nature could cause thread exhaustion under high load
3.  **Follow-up Questions**
    
    -   How does the system handle batch IP lookups efficiently when the API supports multiple IPs per request?
    -   What happens when the IP Permission Provider is down or responding slowly?
    -   How do you prevent the API Gateway from becoming a bottleneck when it's handling both IP checks and audit logging?
    -   What's the strategy for cache warming to avoid cold start issues?
    -   How does the system ensure the 50ms p95 latency when cache misses require external API calls?

_Warning: This feedback was AI generated and may contain errors._

Show more

2

Reply