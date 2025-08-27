##### Design a Global IP Address Blocking System

by TechnologicalHarlequinRodent626• Manager• 10 days ago

Designing a Global IP Address Blocking System that extracts blocked IPs from government sources and restricts access involves several components to ensure scalability, real-time updates, low latency, and reliability.

1.  The rate limiter will be implemented as a separate service.
    
2.  The clients will primarily be internet service providers.
    
3.  The Government Agency will share IPs through a different channel or service.
    

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

[• 10 days ago](https://www.hellointerview.com/community/submissions/cmee09r6000xwad08iseywa5e#comment-cmee0bgf60014ad3b60f8182v)

Nice design! You've recognized the read-heavy nature of the workload and made some clever architectural choices like using CDN for distribution and Redis with consistent hashing for caching. The separation of read and write services shows good understanding of scaling patterns, and using CDC from DynamoDB to Redis is a solid approach for keeping caches updated.

1.  **Positives**
    
    -   Smart use of CDN to distribute IP block lists to ISPs, reducing load on central servers
    -   Good recognition of read-heavy workload (10M:1) and separating read/write services accordingly
    -   Using consistent hashing for Redis cluster to distribute load across 100 nodes effectively
2.  **Potential Issues** **Conflicting data distribution strategies**
    
    -   You have both CDN serving JSON files from S3 and Redis cache serving the same IP data
    -   This creates two separate read paths that could become inconsistent
    -   Maintaining two different data formats and update mechanisms adds unnecessary complexity
    
    **CDN approach won't scale for 1B IPs**
    
    -   1B IPs even in compact format would create massive JSON files (multiple GBs)
    -   CDN is designed for static content, but IP lists change frequently
    -   5-minute refresh cycle means very stale data compared to Redis which updates via CDC
    
    **Confusing dual database architecture**
    
    -   You show both "primary DB" and "DynamoDB" without explaining their relationship
    -   Unclear why you need two different database systems for the same data
    -   This adds operational complexity and potential consistency issues
3.  **Follow-up Questions**
    
    -   How do you handle IP range blocking (CIDR blocks) which is common for blocking entire subnets?
    -   What happens when an ISP needs to check an IP that's not in their regional JSON file?
    -   How do you ensure consistency between the CDN-delivered files and Redis cache data?
    -   Why use both a primary DB and DynamoDB instead of just one database system?
    -   How do you handle the storage requirement of 1TB across your 100 Redis nodes when each is sized for throughput not capacity?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply