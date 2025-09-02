##### Design a Global IP Address Blocking System

by SpotlessPurpleDinosaur885• Manager• 9 days ago

Design a distributed system that blocks requests from IP addresses globally. The system should adhere to a list of blocked IP addresses provided by various governments and ensure that access is restricted globally. The system should be scalable and handle updates to the blocked IP list efficiently.

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

[• 9 days ago](https://www.hellointerview.com/community/submissions/cmefluff901lead07npxuja43#comment-cmeflvkj20016ad3934bdjr0n)

Nice design! You've built a multi-layered blocking system that smartly uses WAF at the edge to reject requests early, combined with Redis for fast lookups and DynamoDB for durability. The use of S3 snapshots for version control and Kafka for propagation shows good thinking about distributed state management, though there are some scaling calculations and consistency mechanisms that need refinement.

1.  **Positives**
    
    -   Using WAF at the edge to block IPs before they hit your infrastructure is an excellent architectural choice that reduces load and improves security
    -   Multi-tier storage with Redis (microsecond lookups) backed by DynamoDB provides good balance of speed and durability
    -   S3 snapshots organized by version/geo (s3://ips/v-date/geo/fr.json) enable rollback and regional policy management
2.  **Potential Issues** **Redis capacity severely underestimated**
    
    -   You calculated 25 Redis nodes for 5M RPS globally (200k RPS per node)
    -   Modern Redis handles ~100k ops/second per instance maximum
    -   You'd need at least 50-100 Redis nodes to handle 5M RPS reliably
    
    **Unrealistic write volume requirements**
    
    -   You're designing for 1M writes/second but only storing 100k-1M total IPs
    -   This means rewriting the entire dataset every second, which doesn't match real blocking list update patterns
    -   Government IP blocklists typically update hourly/daily, not millions of times per second
    
    **Unclear global consistency model**
    
    -   No clear mechanism to ensure all regions have consistent blocklists during the 30s propagation window
    -   Could result in IP being blocked in one region but allowed in another
    -   Missing conflict resolution strategy if different governments block/unblock same IP
3.  **Follow-up Questions**
    
    -   How do you handle the authentication flow when Lambda is positioned after WAF in your design?
    -   What happens when different governments provide conflicting rules for the same IP range?
    -   How do you ensure Redis nodes in different regions stay synchronized within the 200ms regional update requirement?
    -   What's your strategy for handling CIDR blocks efficiently in Redis for fast lookup?
    -   How do you prevent the S3 snapshot size from growing unbounded with version history?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

S

SpotlessPurpleDinosaur885

[• 9 days ago• edited 7 days ago](https://www.hellointerview.com/community/submissions/cmefluff901lead07npxuja43#comment-cmefmcgtp01pcad07oyyiluzk)

-   correction: 50-100 redis nodes acreoss sharded clusters
-   correction: real blocklists update at most hourly, so 1M writes/second is a ceiling , reduce to a 10-50
-   correction: during 30s window, regions can disagree. we need to track last applied and include propagation lag alarms if a region is behind

How do you handle the authentication flow when Lambda is positioned after WAF in your design?: WAF blocks traffic early. Lambda validates authenticated context later.

What happens when different governments provide conflicting rules for the same IP range?: rule based on jurisdiction, and resolve based on precednence too. tag entries to global or scoped to a region

How do you ensure Redis nodes in different regions stay synchronized within the 200ms regional update requirement?: Push policy snapshots via CDN/Cloudflare KV to pre-stage edge data or use redis global datastore and put alarms if apply time > 200ms from publish time

What's your strategy for handling CIDR blocks efficiently in Redis for fast lookup?: compressed trie stored as binary or use flat hash

How do you prevent the S3 snapshot size from growing unbounded with version history?: through retention policy and lifecycle rules. we can use glacier for more retention

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 days ago](https://www.hellointerview.com/community/submissions/cmefluff901lead07npxuja43#comment-cmeit7vly01e9ad082q9depcn)

FYI! You can respond to the bottom comment above directly and it will keep responding to you, answering questions, etc. Could just copy this reply up to directly reply to it.

Show more

0

Reply