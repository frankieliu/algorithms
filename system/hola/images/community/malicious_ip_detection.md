##### Design a Malicious IP Detection System

by VoluntarySalmonCondor836• Mid-level• 4 days ago

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

[• 4 days ago](https://www.hellointerview.com/community/submissions/cmem5c74w05ukad075frxs9bh#comment-cmem5d0w90005ad3axgg59rzn)

Nice design! You've identified key components for a malicious IP detection system and made some solid technology choices like using a Radix Trie for efficient IP lookups and Redis for caching. However, there are some architectural concerns around separation of concerns and component responsibilities that would need refinement before this could handle production traffic at scale.

1.  **Positives**
    
    -   Using Radix Trie for IP lookup is an excellent choice - it provides O(k) lookup time where k is the IP length, perfect for IP prefix matching
    -   Redis cluster for caching frequent reads is appropriate given the 100:1 read/write ratio and helps achieve the <50ms latency requirement
    -   Geo-replication strategy for both Redis and the service recognizes the global nature of the problem and helps with latency requirements
2.  **Potential Issues** **API Gateway doing too much**
    
    -   Your API Gateway is handling authentication, rate limiting, AND malicious IP checking, creating a complex single point of failure
    -   This violates separation of concerns and makes the gateway a bottleneck that needs to handle 100k RPS while doing multiple operations
    -   The gateway should only route requests - IP checking should be a separate service layer
    
    **Kafka overkill for write path**
    
    -   With only 1k writes/second (100k reads/100), adding Kafka introduces unnecessary complexity and latency
    -   The 5-minute update tolerance doesn't require async processing - direct writes would be simpler and faster
    -   You're adding 10-50ms of latency for a use case that doesn't need event streaming capabilities
    
    **Unclear update propagation**
    
    -   How do updates flow from Kafka to the Radix Trie to Redis is not specified
    -   Without clear cache invalidation, you risk serving stale data beyond the 5-minute tolerance
    -   The system needs a clear mechanism to keep Redis and the Radix Trie synchronized
3.  **Follow-up Questions**
    
    -   How does the Radix Trie get updated when new malicious IPs are added via Kafka?
    -   What happens on a Redis cache miss - does it fall back to the Malicious IP Service?
    -   How do you handle the "random char to IP" for hot reads - what does this mean exactly?
    -   Why separate auth service calls for every request when authentication is only needed for writes?
    -   How do you ensure the 5-minute staleness requirement when using async Kafka processing?
    -   What's your strategy for keeping the geo-replicated instances in sync?

_Note: This feedback is based on standard hardware capabilities circa 2025, where a single server can handle significant loads before requiring complex distributed architectures._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

V

VoluntarySalmonCondor836

[• 4 days ago](https://www.hellointerview.com/community/submissions/cmem5c74w05ukad075frxs9bh#comment-cmem5nuab0008ad08gplees08)

I am replicating the gateway as well, so it shouldn't be a single point of failure. Yes realized it a bit late into the design that I should have separated Rate Limiter from the Load Balancer. I agree Kafka is an overkill in the current metrics, added it just as a backup in case if the rate limiter goes down and we are flooded with DDOS requests. Cache invalidation is done and we are setting a TTL in redis for every insert/update of an IP (so we can set that to max 5 mins).

Show more

0

Reply

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 4 days ago](https://www.hellointerview.com/community/submissions/cmem5c74w05ukad075frxs9bh#comment-cmem5otps0007ad3bjf96orv9)

Great clarifications!

-   Replicating the gateway helps availability, but the “doing too much” concern still stands. You’ll get cleaner ops if you slim the gateway to routing and put:
    
    -   WAF/rate limiting at the edge (or as an Envoy/NGINX filter)
    -   IP reputation check as a separate service or L7 filter/sidecar that can scale independently
    -   Auth only on the write/delete path
-   Kafka as a DDoS fallback: it won’t help much in the hot path. DDoS mitigation needs to drop traffic before it hits your app:
    
    -   Use provider DDoS/WAF (e.g., Cloudflare/AWS Shield) + L4 ACLs and per-IP/ASN throttling
    -   Local token-bucket limiters per instance with a lightweight shared state (Redis) for sliding windows; if Redis is down, degrade to local-only limits
    -   Circuit breakers/request shedding; default-closed for obviously abusive categories while staying default-open for known-good
    -   Queues help for write durability, but they don’t protect read paths from floods
-   TTL-based cache invalidation: good baseline, but tighten it to meet the 5-minute window reliably:
    
    -   Don’t rely only on TTL for deletes; push an immediate invalidate (Redis Pub/Sub/Streams or a lightweight fan-out) so regions drop the key right away
    -   Use short TTLs (e.g., 1–2 minutes) plus push invalidations to guarantee <5 minutes even with clock skew/replication lag
    -   Add negative caching (short TTL) to avoid hot-miss stampedes
    -   For indefinite blocks, avoid TTL expiry causing accidental allow; use TTL only for cache freshness, and explicit expiry if the block is time-bound
    -   Consider refresh-ahead for hot keys to avoid dogpiles
-   Keeping in-memory tries in sync: publish updates to all instances with an epoch/version. Instances apply deltas and periodically reconcile with a snapshot to avoid drift.
    

A couple of quick questions to steer the design:

1.  What’s your source of truth for the blocklist—DB, Redis, or the in-memory trie?
2.  Do entries have expirations, or are they indefinite by default?
3.  After you split rate limiting from the LB, where will the IP check live—edge filter/gateway, or per-service sidecar/library?

With those answers, I can suggest a crisp control-plane/data-plane flow that meets the 50 ms p95 and the 5-minute staleness bound.

Show more

2

Reply