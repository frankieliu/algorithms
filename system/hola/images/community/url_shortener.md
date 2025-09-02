##### Design a URL Shortener

by 77teamqn• Senior• 3 days ago

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

[• 3 days ago](https://www.hellointerview.com/community/submissions/cmeobw2mn01whad08n4mtzlt0#comment-cmeobwmfz000dad3coxb6xqrx)

Nice design! You've made some solid architectural choices, particularly using a Redis counter with base62 encoding for generating unique short codes - this is an elegant solution that avoids collision checks. However, the design has some critical gaps around handling custom aliases, Redis failures, and security that would need to be addressed before production deployment.

1.  **Positives**
    
    -   Using Redis counter with base62 encoding for URL generation is clever - it guarantees uniqueness without collision checks and provides good performance
    -   Separating read and write services allows independent horizontal scaling based on the read-heavy workload pattern
    -   Adding Redis cache between application and database for frequently accessed mappings will significantly improve redirect latency
2.  **Potential Issues** **Single Point of Failure with Redis Counter**
    
    -   Your Redis counter is the only way to generate new short codes, creating a critical dependency
    -   If Redis crashes or loses data, you lose track of the counter value and risk generating duplicate short codes
    -   This violates your high availability requirement since write operations completely depend on Redis being up
    
    **Missing Custom Alias Collision Handling**
    
    -   Your API accepts custom aliases but there's no explanation of how you check if an alias is already taken
    -   Without proper uniqueness checks, multiple users could claim the same custom alias leading to data corruption
    -   This breaks the basic functional requirement of unique short URLs mapping to original URLs
    
    **No URL Validation or Security Measures**
    
    -   There's no mention of validating input URLs or preventing malicious redirects
    -   Users could create short links to phishing sites, malware, or internal network resources
    -   This creates serious security vulnerabilities that could damage user trust and system reputation
    
    **Incomplete Database Design**
    
    -   The schema lacks critical indexes on short\_url\_code for fast lookups during redirects
    -   No user authentication/management system despite "User" being listed as a core entity
    -   Missing details on how you handle URL expiration cleanup and prevent accessing expired links
3.  **Follow-up Questions**
    
    -   How does your system recover the counter value if Redis crashes and restarts?
    -   What happens when a user tries to create a custom alias that already exists?
    -   How do you handle URL expiration - do you actively delete expired entries or check on access?
    -   What's your strategy for analytics and tracking URL usage statistics?
    -   How do you prevent abuse like users creating millions of URLs or shortening malicious content?
    -   What happens to the counter when you need to scale across multiple Redis instances?
    -   How do you ensure the write service and Redis counter stay in sync during network partitions?

_Remember to address critical failure scenarios and security considerations in system designs - they're often as important as the happy path functionality!_

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply