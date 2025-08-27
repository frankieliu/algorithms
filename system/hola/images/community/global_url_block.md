##### Design a Global URL Blocking System

by Alex Ilyenko• Senior• 27 days ago

Design a distributed system that provides blocked URLs globally. The system should adhere to a list of blocked URLs provided by various governments and ensure that access is restricted globally. The system should be scalable and handle updates to the blocked URLs list efficiently.

Collapse Description

4

5

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdqanilm06cwad08ve2govf3#comment-cmdqanw1n009dad3iwf6zodst)

Great work on this system design! You've laid out a solid architecture for a global IP address blocking system that handles the key requirements. The use of a CDN, read/write services, and a combination of Redis caching and DynamoDB storage is a good approach.

1.  **Positives**
    
    -   Using a CDN with a 5-minute cache TTL effectively offloads the majority of read traffic
    -   Separating the read and write services allows optimizing each for their specific access patterns
    -   Leveraging Redis as a read-through cache with DynamoDB as the durable store is a proven pattern for read-heavy workloads
2.  **Potential Issues** **Unclear how IP blocking rules are propagated**
    
    -   It's not specified how the IP blocking rules get from the Write Service to the CDN
    -   If there's a delay in propagation, newly blocked IPs may still be accessible for some period
    -   Inconsistent rule enforcement could allow blocked IPs to bypass restrictions
    
    **No mention of rate limiting or abuse prevention**
    
    -   Without rate limiting, a single client could overwhelm the system with a high volume of requests
    -   Attackers could potentially use the service to determine which IPs are blocked in each country
    -   Lack of abuse prevention could lead to denial of service or excessive costs
    
    **Potential hot partition issue in Redis**
    
    -   Using the country as the Redis key could lead to hot partitions for countries with large IP lists
    -   A single Redis instance may not be able to handle the read volume for large countries
    -   Insufficient Redis memory could cause thrashing and performance degradation
3.  **Follow-up Questions**
    
    -   How do IP blocking rules get from the Write Service to the CDN and how quickly?
    -   What rate limiting is in place for clients accessing the APIs?
    -   How do you prevent abuse, such as clients crawling to determine all blocked IPs?
    -   What happens if a single country's IP list exceeds the memory capacity of a Redis instance?
    -   How are reads handled if Redis is unavailable or degraded?
    -   What monitoring and alerting is in place to detect issues like CDN stale data or Redis OOM?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

![Alex Ilyenko](https://lh3.googleusercontent.com/a/ACg8ocItEb6gSi3Y9bGtUXZZJw0xFkFJkFq1AIVmq95nRJ4wQcw3-W5l9g=s96-c)

Alex Ilyenko

[• 27 days ago• edited 27 days ago](https://www.hellointerview.com/community/submissions/cmdqanilm06cwad08ve2govf3#comment-cmdqdksas07f8ad085g8abldh)

Thanks! I realised I implemented URL blocking system, not IP one. So updated the description.

-   CDN will automatically check for update every 5 mins with ETag as versioning. If no updates, will update max-age. S3 will also make notifications on change so we provide pull and push updates. Client always trusts CDN as a source of truth.
-   Added rate limiter - nice suggestion! It will block by requester IP with 1 sec sliding window.
-   Included extra read replicas for Redis and hot partitioning.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdqanilm06cwad08ve2govf3#comment-cmdqdn4pp07o2ad08jkz2vhit)

You could also just key Redis off of a smaller granularity than country. This will allow a more even distribution :)

Show more

2

Reply

![Alex Ilyenko](https://lh3.googleusercontent.com/a/ACg8ocItEb6gSi3Y9bGtUXZZJw0xFkFJkFq1AIVmq95nRJ4wQcw3-W5l9g=s96-c)

Alex Ilyenko

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdqanilm06cwad08ve2govf3#comment-cmdqe3hm507mkad08itjcse7z)

Thanks Evan!

Show more

0

Reply