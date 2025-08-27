##### Design a Web Crawler with Limited Communication

by Noam• Staff• 21 days ago

This crawler design uses 10,000 distributed workers, each responsible for URLs based on hash(url) % N (consistent hashing), eliminating any single point of failure. Each worker maintains a local Frontier queue for scheduling URLs and a Sites table to track crawl status, deduplication, and retry logic. When a worker encounters a URL it doesn’t own, it forwards it to the correct worker, which then adds it to its own Frontier.

### **Deep Dive 1: Resilience & Durability**

-   A few things can go wrong:
    
    -   Website could be down → need retries
        
    -   Crawler could crash → need backup node handling
        
    
-   **Retry Handling**:
    
    -   Add a retry table: url, last\_runtime, retry\_count
        
    -   When a fetch fails, mark as done in frontier but add to retry table
        
    -   next\_run is calculated using exponential backoff based on retry count
        
    -   Limit max retries (e.g., 5 or 10)
        
    -   Failed URLs after limit can be logged
        
    
-   **Retry Execution**:
    
    -   Background process checks retry table
        
    -   If next\_run has passed → add back to frontier
        
    -   next\_run is a _minimum_, actual run may be later
        
    
-   **Parsing Separation**:
    
    -   Downloading, parsing, and extracting links were originally in one step
        
    -   Separated into two steps so if parsing fails, don’t re-download
        
    -   New parsing queue: stores link to interim file + needed metadata
        
    -   Second process parses HTML → saves text to file system → extracts links
        
    
-   **Backup Crawler Assignment**:
    
    -   Each crawler has a backup crawler number
        
    -   If sending a URL to a crawler and it fails:
        
        -   Retry once
            
        -   If still fails → use backup hash function → send to backup crawler
            
        
    -   Backup crawler stores URL in its own frontier
        
    
-   **File Durability**:
    
    -   Need a way to back up files in case a node’s storage goes down
        
    -   Solution: after saving locally, compress file and send to backup crawler
        
    -   Now file is saved in two places
        
    

* * *

### **Deep Dive 2: Politeness**

-   Need to respect **robots.txt** and rate limits
    
-   **Robots.txt**:
    
    -   Can be stored in sites table or in a dedicated robots table
        
    -   Keyed by domain
        
    -   Contains:
        
        -   Limitations (disallowed paths)
            
        -   Crawl-delay / rate limits
            
        
    
-   **Rate Limit Enforcement**:
    
    -   When crawling, check robots data for domain rules
        
    -   Keep in-memory structure (or local Redis) for domains with TTL
        
    -   Before fetching: try to set TTL key for domain
        
        -   If not possible → domain hit rate limit → re-add URL to queue
            
        
    -   Initially: planned to put back into frontier
        
    -   Final decision: put into retry table
        
        -   **Do not** increment retry count because it’s not a failure
            
        
    

* * *

### **Deep Dive 3: Scaling & Efficiency**

-   **Duplicate Content Avoidance**:
    
    -   Before saving page, hash the content
        
    -   Store hash in sites table
        
    -   If hash already exists → skip parsing
        
    -   Still must download before hashing
        
    -   This optimization saves parsing work, not download time
        
    
-   **Depth Tracking**:
    
    -   Each time following a link: depth = parent\_depth + 1
        
    -   Store depth in sites table
        
    -   Avoids spider traps (links leading to more links in deep loops)
        
    -   Helps prevent certain domains from consuming too many resources
        
    
-   **Skew Awareness**:
    
    -   Domain space can be skewed due to hashing
        
    -   Certain hash ranges can have more or heavier sites
        
    -   Acknowledged, but not addressed at this stage
        
    

Collapse Description

4

4

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 21 days ago](https://www.hellointerview.com/community/submissions/cmdy7pqug056dad07octdds1a#comment-cmdy7q53u006ead3csyllvt8i)

Nice design! The decentralized approach using consistent hashing to distribute URLs across 10,000 workers is clever and eliminates single points of failure. The use of a secondary hash function for handling offline nodes shows good thinking about fault tolerance, and the local SQLite databases with Frontier queues demonstrate understanding of distributed system design principles.

1.  **Positives**
    
    -   Consistent hashing with hash(url) % N effectively distributes crawling responsibility across workers without centralized coordination
    -   Secondary hash function fallback mechanism provides fault tolerance when nodes go offline
    -   Local SQLite databases with Sites table for deduplication and retry logic keeps each worker autonomous
2.  **Potential Issues** **Missing Politeness Controls**
    
    -   No implementation of rate limiting per domain or robots.txt checking
    -   This will cause the crawler to overwhelm websites and violate crawling etiquette
    -   The requirements specifically state "Politeness: Adhere to robots.txt and rate-limit per domain"
    
    **Inefficient URL Distribution**
    
    -   Using simple modulo hashing (hash(url) % N) will cause severe load imbalance since URLs from the same domain will scatter across all workers
    -   This prevents effective per-domain rate limiting and causes redundant DNS lookups
    -   Domain-based partitioning would allow proper rate limiting and better cache locality
    
    **SQLite Performance Bottleneck**
    
    -   SQLite with single-writer limitation will bottleneck at ~100-1000 writes/second
    -   Each worker needs to process 100 URLs/second (10B pages / 5 days / 10k workers)
    -   The Frontier queue implementation using ORDER BY on every dequeue will become extremely slow as the queue grows
    
    **No Duplicate Detection Across Workers**
    
    -   Each worker only checks its local Sites table for deduplication
    -   The same URL could be crawled by multiple workers if discovered through different paths
    -   This wastes resources and violates the deduplication requirement
3.  **Follow-up Questions**
    
    -   How do you implement per-domain rate limiting when URLs from the same domain are scattered across all 10,000 workers?
    -   What happens when a worker's SQLite database grows beyond available disk space?
    -   How do you handle the initial seed URLs and ensure work begins distributed across workers?
    -   How do you monitor crawl progress and identify workers that have stalled or failed?
    -   What's your strategy for parsing and normalizing URLs to ensure consistent hashing?
    -   How do you handle redirects that might cause the same content to be crawled under different URLs?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

N

Noam

[• 21 days ago• edited 21 days ago](https://www.hellointerview.com/community/submissions/cmdy7pqug056dad07octdds1a#comment-cmdy89lt605anad07exouqyqn)

> Using simple modulo hashing (hash(url) % N) will cause severe load imbalance since URLs from the same domain will scatter across all workers

Actually we are modding URLs not domain. So it's evenly distributed.

> The same URL could be crawled by multiple workers if discovered through different paths

A URL will only be crawled by it's assigned crawler (based on hash).

Show more

0

Reply

![Priyankar Raj gupta](https://lh3.googleusercontent.com/a/ACg8ocLj4znexnJYoaFwdkTmM26gju9vXeJeZHeGkBO0YPITob8d3Rsl=s96-c)

Priyankar Raj gupta

[• 21 days ago• edited 21 days ago](https://www.hellointerview.com/community/submissions/cmdy7pqug056dad07octdds1a#comment-cmdyhmkq906rxad089dbqe6vf)

I think robot.txt limit is per domain not URL. So if you visiting multiple URLs of same domain within certain time from multiple workers then its not following robot.txt rules. We should hash by domain not URL.

Show more

1

Reply

![Priyankar Raj gupta](https://lh3.googleusercontent.com/a/ACg8ocLj4znexnJYoaFwdkTmM26gju9vXeJeZHeGkBO0YPITob8d3Rsl=s96-c)

Priyankar Raj gupta

[• 21 days ago• edited 21 days ago](https://www.hellointerview.com/community/submissions/cmdy7pqug056dad07octdds1a#comment-cmdyhoq4006sdad08zej7qmxi)

What about same content, diff. URLs can have same content, are you storing duplicate contents all over nodes ?

Show more

0

Reply