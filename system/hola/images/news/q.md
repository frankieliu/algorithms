# Question 1 of 15

Why do news aggregators like Google News download and store their own copies of publisher thumbnails rather than linking directly to publisher images?
1
Consistent load times and availability
Correct!

Publisher images can be slow to load, change URLs, or become unavailable, which would break the feed experience. Storing local copies ensures consistent load times and prevents broken images from disrupting the user experience across millions of daily users.

2
Copyright protection

3
SEO optimization

4
Image compression

# Question 2 of 15

Cursor-based pagination prevents duplicate results when new data is added during user browsing sessions.
1
True
Correct!

Cursor-based pagination uses stable reference points (like timestamps or IDs) rather than offset calculations. When new data is added above the cursor position, it doesn't affect subsequent pages, eliminating the pagination drift problem common with offset-based approaches.

2
False

# Question 3 of 15

Why might implementing webhooks from publishers be preferred over frequent RSS polling for breaking news delivery?
1
Webhooks automatically handle content categorization better than RSS parsing

2
RSS feeds are inherently less secure than webhook endpoints

3
Webhooks eliminate the need for any fallback mechanisms

4
Webhooks provide immediate notification when content is published, reducing discovery latency from minutes to seconds

Correct!

Webhooks provide push-based notifications that allow publishers to immediately notify the news aggregator when new content is published, reducing discovery latency from the polling interval (5-10 minutes) to seconds. This is crucial for breaking news where every minute matters. While webhooks require publisher cooperation and need fallback mechanisms, they provide the fastest possible content discovery for participating publishers.

# Question 4 of 15

When implementing personalized news feeds, why might 'pre-computed user caches' be worse than 'dynamic feed assembly' despite being faster?
1
Security vulnerabilities

2
Memory costs explode with millions of users

3
Both: Memory costs explode with millions of users AND Cache invalidation complexity
Correct!

Pre-computed user caches require enormous additional memory and introduce complex cache invalidation logic. Together these two factors make them worse than dynamic assembly despite faster reads.

4
Cache invalidation complexity

# Question 5 of 15

Auto-scaling groups automatically adjust server capacity based on demand metrics.
1
True
Correct!

Auto-scaling groups monitor metrics like CPU utilization, memory usage, or request count and automatically provision or terminate instances based on predefined thresholds. This enables systems to handle traffic spikes while controlling costs during low-traffic periods.
2
False

# Question 6 of 15

Why do news aggregators implement Change Data Capture (CDC) instead of simple database polling for cache updates?
1
Lower storage costs

2
Sub-second cache freshness without overwhelming the database
Correct!

CDC triggers cache updates immediately when new articles are inserted, providing sub-second freshness. Database polling would require constant queries that scale poorly, while CDC scales effortlessly and maintains real-time cache synchronization without database load.

3
Better security

4
Easier implementation

# Question 7 of 15

During a major election, your Redis cache serving 100M users gets overwhelmed at 100k requests/second. What's the BEST immediate scaling solution?
1
Upgrade to larger Redis instance

2
Switch to database queries

3
Implement cache sharding

4
Add read replicas to distribute query load
Correct!

Read replicas are the fastest solution since each replica can handle ~100k requests/second. With just 10 replicas, you can handle 1M requests/second. Cache sharding takes longer to implement, and database queries would make the problem worse.

# Question 8 of 15

A system's database must serve 100,000 read requests per second. Which scaling approach handles this load most effectively?
1
Single database instance

2
Vertical scaling only

3
Read replicas
Correct!

Read replicas distribute query load across multiple database instances, each capable of handling a portion of the 100k requests. This is the most effective approach for read-heavy workloads, as vertical scaling has limits and write sharding doesn't help with read throughput.

4
Write sharding

# Question 9 of 15

A news publisher's RSS feed is down for 2 hours during breaking news. What's the BEST fallback strategy?
1
Scrape their website directly
Correct!

Web scraping provides immediate access to new content when RSS feeds fail. News aggregators maintain scraping capabilities specifically for these scenarios, since missing breaking news from major publishers significantly impacts user experience and competitive positioning.

2
Wait for RSS to recover

3
Skip this publisher temporarily

4
Use cached content only

# Question 10 of 15

What happens when cached data becomes stale in high-frequency update systems?
1
Performance improves

2
Database load decreases

3
Users see outdated information
Correct!

Stale cached data means users receive outdated information that doesn't reflect recent changes. While cache staleness can improve performance and reduce database load, it creates a trade-off between speed and data freshness that must be carefully managed.
4
Response times are faster

# Question 11 of 15

All of the following improve content freshness EXCEPT:
1
Cache TTL increase
Correct!

Increasing cache TTL (time-to-live) actually reduces content freshness by keeping stale data longer. The other options all improve freshness: webhooks provide immediate notifications, frequent polling discovers content faster, and CDC triggers real-time updates.

2
Real-time webhooks

3
Change data capture

4
Frequent polling

# Question 12 of 15

Geographic data distribution reduces latency by serving content from nearby locations.
1
True
Correct!

Geographic distribution leverages the principle of data locality - serving content from locations physically closer to users reduces network round-trip time. This is fundamental to CDN design and regional deployment strategies for global applications.

2
False

# Question 13 of 15

During traffic spikes, which component typically becomes the bottleneck in read-heavy systems?
1
Database or cache layer
Correct!

In read-heavy systems, the database or cache layer typically becomes the bottleneck first because data retrieval operations are more resource-intensive than request routing or basic processing. Application servers and load balancers can usually be scaled more easily than data layers.

2
Load balancers

3
Application servers

4
Network bandwidth

# Question 14 of 15

Eventual consistency is acceptable for news feeds where availability matters more than perfect synchronization.
1
True
Correct!

According to the CAP theorem, news aggregation systems often prioritize availability over strict consistency. Users prefer access to slightly outdated content rather than no content at all, making eventual consistency an appropriate choice for this use case.

2
False

# Question 15 of 15

When implementing category-based news feeds (Sports, Politics, Tech), which approach provides the best balance of performance and resource efficiency?
1
Store each category in separate Redis caches (feed:sports:US, feed:politics:US)

2
Use machine learning to predict user category preferences dynamically

3
Cache complete article metadata in regional feeds and filter in-memory by category

Correct!

In-memory filtering provides the best balance by storing complete article metadata in regional caches (like feed:US) and filtering by category when requested. This avoids the memory explosion of separate category caches (25 categories × 10 regions = 250 cache keys) while maintaining sub-200ms performance. Reading 1,000 articles from Redis takes ~10ms, and in-memory filtering adds only 1-2ms, making it much faster than real-time database queries while using significantly less memory than pre-computed category caches.

4
Query the database with category filters in real-time for each request