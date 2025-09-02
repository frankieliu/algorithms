##### Design a Distributed Logger

by Vardhan• Mid-level• 11 days ago

I’m designing a distributed logging service that supports wildcard querying and custom aggregations at 1-minute granularity, scaling to 1M+ events/sec.

Ingestion: Logs are ingested via a PUB/SUB pipeline (Kinesis/Kafka), allowing fan-out to multiple parallel writers and downstream aggregation services without overloading the database.

Storage: Using Amazon Timestream, where serviceId + timeBucket are modeled as dimensions. Timestream hashes these dimensions to route data into partitions. Each node buffers incoming records in LSM-tree–like memory structures, then flushes them in bulk to Parquet-like columnar files that store min/max time and dimension stats in their footers for efficient pruning.

Querying: This layout allows fast time-range queries over a given service or bucket without scanning irrelevant files.

Aggregation: Custom aggregations are performed in the streaming layer; Timestream is optimized for storage and retrieval, not heavy OLAP rollups in the hot path.

Evan / Stefan — from a scalability and maintainability standpoint, does this approach feel solid enough to productionize for a large-scale distributed logging service? Are there any trade-offs or bottlenecks you’d want addressed before moving forward?

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

[• 11 days ago](https://www.hellointerview.com/community/submissions/cmed82hot01head08gq6aycye#comment-cmed84io1001nad3ap46yl9xs)

Good job on this distributed logger design! Your choice of Amazon Timestream as the primary storage is excellent for time-series log data, and using Kinesis for ingestion provides the necessary scalability for 1M+ events/sec. The dimension-based partitioning strategy with serviceId and timeBucket shows solid understanding of distributed storage patterns. However, the dual-storage architecture with ElasticSearch introduces unnecessary complexity and consistency challenges that could impact your 20-30s eventual consistency requirement.

1.  **Positives**
    
    -   Using Kinesis/Kafka for ingestion handles 1M+ events/sec well with built-in partitioning and fan-out capabilities
    -   Amazon Timestream is purpose-built for time-series data with efficient columnar storage and time-based pruning
    -   Dimension-based hashing on serviceId + timeBucket provides good data distribution and query locality
2.  **Potential Issues** **Duplicate storage architecture**
    
    -   You're maintaining data in both Timestream and ElasticSearch, requiring synchronization via cron job
    -   The 1-minute cron job introduces up to 60s delay for regex queries, violating your 20-30s consistency requirement
    -   This doubles storage costs and creates potential consistency issues between the two systems
    
    **Unnecessary ElasticSearch dependency**
    
    -   Timestream natively supports pattern matching with LIKE operators for wildcard queries
    -   Adding ES just for regex creates operational overhead without clear benefit
    -   The additional indexing step could become a bottleneck as data volume grows
    
    **Single-log POST API bottleneck**
    
    -   Your API shows POST /logs/:serviceId for individual logs
    -   At 1M+ events/sec, single-log HTTP requests would require millions of connections
    -   No batch ingestion endpoint shown, which is essential at this scale
3.  **Follow-up Questions**
    
    -   How do you define and execute custom aggregations in the streaming layer? What aggregation framework are you using?
    -   Why add ElasticSearch when Timestream supports wildcard queries with LIKE operators?
    -   How do services batch logs for ingestion? Are they writing directly to Kinesis or through your API?
    -   What happens if the ES indexing cron job fails or can't keep up with incoming data rate?
    -   How are permissions cached to avoid hitting DynamoDB on every query request?
    -   How do you handle backpressure when Kinesis consumers can't keep up with producers?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply