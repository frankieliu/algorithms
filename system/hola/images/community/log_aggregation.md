##### Design a Log Aggregation Service

by Manasi• Manager• 20 days ago

Picked up question from hello interview question stack as below--- Design a log aggregation service that handles 10TB/hr of data from various services. The system should support real-time data viewing for debugging, allow regex queries for exact data retrieval in real-time, and provide the analytics team with data from the past 24-48 hours in the form of a tar file.

10TB/hr of data being sent from a lot of services. there are 3 main flows -

1.  teams should be able to see the data in real time for debugging
    
2.  The team can get exact data by issuing a regex query to the logs in real time.
    
3.  The analytics team can look at data from 24 - 48 hrs, which hey receive in form of a tar file.
    

Collapse Description

5

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

[• 20 days ago](https://www.hellointerview.com/community/submissions/cme0g11vw00j8ad08asry9rjt#comment-cme0g1ij40321ad3c0ph1e8gi)

Nice design! You've demonstrated strong understanding of log aggregation patterns with appropriate technology choices like Kafka for ingestion, Flink for stream processing, and a hybrid storage approach. Your back-of-envelope calculations for throughput (2.78 GB/sec, 2.78M logs/sec) and partition sizing are spot-on, and the dual storage strategy with OLAP for real-time queries and S3 for analytics exports shows good architectural thinking. However, there are some critical gaps around regex query performance at this scale and unclear data flow paths that need addressing.

1.  **Positives**
    
    -   Excellent throughput calculations and Kafka partition sizing (50-100 partitions for 2.78 GB/sec) shows proper capacity planning
    -   Smart hybrid storage approach using OLAP DB for real-time queries with 48h retention and S3 for tar exports balances cost and performance
    -   Good use of Flink for stream processing with proper checkpointing strategy and inverted indexing for regex search support
2.  **Potential Issues** **Regex Query Performance at Scale**
    
    -   You're proposing regex queries on 480TB of data (48h retention) with <2s latency requirement
    -   Even with inverted indexes and bloom filters, regex on unstructured log messages at this scale will likely exceed latency SLAs
    -   The system needs either pre-aggregation, sampling, or time-bounded queries to meet the 2-second requirement
    
    **Unclear Real-time Viewing Architecture**
    
    -   The design shows Flink writing to OLAP DB, but real-time viewing typically needs sub-second latency
    -   There's no clear path for how live logs get from Kafka to viewing clients without going through the OLAP DB first
    -   This adds unnecessary latency for the debugging use case which needs immediate visibility
    
    **Missing Query Result Size Management**
    
    -   Regex queries could match millions of log lines from 480TB of data
    -   No pagination, result limiting, or streaming response mechanism shown
    -   This could cause OOM errors on both server and client side, or network timeouts
3.  **Follow-up Questions**
    
    -   How does the system handle regex queries that could potentially match millions of logs without overwhelming the client?
    -   What happens when the OLAP database becomes the bottleneck for concurrent regex queries from multiple teams?
    -   How do you ensure the tar file generation doesn't impact real-time query performance when processing 10-20TB files?
    -   Why use Redis with 1hr TTL for "frequent queries" when the OLAP DB should handle caching internally?
    -   How does the real-time viewing actually work - does it query OLAP or stream directly from Kafka?
    -   What's the actual data flow from services to the real-time viewing UI?

_Warning: This feedback was AI generated and may contain errors._

Show more

2

Reply