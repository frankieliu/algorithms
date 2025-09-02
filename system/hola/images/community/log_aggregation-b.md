##### Design a log aggregation Service

by AbstractGoldCatfish393• Manager• 9 days ago

Design a log aggregation service that handles 10TB/hr of data from various services. The system should support real-time data viewing for debugging, allow regex queries for exact data retrieval in real-time, and provide the analytics team with data from the past 24-48 hours in the form of a tar file.

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

[• 9 days ago](https://www.hellointerview.com/community/submissions/cmefcle9q087fad08qjzte1so#comment-cmefcm7a3001had3aeg8id7jd)

Good job on thinking about data segregation and using different storage tiers for recent vs historical data! Your design shows solid understanding of some distributed systems concepts, particularly around using WebSockets for real-time updates and CDC for search capabilities. However, the architecture faces significant challenges with the massive 10TB/hr scale that would prevent it from working in production without major revisions.

1.  **Positives**
    
    -   Separating recent (24h) and historical (24-48h) data into different tables is a smart optimization for access patterns
    -   Using CDC to ElasticSearch for regex queries shows good understanding of search infrastructure needs
    -   WebSocket connections for real-time log streaming provides appropriate low-latency updates to users
2.  **Potential Issues** **Lambda-based Log Collection Won't Scale**
    
    -   Using Lambda to continuously poll external services for 10TB/hr (2.8GB/s) of logs is fundamentally unworkable
    -   Lambda has 15-minute execution limits and would constantly timeout trying to pull this volume
    -   This polling approach would cost hundreds of thousands of dollars per month and still fail to keep up
    
    **ElasticSearch Would Collapse Under Load**
    
    -   Pumping 240TB/day into ElasticSearch via CDC from DynamoDB streams is far beyond reasonable ES cluster sizes
    -   A cluster capable of ingesting and searching this volume would require 100+ nodes and cost millions annually
    -   No mention of data retention, compression, or index optimization strategies that are critical at this scale
    
    **DynamoDB is Wrong Choice for Log Storage**
    
    -   At 10TB/hr with 1KB average log size, you're looking at ~2.8M writes/second to DynamoDB
    -   This would cost approximately $1.4M/month just in write capacity, not counting storage
    -   DynamoDB's 400KB item limit and lack of native compression make it unsuitable for log data
    
    **Missing Core Log Infrastructure**
    
    -   No log collection agents (like Fluentd/Filebeat) on source services to efficiently ship logs
    -   No message queue or streaming platform (Kafka/Kinesis) to buffer the massive data flow
    -   No mention of log compression which could reduce data volume by 10x or more
3.  **Follow-up Questions**
    
    -   How do the external services actually send their 2.8GB/s of log data to your system?
    -   What's your strategy for log compression and could you use columnar formats like Parquet?
    -   How would you partition the data to enable efficient time-range and service-based queries?
    -   Instead of on-demand TAR generation, could you pre-generate hourly archives in the background?
    -   Have you considered using S3 for cold storage with Athena for analytics instead of keeping everything in databases?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply