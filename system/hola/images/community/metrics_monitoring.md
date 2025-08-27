##### Design Metrics monitoring and alert system

by shivadeepthi toopran• Senior• 16 days ago

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

[• 16 days ago](https://www.hellointerview.com/community/submissions/cme5d3ulh0gw5ad07w8w9m49w#comment-cme5d4tdm0017ad3blaixt492)

Nice design! You've built a comprehensive metrics monitoring system with good separation of concerns between collection, processing, storage, and alerting. The use of Kafka for high-throughput ingestion and Flink for stream processing shows solid understanding of modern data pipeline architectures, and the reconciliation job demonstrates thoughtful consideration of data accuracy requirements.

1.  **Positives**
    
    -   Using Kafka as the central message bus is excellent for handling thousands of metrics per millisecond - it provides durability, scalability, and decouples collectors from downstream processing
    -   The dual-path architecture with both real-time aggregation (Flink) and batch reconciliation (S3 → cron job) ensures data accuracy while maintaining low latency for dashboards
    -   Separating the QuerySvc from the OLAP store provides a clean API abstraction and allows for caching/optimization without exposing storage implementation details
2.  **Potential Issues** **Missing Time-Series Database**
    
    -   You're using a generic OLAP store for metrics data, but metrics have unique access patterns (time-ordered, append-only, high cardinality)
    -   Generic OLAP systems will struggle with efficient storage and querying of millions of time-series, leading to slow dashboard loads and high storage costs
    -   Time-series databases like InfluxDB, Prometheus, or TimescaleDB are specifically optimized for this use case with compression, downsampling, and time-based indexing
    
    **Alert Evaluation Architecture**
    
    -   The CronJob checking AlertDB against OLAP for alerts will have significant latency - potentially minutes between metric threshold breach and notification
    -   This batch approach contradicts the millisecond collection requirement and could miss critical alerts during outages
    -   Real-time alerting should evaluate rules in the Flink streaming pipeline for immediate detection of threshold violations
    
    **Single CollectorSvc Bottleneck**
    
    -   All agents route through a load balancer to a single CollectorSvc, creating a potential bottleneck for thousands of servers sending metrics every millisecond
    -   Even with horizontal scaling, this synchronous collection pattern will struggle with network timeouts and backpressure
    -   Agents should write directly to Kafka or use multiple regional collectors to distribute load
3.  **Follow-up Questions**
    
    -   How do you handle metric cardinality explosion when monitoring thousands of servers with multiple metrics each?
    -   What happens when an agent can't reach the CollectorSvc - do metrics get buffered locally or lost?
    -   How does the dashboard query specific time ranges efficiently from the OLAP store without scanning massive amounts of data?
    -   What's the data retention policy and how do you downsample older metrics to manage storage costs?
    -   How do you ensure the AlertDB rules are evaluated quickly enough to meet real-time alerting requirements?
    -   What prevents duplicate alerts when both the streaming and batch paths detect the same threshold violation?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply