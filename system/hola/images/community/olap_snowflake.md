##### Desgin OLAP like Snowflake

by StrategicSilverGecko559• Staff• 17 days ago

Scope narrowed to only processing user queries and getting results. Its assumed that required data is ingested and available in the required format

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

[• 17 days ago](https://www.hellointerview.com/community/submissions/cme4ik12h08load08wv6jquc2#comment-cme4ikzn8003vad3b8s9issy8)

Nice design! You've created a solid foundation for an OLAP system with good separation of concerns between query submission, job execution, and distributed processing. Your use of Kubernetes for dynamic resource allocation and a coordinator-worker pattern for query execution shows understanding of how modern data processing systems scale. However, there are some critical issues around job reliability and resource management that would need to be addressed before this could handle production workloads.

1.  **Positives**
    
    -   Using Kubernetes for dynamic job execution is excellent - it provides automatic resource allocation, scaling, and isolation between different query workloads
    -   Separating query parsing/planning from execution with a queue-based architecture allows for better load management and prevents the API from blocking on long-running queries
    -   The coordinator-worker pattern with Zookeeper for distributed state management is a proven approach used by systems like Spark and provides good fault tolerance
2.  **Potential Issues** **Job Recovery and Durability**
    
    -   The design shows jobs being submitted to a queue but doesn't specify what happens when a job executor crashes mid-execution
    -   This means partially completed jobs could be lost, requiring users to resubmit queries and losing all computation progress
    -   This violates the durability requirement and would cause significant user frustration with long-running analytical queries
    
    **Resource Management and Multi-tenancy**
    
    -   There's no mention of resource limits or quotas for Kubernetes deployments launched by the job executor
    -   A single large query could consume all cluster resources, starving other users' queries
    -   This breaks the "reasonably fast" requirement as users would experience unpredictable performance based on what other queries are running
    
    **Query Result Storage**
    
    -   The design mentions "resultsUrl" but doesn't specify where or how query results are stored
    -   Without a clear result storage strategy, large query results could overwhelm the Jobs DB or be lost after job completion
    -   This makes it impossible to retrieve results reliably, especially for large analytical queries that might produce gigabytes of data
3.  **Follow-up Questions**
    
    -   How does the system handle job executor failures - is there checkpointing of intermediate query execution state?
    -   What happens when multiple users submit resource-intensive queries simultaneously - how do you prevent resource starvation?
    -   How are query results stored and served - do they go to S3, remain in the Jobs DB, or use a separate result store?
    -   How does the authorization service determine which tables/columns a user can access within their SQL query?
    -   What's the strategy for cleaning up completed Kubernetes jobs and their associated resources?
    -   How do you handle queries that run longer than expected - is there a timeout or resource limit mechanism?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply