##### Design a Billing System

by canirban99• Staff• 12 days ago

Design a Billing System where different services should be able to send usage data to the billing service at any granularity, the billing service/infra needs to generate bills for different accounts and account users should be able to view their service usage through the portal.

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

[• 12 days ago](https://www.hellointerview.com/community/submissions/cmebepu4a0707ad07nnm2qdp2#comment-cmebpls2t000zad3cnmmgmve6)

Nice design! You've created a thoughtful billing architecture with good separation of concerns between charge ingestion, bill generation, and customer-facing APIs. The use of Kafka for high-volume event streaming and DynamoDB for charge storage shows solid understanding of scalability patterns. However, there are some critical issues around data consistency and failure handling that need attention, particularly the hot partition problem you identified and the potential for data loss during accumulator failures.

1.  **Positives**
    
    -   Using Kafka for charge event streaming with accumulator pattern is excellent for handling high-volume usage data from AWS services while decoupling ingestion from processing
    -   Separating read and write paths with dedicated services and caching for bill retrieval reduces load on the primary data stores and improves customer experience
    -   Choosing DynamoDB for charges storage is appropriate given the high write throughput requirements and variable usage patterns across accounts
2.  **Potential Issues** **Hot Partition Problem Without Solution**
    
    -   You correctly identified that using account\_id as partition key will create hot partitions for high-usage accounts
    -   This will cause throttling and rejected writes for your biggest customers who generate the most revenue
    -   While you mention adding entropy to the PK, this breaks your query pattern and makes bill generation extremely inefficient
    
    **Missing Rate Limiting on Charge Ingestion**
    
    -   The charge ingestion API has no rate limiting, allowing any service to flood the system
    -   A misbehaving service could overwhelm Kafka and downstream processors
    -   This violates the requirement that "billing should never cause a service to go down" by creating a denial of service vector
    
    **Data Loss During Accumulator Failures**
    
    -   Your 60-second accumulation window commits offsets before data is persisted to DynamoDB
    -   If the accumulator crashes after committing but before writing, you lose up to 60 seconds of charges
    -   This directly violates the business requirement to "prefer to lose $$ than have billing cause a service outage"
3.  **Follow-up Questions**
    
    -   How do you handle the hot partition problem in practice? Would you consider time-based partition keys like (account\_id, month)?
    -   What happens when the reconciliation worker finds discrepancies between Kafka and DynamoDB?
    -   How do you ensure exactly-once processing when services retry failed charge submissions?
    -   Why use both Postgres for bills and DynamoDB for charges instead of one database technology?
    -   How do you handle idempotency when the monthly cron retries failed bill generation requests?
    -   What's your strategy for handling partial failures during bill generation when some charges are read but the bill write fails?

_Warning: This feedback was AI generated and may contain errors._

Show more

2

Reply