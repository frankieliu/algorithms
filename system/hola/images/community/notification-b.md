##### Design a Notification System

by SuddenMagentaAnt951• Senior• 5 days ago

Design Notification System

Design a notification system to support critical time-sensitive notifications and promotional notifications that are not time-sensitive but should not be sent after the promotion or offer has expired.

The system should scale to handle 1 million notifications per second with an 80/20 distribution between critical and promotional notifications, where promotional notifications may target thousands of customers.

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

[• 5 days ago](https://www.hellointerview.com/community/submissions/cmekv6inj05i1ad08vqjs291t#comment-cmekv897i0012ad3bkftiuc8s)

Nice design! You've clearly thought about separating critical and promotional notifications, and your use of bloom filters for deduplication shows good consideration of the at-most-once delivery requirement. However, the architecture has some complexity and scalability concerns that would prevent it from handling the 1M notifications/second requirement effectively, particularly around data modeling and the scheduling flow.

1.  **Positives**
    
    -   Separating critical and promotional notifications into different queues is smart for prioritization and allows different processing strategies for each type
    -   Using Redis sorted sets for scheduling provides efficient time-based retrieval of notifications that need to be sent
    -   Including expiry time checking in delivery workers correctly ensures promotional notifications aren't sent after they expire
2.  **Potential Issues** **Notification data model won't scale**
    
    -   Storing userIds\[\] array in each notification record means promotional notifications targeting thousands of users create massive records
    -   This causes excessive storage overhead and makes querying/updating individual user preferences impossible
    -   The design should use a separate mapping table or segment-based approach to handle large recipient lists efficiently
    
    **Redis as primary scheduling store lacks durability**
    
    -   Using Redis sorted set as the sole scheduling store means all scheduled notifications are lost if Redis crashes
    -   This violates the delivery guarantee since scheduled notifications would simply disappear
    -   You need persistent storage backing Redis or use a durable queue like Kafka for scheduling
    
    **Unclear and complex scheduling flow**
    
    -   The flow from scheduling workers → Redis → Coordinating Service → queues → delivery workers is convoluted
    -   Multiple components (scheduling workers, coordinating service, fan out service) seem to duplicate responsibility
    -   This complexity increases failure points and makes debugging/monitoring extremely difficult
3.  **Follow-up Questions**
    
    -   How do you handle batching for promotional notifications that target thousands of users to avoid overwhelming the push services?
    -   What happens when Redis crashes - how do you recover the scheduled notifications that were in memory?
    -   How does the system prevent duplicate notifications when a delivery worker crashes after pulling from queue but before sending?
    -   Why use both relational DB for users and NoSQL for notifications - what drove this technology split?
    -   How do you ensure critical notifications get priority processing when both queues are full?
    -   What's the purpose of having both "notif queue" and "batched queue" - when would notifications go to each?

_Note: These suggestions are meant to help you refine your design - the core ideas around separation of notification types and deduplication are solid foundations to build upon._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply