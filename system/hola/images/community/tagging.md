##### Design a tagging system

by FutureHarlequinMeerkat461• Staff• 13 days ago

Atlassian has a number of different products, each containing different types of content. As an example, let’s look at three of our products:

Jira - Issues Confluence - Pages Bitbucket - Pull requests

We want to build a system that allows users to tag content from different products, and then to view content by tags. A goal of this system is that it should be built in a product-agnostic way so that we could add new products in the future without a large amount of work.

There are three key experiences that we want to build here:

1.  As a user, I can add, remove, and update tags on content.
    
2.  As a user, I can click on a tag and see all content that has been associated with that tag.
    
3.  As a user, I can see a dashboard of popular tags
    

Collapse Description

1

3

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 13 days ago](https://www.hellointerview.com/community/submissions/cme9lbjcb09zaad08om48sjhe#comment-cme9lcs6i003pad3biu8uflej)

Nice design! You've created a comprehensive tagging system that handles cross-product functionality well. Your use of Kafka for decoupling writes, Elasticsearch for efficient tag searches, and the consideration of both Redis and DynamoDB Streams for popular tags shows good architectural thinking. The most critical issue is the potential for tag creation race conditions that could lead to duplicate tags in the system.

1.  **Positives**
    
    -   Product-agnostic design with separate Elasticsearch indices per product allows easy addition of new products while maintaining search isolation
    -   Smart use of CDC (Change Data Capture) for keeping caches synchronized and Elasticsearch updated without impacting write performance
    -   Good separation of concerns with dedicated services for tag creation vs search, allowing independent scaling of read and write paths
2.  **Potential Issues** **Tag Creation Race Condition**
    
    -   Multiple concurrent requests to create the same tag name will check cache/DB simultaneously and find no existing tag
    -   This causes duplicate tag entries with different tag\_ids for the same tag\_name
    -   Violates the unique constraint on tag\_name and creates data inconsistency where documents get tagged with different IDs for semantically identical tags
    
    **Missing API Design**
    
    -   The submission explicitly states "TBD" for API design despite it being a core requirement
    -   This prevents evaluation of critical aspects like pagination strategy, error handling, and idempotency
    -   Without defined APIs, it's unclear how clients interact with the system or handle failures
    
    **Unclear Consistency Model**
    
    -   The design mixes strongly consistent (PostgreSQL) and eventually consistent (DynamoDB, Elasticsearch) stores
    -   No clear strategy for handling scenarios where tag exists in PostgreSQL but not yet in Elasticsearch
    -   This could lead to users creating duplicate tags or failing to find recently tagged documents
3.  **Follow-up Questions**
    
    -   How do you handle the race condition when two users simultaneously create a tag with the same name?
    -   What happens when Elasticsearch is out of sync - do searches return stale results or errors?
    -   How do you paginate through millions of documents for a popular tag efficiently?
    -   What's your strategy for handling Kafka consumer failures that could lose tag associations?
    -   How do you ensure exactly-once processing when updating tag counts to avoid incorrect popularity metrics?
    -   What happens if the Flink job crashes while processing tag aggregations?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

F

FutureHarlequinMeerkat461

[• 12 days ago](https://www.hellointerview.com/community/submissions/cme9lbjcb09zaad08om48sjhe#comment-cmeb5ahwr05phad08x62f59b4)

"How do you handle the race condition when two users simultaneously create a tag with the same name?" - I've used a unique constraint on the tag name in the Postgres table - this should handle the race condition for duplicate addition as well - am I missing something here?

Show more

0

Reply

F

FutureHarlequinMeerkat461

[• 12 days ago](https://www.hellointerview.com/community/submissions/cme9lbjcb09zaad08om48sjhe#comment-cmeb5bugl05pxad0864upu6d8)

Of course, if the tag id creation fails we handle that in the applicate and query again to fetch the tag\_id for the tag\_name during conflicts.

Show more

0

Reply