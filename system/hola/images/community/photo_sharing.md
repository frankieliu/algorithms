##### Design a photo sharing system like Google Photos

by AbstractGoldCatfish393• Senior• 11 days ago

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

[• 11 days ago](https://www.hellointerview.com/community/submissions/cmeciyqjv03tzad0875v21zh4#comment-cmecj0j8t0039ad39o5js4kd0)

Nice design! You've identified several key architectural patterns that would work well for a photo sharing system, particularly the use of CDN for content delivery and direct S3 uploads to offload your application servers. However, there are some significant issues with the database design and consistency model that would prevent this system from working correctly in production.

1.  **Positives**
    
    -   Using pre-signed URLs for direct S3 uploads is an excellent choice - this offloads the heavy upload traffic from your application servers and scales naturally with S3's capabilities
    -   CDN (CloudFront) for content delivery is perfect for this read-heavy workload and will handle the bursty traffic from viral media efficiently
    -   Redis caching for frequently accessed metadata is a smart optimization that will significantly reduce database load for popular content
2.  **Potential Issues** **Incompatible DynamoDB sharding strategy**
    
    -   You're trying to shard DynamoDB by user\_id ranges (1-100k, 100k-200k), but DynamoDB uses hash-based partitioning, not range-based sharding
    -   This fundamental misunderstanding means your sharding strategy won't work as designed
    -   DynamoDB will automatically distribute data based on the partition key hash, making manual range-based sharding impossible
    
    **Share table primary key prevents multiple shares**
    
    -   Your PostgreSQL share table uses media\_id as the primary key, meaning each media can only be shared once
    -   When a user tries to share the same photo with multiple friends, the second insert will fail with a primary key violation
    -   You need a composite key or different schema to support sharing the same media with multiple users
    
    **Inconsistent upload flow**
    
    -   Your design shows both direct S3 uploads with pre-signed URLs AND SQS chunk-by-chunk uploads, creating confusion about the actual flow
    -   These are contradictory approaches - direct uploads bypass your servers entirely while SQS queuing requires server processing
    -   This ambiguity makes it unclear how upload failures are actually handled or how you track upload progress
3.  **Follow-up Questions**
    
    -   How does the share link generation work, and how do you validate that a user has permission to access a shared photo?
    -   What happens when the media metadata database update fails after a successful S3 upload completes?
    -   How do you prevent users from directly accessing S3 objects if they guess or discover the S3 URLs?
    -   How do you handle large video uploads that might take several minutes and could timeout?
    -   Why use both DynamoDB for media metadata and PostgreSQL for shares instead of consolidating on one database technology?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply