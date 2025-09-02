##### Design a Donations Website

by Manasi• Manager• 21 days ago

7

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

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdzfedmh06xbad08fheaxgqq#comment-cmdzfermv002zad3atmfnzfvp)

Nice design! You've created a comprehensive donation system with strong security considerations, proper event-driven architecture, and thoughtful handling of payment failures. Your use of idempotency keys, webhook validation, and eventual consistency for aggregates shows good understanding of distributed systems. The main areas for improvement are around potential race conditions in the aggregation logic and some missing details about how the system handles edge cases during failures.

1.  **Positives**
    
    -   Excellent payment security design with PCI compliance through Stripe integration, no sensitive card data storage, and proper webhook signature validation
    -   Smart use of event-driven architecture with Kafka for asynchronous processing, enabling reliable donation tracking without blocking the payment flow
    -   Well-thought-out failure handling with both immediate and delayed failure scenarios, including retry logic and reconciliation workers
2.  **Potential Issues** **Race Condition in Optimistic Locking**
    
    -   Your optimistic locking approach reads totalAmount, computes new total, then updates with version check
    -   This creates a window where the read value could be stale, causing lost updates if multiple workers process donations for the same charity simultaneously
    -   Even with Kafka partitioning by charityId, multiple workers consuming from the same partition could still race
    
    **Missing Idempotency for Webhook Processing**
    
    -   Stripe can send duplicate webhooks, but your design doesn't show deduplication logic
    -   Processing the same webhook twice could result in duplicate transaction records or incorrect status updates
    -   This violates the idempotency requirement and could corrupt donation history
    
    **Undefined Behavior for Partial Failures**
    
    -   When donation intent is created but Stripe payment fails, the system state is unclear
    -   No clear rollback mechanism if transaction service succeeds but notification fails
    -   This could leave donations in limbo states that confuse users
3.  **Follow-up Questions**
    
    -   How does your CDC (Change Data Capture) mechanism handle replication lag when reading from Postgres?
    -   What happens if the aggregation worker crashes after reading but before updating the charity total?
    -   How do you prevent duplicate donations if a user clicks submit multiple times before the first request completes?
    -   Why shard DonationIntent by charityId instead of userId when users query their own donation history?
    -   How does the reconciliation worker differentiate between retryable vs non-retryable failures?
    -   What's the purpose of the unnamed component between Postgres and Kafka in your diagram?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

![Manasi](https://lh3.googleusercontent.com/a/ACg8ocI7UXL3f0aW-ZDAb0RPYCgcXfhirvXGD2ZEKHcFJ9kup84jcCo=s96-c)

Manasi

[• 20 days ago• edited 20 days ago](https://www.hellointerview.com/community/submissions/cmdzfedmh06xbad08fheaxgqq#comment-cmdzg8gh00798ad08j8b8nt0m)

2.1) To handle the race condition with optimistic locking, I ensure that only one Aggregation Worker consumes each Kafka partition. Since the Kafka topic is already partitioned by charityId, and each worker is assigned a unique partition, all updates for a given charity are processed sequentially by a single worker, effectively eliminating concurrent writes for the same charity.

2.2) For the missing idempotency for web hook, I updated the design to add another cache that stores web hook event id (each stripe web hook event has a eventId) with a ttl.

2.3) I am handling partial failures through my reconciliation worker.

Show more

0

Reply

![Manasi](https://lh3.googleusercontent.com/a/ACg8ocI7UXL3f0aW-ZDAb0RPYCgcXfhirvXGD2ZEKHcFJ9kup84jcCo=s96-c)

Manasi

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdzfedmh06xbad08fheaxgqq#comment-cmdzgoxiu07foad08gsmuf47h)

3.1) How does your CDC (Change Data Capture) mechanism handle replication lag when reading from Postgres? We only emit events after the transaction is committed in DB, ensuring strong consistency from the source DB.

3.2) What happens if the aggregation worker crashes after reading but before updating the charity total? Kafka doesn’t mark the message as consumed until the Aggregation Worker finishes processing and commits the offset. On crash, the same message is redelivered.

3.3) How do you prevent duplicate donations if a user clicks submit multiple times before the first request completes? We use idempotency keys on the POST /donations API.

3.4) Why shard DonationIntent by charityId instead of userId when users query their own donation history? Sharding by charityId is chosen because aggregation workloads (donation totals, campaign stats) are write-heavy and latency-sensitive, while GET /users/me/donations is: read-only, lower frequency, served from replica or denormalized view.

3.5) How does the reconciliation worker differentiate between retryable vs non-retryable failures? The reconciliation worker uses transaction status + failure reason + retry count to classify failures

Show more

0

Reply