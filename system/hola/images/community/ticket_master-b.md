##### Ticketmaster General Admission

by FutureHarlequinMeerkat461• Senior• 4 days ago

User Flow:

1.  Users make a request to Waiting Service and establish a persistent SSE connection with the service. The request is enqueued in SQS (along-with the the SSE connection ID).
    
2.  A worker which subscribes to ticket availability notifications for an event pulls the user requests off of the queue, generates an access token for the user if tickets are available with a validity of 5 minutes and passes the token back to the user via the SSE connection.
    
3.  The user, passes the Access token in a POST request to the booking service to create the booking. Booking service in a transaction - (1) Queries Ticket Service to check if the requested number of tickets are available (2a) If available, reduces the ticket count by the requested amount and proceeds to make a booking in the Booking DB with status as "Processing". (2b) If not available, rejects the booking request with an error to the user.
    

4a. Booking Service upon creating the booking, passes the booking ID to the user in the response along-with a Stripe URL for completing payment.

4b. Simultaneously, Booking Service pushes a message into an SQS queue with delivery delay of 5 mins (time by which user should complete their booking).

1.  User complete their payment on Stripe, which registers a webhook with the booking service, passing in the booking ID and transaction status to booking service.
    

6a. If payment failed, booking service (in a transaction) marks the booking as failed and returns the tickets to the ticket pool.

6b. If payment succeeds, booking service marks the booking as SUCCEEDED.

6c. If no response is received from the user (e.g. they closed the browser), a worker reads the message from the SQS queue post the delivery delay, checks the Booking status from the Bookings DB - and if it finds the booking is still processing, cancels the booking and returns the tickets back to the pool.

1.  Ticket Service pushes current ticket availablity upon each update (increment / decrement) of the ticket count into a queue, which is read by workers to allow users waiting in the queue to continue with ticket booking.
    

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

[• 4 days ago](https://www.hellointerview.com/community/submissions/cmemtygst05fyad08eqfuoj4y#comment-cmemu0bu000cvad39f34hiocy)

Nice design! You've built a thoughtful system that handles the complex orchestration of queue management, ticket reservation, and payment processing. Your use of SQS with delivery delays for timeout handling and transactional approaches to prevent overbooking show good architectural thinking. However, there are some scalability concerns with SSE connection management and queue persistence that would need addressing for production deployment.

1.  **Positives**
    
    -   Using SQS with delivery delay for automatic booking timeouts is elegant - this leverages AWS infrastructure to handle the 5-minute reservation window without custom timer management
    -   Transactional approach in booking service (checking availability and reducing count atomically) properly prevents overbooking scenarios
    -   Separating queue management workers from timeout workers provides good separation of concerns and allows independent scaling
2.  **Potential Issues** **SSE Connection Scalability**
    
    -   Maintaining 500K persistent SSE connections across your infrastructure is extremely challenging - each connection consumes server resources and requires sticky sessions
    -   SSE connections can't be load balanced effectively, leading to uneven server utilization and potential cascading failures when servers crash
    -   Modern alternatives like WebSockets with connection multiplexing or polling-based approaches would scale better for this volume
    
    **Queue State Persistence**
    
    -   Queue positions only exist as messages in SQS without any persistent backing store
    -   If the system crashes or SQS has issues, all users lose their queue positions with no recovery mechanism
    -   This violates the implicit requirement that users who wait should maintain their position in line
    
    **Ticket Allocation Race Conditions**
    
    -   Multiple workers pulling from the queue could attempt to allocate the same available tickets simultaneously
    -   No distributed locking mechanism shown between workers checking ticket availability and the booking service
    -   This could lead to allocation conflicts where multiple users receive tokens for the same tickets
    
    **Single Tickets DB Bottleneck**
    
    -   All 500K users will be hitting a single tickets database for availability checks and updates
    -   Even with just increment/decrement operations, this creates a write bottleneck at scale
    -   Modern databases can handle ~10K write TPS, but 500K users checking availability would far exceed this
3.  **Follow-up Questions**
    
    -   How do you maintain strict FIFO ordering when multiple workers are processing the queue in parallel?
    -   What happens to users' SSE connections when API Gateway instances fail or need to be restarted?
    -   How do you prevent the same tickets from being allocated to multiple users when workers process queue messages concurrently?
    -   How would you handle database connection pool exhaustion when 500K users are actively checking ticket availability?
    -   What's your strategy for reconnecting users to their exact queue position if their connection drops?

_Note: These examples focus on specific technical issues that would prevent the system from working correctly in production, not minor optimizations or preferences._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

F

FutureHarlequinMeerkat461

[• 4 days ago](https://www.hellointerview.com/community/submissions/cmemtygst05fyad08eqfuoj4y#comment-cmemuc8hs05j2ad08rt5s2obz)

Queue State Persistence - Durability in Message queues (Kafka / SQS) are taken for granted - is this something which is to be considered as a point of failure too and discussed?

Show more

0

Reply

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 4 days ago](https://www.hellointerview.com/community/submissions/cmemtygst05fyad08eqfuoj4y#comment-cmemud0rd0005ad3c5cfra5z5)

Great question. In interviews it’s reasonable to trust SQS/Kafka for very high durability, but you should still discuss them as a dependency with specific failure modes and how you mitigate them. Also, my earlier concern wasn’t “SQS will lose messages,” but that using the queue as the only source of truth for a waiting room makes recovery/restarts and fairness hard.

What to acknowledge and mitigate:

-   Duplicates and reordering (SQS Standard) or partitioned FIFO (Kafka): design idempotent consumers and don’t rely on strict global order unless you control it (SQS FIFO with limits, or a single Kafka partition which won’t scale).
-   Visibility/processing timeouts causing redelivery: use idempotency keys and conditional writes.
-   Retention expiry and accidental purges: size retention for worst-case backlog, add alarms on queue depth/age, and protect against destructive ops.
-   Throughput/quotas/backpressure: monitor and autoscale workers; use DLQs for poison messages.
-   Regional availability: SQS is multi-AZ within a region but still a regional dependency. Call out DR: active-passive to a secondary region, or mirrored Kafka with clear leader region per event.
-   Configuration choices: Kafka replication factor, min.insync.replicas, acks=all; SQS FIFO vs Standard trade-offs.

For the waiting room specifically:

-   Don’t use the queue as the sole record of “who is where in line.” Keep an authoritative Waitlist store (e.g., DynamoDB) with user\_id, position, status (waiting/notified/expired), and a pointer for next\_to\_notify. Workers read from this store to issue tokens and use the queue only as a work/notification mechanism.
-   Make token issuance idempotent via conditional writes (attribute\_not\_exists) so two workers can’t allocate the same spot.
-   On restart or if the queue is unavailable briefly, you can rebuild state from the Waitlist and resume without losing users’ positions.

How to phrase it in an interview:

-   “I’ll rely on SQS/Kafka for durability within a region, but I won’t treat the queue as my source of truth for user positions. I’ll persist the waitlist in DynamoDB, make consumers idempotent, use DLQs and alarms, and have a DR plan for regional issues. That way, even if the queue is delayed or reset, the system can recover users’ exact positions and continue fairly.”

So yes—discuss queues as a potential point of failure, but focus on their semantics (duplicates, ordering, availability) and ensure business-critical state (the line) lives in a durable datastore you control.

Show more

0

Reply