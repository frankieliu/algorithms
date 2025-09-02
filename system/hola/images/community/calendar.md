##### Design a Calendar System

by VoluntarySalmonCondor836• Mid-level• 3 days ago

On a high level: user can create/view/update/delete the event through event service and this updates the DB. Though ensure that only host can ensure to edit the events and the others can view it. When there is CDC on the DB, we send a notification to all the recipients present in the event. Also we have a watcher polling the DB which places all the upcoming events on a message queue like SQS. The notification service consumes the messages from SQS and triggers a lambda function to send out the notification emails to the recipients, also this service should ensure we send the notification email only once to the user.

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

[• 2 days ago](https://www.hellointerview.com/community/submissions/cmenmaucn02svad07d6wnwiss#comment-cmepvkhs7000bad3by7ixqp8u)

Nice design! You've thought through the separation of concerns between event management and notifications well, and using CDC for real-time updates with SQS for delayed notifications shows good understanding of event-driven architecture. However, there are some scalability concerns with the polling approach and data consistency issues that need addressing.

1.  **Positives**
    
    -   Using CDC (Change Data Capture) for real-time event updates is a solid choice that avoids polling overhead and provides near real-time notification triggers
    -   Leveraging SQS with delayed message delivery for the 30-minute reminders is clever - it offloads timing logic from your application and provides reliable delivery
    -   Separating event service from notification service with clear boundaries creates good modularity and allows independent scaling
2.  **Potential Issues** **Database Polling Bottleneck**
    
    -   The watcher polling Postgres every 35 minutes for upcoming events will struggle with 50M daily events across sharded databases
    -   This creates O(n) database scans that will degrade as data grows, causing missed notifications
    -   A time-series database or sorted queue (like Redis sorted sets) would handle temporal queries more efficiently
    
    **Conflicting Sharding Strategy**
    
    -   You mention sharding by both event\_id and user\_id, but these create incompatible access patterns
    -   Querying all events for a user becomes a scatter-gather across all shards if partitioned by event\_id
    -   This violates your read-heavy requirement by making the most common query inefficient
    
    **Missing Recurring Event Logic**
    
    -   Your schema includes RRule but there's no explanation of how recurring events are expanded or stored
    -   Without proper handling, you can't query "all events in a time range" or modify recurring series
    -   This fundamental calendar feature will require significant rearchitecture if not designed upfront
    
    **No Notification Deduplication**
    
    -   You mention ensuring notifications are sent only once but provide no mechanism
    -   With distributed workers and potential retries, you need explicit deduplication (like using notification\_id as idempotency key)
    -   Without this, users will receive duplicate reminder emails during system retries
3.  **Follow-up Questions**
    
    -   How do you efficiently query all events for a specific user if the database is sharded by event\_id?
    -   What's your strategy for expanding recurring events - do you materialize all instances or compute them on-demand?
    -   How do you prevent duplicate notifications when Lambda functions retry or multiple workers process the same message?
    -   How does the system handle timezone differences for users in different locations?
    -   What happens when someone modifies a single instance of a recurring event series?
    -   How do you detect and prevent double-booking when users have overlapping events?

_Note: This evaluation focuses on technical system design quality, not documentation completeness or presentation style._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply