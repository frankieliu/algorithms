##### Design DoorDash

by Vardhan• Mid-level• 20 days ago

Felt like a combination of uber+Ticketmaster Covered major aspects minus feed generation Took me around 90 minutes to come up with this From the writeups, this should be passing for mid-level/senior but I felt that it lacks failure handling for Staff and above, but its hard to come up with in a hour - 90 minutes. Feedback from the team would be great.

Collapse Description

5

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

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdzwfj8b0b57ad08nwvujqr1#comment-cmdzwfxsu007bad3ayf98uvtp)

Nice design! You've created a comprehensive DoorDash system that covers the major components needed for food delivery, including search, ordering, and driver management. Your use of geo-indexing for driver location tracking and ElasticSearch for restaurant/item search shows good understanding of appropriate technology choices. While the design has a solid foundation, there are some critical areas around order assignment logic, payment processing, and failure handling that would need refinement for a production system.

1.  **Positives**
    
    -   Using Redis with geo-indexing for real-time driver location tracking is an excellent choice that enables efficient proximity-based driver matching
    -   Implementing a retry queue for ignored driver requests with TTL-based locking prevents driver spam while ensuring orders get fulfilled
    -   Separating search functionality with ElasticSearch for items/restaurants while using DynamoDB for core data provides appropriate read/write optimization
2.  **Potential Issues** **Missing Order Assignment Logic**
    
    -   The design doesn't specify how drivers are selected from the geo-index results or how optimal route calculation happens
    -   This could lead to inefficient driver assignments where the closest driver isn't necessarily the best choice considering their current route
    -   The functional requirement for "optimal route for destinations" is not addressed in the current architecture
    
    **Synchronous Payment Processing Bottleneck**
    
    -   The Order Service directly calls Stripe synchronously before notifying drivers
    -   This creates a bottleneck where slow payment processing (2-5 seconds) blocks the entire order flow
    -   At 1M orders/day (~12 orders/second peak), this could cause significant delays and timeouts
    
    **Undefined Driver State Management**
    
    -   There's no clear mechanism for tracking driver availability (busy, available, offline)
    -   Multiple orders could be assigned to the same driver simultaneously if they're just using geo-proximity
    -   This violates the requirement that "drivers should not be spammed with multiple orders"
3.  **Follow-up Questions**
    
    -   How does the system handle partial order failures when ordering from multiple restaurants?
    -   What happens when a driver accepts an order but then becomes unavailable (app crash, network issues)?
    -   How do you ensure consistency between the driver's location in Redis and their actual order assignment status?
    -   How does the retry queue determine which drivers to try next after an ignore?
    -   What's the strategy for handling peak hours when driver availability is low?
    -   How do you prevent race conditions when multiple orders are trying to lock the same driver?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply