##### Design Flight Booking System

by Rahul Dobriyal• Senior• 24 days ago

Design a flight booking system which aggregates flight details from 3rd parties and provides a platform to book. Payment is handled by a 3rd party payment service.

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

[• 24 days ago](https://www.hellointerview.com/community/submissions/cmdupjzto0b82ad070diamqux#comment-cmdupke140019ad3ac1gw3qyx)

Nice design! You've created a well-structured flight booking system with good separation of concerns, appropriate use of message queues for asynchronous processing, and smart caching strategies. Your strongest insight is using distributed locking with Redis to prevent double-booking during the critical reservation phase, which shows solid understanding of concurrency challenges. However, the design has some critical issues around data consistency between the booking service and flight service that could lead to overselling seats, and the single-point-of-failure in your Redis lock could bring down the entire booking system.

1.  **Positives**
    
    -   Using Kafka queues for reservation and booking flows provides excellent decoupling and helps handle peak loads by buffering requests during high-traffic periods
    -   Implementing Redis distributed locking for seat reservation shows good understanding of preventing race conditions in a distributed system
    -   Separating flight search into a dedicated service with Elasticsearch indexing and Redis caching is a smart optimization for the low-latency search requirement
2.  **Potential Issues** **Data Consistency Between Services**
    
    -   The flight service and booking service maintain separate databases but need to coordinate seat availability updates
    -   This creates a distributed transaction problem where a booking could be confirmed in the booking DB while the flight DB still shows seats as available
    -   This violates the high consistency requirement and could lead to overselling flights
    
    **Single Point of Failure in Redis Lock**
    
    -   All booking operations depend on acquiring a lock from a single Redis instance
    -   If Redis fails or becomes unavailable, the entire booking system stops working
    -   This violates the high availability requirement and creates an unacceptable failure mode for a critical business function
    
    **Missing Seat Inventory Management**
    
    -   The design doesn't clearly show how available seats are tracked and decremented atomically
    -   Without a dedicated inventory service or clear ownership of seat counts, concurrent bookings could exceed actual capacity
    -   This breaks the fundamental requirement of preventing overbooking
3.  **Follow-up Questions**
    
    -   How does the system handle partial failures when payment succeeds but seat confirmation fails?
    -   What happens to reserved seats if a user abandons the booking flow - how long are they held?
    -   How do you ensure consistency when flight data is updated by 3rd party aggregators while bookings are in progress?
    -   What's your strategy for handling the 1 million concurrent bookings requirement - wouldn't the single Redis lock become a bottleneck?
    -   How does the system reconcile differences between your cached flight data and the actual availability from 3rd party providers?

_Warning: This feedback was AI generated and may contain errors._

Show more

4

Reply