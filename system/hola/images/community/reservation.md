##### Design a Reservation System for Airbnb

by ElderlyHarlequinMongoose590• Staff• 14 days ago

Design Airbnb’s Reservation System that supports guests searching for listings, viewing availability, placing holds and completing reservations (payment), and prevents double-booking. The system must also support host-configurable settings like “instant book” vs “request to book,” cancellations and refunds, and display accurate availability across multiple time zones. Focus on correctness of bookings, user experience during booking flow (low latency), and reliable payment handling. You should discuss data models, APIs, high-level architecture, concurrency control for availability, failure modes and recovery, metrics, and testing strategies.

Constraints / expectations:

Design for large scale (multi-region), many concurrent searches and bookings.

Prioritize no double-booking, reasonable UX (fast search + booking), and durable bookings.

Assume listings have per-night availability only (no hourly).

Don’t invent precise business numbers — describe scale qualitatively (e.g., “millions of listings, tens of thousands of bookings per minute”).

Collapse Description

2

2

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 14 days ago](https://www.hellointerview.com/community/submissions/cme8o2kyv04nkad08u9elt589#comment-cme8o3c14002wad3adu27adg1)

Nice design! You've created a comprehensive reservation system with good attention to preventing double-booking through distributed locking and handling both instant-book and request-to-book flows. Your use of regional Redis locks for low-latency holds while maintaining global consistency is a particularly clever approach, and the composite key strategy for availability calendar lookups shows solid understanding of database optimization.

1.  **Positives**
    
    -   Regional Redis locks with TTL for temporary holds is an excellent pattern - this provides low-latency booking flow while preventing double-booking race conditions
    -   Composite key (listing\_id+month+year) for availability calendar is smart - this reduces query complexity and improves performance for date range searches
    -   Separating instant-book vs request-to-book flows with different locking strategies shows good understanding of business requirements and their technical implications
2.  **Potential Issues** **Availability Calendar Data Model**
    
    -   Storing availability as an array within a single row (listing\_id, month, year) creates contention when multiple users book different dates in the same month
    -   This design requires locking the entire month's data for any single-day update, causing unnecessary blocking
    -   Should use individual rows per date to allow concurrent bookings for different dates in the same listing
    
    **Missing Payment Failure Handling**
    
    -   No clear rollback mechanism when Stripe payment fails after acquiring the Redis lock
    -   This could leave listings in limbo state where they appear booked but payment never completed
    -   Need explicit compensation logic to release locks and update availability on payment failures
    
    **Cross-Region Consistency Gap**
    
    -   Regional Redis locks without global coordination could allow double-booking across regions
    -   Two users in different regions could simultaneously acquire locks for the same dates
    -   Need either global lock coordination or strong consistency checks before final commit
3.  **Follow-up Questions**
    
    -   How do you handle the race condition when the Redis lock expires while payment is still processing?
    -   What happens to pending host confirmations if the host never responds within 24 hours?
    -   How do you ensure availability calendar updates are atomic when a booking spans multiple months?
    -   What's your strategy for handling partial cancellations (e.g., canceling 2 nights of a 5-night booking)?
    -   How do you prevent overselling when multiple regional Redis instances have network partitions?
    -   What's the reconciliation process if Redis and the database get out of sync?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

E

ElderlyHarlequinMongoose590

[• 13 days ago• edited 13 days ago](https://www.hellointerview.com/community/submissions/cme8o2kyv04nkad08u9elt589#comment-cme9fq23m082cad08vdtafogp)

Availability Calendar Data Model -- Regarding this.. I was divided between having one entry per day for every listing, but the data overhead and read overhead for this when we fetch the calendar is pretty high. My assumption is that OCC would avoid the lock on reads even when this row is being accessed and was planning to use that. but let me know your thoughts. A single row for a month can help in faster searches for the month as well. and it is edge casey that a lot of people will be fetching the same calendar for the same month and performing updates together. So for majority of the cases, this could work. but let me know if I am mistaken for celebrity cases in popular cities; and if we should think of those cases in a different way.

Missing Payment Failure Handling -- For this I was thinking the extended TTL expiration would enable in releasing those listings. Would we need another clean up job in this case as well? Is that overengineering necessary when redis can handle the lock release?

Cross-Region Consistency Gap Probably should have been clearer, but intended to do a global commit check on this reservation before confirmation on this listing. Any conflict at this point/after payment, which should be edge casey would need a refund flow/rejection of confirmation.

Let me know your thoughts.

Show more

0

Reply