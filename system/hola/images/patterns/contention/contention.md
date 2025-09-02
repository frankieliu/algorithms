# Dealing with Contention

Learn about how to deal with high contention in your system design interview.

Dealing with Contention

* * *

🔒 **Contention** occurs when multiple processes compete for the same resource simultaneously. This could be booking the last concert ticket, bidding on an auction item, or any similar scenario. Without proper handling, you get race conditions, double-bookings, and inconsistent state. This pattern walks you through solutions from simple database transactions to more complex distributed coordination, showing when optimistic concurrency beats pessimistic locking and how to scale beyond single-node constraints.

## The Problem

Consider buying concert tickets online. There's 1 seat left for The Weeknd concert. Alice and Bob both want this last seat and click "Buy Now" at exactly the same moment. Without proper coordination, here's what happens:

1.  Alice's request reads: "1 seat available"
    
2.  Bob's request reads: "1 seat available" (both reads happen before either write)
    
3.  Alice's request checks if 1 ≥ 1 (yes, there is a seat available), proceeds to payment
    
4.  Bob's request checks if 1 ≥ 1 (yes, there is a seat available), proceeds to payment
    
5.  Alice gets charged $500, seat count decremented to 0
    
6.  Bob gets charged $500, seat count decremented to -1
    

Both Alice and Bob receive confirmation emails with the exact same seat number. They both show up to the concert thinking they own Row 5, Seat 12. One of them is getting kicked out, and the venue has to issue a refund while dealing with two very angry customers.

Race Condition Timeline

The race condition happens because both Alice and Bob read the same initial state (1 seat available) before either of their updates takes effect. By the time Bob's update runs, Alice has already reduced the count to 0, but Bob's logic was based on the stale reading of 1 seat.

This race condition occurs because reading and writing aren't atomic. There's a gap between "check the current state" and "update based on that state" where the world can change. In that tiny window (microseconds in memory, milliseconds over a network) lies chaos.

The problem only gets worse when you scale. With 10,000 concurrent users hitting the same resource, even small race condition windows create massive conflicts. As you continue to grow, it's likely that you'll need to coordinate across multiple nodes which adds to the complexity.

To get this right, we need some form of synchronization.

## The Solution

The solution to contention problems follows a natural progression of complexity. We start with single-database solutions using atomicity and transactions, then add coordination mechanisms when concurrent access creates conflicts, and finally move to distributed coordination when multiple databases are involved.

### Single Node Solutions

When all your data exists in a single database, contention solutions are more straightforward but still have important gotchas to watch out for. Let's walk through the possible solutions for handling contention within a single node.

#### Atomicity

Before reaching for complex coordination mechanisms, atomicity solves many contention problems. **Atomicity** means that a group of operations either all succeed or all fail. There's no partial completion. If you're transferring money between accounts, either both the debit and credit happen, or neither does.

**Transactions** are how databases provide atomicity. A transaction is a group of database operations treated as a single unit. You start with BEGIN TRANSACTION, perform your operations, and finish with COMMIT (to save changes) or ROLLBACK (to undo everything).

`BEGIN TRANSACTION; -- Debit Alice's account UPDATE accounts SET balance = balance - 100 WHERE user_id = 'alice'; -- Credit Bob's account  UPDATE accounts SET balance = balance + 100 WHERE user_id = 'bob'; COMMIT; -- Both operations succeed together`

If anything goes wrong during this transaction, like Alice has insufficient funds, Bob's account doesn't exist, or the database crashes, the entire transaction gets rolled back. This prevents money from disappearing or appearing out of nowhere.

These examples use SQL because relational databases are well-known for their strong ACID guarantees (where the "A" in ACID stands for Atomicity). However, many databases support transactions, including some NoSQL databases like MongoDB (multi-document transactions), DynamoDB (transaction operations), and CockroachDB (distributed ACID transactions). The concepts apply regardless of the specific database technology.

For a concert ticket purchase, atomicity ensures that multiple related operations happen together. A ticket purchase isn't just decrementing a seat count - you also need to create a ticket record:

`BEGIN TRANSACTION; -- Check and reserve the seat UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'  -- Create the ticket record INSERT INTO tickets (user_id, concert_id, seat_number, purchase_time) VALUES ('user123', 'weeknd_tour', 'A15', NOW()); COMMIT;`

If any of these operations fail, the entire transaction rolls back. You don't end up with a seat reserved but no ticket created.

But even with this atomic transaction, there's a subtle problem that atomicity alone doesn't solve. Two people can still book the same seat. Here's why: Alice and Bob can both start their transactions simultaneously, both check that available\_seats >= 1 (both see 1 seat available), and both execute their UPDATE statements. Since each transaction is atomic, both succeed, but now we've sold 2 tickets for 1 seat.

The issue is that transactions provide atomicity within themselves, but don't prevent other transactions from reading the same data concurrently. We need coordination mechanisms to solve this.

#### Pessimistic Locking

Pessimistic locking prevents conflicts by acquiring locks upfront. The name comes from being "pessimistic" about conflicts - assuming they will happen and preventing them.

We can fix our race condition using explicit row locks:

Of course, in a real ticketing system you'd have the concept of ticket reservations to improve the user experience. But for the sake of this example, we'll keep it simple and we talk about how to handle reservations in a later section.

`BEGIN TRANSACTION; -- Lock the row first to prevent race conditions SELECT available_seats FROM concerts  WHERE concert_id = 'weeknd_tour'  FOR UPDATE; -- Now safely update the seat count UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'  -- Create the ticket record INSERT INTO tickets (user_id, concert_id, seat_number, purchase_time) VALUES ('user123', 'weeknd_tour', 'A15', NOW()); COMMIT;`

The FOR UPDATE clause acquires an exclusive lock on the concert row before reading. When Alice runs this code, Bob's identical transaction will block at the SELECT statement until Alice's transaction completes. This prevents both from seeing the same initial seat count and ensures only one person can check and update at a time.

A **lock** in this context is a mechanism that prevents other database connections from accessing the same data until the lock is released. Databases like PostgreSQL and MySQL are multi-threaded systems that can handle thousands of concurrent connections, but locks ensure that only one connection can modify a specific row (or set of rows) at a time.

Explicit Row Locks

Performance considerations are really important when using locks. You want to lock as few rows as possible for as short a time as possible. Lock entire tables and you kill concurrency. Hold locks for seconds instead of milliseconds and you create bottlenecks. In our example, we're only locking one specific concert row briefly during the purchase.

#### Isolation Levels

Instead of explicitly locking rows with FOR UPDATE, you can let the database automatically handle conflicts by raising what's called the isolation level. **Isolation levels** control how much concurrent transactions can see of each other's changes. Think of it as how "isolated" each transaction is from seeing other transactions' work.

Most databases support four standard isolation levels (these are different options, not a progression):

-   READ UNCOMMITTED - Can see uncommitted changes from other transactions (rarely used)
    
-   READ COMMITTED - Can only see committed changes (default in PostgreSQL)
    
-   REPEATABLE READ - Same data read multiple times within a transaction stays consistent (default in MySQL)
    
-   SERIALIZABLE - Strongest isolation, transactions appear to run one after another
    

The defaults of either READ COMMITTED or REPEATABLE READ still allows our concert ticket race condition because both Alice and Bob can read "1 seat available" simultaneously before updating. The SERIALIZABLE isolation level solves this by making transactions appear to run one at a time:

`-- Set isolation level for this transaction BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'  -- Create the ticket record INSERT INTO tickets (user_id, concert_id, seat_number, purchase_time) VALUES ('user123', 'weeknd_tour', 'A15', NOW()); COMMIT;`

With SERIALIZABLE, the database automatically detects conflicts and aborts one transaction if they would interfere with each other. The aborted transaction must retry.

Isolation Levels

The tradeoff is that SERIALIZABLE isolation is much more expensive than explicit locks. It requires the database to track all reads and writes to detect potential conflicts, and transaction aborts waste work that must be redone. Explicit locks give you precise control over what gets locked and when, making them more efficient for scenarios where you know exactly which resources need coordination.

#### Optimistic Concurrency Control

Pessimistic locking assumes conflicts will happen and prevents them upfront. **Optimistic concurrency control (OCC)** takes the opposite approach in that it assumes conflicts are rare and detects them after they occur.

The performance benefit is significant. Instead of blocking transactions waiting for locks, you let them all proceed and only retry the ones that conflict. Under low contention, this eliminates locking overhead entirely.

The pattern is simple, you can include a version number with your data. Every time you update a record, increment the version. When updating, specify both the new value and the expected current version.

`-- Alice reads: concert has 1 seat, version 42 -- Bob reads: concert has 1 seat, version 42 -- Alice tries to update first: BEGIN TRANSACTION; UPDATE concerts  SET available_seats = available_seats - 1, version = version + 1 WHERE concert_id = 'weeknd_tour'    AND version = 42;  -- Expected version INSERT INTO tickets (user_id, concert_id, seat_number, purchase_time) VALUES ('alice', 'weeknd_tour', 'A15', NOW()); COMMIT; -- Alice's update succeeds, seats = 0, version = 43 -- Bob tries to update: BEGIN TRANSACTION; UPDATE concerts  SET available_seats = available_seats - 1, version = version + 1 WHERE concert_id = 'weeknd_tour'   AND version = 42;  -- Stale version! -- Bob's update affects 0 rows - conflict detected, transaction rolls back`

When Bob's update fails, he knows someone else modified the record. He can re-read the current state, check if seats are still available, and retry with the new version number. If seats are gone, he gets a clear "sold out" message instead of a mysterious failure.

Importantly, the "version" doesn't have to be a separate column. You can use existing data that naturally changes when the record is updated. In our concert example, the available seats count itself serves as the version. Here's how it works:

`-- Alice reads: 1 seat available -- Bob reads: 1 seat available -- Alice tries to update first: BEGIN TRANSACTION; UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'    AND available_seats = 1;  -- Expected current value INSERT INTO tickets (user_id, concert_id, seat_number, purchase_time) VALUES ('alice', 'weeknd_tour', 'A15', NOW()); COMMIT; -- Alice's update succeeds, seats now = 0 -- Bob tries to update: BEGIN TRANSACTION; UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'    AND available_seats = 1;  -- Stale value! -- Bob's update affects 0 rows - conflict detected, transaction rolls back`

This approach works because we're checking that the current seat count matches what we read earlier. If someone else bought the ticket first, the seat count changed and our update fails.

The same pattern applies to other scenarios. For eBay bidding, use the current highest bid amount as the version. For bank transfers, use the account balance. For inventory systems, use the stock count. Any value that changes when the record is updated can serve as your optimistic concurrency control mechanism.

You'll need to be careful to avoid what is called the ABA problem. Where thread A reads a value (A), thread B changes it to B then back to A, and thread A thinks nothing changed when it does its compare-and-swap. This can happen with optimistic locking when using simple version numbers that wrap around, or when memory gets reused. More on this when we get into Common Deep Dives.

This approach makes sense when conflicts are uncommon. For most e-commerce scenarios, the chance of two people buying the exact same item at the exact same moment is low. The occasional retry is worth avoiding the overhead of pessimistic locking.

### Multiple Nodes

All the approaches we've covered so far work within a single database. But what happens when you need to coordinate updates across multiple databases? This is where things get significantly more complex.

If you identify that your system needs strong consistency guarantees during high-contention scenarios, you should do all you can to keep the relevant data in a single database. Nine times out of ten, this is entirely possible and avoids the need for distributed coordination, which can get ugly fast.

Consider a bank transfer where Alice and Bob have accounts in different databases. Maybe your bank grew large enough that you had to shard user accounts across multiple databases. Alice's account lives in Database A while Bob's account lives in Database B. Now you can't use a single database transaction to handle the transfer. Database A needs to debit $100 from Alice's account while Database B needs to credit $100 to Bob's account. Both operations must succeed or both must fail. If Database A debits Alice but Database B fails to credit Bob, money disappears from the system.

You have several options for distributed coordination, each with different trade-offs:

#### Two-Phase Commit (2PC)

The classic solution is two-phase commit, where your transfer service acts as the coordinator managing the transaction across multiple database participants. The coordinator (your service) asks all participants to prepare the transaction in the first phase, then tells them to commit or abort in the second phase based on whether everyone successfully prepared.

Two-Phase Commit

Critically, the coordinator must write to a persistent log before sending any commit or abort decisions. This log records which participants are involved and the current state of the transaction. Without this log, coordinator crashes create unrecoverable situations where participants don't know whether to commit or abort their prepared transactions.

Keeping transactions open across network calls is extremely dangerous. Those open transactions hold locks on Alice's and Bob's account rows, blocking any other operations on those accounts. If your coordinator service crashes, those transactions stay open indefinitely, potentially locking the accounts forever. Production systems add timeouts to automatically rollback prepared transactions after 30-60 seconds, but this creates other problems like legitimate slow operations might get rolled back, causing the transfer to fail even when it should have succeeded.

The prepare phase is where each database does all the work except the final commit. Database A starts a transaction, verifies Alice has sufficient funds, places a hold on $100, but doesn't commit yet. The changes are made but not permanent, and other transactions can't see them. Database B starts a transaction, verifies Bob's account exists, prepares to add $100, but doesn't commit yet. In SQL terms, this looks like:

`-- Database A during prepare phase BEGIN TRANSACTION; SELECT balance FROM accounts WHERE user_id = 'alice' FOR UPDATE; -- Check if balance >= 100 UPDATE accounts SET balance = balance - 100 WHERE user_id = 'alice'; -- Transaction stays open, waiting for coordinator's decision -- Database B during prepare phase  BEGIN TRANSACTION; SELECT * FROM accounts WHERE user_id = 'bob' FOR UPDATE; -- Verify account exists and is active UPDATE accounts SET balance = balance + 100 WHERE user_id = 'bob'; -- Transaction stays open, waiting for coordinator's decision`

If both databases can prepare successfully, your service tells them to commit their open transactions. If either fails, both roll back their open transactions.

Two-phase commit guarantees atomicity across multiple systems, but it's expensive and fragile. If your service crashes between prepare and commit, both databases are left with open transactions in an uncertain state. If any database is slow or unavailable, the entire transfer blocks. Network partitions can leave the system in an inconsistent state.

#### Distributed Locks

For simpler coordination needs, you can use distributed locks. Instead of coordinating complex transactions, you just ensure only one process can work on a particular resource at a time across your entire system.

For our bank transfer, you could acquire locks on both Alice's and Bob's account IDs before starting any operations. This prevents concurrent transfers from interfering with each other:

Distributed locks can be implemented with several technologies, each with different characteristics:

**Redis with TTL** - Redis provides atomic operations with automatic expiration, making it ideal for distributed locks. The SET command with expiration atomically creates a lock that Redis will automatically remove after the TTL expires. This eliminates the need for cleanup jobs since Redis handles expiration in the background. The lock is distributed because all your application servers can access the same Redis instance and see consistent state. When the lock expires or is explicitly deleted, the resource becomes available again. The advantage is speed and simplicity. Redis operations are very fast and the TTL handles cleanup automatically. The disadvantage is that Redis becomes a single point of failure, and you need to handle scenarios where Redis is unavailable.

**Database columns** - You can implement distributed locks using your existing database by adding status and expiration columns to track which resources are locked. This approach keeps everything in one place and leverages your database's ACID properties to ensure atomicity when acquiring locks. A background job periodically cleans up expired locks, though you need to handle race conditions between the cleanup job and users trying to extend their locks. The advantage is consistency with your existing data and no additional infrastructure. The disadvantage is that database operations are slower than cache operations, and you need to implement and maintain cleanup logic.

**[ZooKeeper](https://www.hellointerview.com/learn/system-design/deep-dives/zookeeper)/etcd** - These are purpose-built coordination services designed specifically for distributed systems. They provide strong consistency guarantees even during network partitions and leader failures. ZooKeeper uses ephemeral nodes that automatically disappear when the client session ends, providing natural cleanup for crashed processes. Both systems use consensus algorithms (Raft for etcd, ZAB for ZooKeeper) to maintain consistency across multiple nodes.

The advantage is robustness. These systems are designed to handle the complex failure scenarios that Redis and database approaches struggle with. The disadvantage is operational complexity, as you need to run and maintain a separate coordination cluster.

Distributed locks aren't just for technical coordination either, they can dramatically improve user experience by preventing contention before it happens. Instead of letting users compete for the same resource, create intermediate states that give temporary exclusive access.

Consider Ticketmaster seat reservations. When you select a seat, it doesn't immediately go from "available" to "sold." Instead, it goes to a "reserved" state that gives you time to complete payment while preventing others from selecting the same seat. The contention window shrinks from the entire purchase process (5 minutes) to just the reservation step (milliseconds).

The same pattern appears everywhere. Uber sets driver status to "pending\_request," e-commerce sites put items "on hold" in shopping carts, and meeting room booking systems create temporary holds.

The advantage is simplicity compared to complex transaction coordination. The disadvantage is that distributed locks can become bottlenecks under high contention, and you need to handle lock timeouts and failure scenarios.

#### Saga Pattern

The saga pattern takes a different approach. Instead of trying to coordinate everything atomically like 2PC, it breaks the operation into a sequence of independent steps that can each be undone if something goes wrong.

Think of it like this. Instead of holding both Alice's and Bob's accounts locked while coordinating, you just do the operations one by one. First, debit Alice's account and commit that transaction immediately. Then, credit Bob's account and commit that transaction. If the second step fails, you "compensate" by crediting Alice's account back to undo the first step.

For our bank transfer example

1.  **Step 1** - Debit $100 from Alice's account in Database A, commit immediately
    
2.  **Step 2** - Credit $100 to Bob's account in Database B, commit immediately
    
3.  **Step 3** - Send confirmation notifications
    

If Step 2 fails (Bob's account doesn't exist), you run the compensation for Step 1. You credit $100 back to Alice's account. If Step 3 fails, you compensate both Step 2 (debit Bob's account) and Step 1 (credit Alice's account).

Each step is a complete, committed transaction. There are no long-running open transactions and no coordinator crashes leaving things in limbo. Each database operation succeeds or fails independently.

But there is (of course) an important tradeoff. During saga execution, the system is temporarily inconsistent. After Step 1 completes, Alice's account is debited but Bob's account isn't credited yet. Other processes might see Alice's balance as $100 lower during this window. If someone checks the total money in the system, it appears to have decreased temporarily.

This eventual consistency is what makes sagas practical. You avoid the fragility of 2PC by accepting that the system will be briefly inconsistent. You handle this by designing your application to understand these intermediate states. For example, you might show transfers as "pending" until all steps complete.

## Choosing the Right Approach

Keep in mind, like with much of system design, there isn't always a clear-cut answer. You'll need to consider the tradeoffs of each approach based on your specific use case and make the appropriate justification for your choice.

**Start here. Can you keep all the contended data in a single database?** If yes, use pessimistic locking or optimistic concurrency based on your conflict frequency.

**Single database, high contention:** Pessimistic locking with explicit locks (FOR UPDATE). This provides predictable performance, is simple to reason about, and handles worst-case scenarios well.

**Single database, low contention:** Optimistic concurrency control using existing columns as versions. This provides better performance when conflicts are rare and has no blocking.

**Multiple databases, must be atomic:** Distributed transactions (2PC for strong consistency, Sagas for resilience). Use only when you absolutely need atomicity across systems.

**User experience matters:** Distributed locks with reservations to prevent users from entering contention scenarios. This is great for ticketing, e-commerce, and any user-facing competitive flows.

Approach

Use When

Avoid When

Typical Latency

Complexity

**Pessimistic Locking**

High contention, critical consistency, single database

Low contention, high throughput needs

Low (single DB query)

Low

**SERIALIZABLE Isolation**

Need automatic conflict detection, can't identify specific locks

Performance critical, high contention

Medium (conflict detection overhead)

Low

**Optimistic Concurrency**

Low contention, high read/write ratio, performance critical

High contention, can't tolerate retries

Low (when no conflicts)

Medium

**Distributed Transactions**

Must have atomicity across systems, can tolerate complexity

High availability requirements, performance critical

High (network coordination)

Very High

**Distributed Locks**

User-facing flows, need reservations, simpler than 2PC

No alternatives available, purely technical coordination

Low (simple status updates)

Medium

Flow Chart

When in doubt, start with pessimistic locking in a single database. It's simple, predictable, and you can always improve it later.

## When to Use in Interviews

Don't wait for the interviewer to ask about contention. Be proactive in recognizing scenarios where multiple processes might compete for the same resource and suggest appropriate coordination mechanisms. This is typically when you determine during your non-functional requirements that your system requires strong consistency.

### Recognition Signals

Here are some bang on examples of when you might need to use contention patterns:

**Multiple users competing for limited resources** such as concert tickets, auction bidding, flash sale inventory, or matching drivers with riders

**Prevent double-booking or double-charging** in scenarios like payment processing, seat reservations, or meeting room scheduling

**Ensure data consistency under high concurrency** for operations like account balance updates, inventory management, or collaborative editing

**Handle race conditions in distributed systems** in any scenario where the same operation might happen simultaneously across multiple servers and where the outcome is sensitive to the order of operations.

### Common Interview Scenarios

This shows up A LOT in common interview questions. It's one of the most popular patterns and interviewers love to ask about it. Here are some examples of places where you might need to use contention patterns:

**[Online Auction Systems](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction)** - Perfect for demonstrating optimistic concurrency control because multiple bidders compete for the same item. You can use the current high bid as the "version" and only accept new bids if they're higher than the expected current bid. Application-level status coordination also helps by marking items as "bidding ends in 30 seconds" to prevent last-second contention scenarios.

**[Ticketmaster/Event Booking](https://www.hellointerview.com/learn/system-design/problem-breakdowns/ticketmaster)** - While this seems like a classic pessimistic locking scenario for seat selection, application-level status coordination is actually the bigger win. When users select seats, you immediately reserve them with a 10-minute expiration, which prevents the terrible UX of users filling out payment info only to find the seat was taken by someone else.

**Banking/Payment Systems** - Great place to showcase distributed transactions since account transfers between different banks or services need atomic operations across multiple systems. You should start with saga pattern for resilience and mention 2PC only if the interviewer pushes for strict consistency requirements.

**[Ride Sharing Dispatch](https://www.hellointerview.com/learn/system-design/problem-breakdowns/uber)** - Application-level status coordination shines here because you can set driver status to "pending\_request" when sending ride requests, which prevents multiple simultaneous requests to the same driver. You can use either caches with TTL for automatic cleanup when drivers don't respond within 10 seconds, or database status fields with periodic cleanup jobs.

**Flash Sale/Inventory Systems** - Perfect for demonstrating a mix of approaches. You can use optimistic concurrency for inventory updates with the current stock count as your version, but you should also implement application-level coordination for shopping cart "holds" to improve user experience and reduce contention at checkout.

**[Yelp/Review Systems](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp)** - Great example of optimistic concurrency control because when users submit reviews, you need to update the business's average rating. Multiple concurrent reviews for the same restaurant create contention, so you can use the current rating and review count as your "version" and only update if they match what you read initially. This prevents rating calculations from getting corrupted when reviews arrive simultaneously.

The best candidates identify contention problems before they're asked. When designing any system with shared resources, immediately address coordination:

"This auction system will have multiple bidders competing for items, so I'll use optimistic concurrency control with the current high bid as my version check."

"For the ticketing system, I want to avoid users losing seats after filling out payment info, so I'll implement seat reservations with a 10-minute timeout."

"Since we're sharding user accounts across databases, transfers between different shards will need distributed transactions. I'll use the saga pattern for resilience."

### When NOT to overcomplicate

Don't reach for complex coordination mechanisms when simpler solutions work.

A common mistake I see is candidates reaching for distributed locks (Redis, etc) when a simple database transaction with row locking or OCC is sufficient. Keep in mind that adding new components adds system complexity and introduces new failure modes so do what you can to avoid them.

**Low contention scenarios** where conflicts are rare (like updating product descriptions where only admins can edit) can use basic optimistic concurrency with retry logic. Don't implement elaborate locking schemes when simple retry logic handles the occasional conflict.

**Single-user operations** like personal todo lists, private documents, or user preferences have no contention, so no coordination is needed.

**Read-heavy workloads** where most operations are reads with occasional writes can use simple optimistic concurrency to handle the rare write conflicts without impacting read performance.

## Common Deep Dives

Interviewers love to probe your understanding of edge cases and failure scenarios. Here are the most common follow-up questions when discussing contention patterns that end up being most common.

### "How do you prevent deadlocks with pessimistic locking?"

Consider a bank transfer between two accounts. Alice wants to transfer $100 to Bob, while Bob simultaneously wants to transfer $50 to Alice. Transaction A needs to debit Alice's account and credit Bob's account. Transaction B needs to debit Bob's account and credit Alice's account. Transaction A locks Alice's account first, then tries to lock Bob's account. Transaction B locks Bob's account first, then tries to lock Alice's account. Both transactions wait forever for the other to release their lock.

Deadlock

This deadlock happens because the transactions acquire locks in different orders. The business logic doesn't care about order, it just wants to update both users when they interact. But databases can't read your mind about which locks are safe to acquire simultaneously.

The standard solution is ordered locking, which means always acquiring locks in a consistent order regardless of your business logic flow. Sort the resources you need to lock by some deterministic key like user ID, database primary key, or even memory address. If you need to lock users 123 and 456, always lock 123 first even if your business logic processes 456 first. This prevents circular waiting because all transactions follow the same acquisition order.

In practice, this might mean sorting your user IDs before locking them, or establishing conventions like "always lock the user initiating the action first, then lock recipients in ascending ID order." The exact ordering doesn't matter as long as it's consistent across all transactions in your system.

As a fallback, database timeout configurations serve as your safety net when ordered locking isn't practical or when you miss edge cases. Set transaction timeouts so deadlocked transactions get killed after a reasonable wait period and can retry with proper ordering. Most modern databases also have automatic deadlock detection that kills one transaction when cycles are detected, but this should be your fallback, not your primary strategy.

### "What if your coordinator service crashes during a distributed transaction?"

This is the classic 2PC failure scenario. Databases are sitting with prepared transactions, waiting for commit or abort instructions that never come. Those transactions hold locks on resources, potentially blocking other operations indefinitely.

2PC Failure

Production systems handle this with coordinator failover and transaction recovery. When a new coordinator starts up, it reads persistent logs to determine which transactions were in-flight and completes them. Most enterprise transaction managers (like Java's JTA) handle this automatically, but you still need to design for coordinator high availability and maintain transaction state across failures.

Sagas are more resilient here (as discussed earlier) because they don't hold locks across network calls. Coordinator failure just pauses progress rather than leaving participants in limbo.

### "How do you handle the ABA problem with optimistic concurrency?"

Sneaky question that tests deeper understanding. The ABA problem occurs when a value changes from A to B and back to A between your read and write. Your optimistic check sees the same value and assumes nothing changed, but important state transitions happened.

Consider a review system like Yelp, where users can review businesses and each business tracks an average rating so we don't need to recalculate it each time. A restaurant starts with 4.0 stars and 100 reviews. Two new reviews come in simultaneously - one gives 5 stars, another gives 3 stars. Both reviews see the current average as 4.0 and calculate the new average. Due to the math, the final average might still end up at 4.0 stars, but now with 102 reviews. If you use just the average rating as your "version," both updates would succeed because they see the same 4.0 value, but you'd miss one of the reviews.

The solution is using a column that you know will always change. In the Yelp case, use the review count instead of the average rating as your optimistic concurrency check. Every new review increases the count, so it's a perfect monotonically increasing version. Your update becomes "set new average and increment count to 101, but only if current count is 100."

`-- Use review count as the "version" since it always increases UPDATE restaurants  SET avg_rating = 4.1, review_count = review_count + 1 WHERE restaurant_id = 'pizza_palace'    AND review_count = 100;  -- Expected current count`

When you can't find a naturally changing column, fall back to an explicit version column that increments on every update regardless of whether your business data actually changed. Even if the average rating stays the same, the version number shows that processing occurred.

Some databases solve this automatically. DynamoDB's conditional writes include an internal version that increments on every update, not just when your data changes.

### "What about performance when everyone wants the same resource?"

This is the hot partition or celebrity problem, where your carefully designed distributed system suddenly has everyone hammering the same single resource. Think about what happens when a celebrity joins Twitter and millions of users try to follow them simultaneously, or when a rare collectible drops on eBay and thousands of people bid on the same item, or when Taylor Swift announces a surprise concert and everyone tries to buy tickets at the exact same time.

The fundamental issue is that normal scaling strategies break down when demand concentrates on a single point. Sharding doesn't help because you can't split one Taylor Swift concert across multiple databases because everyone wants that specific resource. Load balancing doesn't help because all the load balancer does is distribute requests to different servers that then compete for the same database row. Even read replicas don't help because the bottleneck is on the writes.

Your first strategy should be questioning whether you can change the problem itself rather than throwing more infrastructure at it. Maybe instead of one auction item, you actually have 10 identical items and can run separate auctions for each. Maybe instead of requiring immediate consistency for social media interactions, you can make likes and follows eventually consistent - users won't notice if their follow takes a few seconds to appear on the celebrity's follower count.

Queue-Based Serialization

For cases where you truly need strong consistency on a hot resource, implement queue-based serialization. Put all requests for that specific resource into a dedicated queue that gets processed by a single worker thread. This eliminates contention entirely by making operations sequential rather than concurrent. The queue acts as a buffer that can absorb traffic spikes while the worker processes requests at a sustainable rate.

The tradeoff is latency. Users might wait longer for their requests to be processed. But this is often better than the alternative of having your entire system grind to a halt under the contention.

## Conclusion

Contention handling is crucial for reliable systems, but the path to success isn't what most engineers think. You should exhaust every single-database solution before even considering distributed coordination, since modern databases like PostgreSQL can handle tens of terabytes and thousands of concurrent connections. This covers the vast majority of applications you'll ever build, and the complexity jump to distributed coordination comes with enormous overhead and often worse performance.

You should stay within a single database as long as possible because both pessimistic locking and optimistic concurrency give you simple, battle-tested solutions with ACID guarantees. Pessimistic locking handles high contention predictably, while optimistic concurrency delivers excellent performance when conflicts are rare. Only move to distributed coordination when you've truly outgrown vertical scaling or need geographic distribution, which happens much later than most engineers think.

The best system designers fight hard to keep their data together and choose the right coordination pattern for their specific consistency requirements. Master these fundamentals, but always remember that the simplest solution that works is almost always the right choice.

###### Test Your Knowledge

Take a quick 15 question quiz to test what you've learned.

Start Quiz

Mark as read

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd0errt9060uad08d9rhsgf4)

great atricle , but in banking system in general what pattern do we use 2 phase commit or SAGA?

Show more

8

Reply

P

PracticalRedFowl682

[• 8 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmef3y58805dcad08jsd9bliw)

It said "You should start with saga pattern for resilience and mention 2PC only if the interviewer pushes for strict consistency requirements."

Show more

0

Reply

S

SmoothSilverHippopotamus562

[• 3 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmemh2n8u03jdad08wkqdrd3t)

yeah I saw this, but banking systems should be CP systems, does saga pattern guarantee strong consistency?

Show more

0

Reply

![Aditya Bharadwaj](https://lh3.googleusercontent.com/a/ACg8ocL4W4AisDpuNn9_Tx40qhOHqknN9Po70gB5lkxD6TJFGGKfPZ4=s96-c)

Aditya Bharadwaj

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd0hm7bk00duad084ak7zmfv)

Loving these focused pattern breakdowns!

Show more

8

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd0ie76e00m4ad08ogjwav91)

Glad you like them!

Show more

3

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd0ib1so00k3ad08p60lwu8m)

Maybe instead of one auction item, you actually have 10 identical items and can run separate auctions for each- sorry unable to get this what if I have actually one auction and million people want to auction that? how we will create 10 auction - please clarify more

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd0ie1qu00lzad082c8oeh7c)

Then it would be high contention :) That paragraph is just saying that, if possible, can you find a way to change the problem to avoid the contention. If everyone wants a single auction, then the answer is no!

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd0igasb00lfad082ertx1v5)

ok , so shall I use locking or shall I use queues for this?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd0isc6q00pdad08i5rtw30d)

Depends!

If you do the math for the given problem's scale and find that your infrastructure as it is can handle whatever maximum TPS your auction will have, then transactions with locking are sufficient, and a queue would be overengineering. If, on the other hand, you do the math and justify to your interviewers that you expect large spikes (maybe toward the ends of auctions) that could overwhelm your system and exceed the maximum write throughput (when using locking), then it would be easy to justify adding a queue partitioned by auctionId under the justification that it both buffers and serializes.

Show more

10

Reply

W

walnatara2

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd1wyapi0ckead082awbu77j)

I think more great example of that paragraph would be

when there is a sudden hot tweet/post in Twitter. Usually you need to count the likes. If a sudden 1 1 millions likes happen, then it would be hard to increase the likes count due to contention.

Solution is we can a have 10 separate counter and then every likes by users will increase one of those counter (you can use userId hash and modulo).

since we have 10 counters. We trade off read operation, so every users that see the hot tweet, will need to aggregate those 10 counters to calculate real likes count.

Show more

2

Reply

C

CasualAquamarineRhinoceros625

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd12ml7o06bzad084q5110f3)

Impressive article, immensely helpful, thank you!

Show more

1

Reply

C

CasualAquamarineRhinoceros625

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd1g9kbb0ab2ad08gj7am63x)

Under OCC you have "For most e-commerce scenarios, the chance of two people buying the exact same item at the exact same moment is low." - is this true at Amazon's scale and during events like Prime Day or in Flash sale type situations as well ?

Show more

1

Reply

B

BottomTealSkunk679

[• 25 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdr1kkda04dfad085snrzea3)

For those situations, you can use a good and fast, but imperfect, lock in front of the DB. I think this is covered in the Ticketmaster question

Show more

0

Reply

![Sai Nikhil](https://lh3.googleusercontent.com/a/ACg8ocLn9Zn2QRZJuYG2tXubcCVGjaKoiWtJYFn0nBxLe1nWcS5auQ=s96-c)

Sai Nikhil

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd1y15y90epzad08ey46ahrj)

This pattern section is a goldmine.

Show more

3

Reply

R

RetiredHarlequinHarrier460

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd2eeer602gsad083yklm48g)

Correction: The default isolation level for MySQL is Repeatable Read, not Read Comitted.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd2fz86a02z1ad08jylfyvf2)

You're right! MySQL's default isolation level is indeed REPEATABLE READ, while PostgreSQL's default is READ COMMITTED. Will update

Show more

1

Reply

![Sudipta Pal](https://lh3.googleusercontent.com/a/ACg8ocJ8n9uzrli5knX8X8KUrMU7OwTe-Gq1DmDeAMSQq5tAgGHAlA=s96-c)

Sudipta Pal

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd2ky65b04wgad07o8molhab)

> BEGIN TRANSACTION;
> 
> \-- Check and reserve the seat UPDATE concerts SET available\_seats = available\_seats - 1 WHERE concert\_id = 'weeknd\_tour' AND available\_seats >= 1;
> 
> \-- Create the ticket record INSERT INTO tickets (user\_id, concert\_id, seat\_number, purchase\_time) VALUES ('user123', 'weeknd\_tour', 'A15', NOW());
> 
> COMMIT;

Doesn't the database automatically implicitly lock rows during an update if multiple requests try to modify the same row(Update statement) at the same time even if no explicit lock is specified?

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd2kzo7804s8ad086fqj7n7e)

You're right that databases typically implement row-level locking during updates, but this isn't enough to prevent the race condition in the example. The issue is that both transactions can read the initial seat count (1) before either update occurs, then both proceed to update since they both saw available\_seats >= 1. The implicit locks only prevent simultaneous updates to the same row, but don't help with the read-then-update pattern where the read becomes stale.

Show more

1

Reply

![Dmitry Khamitov](https://lh3.googleusercontent.com/a/ACg8ocISRfxkb6-w6m-2ydOWrvAjG_t-Te9aJAYr48_xNKO_pPC1hHm7=s96-c)

Dmitry Khamitov

[• 1 month ago• edited 30 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdhzqhnr037had08qa7btl0i)

The WHERE clause is actually re-evaluated in the second (blocked/waiting) transaction. So there is no race condition in there - [https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED](https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED)

> The search condition of the command (the WHERE clause) is re-evaluated to see if the updated version of the row still matches the search condition. If so, the second updater proceeds with its operation using the updated version of the row.

Thanks for the great and comprehensive article!

Show more

7

Reply

C

ChristianPlumTakin500

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd2pt2du06o5ad073uwhoe3x)

Great write up!

1.  No alternatives available, purely technical coordination - could you clarify why you have this under "Avoid When" for Distributed Locks?
2.  In case of Optimistic locking, why bother using existing data attribute as version when you can just use a separate version attribute that works 100% of the use cases? I believe in general this is a standard pattern that's agnostic to business logic. (However, if your approach doesn't require read + write, and can directly make an update, then it leads to fewer conflicts, higher throughput).

Show more

3

Reply

![HONGJI WANG](https://lh3.googleusercontent.com/a/ACg8ocItSTa09fpqtWasO0RUi3l9CeJ42tH-dDmFS_28f0RcEsEQ7fVh=s96-c)

HONGJI WANG

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd2qs08t06phad08w3vhuvya)

great writeup, and would like to add some other common patterns to tackle distributed txs:

1.  outbox pattern: where you write to an outbox table and have a background task to take care of the msg sending.
2.  transactional msg: a variation of #1, where you send the msg / update the db atomically
3.  tcc: try, confirm, cancel, application level 2-pc
4.  cdc: change data capture, read from the binlog etc, final consistency

Show more

1

Reply

O

OutdoorOrangeRhinoceros745

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd31c9nk08riad08az2wzllf)

> \-- Alice reads: concert has 1 seat, version 42 -- Bob reads: concert has 1 seat, version 42
> 
> \-- Alice tries to update first: BEGIN TRANSACTION; UPDATE concerts SET available\_seats = available\_seats - 1, version = version + 1 WHERE concert\_id = 'weeknd\_tour' AND version = 42; -- Expected version
> 
> INSERT INTO tickets (user\_id, concert\_id, seat\_number, purchase\_time) VALUES ('alice', 'weeknd\_tour', 'A15', NOW()); COMMIT;
> 
> \-- Alice's update succeeds, seats = 0, version = 43
> 
> \-- Bob tries to update: BEGIN TRANSACTION; UPDATE concerts SET available\_seats = available\_seats - 1, version = version + 1 WHERE concert\_id = 'weeknd\_tour' AND version = 42; -- Stale version!
> 
> \-- Bob's update affects 0 rows - conflict detected, transaction rolls back

As there is a INSERT statement after UPDATE who will check if 0 rows are affected and do the transaction roll back ? Isn't it true that conditional update for most relational databases don't fail if where clause doesn't match any row ? In that case this will not work and same seat will still be assigned to multiple folks.

Show more

5

Reply

![Aditya Naidu](https://lh3.googleusercontent.com/a/ACg8ocLdbPoctMI76a9OcmikOO6pCdmjrZ2vbkeRnfTjrjEzpsH4R1Q=s96-c)

Aditya Naidu

[• 25 days ago• edited 25 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdqqhp0200idad08586zu0kf)

Even I have this question!

Show more

2

Reply

![Peter Leung](https://lh3.googleusercontent.com/a/ACg8ocIgPnOOUWW713aLIcFd8ZdjkkHbrpYifk8xm1_P0pFGR-CeFKhzyQ=s96-c)

Peter Leung

[• 19 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdysc8kx09t2ad07wkju8bsq)

So we'd have to set the Isolation to Repeatable Read mode?

Show more

0

Reply

![Sudipta Pal](https://lh3.googleusercontent.com/a/ACg8ocJ8n9uzrli5knX8X8KUrMU7OwTe-Gq1DmDeAMSQq5tAgGHAlA=s96-c)

Sudipta Pal

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd3gozrf00toad08k6v1zz7z)

In a Two-Phase Commit (2PC) setup, is it always necessary for the coordinator service to have direct read and write access to the database?

Is it possible to implement 2PC if the coordinator interacts with the database indirectly through APIs rather than direct connections? I felt it would be very hard to achieve 2PC if we go via api route, Thoughts?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd3gqc4c0dubad08vw17blnz)

No, the coordinator doesn't need direct database access. It just needs to coordinate the prepare/commit/abort messages between participants. The coordinator can work through APIs as long as those APIs expose the necessary transaction control operations.

Show more

0

Reply

![Sudipta Pal](https://lh3.googleusercontent.com/a/ACg8ocJ8n9uzrli5knX8X8KUrMU7OwTe-Gq1DmDeAMSQq5tAgGHAlA=s96-c)

Sudipta Pal

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd3h4nz40ebrad086pqyb65p)

If request times out after aquiring lock via api communication during returning response to coordinator service due to some reason. In retry db will throw exception from that we will know row is locked but our service will asssume that lock is aquired by different transaction. So it will wait..seems deadlock condition here

Show more

0

Reply

F

FunBlackBadger208

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd3oztfx0135ad08iahhyuxf)

In the atomicity section, we have this

> \-- Create the ticket record INSERT INTO tickets (user\_id, concert\_id, seat\_number, purchase\_time) VALUES ('user123', 'weeknd\_tour', 'A15', NOW());

Isn't it simpler to have a UNIQUE constraint on the seat\_number or ticket\_id column so that the second insert operation fails automatically and avoids double booking? Basically pushing the responsibility to the database rather than handling this in the application code, or are there any downsides with that approach?

I would assume this works in all scenarios, i.e. whatever the resource that is being contended for, we can always have a id for that resource and make sure it is unique in that table

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd3p4nyb015mad08tqrzmx6q)

The crux of that section is showing we need to first read the availability column to see if any seats are available (before booking a new ticket) and a unique constraints isn't related to that or am I misunderstanding you?

The general pattern you're referring to is certainly valid, just does not apply to this case.

Show more

0

Reply

W

WilyBronzeOctopus126

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd3unu8i0989ad08zdehcb0t)

These pattern articles are great, love them. Especially appreciate the common deep dives!

Show more

1

Reply

C

CooperativeRedAntelope278

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd3wiuht0jgnad0851qmw7n2)

Unreal article. Simply amazing. I had SUCH a huge mental blockage when it comes to this. I can't believe how complex I was making it in my head. THANK YOU!!!!!

Show more

0

Reply

![paiam](https://lh3.googleusercontent.com/a/ACg8ocKepEYGK0Q3qL16vSB1PmZ0sDjCiGzxHXiz5lRhkyVQ3snMfw=s96-c)

paiam

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd4170i905w8ad07qjnotnhy)

Hey, quick question about the transaction below; what’s stopping the INSERT from running if available\_seats is already 0?

`BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'    AND available_seats >= 1; INSERT INTO tickets (user_id, concert_id, seat_number, purchase_time) VALUES ('user123', 'weeknd_tour', 'A15', NOW()); COMMIT;`

As I understand it, if available\_seats is 0, the UPDATE won’t do anything (since the condition fails), but the INSERT will still run, which could lead to overselling tickets, right?

Wouldn't it be safer to use something like PostgreSQL’s RETURNING clause here to check if the update actually succeeded before creating the ticket? For example:

`WITH updated AS (   UPDATE concerts  SET available_seats = available_seats - 1   WHERE concert_id = 'weeknd_tour' AND available_seats >= 1   RETURNING concert_id ) INSERT INTO tickets (user_id, concert_id, seat_number, purchase_time) SELECT 'user123', concert_id, 'A15', NOW() FROM updated;`

This way, the ticket only gets inserted if the seat was successfully reserved. Just curious if I'm missing anything. Happy to hear other thoughts!

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd63mqdg02vmad08i0z201qf)

You're absolutely right that the original code is unsafe. The UPDATE's WHERE clause prevents decrementing when seats=0, but the INSERT would still run. The WITH/RETURNING approach is a good pattern here. Another approach would be checking UPDATE's affected row count (if 0, rollback), but the WITH/RETURNING is cleaner. Good catch! Will update

Show more

3

Reply

S

SmallAzureTakin612

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd5z8lj901tqad07fy0e5yvo)

Thank you for this blog , this is very exhaustive and helpful.

Show more

0

Reply

![Rahul Garg](https://lh3.googleusercontent.com/a/ACg8ocLm_DZbLjHl5mwOa3vn-0yvDWZ11QimvH5H60hFuiUQG1U__KhI=s96-c)

Rahul Garg

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd6086f1023jad08tx471r7c)

SELECT FOR UPDATE does not block other reads. It only blocks other SELECT FOR UPDATE. Explanation is unclear in atomic section.

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd63l5ds02vaad08ux4clzrp)

SELECT FOR UPDATE does block other reads in most databases when using the default isolation level (at least READ COMMITTED). It prevents dirty reads by blocking other transactions from reading the locked rows until the locking transaction commits or rolls back. The exception is if you explicitly use READ UNCOMMITTED isolation level, but that's rarely used in practice.

Show more

0

Reply

![Rahul Garg](https://lh3.googleusercontent.com/a/ACg8ocLm_DZbLjHl5mwOa3vn-0yvDWZ11QimvH5H60hFuiUQG1U__KhI=s96-c)

Rahul Garg

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd65ieut03k3ad08notea5j0)

Asked GPT for quick answer. It says you are incorrect here.

Regular SELECT reads in READ COMMITTED mode:

-   Read the last committed version of the row.
-   Ignore uncommitted changes, hence avoid dirty reads.
-   Don’t care if another transaction has a row locked for update.

Show more

0

Reply

![Rahul Garg](https://lh3.googleusercontent.com/a/ACg8ocLm_DZbLjHl5mwOa3vn-0yvDWZ11QimvH5H60hFuiUQG1U__KhI=s96-c)

Rahul Garg

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd65n72c03llad08vx96z44u)

"while SELECT FOR UPDATE prevents other transactions from acquiring conflicting locks on the same rows, it doesn’t lock reads. Other transactions can still read those rows using regular SELECT statements (assuming you’re using the default READ COMMITTED isolation level). They just can’t modify or lock those rows themselves." Also same is mentioned in this article here: https://stormatics.tech/blogs/select-for-update-in-postgresql

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd67zmjh04buad0894glx2nv)

Aw yah ok let me be more clear. First you're totally right. But, in this case both transactions are using SELECT FOR UPDATE, so only one can acquire the lock and read the row. Others can still read (like if they just want to check current avialability to display to users), but they can't book because any booking flow uses the transaction with the SELECT FOR UPDATE.

Does this make sense? :) Let me see how I can update the doc to be clearer too.

Show more

2

Reply

![Rahul Garg](https://lh3.googleusercontent.com/a/ACg8ocLm_DZbLjHl5mwOa3vn-0yvDWZ11QimvH5H60hFuiUQG1U__KhI=s96-c)

Rahul Garg

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd69muzv04xcad08n0ngoptc)

Yes, this makes sense and what I was trying to convey as well. Thanks.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd69nxlg04tcad08ozubev46)

Sweet :)

Show more

1

Reply

R

RainyCyanHippopotamus256

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd63gvll02thad07nw0jgurc)

With the release of MemoryDB which offers strong consistency guarantees & lossless failovers with distributed transaction log, would it make sense to use MemoryDB (redis-compatible) for distributed locks over other solutions like ZooKeeper? It might be less complex to set it up and use it than using Zookeeper.

Show more

0

Reply

![Mark Wang](https://lh3.googleusercontent.com/a/ACg8ocLV06fE70xn_ppHnHLBOD4QScQPlIx-rEm7XYiu5ejWgb-ZssdsnQ=s96-c)

Mark Wang

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmd8akogb036dad084qftph7r)

I don't understand what the difference is between the "bad" transaction

`BEGIN TRANSACTION; -- Check and reserve the seat UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'    AND available_seats >= 1; -- Create the ticket record INSERT INTO tickets (user_id, concert_id, seat_number, purchase_time) VALUES ('user123', 'weeknd_tour', 'A15', NOW()); COMMIT;`

and the good one later for optimistic locking. The only code difference is this uses available\_seats >= 1; instead of available\_seats = 1; But if there's only 1 available\_seat as assumed there's no functional difference

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdbzvu810a6tad09oupss6sv)

~~With \>=1 both transactions can succeed because they both check if seats are greater than or equal to 1, see that condition is true (1 >= 1), and proceed to decrement. With \=1, the second transaction will fail because after the first transaction decrements the count to 0, the condition available\_seats = 1 is no longer true. This is what makes it optimistic, we're checking for the exact value we expect, not just a range condition. If that value changed, the transaction fails and we know there was a conflict.

^^ Ignore this, I was confused about the question. See below

Show more

0

Reply

T

tinja24

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdc7wxws02dtad079ssyo6r2)

The issue is that both transactions can read the initial seat count (1) before either update occurs, then both proceed to update since they both saw available\_seats >= 1.

Show more

0

Reply

![Mark Wang](https://lh3.googleusercontent.com/a/ACg8ocLV06fE70xn_ppHnHLBOD4QScQPlIx-rEm7XYiu5ejWgb-ZssdsnQ=s96-c)

Mark Wang

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmddj3wwt0195ad091wfafl5c)

> With >=1 both transactions can succeed because they both check if seats are greater than or equal to 1, see that condition is true (1 >= 1), and proceed to decrement. With =1, the second transaction will fail because after the first transaction decrements the count to 0, the condition available\_seats = 1 is no longer true.

Since the >= 1 also decrements, I don't see how this can be true unless sql internally treats range checks as different from equality checks, since the code is exactly the same otherwise. In this quoted block you can replace >= with = and vice versa and there's no logical difference if there's only 1 ticket available.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmddklydl01mead088vell5jb)

Yah shoot my bad I did not realize you all were referring to the one in the atommicity section. Ignore what I said above, sorry. You are right that these are the same and that BOTH work. The one in the atomicity section which we say is bad is not bad. That should be where concertId = "something" not the count. Fixing!!

Sorry for the confusion.

Show more

3

Reply

F

FullTealTrout477

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdadcmhw06ypad082jnw60b5)

what happens when one of the several nodes fail after prepared state in 2 phase commit? How would other nodes know if other nodes are committed/abort? Is that the responsibility of coordinator?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdbzupqm0a6jad09o71zwnt0)

Yah the coordinator is 100% responsible here. If a participant node fails after prepare, the coordinator will timeout waiting for their response and tell everyone to abort. If the coordinator itself fails after getting all prepare-OKs but before sending commit/abort, it must have written its decision to persistent storage first. When it recovers, it reads this log and resends the commit/abort message. If it didn't write to persistent storage, the only safe option is to abort since we can't know if any node got a commit message.

Show more

0

Reply

![Frank Zhao](https://lh3.googleusercontent.com/a/ACg8ocKCCx9RIAiPS5xtHnwyirK6QnyJYHm4QJaTmglmaeBfU9zqfH0=s96-c)

Frank Zhao

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdaktfa0091pad088q20jmal)

> Yelp/Review Systems - Great example of optimistic concurrency control because when users submit reviews, you need to update the business's average rating. Multiple concurrent reviews for the same restaurant create contention, so you can use the current rating and review count as your "version" and only update if they match what you read initially. This prevents rating calculations from getting corrupted when reviews arrive simultaneously.

Thanks for the great content! But I’m curious—why is concurrency control necessary in this specific case? IMHO:

-   The Yelp/Review System can tolerate eventual consistency.
-   Ratings and reviews can be reconciled; exact accuracy isn't critical. An estimated star rating and slightly delayed or stale comments are usually acceptable.
-   Rejecting a user review/rating due to concurrency issues would result in poor UX.

Show more

2

Reply

![Qing Zhao](https://lh3.googleusercontent.com/a/ACg8ocLRejhUyGqwRP9HoTZwfn3nMJjd2pKDLTaeaLGjhUvv7JRcFCU=s96-c)

Qing Zhao

[• 12 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme9x9f6z0cbpad071qmcae5y)

You can auto retry multiple times when concurrency issue detected?

Show more

0

Reply

![Ashish Rana](https://lh3.googleusercontent.com/a/ACg8ocKYKTq4kDPe11S64YU_gBc1k8QZ1BJyPE3-Cscxf7imdhaBgw=s96-c)

Ashish Rana

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdanj3hx09nqad08xohcqfgw)

I have one doubts. If there are millions of request is coming and we have a Pessimistic locking. Then It would slow our write. won't it cause system slow down due to thread are waiting and request timeouts?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdbztpnu0a65ad09sh6iph3u)

Yah excatly! This is why you'd either use OCC if contention is low or buffer with a message queue. See more in the "What about performance when everyone wants the same resource?" deep dive

Show more

0

Reply

![AdvisoryMaroonPanda905](https://lh3.googleusercontent.com/a/ACg8ocKINVTuM6iGCkKoAwJgi7DDM92miXv3WYeZLJtkZCy9GUsubd4F=s96-c)

AdvisoryMaroonPanda905

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdbur2an08qnad092twqt0tc)

hi Evan, Stefan,

I got probed on handing contentions woth accessing multiple vendor APIs and potential limitations with vendor APIs Oauth tokens using refresh tokens. I mentioned using a caching mechanism to store the tokens with set TTL where the server can reuse valid tokens and handle refreshing tokens as needed. The interviewer was pushing for storing the tokens in the DB which I challenged since the caching layer is quicker and can prevent a trip to the DB. what trade offs have I not considered that would make the DB solution more preferred? Are there resources thst you recommend that has explores challenged with handling integrations with external services? much thanks

Show more

0

Reply

BC

Bhaskar Chatterjee

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmddov3ag030zad09xxct7xoa)

in sharded database with replica count N what should be nature of transactions , distributed or single node? How to resolve conflicts in such cases?

Show more

0

Reply

![sameer A](https://lh3.googleusercontent.com/a/ACg8ocIMZiadviONoTQ3nL2TfXxwutpfpCO_89_A_IHuZaK1Mi9jWg=s96-c)

sameer A

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdeisch302apad08e83o8eqo)

I don't quite understand why we'd need concurrency control for a Yelp review/rating system. Also why would we need to store review\_count in the table.

Can we not just do something like _SELECT AVG(ratings) FROM restaurants_ and then maybe cache that value in Redis{business\_id: avg rating}. Later we just run a batch job to update that value every few hours once new reviews come in. Would that be ok?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdeo5f9503eoad08iplnjbn8)

Checkout the yelp breakdown. What you describe is certainly an option. Tradeoff is just how much eventual consistency you want to tolerate.

Show more

0

Reply

Z

znoori455

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdgwf6um0247ad07pjzad7h4)

This is absolute gold, thank you for taking the time!

One question that I still have is when should we use Pessimistic Locking like below:

`BEGIN TRANSACTION; -- Lock the row first to prevent race conditions SELECT available_seats FROM concerts  WHERE concert_id = 'weeknd_tour'  FOR UPDATE; -- Now safely update the seat count UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'    AND available_seats >= 1;`

vs a database constraint on the available\_seats column:

`CREATE TABLE concerts (     available_seats numeric CONSTRAINT positive_seats CHECK (available_seats > 0)     ... );`

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdi35owi04ahad08gkeo2wu0)

The constraint prevents negative values but doesn't prevent double-booking. Two concurrent transactions could both read 1 seat, both calculate 1-1=0 (which satisfies the constraint), and you end up overselling tickets even though the final seat count appears valid.

So you always still need proper concurrency control, the constraint does not solve that issue

Show more

0

Reply

Z

znoori455

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdi4kosl04hiad08lhx324fp)

Ah ok, I was under the impression constraints provide atomic read-modify-write operations similar to a single threaded Redis instance INCR operation, but thanks for clarifying this is not the case.

This concept was suggested in Ch. 7 Page 214 of "System Design Interview - Volume 2" by Alex Xu. I'll just avoid suggesting database constraints all together during my interviews. Thanks for the help!

Show more

0

Reply

![Reetuparna Mukherjee](https://lh3.googleusercontent.com/a/ACg8ocI9LjcG7cfVG_hLWZ4zrXNU-xqiVVA9JcmMaH6e3cGQwu25iMmC=s96-c)

Reetuparna Mukherjee

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdhthinv01head08t44k63ub)

"You should stay within a single database" - does this mean a single node instance? In what scenarios would this be the right choice? Also, doesn't this imply the risk of a single point of failure?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdhtv94r01lrad089gmq4cmm)

You can (and should) have replicas, failover, etc. The point is to keep related data that needs strong consistency in the same database system rather than splitting across different database types/systems. Modern databases can handle enormous scale (Instagram ran on a single PostgreSQL instance for years), and the complexity cost of distributed transactions is massive. So unless you have a genuine need to split data (like geographic distribution requirements), keeping it in one database system with proper replication is almost always the right choice.

Show more

1

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 2 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmenym2fe05scad07za5uzepe)

HI, I am looking to understand when we should go for multi region , and does multi region means data is sharded?

Show more

0

Reply

BC

Bhaskar Chatterjee

[• 30 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdjd5sur028had08ozhrph97)

Hi Evan , could you please explain how distributed transaction works in modern cloud native SQL databases like Cockroach DB? It seems a combination of MVCC and 2PC i suppose that kind of deal with the contention also guarantees ACID.

Show more

0

Reply

T

ToxicSushi

[• 29 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdkpyvaq00h3ad07hhlajchb)

Love the write-up - thanks for putting so much relevant info into a single document! Small suggestion - could you guys also mention "fencing tokens" here please?

When distributed locks rely on timing and TTL (which they probably will if you want to guarantee progress) you potentially need to consider race conditions that could happen around the time of lock expiry (due to network delays or the process stalling / GCing etc). The risk is that process A still thinks it has a lock, but actually the lock has expired and is now being held by process B.

A solution to this is a "fencing token", which is basically a monotonically increasing "version" associated with the lock. You validate these tokens at the storage/DB layer (essentially using a form of optimistic concurrency control) and reject any updates associated with tokens that go back-in-time.

Source: https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html

Show more

0

Reply

![HottieAsian812](https://lh3.googleusercontent.com/a/ACg8ocKai4Y6bQ-snSpt6qK5zD7pHvwBjHMSL-EvCgEY6Z4zsFUHGw=s96-c)

HottieAsian812

[• 27 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdneyx6r08a5ad08g7u1k3f7)

A bit confused on the race condition timeline at the very beginning. Should look at the image from right to left and shouldn't it say "Decrement seat count to -1" for Bob? Thanks.

Show more

1

Reply

![MOHIT ANAND](https://lh3.googleusercontent.com/a/ACg8ocIDaqPf5hGoO6nZ61RcRAqxdhKBS2Tn_93V9IBHquWHJeTPnw=s96-c)

MOHIT ANAND

[• 25 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdqdexdv07doad08ejfvk0od)

what is the difference between a locking mechanism (pessimistic/optimistic) and isolation levels. Which one should one prefer? From whatever I read, it feels like locking is a lower implementation level detail, which is used by isolation levels. Can you please confirm? Thanks

Show more

0

Reply

![Kartik Dutta](https://lh3.googleusercontent.com/a/ACg8ocKYwiZrWfQW4-orGP3B5id1__SEB1u-g4_g0uwcv4u6y57Wk7yk=s96-c)

Kartik Dutta

[• 24 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdsk0s3r07vmad088syaie8y)

I see this phrase mentioned a few times, but it is not clarified: "Application-level status coordination". Could someone be kind enough to explain this? Thank you

Show more

0

Reply

B

BottomTealSkunk679

[• 23 days ago• edited 23 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdt0uixr0b7gad08oz8gf1fg)

I don't see the difference between the first example and the OCC example.

The first example has this:

`UPDATE concerts  SET available_seats = available_seats - 1 WHERE concert_id = 'weeknd_tour'    AND available_seats >= 1;`

and the OCC example has this:

`UPDATE concerts  SET available_seats = available_seats - 1, version = version + 1 WHERE concert_id = 'weeknd_tour'    AND version = 42;`

In both situations the where is confirming the current row has a certain value set (available seats or version) and only making the update if the conditions is met. I don't see how they are really different.

Should the first example, the non-OCC example, instead have a SELECT check the available seats in code and then make an update w/o a WHERE condition?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 23 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdt0vzgm0b7sad08lfkkh8vv)

Could've sworn i removed that. Imagine no AND available\_seats >= 1; on the first one. That should not be there. Removing!

Show more

0

Reply

B

BottomTealSkunk679

[• 23 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmdt1jjha0bm7ad08c72q51d2)

Great, makes perfect sense to me now. Thanks for the lightning fast update and great write-up!

Show more

0

Reply

![John Moody](https://lh3.googleusercontent.com/a/ACg8ocJXQvSm_VtyVdQ6FWR27cT1TEAPlUdnI9b1ZSNVLvTWZopRSQ=s96-c)

John Moody

[• 22 days ago• edited 16 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmduw0nz10dexad08edjc0253)

""always lock the user initiating the action first, then lock recipients in ascending ID order." - doesn't this still allow the Alice<->Bob deadlock scenario it's meant to resolve?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 16 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme3b3wta01lzad08sojc163q)

If both Alice (ID: 123) and Bob (ID: 456) try to transfer to each other simultaneously, both transactions will attempt to lock Alice first (ID 123 < 456), then Bob. This means one transaction will get Alice's lock first and proceed, while the other waits and no circular dependency are possible. The key is that BOTH transactions follow the same order regardless of who initiates the transfer.

Show more

2

Reply

![John Moody](https://lh3.googleusercontent.com/a/ACg8ocJXQvSm_VtyVdQ6FWR27cT1TEAPlUdnI9b1ZSNVLvTWZopRSQ=s96-c)

John Moody

[• 16 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme3bj2yx01ruad08kvzlyiiv)

I must just be misreading "always lock the user initiating the action first," because what you're saying here makes sense, but I read the quoted sentence as Alice locks Alice first then everything else in order, and Bob locks Bob first and then everything else in order; which (to my mind), still allows for the circular dependency. If Bob locks Alice first, even though Bob initiated the action, then agreed.

Show more

0

Reply

![Shanc](https://lh3.googleusercontent.com/a/ACg8ocJBzCqumXHIHmw1uJgZu9b-hYbCn45YlDlhiveYiE_dMIG7DA=s96-c)

Shanc

[• 17 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme2jcx660539ad08e6yr5ymn)

Nice article and walk throughs for each options and summarizes it very well. I could easily summarize all the info here into a page for revision before an interview. Thanks @helloInterview. very well done and the tagging of different sys design problems with these patterns is really helpful as well.

Show more

0

Reply

C

CommercialPinkMarmot541

[• 16 days ago• edited 16 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme35lssd002yad089fsyt1es)

You explain how to prevent a deadlock "or establishing conventions like "always lock the user initiating the action first, then lock recipients in ascending ID order."

Wouldn't that exactly what will create the deadlock and not prevent it?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 16 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme3b2vvq01llad08in9gjkej)

Having a consistent ordering is what prevents deadlocks :) If both transactions always lock users in ascending ID order (e.g. always lock ID 123 before 456), then it's impossible to get into a circular wait. Transaction A can't hold 123 while waiting for 456 while Transaction B holds 456 while waiting for 123, because they both must acquire 123 first. One will get it, the other will wait, then the first will get 456 and complete, allowing the second to proceed. Hope that makes sense!

Show more

0

Reply

C

CommercialPinkMarmot541

[• 16 days ago• edited 16 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme3bmqow01srad08br42025b)

Thanks, the concept of ordering is clear, but it is said that: "or establishing conventions like "always lock the user initiating the action first, then lock recipients in ascending ID order." And its still not clear to me why do I need to lock the initiating user 1st? I would rather lock them both by asc IDs as you suggested in the reply, regardless of **who initiated** the trxns.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 16 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme3bn6tu01tkad081fpizlgr)

I agree its confusing. let me reword

Show more

1

Reply

![Qing Zhao](https://lh3.googleusercontent.com/a/ACg8ocLRejhUyGqwRP9HoTZwfn3nMJjd2pKDLTaeaLGjhUvv7JRcFCU=s96-c)

Qing Zhao

[• 13 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cme8ex3ci02t6ad07m4wmli0o)

Great article! One question re Distributed lock, DB column approach. Seems like it only works if you store the locks in a shared centralised database? So sounds like it is still single database, not multiple DBs in a distributed environment?

Show more

0

Reply

![Kongtao Chen](https://lh3.googleusercontent.com/a/ACg8ocJZ6I11Vd_K4yj0kN8fOd3RDAKvSpLwZ4u1FLB_-4lWX3xyUw=s96-c)

Kongtao Chen

[• 7 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmeg6hauq04w9ad07pds764h9)

An in-memory, single-thread-per-partition database can also solve contention issues if the data fits into memory.

Show more

1

Reply

JN

Joshua Nance

[• 3 days ago](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention#comment-cmembjgrj021mad081hbrwe4u)

For the hot partition problem, using the concert example, would a queue open you up to a potential issue where scalpers fill up much of the queue before real users?

I'm also wondering if you could have exceptional events like that be modeled as the sum of multiple "sub-events," so to speak, each with their own ticket totals and which could be sharded on subevent ID.

Show more

0

Reply

