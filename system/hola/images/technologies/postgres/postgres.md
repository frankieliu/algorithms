# PostgreSQL

Learn when and how to use PostgreSQL in your system design interviews

* * *

There's a good chance you'll find yourself discussing PostgreSQL in your system design interview. After all, it's consistently ranked as the most beloved database in [Stack Overflow's developer survey](https://survey.stackoverflow.co/2023/#section-most-popular-technologies-databases) and is used by companies from Reddit to Instagram and even the very website you're reading right now.

That said, it's important to understand that while PostgreSQL is packed with features and capabilities, your interviewer isn't looking for a database administrator. They want to see that you can make informed architectural decisions. When should you choose PostgreSQL? When should you look elsewhere? What are the key trade-offs to consider?

I often see candidates get tripped up here. They either dive too deep into PostgreSQL internals (talking about MVCC and WAL when the interviewer just wants to know if it can handle their data relationships), or they make overly broad statements like "NoSQL scales better than PostgreSQL" without understanding the nuances.

In this deep dive, we'll focus specifically on what you need to know about PostgreSQL for system design interviews. We'll start with a practical example, explore the key capabilities and limits that should inform your choices, and build up to common interview scenarios.

For this deep dive, we're going to assume you have a basic understanding of SQL. If you don't, I've added an [Appendix: Basic SQL Concepts](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#appendix-basic-sql-concepts) at the end of this page for you to review.

Let's get started.

### A Motivating Example

Let's build up our intuition about PostgreSQL through a concrete example. Imagine we're designing a social media platform - not a massive one like Facebook, but one that's growing and needs a solid foundation.

Our platform needs to handle some fundamental relationships:

-   Users can create posts
    
-   Users can comment on posts
    
-   Users can follow other users
    
-   Users can like both posts and comments
    
-   Users can create direct messages (DMs) with other users
    

This is exactly the kind of scenario that comes up in interviews. The relationships between entities are clear but non-trivial, and there are interesting questions about data consistency and scaling.

What makes this interesting from a database perspective? Well, different operations have different requirements:

-   Multi-step operations like creating DM threads need to be atomic (creating the thread, adding participants, and storing the first message must happen together)
    
-   Comment and follow relationships need referential integrity (you can't have a comment without a valid post or follow a non-existent user)
    
-   Like counts can be eventually consistent (it's not critical if it takes a few seconds to update)
    
-   When someone requests a user's profile, we need to efficiently fetch their recent posts, follower count, and other metadata
    
-   Users need to be able to search through posts and find other users
    
-   As our platform grows, we'll need to handle more data and more complex queries
    

This combination of requirements - complex relationships, mixed consistency needs, search capabilities, and room for growth - makes it a perfect example for exploring PostgreSQL's strengths and limitations. Throughout this deep dive, we'll keep coming back to this example to ground our discussion in practical terms.

## Core Capabilities & Limitations

With a motivating example in place, let's dive into what PostgreSQL can and can't do well. Most system design discussions about PostgreSQL will center around its read performance, write capabilities, consistency guarantees, and schema flexibility. Understanding these core characteristics will help you make informed decisions about when to use PostgreSQL in your design.

### Read Performance

First up is read performance - this is critical because in most applications, reads vastly outnumber writes. In our social media example, users spend far more time browsing posts and profiles than they do creating content.

In system design interviews, you don't need to dive into query planner internals. Instead, focus on practical performance patterns and when different types of indexes make sense.

When a user views a profile, we need to efficiently fetch all posts by that user. Without proper indexing, PostgreSQL would need to scan every row in the posts table to find matching posts - a process that gets increasingly expensive as our data grows. This is where indexes come in. By creating an index on the user\_id column of our posts table, we can quickly locate all posts for a given user without scanning the entire table.

#### Basic Indexing

The most fundamental way to speed up reads in PostgreSQL is through indexes. By default, PostgreSQL uses B-tree indexes, which work great for:

-   Exact matches (WHERE email = '[user@example.com](mailto:user@example.com)')
    
-   Range queries (WHERE created\_at > '2024-01-01')
    
-   Sorting (ORDER BY username if the ORDER BY column match the index columns' order)
    

By default, PostgreSQL will create a B-tree index on your primary key column, but you also have the ability to create indexes on other columns as well.

`-- This is your bread and butter index CREATE INDEX idx_users_email ON users(email); -- Multi-column indexes for common query patterns CREATE INDEX idx_posts_user_date ON posts(user_id, created_at);`

A common trap in interviews is to suggest adding indexes for every column. Remember that each index:

-   Makes writes slower (as the index must be updated)
    
-   Takes up disk space
    
-   May not even be used if the query planner thinks a sequential scan would be faster
    

#### Beyond Basic Indexes

Where PostgreSQL really shines is its support for specialized indexes. These come up frequently in system design interviews because they can eliminate the need for separate specialized databases:

**Full-Text Search** using GIN indexes. Postgres supports full-text search out of the box using GIN (Generalized Inverted Index) indexes. GIN indexes work like the index at the back of a book - they store a mapping of each word to all the locations where it appears. This makes them perfect for full-text search where you need to quickly find documents containing specific words:

`-- Add a tsvector column for search ALTER TABLE posts ADD COLUMN search_vector tsvector; CREATE INDEX idx_posts_search ON posts USING GIN(search_vector); -- Now you can do full-text search SELECT * FROM posts  WHERE search_vector @@ to_tsquery('postgresql & database');`        

For many applications, this built-in search capability means you don't need a separate [Elasticsearch](https://www.hellointerview.com/learn/system-design/deep-dives/elasticsearch) cluster. It supports everything from:

1.  Word stemming (finding/find/finds all match)
    
2.  Relevance ranking
    
3.  Multiple languages
    
4.  Complex queries with AND/OR/NOT
    

While PostgreSQL's full-text search is powerful, it may not fully replace Elasticsearch for all use cases. Consider Elasticsearch when you need:

-   More sophisticated relevancy scoring
    
-   Faceted search capabilities
    
-   Fuzzy matching and "search as you type" features
    
-   Distributed search across very large datasets
    
-   Advanced analytics and aggregations
    

Start with PostgreSQL's built-in search for simpler use cases. Only introduce Elasticsearch when you have specific requirements that PostgreSQL's search capabilities can't meet. This keeps your architecture simpler and reduces operational complexity.

JSONB columns with GIN indexes are particularly useful when you need flexible metadata on your posts. For example, in our social media platform, each post might have different attributes like location, mentioned users, hashtags, or attached media. Rather than creating separate columns for each possibility, we can store this in a JSONB column (giving us the flexibility to add new attributes as needed just like we would in a NoSQL database!).

`-- Add a JSONB column for post metadata ALTER TABLE posts ADD COLUMN metadata JSONB; CREATE INDEX idx_posts_metadata ON posts USING GIN(metadata); -- Now we can efficiently query posts with specific metadata SELECT * FROM posts  WHERE metadata @> '{"type": "video"}'    AND metadata @> '{"hashtags": ["coding"]}'; -- Or find all posts that mention a specific user SELECT * FROM posts  WHERE metadata @> '{"mentions": ["user123"]}';`

Geospatial Search with PostGIS. While not built into PostgreSQL core, the PostGIS extension adds powerful spatial capabilities. Just like we can index text for fast searching, PostGIS lets us index location data for efficient geospatial queries. This is perfect for our social media platform when we want to show users posts from their local area:

`-- Enable PostGIS CREATE EXTENSION postgis; -- Add a location column to posts ALTER TABLE posts  ADD COLUMN location geometry(Point); -- Create a spatial index CREATE INDEX idx_posts_location  ON posts USING GIST(location); -- Find all posts within 5km of a user SELECT * FROM posts  WHERE ST_DWithin(     location::geography,     ST_MakePoint(-122.4194, 37.7749)::geography, -- SF coordinates     5000  -- 5km in meters );`

PostGIS is incredibly powerful - it can handle:

-   Different types of spatial data (points, lines, polygons)
    
-   Various distance calculations (as-the-crow-flies, driving distance)
    
-   Spatial operations (intersections, containment)
    
-   Different coordinate systems
    

PostGIS is so capable that companies like Uber initially used it for their entire ride-matching system. While they've since moved to custom solutions for scale, it shows how far you can go with PostgreSQL's extensions before needing specialized databases.

The index type used here (GIST) is specifically optimized for geometric data, using R-tree indexing under the hood. This means queries like "find all posts within X kilometers" or "find posts inside this boundary" can be executed efficiently without having to check every single row.

Just like with full-text search, you should consider PostGIS before reaching for a specialized geospatial database. It's another example of getting sophisticated functionality while keeping your architecture simple.

Better yet, we can combine all these capabilities to create rich search experiences. For example, we can find all video posts within 5km of San Francisco that mention "food" in their content and are tagged with "restaurant":

`SELECT * FROM posts  WHERE search_vector @@ to_tsquery('food')   AND metadata @> '{"type": "video", "hashtags": ["restaurant"]}'   AND ST_DWithin(     location::geography,     ST_MakePoint(-122.4194, 37.7749)::geography,     5000   );`

Let me rewrite this section to better explain the why and connect the concepts:

#### Query Optimization Essentials

So far we've covered the different types of indexes PostgreSQL offers, but there's more to query optimization than just picking the right index type. Let's look at some advanced indexing strategies that can dramatically improve read performance.

##### Covering Indexes

When PostgreSQL uses an index to find a row, it typically needs to do two things:

1.  Look up the value in the index to find the row's location
    
2.  Fetch the actual row from the table to get other columns you need
    

But what if we could store all the data we need right in the index itself? That's what covering indexes do:

`-- Let's say this is a common query in our social media app: SELECT title, created_at  FROM posts  WHERE user_id = 123  ORDER BY created_at DESC; -- A covering index that includes all needed columns CREATE INDEX idx_posts_user_include  ON posts(user_id) INCLUDE (title, created_at);`

Covering indexes can make queries significantly faster because PostgreSQL can satisfy the entire query just from the index without touching the table. The trade-off is that the index takes up more space and writes become slightly slower.

##### Partial Indexes

Sometimes you only need to index a subset of your data. For example, in our social media platform, most queries are probably looking for active users, not deleted ones:

`-- Standard index indexes everything CREATE INDEX idx_users_email ON users(email);  -- Indexes ALL users -- Partial index only indexes active users CREATE INDEX idx_active_users  ON users(email) WHERE status = 'active';  -- Smaller, faster index`

Partial indexes are particularly effective in scenarios where most of your queries only need a subset of rows, when you have many "inactive" or "deleted" records that don't need to be indexed, or when you want to reduce the overall size and maintenance overhead of your indexes. By only indexing the relevant subset of data, partial indexes can significantly improve both query performance and resource utilization.

##### Practical Performance Limits

There is a good chance that during your non-functional requirements you outlined some latency goals. Ideally, you even quantified them! That means that as you go deep into the design, you need some basic performance numbers in mind. These numbers are very rough estimates as real numbers depend heavily on the hardware and the specific workload. That said, estimates should be enough to get you started in an interview.

1.  **Query Performance**:
    
    -   Simple indexed lookups: tens of thousands per second per core
        
    -   Complex joins: thousands per second
        
    -   Full-table scans: depends heavily on whether data fits in memory
        
    
2.  **Scale Limits**:
    
    -   Tables start getting unwieldy past 100M rows
        
    -   Full-text search works well up to tens of millions of documents
        
    -   Complex joins become challenging with tables >10M rows
        
    -   Performance drops significantly when working set exceeds available RAM
        
    

These aren't hard limits - PostgreSQL can handle much more with proper optimization. But they're good rules of thumb for when you should start considering partitioning, sharding, or other scaling strategies.

Keep in mind, memory is king when it comes to performance! Queries that can be satisfied from memory are orders of magnitude faster than those requiring disk access. As a rule of thumb, you should try to keep your working set (frequently accessed data) in RAM for optimal performance.

In your interview, showing knowledge of these practical limits helps demonstrate that you understand not just how to use PostgreSQL, but when you might need to consider alternatives or additional optimization strategies.

### Write Performance

Now let's talk about writes. While reads might dominate most workloads, write performance is often more critical because it directly impacts user experience - nobody wants to wait seconds after hitting "Post" for their content to appear.

When a write occurs in PostgreSQL, several steps happen to ensure both performance and durability:

1.  **Transaction Log (WAL) Write \[Disk\]**: Changes are first written to the Write-Ahead Log (WAL) on disk. This is a sequential write operation, making it relatively fast. The WAL is critical for durability - once changes are written here, the transaction is considered durable because even if the server crashes, PostgreSQL can recover the changes from the WAL.
    
2.  **Buffer Cache Update \[Memory\]**: Changes are made to the data pages in PostgreSQL's shared buffer cache, where the actual tables and indexes live in memory. When pages are modified, they're marked as "dirty" to indicate they need to be written to disk eventually.
    
3.  **Background Writer \[Memory → Disk\]**: Dirty pages in memory are periodically written to the actual data files on disk. This happens asynchronously through the background writer, when memory pressure gets too high, or when a checkpoint occurs. This delayed write strategy allows PostgreSQL to batch multiple changes together for better performance.
    
4.  **Index Updates \[Memory & Disk\]**: Each index needs to be updated to reflect the changes. Like table data, index changes also go through the WAL for durability. This is why having many indexes can significantly slow down writes - each index requires additional WAL entries and memory updates.
    

This architecture is why PostgreSQL can be fast for writes - most of the work happens in memory, while ensuring durability through the WAL. The actual writing of data pages to disk happens later and is optimized for batch operations.

The practical implication of this design is that write performance is typically bounded by how fast you can write to the WAL (disk I/O), how many indexes need to be updated, and how much memory is available for the buffer cache.

#### Throughput Limitations

Now we know about what happens when a write occurs in PostgreSQL, before we go onto optimizations, let's first talk about the practical limits of write throughput. This is important to know as it will help you decide whether PostgreSQL is a good fit for your system.

A well-tuned PostgreSQL instance on good (not great) hardware can handle:

-   Simple inserts: ~5,000 per second per core
    
-   Updates with index modifications: ~1,000-2,000 per second per core
    
-   Complex transactions (multiple tables/indexes): Hundreds per second
    
-   Bulk operations: Tens of thousands of rows per second
    

These numbers assume PostgreSQL's default transaction isolation level (Read Committed), where transactions only see data that was committed before their query began. If you change the default isolation level these numbers can go up or down.

What affects these limits? Several factors:

-   Hardware: Write throughput is often bottlenecked by disk I/O for the WAL
    
-   Indexes: Each additional index reduces write throughput
    
-   Replication: If configured, synchronous replication adds latency as we wait for replicas to confirm
    
-   Transaction Complexity: More tables or indexes touched = slower transactions
    

Remember, were talking about a single node here! So if your system has higher write throughput that, say, 5k writes per second, this does not mean that PostgreSQL is off the table, it just means that you are going to need to shard your data across multiple nodes/machines.

Let me continue from there, building on our social media example:

#### Write Performance Optimizations

Ok, a single node can handle around 5k writes per second, so what can we do? How can we improve our write performance if we require more than that? Let's look at strategies ranging from simple optimizations to architectural changes.

We have a few options:

1.  Batch Processing
    
2.  Vertical Scaling
    
3.  Write Offloading
    
4.  Table Partitioning
    
5.  Sharding
    

Let's discuss each of these in turn.

**1\. Vertical Scaling** Before jumping to complex solutions, we can always consider just upgrading our hardware. This could mean using faster NVMe disks for better WAL performance, adding more RAM to increase the buffer cache size, or upgrading to CPUs with more cores to handle parallel operations more effectively.

This usually isn't the most compelling solution in an interview, but it's a good place to start.

**2\. Batch Processing** The simplest optimization is to batch writes together. Instead of processing each write individually, we collect multiple operations and execute them in a single transaction. For example, instead of inserting 1000 likes one at a time, we can insert them all in a single transaction. This means we're buffering writes in our server's memory before committing them to disk. The risk here is clear, if we crash in the middle of a batch, we'll lose all the writes in that batch.

`-- Instead of 1000 separate inserts: INSERT INTO likes (post_id, user_id) VALUES    (1, 101), (1, 102), ..., (1, 1000);`

**3\. Write Offloading** Some writes don't need to happen synchronously. For example, analytics data, activity logs, or aggregated metrics can often be processed asynchronously. Instead of writing directly to PostgreSQL, we can:

1.  Send writes to a message queue (like Kafka)
    
2.  Have background workers process these queued writes in batches
    
3.  Optionally maintain a separate analytics database
    

This pattern works especially well for handling activity logging, analytics events, metrics aggregation, and non-critical updates like "last seen" timestamps. These types of writes don't need to happen immediately and can be processed in the background without impacting the core user experience.

**4\. Table Partitioning**

For large tables, partitioning can improve both read and write performance by splitting data across multiple physical tables. The most common use case is time-based partitioning. Going back to our social media example, let's say we have a posts table that grows by millions of rows per month:

`CREATE TABLE posts (     id SERIAL,     user_id INT,     content TEXT,     created_at TIMESTAMP ) PARTITION BY RANGE (created_at); -- Create partitions by month CREATE TABLE posts_2024_01 PARTITION OF posts    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');`

Why does this help writes? First, different database sessions can write to different partitions simultaneously, increasing concurrency. Second, when data is inserted, index updates only need to happen on the relevant partition rather than the entire table. Finally, bulk loading operations can be performed partition by partition, making it easier to load large amounts of data efficiently.

Conveniently, it also helps with reads. When users view recent posts, PostgreSQL only needs to scan the recent partitions. No need to wade through years of historical data.

A common pattern is to keep recent partitions on fast storage (like NVMe drives) while moving older partitions to cheaper storage. Users get fast access to recent data, which is typically what they care about most.

This is a great strategy for data that has a natural lifecycle like social media posts where new ones are most relevant.

**5\. Sharding** This is the most common solution in an interview. When a single node isn't enough, sharding lets you distribute writes across multiple PostgreSQL instances. You'll just want to be clear about what you're sharding on and how you're distributing the data.

For example, we may consider sharing our posts by user\_id. This way, all the data for a user lives on a single shard. This is important, because when we go to read the data we want to avoid cross-shard queries where we need to scatter-gather data from multiple shards.

You typically want to shard on the column that you're querying by most often. So if we typically query for all posts from a given user, we'll shard by user\_id.

Sharding adds complexity - you'll need to handle cross-shard queries, maintain consistent schemas across shards, and manage multiple databases. Only introduce it when simpler optimizations aren't sufficient.

Unlike DynamoDB, PostgreSQL doesn't have a built-in sharding solution. You'll need to implement sharding manually, which can be a bit of a challenge. Alternatively, you can use managed services like [Citus](https://www.citusdata.com/) which handles many of the sharding complexities for you.

### Replication

While we've discussed how to optimize write performance on a single node, most real-world deployments use replication for two key purposes:

1.  Scaling reads by distributing queries across replicas
    
2.  Providing high availability in case of node failures
    

Replication is the process of copying data from one database to one or more other databases. This is a key part of PostgreSQL's scalability and availability story.

PostgreSQL supports two main types of replication: synchronous and asynchronous. In synchronous replication, the primary waits for acknowledgment from replicas before confirming the write to the client. With asynchronous replication, the primary confirms the write to the client immediately and replicates changes to replicas in the background. While the technical details may not come up in an interview, understanding these tradeoffs is important - synchronous replication provides stronger consistency but higher latency, while asynchronous replication offers better performance but potential inconsistency between replicas.

Many organizations use a hybrid approach: keeping a small number of synchronous replicas for stronger consistency while maintaining additional asynchronous replicas for read scaling. PostgreSQL allows you to specify which replicas should be synchronous.

#### Scaling reads

The most common use for replication is to scale read performance. By creating read replicas, you can distribute read queries across multiple database instances while sending all writes to the primary. This is particularly effective because most applications are read-heavy.

Let's go back to our social media example. When users browse their feed or view profiles, these are all read operations that can be handled by any replica. Only when they create posts or update their profile do we need to use the primary. Now we have multiplied our read throughput by N where N is the number of replicas.

There's one key caveat with read replicas: replication lag. If a user makes a change and immediately tries to read it back, they might not see their change if they hit a replica that hasn't caught up yet. This is known as "read-your-writes" consistency.

#### High Availability

The second major benefit of replication is high availability. By maintaining copies of your data across multiple nodes, you can handle hardware failures without downtime. If your primary node fails, one of the replicas can be promoted to become the new primary.

This failover process typically involves:

1.  Detecting that the primary is down
    
2.  Promoting a replica to primary
    
3.  Updating connection information
    
4.  Repointing applications to the new primary
    

In your interview, emphasize that replication isn't just about scaling - it's about reliability. You might say: "We'll use replication not just for distributing read load, but also to ensure our service stays available even if we lose a database node."

Most teams use managed PostgreSQL services (like AWS RDS or GCP Cloud SQL) that handle the complexities of failover automatically. In your interview, it's enough to know that failover is possible and roughly how it works - you don't need to get into the details of how to configure it manually.

### Data Consistency

If you've chosen to prioritize consistency over availability in your non-functional requirements, then PostgreSQL is a strong choice. It's built from the ground up to provide strong consistency guarantees through ACID transactions. However, simply choosing PostgreSQL isn't enough - you need to understand how to actually achieve the consistency your system requires.

A common mistake in interviews is to say "We'll use PostgreSQL because it's ACID compliant" without being able to explain how you'll actually use those ACID properties to solve your consistency requirements.

#### Transactions

One of the most common points of discussion in interviews ends up being around transactions. A transaction is a set of operations that are executed together and must either all succeed or all fail together. This is the foundation for ensuring consistency in PostgreSQL.

Let's consider a simple example where we need to transfer money between two bank accounts. We need to ensure that if we deduct money from one account, it must be added to the other account. Neither operation can happen in isolation:

`BEGIN; UPDATE accounts SET balance = balance - 100 WHERE id = 1; UPDATE accounts SET balance = balance + 100 WHERE id = 2; COMMIT;`

This transaction ensures atomicity - either both updates happen or neither does. However, transactions alone don't ensure consistency in all scenarios, particularly when multiple transactions are happening concurrently.

##### Transactions and Concurrent Operations

Transactions ensure consistency for a single series of operations, but things get more complicated when multiple transactions are happening concurrently. Remember, in most real applications, you'll have multiple users or services trying to read and modify data at the same time.

This is where many candidates get tripped up in interviews. They understand basic transactions but haven't thought through how to maintain consistency when multiple operations are happening simultaneously.

Let's look at an auction system as an example. Here users place bids on items and we accept bids only if they're higher than the current max bid. A single transaction can ensure that checking the current bid and placing a new bid happen atomically, but what happens when two users try to bid at the same time?

`BEGIN; -- Get current max bid for item 123 SELECT maxBid from Auction where id = 123; -- Place new bid if it's higher INSERT INTO bids (item_id, user_id, amount)  VALUES (123, 456, 100); -- Update the max bid UPDATE Auction SET maxBid = 100 WHERE id = 123; COMMIT;`

Even though this is in a transaction, with PostgreSQL's default isolation level (Read Committed), we could still have consistency problems if two users bid simultaneously. Both transactions could read the same max bid before either commits.

Here's how this could lead to an inconsistent state:

1.  User A's transaction reads current max bid: $90
    
2.  User B's transaction reads current max bid: $90
    
3.  User A places bid for $100
    
4.  User A commits
    
5.  User B places bid for $95
    
6.  User B commits
    

Now we have an invalid state: a $95 bid was accepted after a $100 bid!

There are two main ways we can solve this concurrency issue:

**1\. Row-Level Locking** The simplest solution is to lock the auction row while we're checking and updating bids. By using the FOR UPDATE clause, we tell PostgreSQL to lock the rows we're reading. Other transactions trying to read these rows with FOR UPDATE will have to wait until our transaction completes. This ensures we have a consistent view of the data while making our changes.

`BEGIN; -- Lock the item and get current max bid SELECT maxBid FROM Auction WHERE id = 123 FOR UPDATE; -- Place new bid if it's higher INSERT INTO bids (item_id, user_id, amount)  VALUES (123, 456, 100); -- Update the max bid UPDATE Auction SET maxBid = 100 WHERE id = 123; COMMIT;`

So when you need to ensure that two operations happen atomically, you'll want to emphasize to your interviewer how you will achieve that beyond simply saying "we'll use transactions". Instead, "we'll use transactions and row-level locking on the auction row", for this case.

**2\. Higher Isolation Level** Alternatively, we can use a stricter isolation level:

`BEGIN; SET TRANSACTION ISOLATION LEVEL SERIALIZABLE; -- Same code as before... COMMIT;`

While serializable isolation prevents all consistency anomalies, it comes with a cost: if two transactions conflict, one will be rolled back and need to retry. Your application needs to be prepared to handle these retry scenarios.

PostgreSQL supports three isolation levels, each providing different consistency guarantees:

1.  **Read Committed** (Default) is PostgreSQL's default isolation level that only sees data that was committed before the query began. As transactions execute, each query within a transaction can see new commits made by other transactions that completed after the transaction started. While this provides good performance, it can lead to non-repeatable reads where the same query returns different results within a transaction.
    
2.  **Repeatable Read** in PostgreSQL provides stronger guarantees than the SQL standard requires. It creates a consistent snapshot of the data as of the start of the transaction, and unlike other databases, PostgreSQL's implementation prevents both non-repeatable reads AND phantom reads. This means not only will the same query return the same results within a transaction, but no new rows will appear that match your query conditions - even if other transactions commit such rows.
    
3.  **Serializable** is the strongest isolation level that makes transactions behave as if they were executed one after another in sequence. This prevents all types of concurrency anomalies but comes with the trade-off of requiring retry logic in your application to handle transaction conflicts.
    

PostgreSQL's implementation of Repeatable Read is notably stronger than what the SQL standard requires. While other databases might allow phantom reads at this isolation level, PostgreSQL prevents them. This means you might not need Serializable isolation in cases where you would in other databases.

So, when should you use row-locking and when should you use a higher isolation level?

Aspect

Serializable Isolation

Row-Level Locking

**Concurrency**

Lower - transactions might need to retry on conflict

Higher - only conflicts when touching same rows

**Performance**

More overhead - must track all read/write dependencies

Less overhead - only locks specific rows

**Use Case**

Complex transactions where it's hard to know what to lock

When you know exactly which rows need atomic updates

**Complexity**

Simple to implement but requires retry logic

More explicit in code but no retries needed

**Error Handling**

Must handle serialization failures

Must handle deadlock scenarios

**Example**

Complex financial calculations across multiple tables

Auction bidding, inventory updates

**Memory Usage**

Higher - tracks entire transaction history

Lower - only tracks locks

**Scalability**

Doesn't scale as well with concurrent transactions

Scales better when conflicts are rare

Row-level locking is generally preferred when you know exactly which rows need to be locked. Save serializable isolation for cases where the transaction is too complex to reason about which locks are needed.

## When to Use PostgreSQL (and When Not To)

Let me revise that opening with more concrete technical advantages:

Here's my advice: in your system design interview, PostgreSQL should be your default choice unless you have a specific reason to use something else. Why? Because PostgreSQL:

1.  Provides strong ACID guarantees while still scaling effectively with replication and partitioning
    
2.  Handles both structured and unstructured data through JSONB support
    
3.  Includes built-in solutions for common needs like full-text search and geospatial queries
    
4.  Can scale reads effectively through replication
    
5.  Offers excellent tooling and a mature ecosystem
    

Start with PostgreSQL, then justify why you might need to deviate. This is much stronger than starting with a niche solution and trying to justify why it's better than PostgreSQL.

PostgreSQL shines when you need:

-   Complex relationships between data
    
-   Strong consistency guarantees
    
-   Rich querying capabilities
    
-   A mix of structured and unstructured data (JSONB)
    
-   Built-in full-text search
    
-   Geospatial queries
    

For example, it's perfect for:

-   E-commerce platforms (inventory, orders, user data)
    
-   Financial systems (transactions, accounts, audit logs)
    
-   Content management systems (posts, comments, users)
    
-   Analytics platforms (up to reasonable scale)
    

### When to Consider Alternatives

That said, we aren't maxis over here. There are legitimate reasons to look beyond PostgreSQL.

**1\. Extreme Write Throughput** If you need to handle millions of writes per second, PostgreSQL will struggle because each write requires a WAL entry and index updates, creating I/O bottlenecks even with the fastest storage. Even with sharding, coordinating writes across many PostgreSQL nodes adds complexity and latency. In these cases, you might consider:

-   NoSQL databases (like [Cassandra](https://www.hellointerview.com/learn/system-design/deep-dives/cassandra)) for event streaming
    
-   Key-value stores (like [Redis](https://www.hellointerview.com/learn/system-design/deep-dives/redis)) for real-time counters
    

**2\. Global Multi-Region Requirements** When you need active-active deployment across regions (where multiple regions accept writes simultaneously), PostgreSQL faces fundamental limitations. Its single-primary architecture means one region must be designated as the primary writer, while others act as read replicas. Attempting true active-active deployment creates significant challenges around data consistency and conflict resolution, as PostgreSQL wasn't designed to handle simultaneous writes from multiple primaries. The synchronous replication needed across regions also introduces substantial latency, as changes must be confirmed by distant replicas before being committed. For these scenarios, consider:

-   CockroachDB for global ACID compliance
    
-   [Cassandra](https://www.hellointerview.com/learn/system-design/deep-dives/cassandra) for eventual consistency at global scale
    
-   [DynamoDB](https://www.hellointerview.com/learn/system-design/deep-dives/dynamodb) for managed global tables
    

**3\. Simple Key-Value Access Patterns** If your access patterns are truly key-value (meaning you're just storing and retrieving values by key without joins or complex queries), PostgreSQL is overkill. Its MVCC architecture, WAL logging, and complex query planner add overhead you don't need. In these cases, consider:

-   [Redis](https://www.hellointerview.com/learn/system-design/deep-dives/redis) for in-memory performance
    
-   [DynamoDB](https://www.hellointerview.com/learn/system-design/deep-dives/dynamodb) for managed scalability
    
-   [Cassandra](https://www.hellointerview.com/learn/system-design/deep-dives/cassandra) for write-heavy workloads
    

Scalability alone is not a good reason to choose an alternative to PostgreSQL. PostgreSQL can handle significant scale with proper design.

## Summary

PostgreSQL should be your default choice in system design interviews unless specific requirements demand otherwise. Its combination of ACID compliance, rich feature set, and scalability options make it suitable for a wide range of use cases, from simple CRUD applications to complex transactional systems.

When discussing PostgreSQL in interviews, focus on analyzing concrete requirements around data consistency, query patterns, and scale, rather than following trends. Be prepared to discuss key trade-offs like ACID vs eventual consistency, read vs write scaling strategies, and indexing decisions. Start simple and add complexity only as needed.

PostgreSQL's rich feature set often eliminates the need for additional systems in your architecture. Its full-text search capabilities might replace Elasticsearch, JSONB support could eliminate the need for MongoDB, and PostGIS handles geospatial needs that might otherwise require specialized databases. Built-in replication often provides sufficient scaling capabilities for many use cases. However, it's equally important to recognize when PostgreSQL might not be the best fit, such as cases requiring extreme write scaling or global distribution, where databases like Cassandra or CockroachDB might be more appropriate.

* * *

## Appendix: Basic SQL Concepts

Before diving into PostgreSQL-specific features, let's review how SQL databases organize data. These core concepts apply to any SQL database, not just PostgreSQL but are a necessary foundation that we will build on throughout this deep dive.

### Relational Database Principles

At its core, PostgreSQL stores data in tables (also called relations). Think of a table like a spreadsheet with rows and columns. Each column has a specific data type (like text, numbers, or dates), and each row represents one complete record.

Let's look at a concrete example. Imagine we're designing a social media platform. We might have a users table that looks like this:

`CREATE TABLE users (     id SERIAL PRIMARY KEY,     username VARCHAR(50) UNIQUE NOT NULL,     email VARCHAR(255) UNIQUE NOT NULL,     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP );`

This command would create the following table (data is just for example):

id

username

email

created\_at

1

johndoe

[john@example.com](mailto:john@example.com)

2024-01-01 10:00:00

2

janedoe

[jane@example.com](mailto:jane@example.com)

2024-01-01 10:05:00

3

bobsmith

[bob@example.com](mailto:bob@example.com)

2024-01-01 10:10:00

When a new user signs up, we create a new row in this table. Each user gets a unique id (that's what PRIMARY KEY means), and we ensure no two users can have the same username or email (that's what UNIQUE does).

But users aren't much fun by themselves. They need to be able to post content. Here's where the "relational" part of relational databases comes in. We can create a posts table that's connected to our users:

`CREATE TABLE posts (     id SERIAL PRIMARY KEY,     user_id INTEGER REFERENCES users(id),     content TEXT NOT NULL,     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP );`

id

user\_id

content

created\_at

1

1

Hello, world!

2024-01-01 10:00:00

2

1

My first post

2024-01-01 10:05:00

3

2

Another post

2024-01-01 10:10:00

See REFERENCES users(id)? That's called a foreign key - it creates a relationship between posts and users. Every post must belong to a valid user, and PostgreSQL will enforce this for us. This is one of the key strengths of relational databases: they help maintain data integrity by enforcing these relationships.

In your interview, being able to explain these relationships is crucial. There are three main types:

-   One-to-One: Like a user and their profile settings
    
-   One-to-Many: Like our users and posts (one user can have many posts)
    
-   Many-to-Many: Like users and the posts they like (which we'll see next)
    

Now, what if we want users to be able to like posts? This introduces a many-to-many relationship - one user can like many posts, and one post can be liked by many users. We handle this with what's called a join table:

`CREATE TABLE likes (     user_id INTEGER REFERENCES users(id),     post_id INTEGER REFERENCES posts(id),     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,     PRIMARY KEY (user_id, post_id) );`

This structure, where we break data into separate tables and connect them through relationships, is called "normalization." It helps us:

1.  Avoid duplicating data (we don't store user information in every post)
    
2.  Maintain data integrity (if a user changes their username, it updates everywhere)
    
3.  Make our data model flexible (we can add new user attributes without touching posts)
    

While normalization is generally good, sometimes we intentionally denormalize data for performance. For example, we might store a post's like count directly in the posts table even though we could calculate it from the likes table. This trade-off between data consistency and query performance is exactly the kind of thing you should discuss in your interview!

Understanding these fundamentals - tables, relationships, and normalization - is important. They're what make SQL databases like PostgreSQL so powerful for applications that need to maintain complex relationships between different types of data. In your interview, being able to explain not just how to use these concepts, but when and why to use them (or break them), will set you apart.

### ACID Properties

One of PostgreSQL's greatest strengths is its strict adherence to ACID (Atomicity, Consistency, Isolation, and Durability) properties. If you've used databases like MongoDB or Cassandra, you're familiar with eventual consistency or relaxed transaction guarantees which are common trade-offs in NoSQL databases. PostgreSQL takes a different approach – it ensures that your data always follows all defined rules and constraints (like foreign keys, unique constraints, and custom checks), and that all transactions complete fully or not at all, even if it means sacrificing some performance.

Let's break down what ACID means using a real-world example: transferring money between bank accounts.

#### Atomicity (All or Nothing)

Imagine you're transferring $100 from your savings to your checking account. This involves two operations:

1.  Deduct $100 from savings
    
2.  Add $100 to checking
    

`BEGIN;   UPDATE accounts SET balance = balance - 100 WHERE account_id = 'savings';   UPDATE accounts SET balance = balance + 100 WHERE account_id = 'checking'; COMMIT;`

Atomicity guarantees that either both operations succeed or neither does. If the system crashes after deducting from savings but before adding to checking, PostgreSQL will roll back the entire transaction. Your money never disappears into thin air.

In your interview, emphasize how atomicity prevents partial failures. Without it, distributed systems can end up in inconsistent states that are very difficult to recover from.

#### Consistency (Data Integrity)

Consistency ensures that transactions can only bring the database from one valid state to another. For example, let's say we have a rule that account balances can't go negative:

`CREATE TABLE accounts (     account_id TEXT PRIMARY KEY,     balance DECIMAL CHECK (balance >= 0),     owner_id INTEGER REFERENCES users(id) );`

If a transaction would make your balance negative, PostgreSQL will reject the entire transaction. This is different from NoSQL databases where you often have to enforce these rules in your application code.

Confusingly, consistency in ACID has a slightly different meaning than consistency in CAP Theorem. In ACID, consistency means that the database always follows all defined rules and constraints. In the CAP Theorem, consistency means that the database always returns the correct result, even if it means sacrificing availability or partition tolerance.

#### Isolation (Concurrent Transactions)

Isolation levels determine how transactions can interact with data that's being modified by other concurrent transactions. PostgreSQL supports four isolation levels, each preventing different types of phenomena:

`BEGIN; SET TRANSACTION ISOLATION LEVEL READ COMMITTED;  -- Default level -- or REPEATABLE READ -- or SERIALIZABLE COMMIT;`

PostgreSQL's default "Read Committed" level only prevents dirty reads (reading uncommitted changes). It allows non-repeatable reads and phantom reads. Check the [PostgreSQL documentation](https://www.postgresql.org/docs/current/transaction-iso.html) for a detailed breakdown of which anomalies each level prevents.

While the SQL standard defines four isolation levels, PostgreSQL implements only three distinct levels internally. Specifically, Read Uncommitted behaves identically to Read Committed in PostgreSQL. This design choice aligns with PostgreSQL's multiversion concurrency control (MVCC) architecture, which always provides snapshot isolation - making it impossible to read uncommitted data.

Isolation Level

Dirty Read

Nonrepeatable Read

Phantom Read

Serialization Anomaly

Read uncommitted

Allowed, but not in PG

Possible

Possible

Possible

Read committed

Not possible

Possible

Possible

Possible

Repeatable read

Not possible

Not possible

Allowed, but not in PG

Possible

Serializable

Not possible

Not possible

Not possible

Not possible

#### Durability (Permanent Storage)

Once PostgreSQL says a transaction is committed, that data is guaranteed to have been written to disk and sync'd, protecting against crashes or power failures. This is achieved through Write-Ahead Logging (WAL):

1.  Changes are first written to a log
    
2.  The log is flushed to disk
    
3.  Only then is the transaction considered committed
    

While durability is guaranteed, there's a performance cost. Some applications might choose to relax durability for speed (like setting synchronous\_commit = off), meaning some writes which haven't been written to disk may be lost in the event of a power outage or crash.

### Why ACID Matters

In your interview, you'll often need to decide between different types of databases. ACID compliance is a crucial factor in this decision. Consider these scenarios:

-   **Financial transactions**: You absolutely need ACID properties to prevent money from being lost or double-spent
    
-   **Social media likes**: You might be okay with eventual consistency
    
-   **User authentication**: You probably want ACID to prevent security issues
    
-   **Analytics data**: You might prioritize performance over strict consistency
    

PostgreSQL's strict ACID compliance makes it an excellent choice for systems where data consistency is crucial. While performance trade-offs nowadays are minor, they're still worth being aware of.

### SQL Language

Let's talk about SQL briefly. While you rarely write actual SQL queries in system design interviews, understanding SQL's capabilities helps you make better architectural decisions. Plus, if you're interviewing for a more junior role, you might be asked to write some basic queries to demonstrate database understanding.

#### SQL Command Types

SQL commands fall into four main categories:

1.  **DDL (Data Definition Language)**
    
    -   Creates and modifies database structure
        
    -   Examples: CREATE TABLE, ALTER TABLE, DROP TABLE
        
    
    `CREATE TABLE users (   id SERIAL PRIMARY KEY,   email VARCHAR(255) UNIQUE ); ALTER TABLE users ADD COLUMN username TEXT;`
    
2.  **DML (Data Manipulation Language)**
    
    -   Manages data within tables
        
    -   Examples: SELECT, INSERT, UPDATE, DELETE
        
    
    `-- Find all users who joined in the last week SELECT * FROM users  WHERE created_at > NOW() - INTERVAL '7 days'; -- Update a user's email UPDATE users SET email = 'new@email.com'  WHERE id = 123;`
    
3.  **DCL (Data Control Language)**
    
    -   Controls access permissions
        
    -   Examples: GRANT, REVOKE
        
    
    `-- Give read access to a specific user GRANT SELECT ON users TO read_only_user;`
    
4.  **TCL (Transaction Control Language)**
    
    -   Manages transactions
        
    -   Examples: BEGIN, COMMIT, ROLLBACK
        
    
    `BEGIN;   -- Multiple operations... COMMIT;`
    

In your interview, you might be asked about database access patterns rather than specific queries. For example, "How would you query this data efficiently?" or "What indexes would you create?" These questions test your understanding of database concepts rather than SQL syntax.

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

M

MinorBlueDove142

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm44z7b9w01263bhc6zcl55ln)

You mentioned the same for DynamoDB as default choice in System Design - b/w Postgres and Dynamo, which one can I chose by default?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm46m6za001t34dcr3quowc7g)

I'll update the DDB article to ensure there is no confusion here. When in doubt, I'd lean towards Postgres. But the reality is, in many cases, there is no "wrong" answer here. It's about how you justify it.

In fact, "either works, but I X is what I have more experience with" is a reasonable justification, believe it or not. And it avoids the false dichotomy.

Show more

5

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm46m8r4n01pdhco91gs412e1)

I actually don't see where I say to choose DDB w/ regards to a default choice. Did you have the specific line? Would love to get that updated.

Show more

4

Reply

M

MinorBlueDove142

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4buetwb00iguriser4crhv5)

this is from DDB article: "For system design interviews in particular, it has just about everything you'd ever need from a database. It even supports transactions now! Which neutralizes one of the biggest criticisms of DynamoDB in the past."

I think I misunderstood it.

Show more

1

Reply

Q

QuietBlackJay631

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm6grly7a07dtsn4hspfae29l)

Funnily enough re: "either works, but X is what I have more experience with", in a recent interview round I got feedback that this was undesirable to say even though everything I said before it was justifications for using DDB instead of Postgres...so I'd caution that some interviewers only hear 'I want to use this because I know it' even if you've provided justifications.

Show more

7

Reply

M

MathematicalPurpleAmphibian147

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm452ieov002g129it894dhyf)

Thanks for putting together this guide - outcome focused and effective. Quick Question - For interview Qs that dont need ACID, why would engineering complexity (developer skill/training, DB Admins, lower flexibility of schema changes, high monitoring effort) compared to a managed solution like DDB not a point of trade-off?

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm46m53j401sy4dcruwll4ggy)

Sounds like a totally valid trade-off to me!

What we're looking to avoid is statements like, "choose NoSQL because of scale," which show a lack of understanding of how far modern RDBMSs have come. But the list you have there are absolutely valid justifications.

Show more

2

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4cfjmpv018fj6nb0rh63zuv)

Hey Evan correct me if I'm wrong, but when discussing DB tradeoffs we should really be focused around the application needs and choosing a DB that supports that use case right?

For instance, you could say you will use Postgres for everything (since it can pretty much handle most application use cases), but then explain as the system scales, you want to add something specific, say for analytical data (and point to a wide column/columnar DB). Or you wouldn't want to choose Mongo right out of the gate just "because its flexible" when a standard RDBMS will often times lead to less headache down the road (for devs just learning about DB management).

About your point on scalability, modern day RDBMS can scale to extremely high volume of data and throughput, but the challenge probably then comes in how you optimize the database with indexes, replicas, partitioning, etc., right? Albeit, scaling out with a RDBMS is probably more difficult than a NoSQL DB.

These are on top of other points, i.e. CAP theorem, if your system even needs a distributed database system right away, emphasis on read or write throughput, ACID properties, etc...

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4cyun9t01gkqfetftechb5u)

Yah, this is great. The key is just to show your interviewer that you understand nuance. This does not need to be a lot of information, but avoid the false SQL vs NoSQL dichotomy. Something as simple as, "Our data has XYZ property, so I'll opt for W because it handles this well and is what I know best," is a great answer.

Show more

0

Reply

E

ExtraAmaranthDuck389

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmauliega02kbad08ld6adwvz)

That's awesome!

Show more

0

Reply

P

PersonalBrownFalcon225

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4fromjt03y1j6nbj69d71g1)

How do we support atomic transactions if we have multiple nodes (I think we need something like 2 phase commit here?)

Show more

3

Reply

B

BeneficialBrownBarnacle904

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4j6uk8300iv12z8gtbq1ew1)

Is there a recommendation on when a dedicated search optimized database like ElasticSearch would be needed vs when Postgres GIN/ PostGIS indexes would be sufficient?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4lk38lq009nzypxxhs37e5c)

Added a section for this! Good question. If you need:

-   More sophisticated relevancy scoring
-   Faceted search capabilities
-   Fuzzy matching and "search as you type" features
-   Real-time index updates
-   Distributed search across very large datasets
-   Advanced analytics and aggregations

Main one being if you're growing into the 100Ms of rows, postgres might not be the best fit for search.

Show more

1

Reply

B

BeneficialBrownBarnacle904

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4lkodhw00c3nbot47sn4gzl)

Thank you for the detailed answer and the bonus section! :D

Show more

0

Reply

R

RainyCyanHippopotamus256

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4oh6ud90199wluf7g43892l)

"Main one being if you're growing into the 100Ms of rows, postgres might not be the best fit for search."

Why postgres might not be the best fit for many rows? Is it because ES can shard the data and horizontally scale the data automatically with ability to search through them by aggregating the data? But with postgres, it is generally a read-replica which could be a bottleneck?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4omp3n201fglkt9a9z38ovs)

PostgreSQL could technically can handle large-scale search through manual sharding, but you'd need to build your own logic to merge and rank results across shards which could be a pain. But, Elasticsearch was purpose-built for this. Automatic sharding and cross-node result aggregation are built into its core architecture.

Show more

1

Reply

R

RainyCyanHippopotamus256

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4p58u6601xqp041nmvymwxy)

Thanks. Yup, that's what I meant.

Show more

0

Reply

![Eneias Silva](https://lh3.googleusercontent.com/a/ACg8ocKqc4qNwlxwdChmgBNcSNI_BmJCE9soiat_1vUiPBOHzpxhXSev=s96-c)

Eneias Silva

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm8wmq519012ead0814fif97b)

Can you elaborate on Real-time index updates? How is the Postgres index updated? Isn’t it in real-time? Also Elastic Search index must be kept in sync with the main DB

Show more

0

Reply

![Jiatang Dong](https://lh3.googleusercontent.com/a/ACg8ocKfQgaYilpR7RBKGa8_AXqyhuDM2GA6B29pLwiJomT1-dI5c0tTlg=s96-c)

Jiatang Dong

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm4xszpha01tby3n1cp75dgl3)

"PostgreSQL supports two main types of replication: synchronous and asynchronous. Synchronous replication ensures that changes are written to disk on the primary node before being acknowledged, while asynchronous replication allows writes to complete on the primary node before being acknowledged."

I find this paragraph is confusing, can you explain more?

Show more

0

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 8 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm50k7eo401fu13vu5z63s0ii)

this is a typical replication setup. Imagine a write to the leader in synchronous replication. The response to the write is blocked until the replica doesn't acknowledge or the request times out. This leads to high consistency since every write is acknowledged. However, it leads to bad experience and affects availability. Async replication is actually async where the leader won't block for an ack from the replica. Though, an organization may choose to have a limited number of synchronous and rest async replication. A consensus algorithm is used for this purpose with a defined quorum.

Show more

0

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm5493yg603r5mqq6vrv6erh7)

I think, there must be some typo. Evan, could you please take a re-look in that paragraph (synchronous vs asynchronous replication) and clarify.

Show more

0

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm54v9iln04b2r5q6l1k1bh1d)

Thanks Evan for a great content. I think this is a game changer in the context of explaining DB choice during the interview. One question when I combine the capabilities of replication (out of the box) and sharing (say, used managed service like Citus)..then could you please clarify my understanding the topology. Is it similar to MongoDB, where each shard has one primary and several replicas? That means, multiple database nodes (primary of respective shard) taking writes in parallel (..and syncing to their respective replicas). Does this setup make it multi primary?

Show more

1

Reply

S

SpecialPlumKrill871

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm55vri4g00442csici03mavu)

Great deep dive. Thanks.

Quick question - how to easily figure out which part of the doc was updated?

Show more

2

Reply

S

SmartAzureTuna666

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm5arckv602lgke2rjnk2at7r)

+1 on this. I like the UPD badge on updated posts. Any thoughts on adding an appendix of # ChangeList at the bottom of the article? (Similar to what Great FE Design system does)

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm5bj80ro0353ke2rsfq98m3b)

Trying to work out something here! Our commit messages are very lazy for a small team, so we need some mechanism to label the changes so you don't see updates for every typo we fix. I like the idea of a changelog.

Show more

1

Reply

S

SmartAzureTuna666

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm5blcauo03b5e1nmx7mn6oxt)

I agree, you can take inspiration from "major" and "minor" update in Amazon wiki. Only expose major or publicly tagged updates. Aka privacy control deep dive in social nw feeds design.

Show more

1

Reply

I

IntermediateSilverGerbil786

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm5fozyho008iyug6a7ckbnxx)

You are missing the "Write Offloading" explanation in the Write Performance section

Show more

3

Reply

M

MinisterialPurpleGiraffe264

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm5fpy2ek00a9yug6381n15xb)

Hey! nice writeup. Were some sections removed altogether? I remember seeing publisher and subscriber section for logical replication, and some more sections which I can't see anymore

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm5xi33vm00yezd19x5u8xwat)

Yah, it was rewritten to be more practical and focused on whats needed for interviews

Show more

0

Reply

C

ChiefFuchsiaPrimate311

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm5zx6zub002x4i6qx31bvl6b)

Hey thanks for the great content as always. Just a question on the "Why ACID Matters" section, it's mentioned "Analytics data: You might prioritise performance over strict consistency" and "Social media likes: You might be okay with eventual consistency" --- are these more closely related to CAP theorem tradeoffs?

Show more

0

Reply

B

BiologicalMoccasinTahr305

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm60yhndj01233lineb1sz8sm)

Evan/Stefan - thank you both for writing this exceptionally informative guide.

I am confused by this statement because of the overloaded nature of the word "consistency" in regards to CAP theorem vs ACID:

"If you've chosen to prioritize consistency over availability in your non-functional requirements, then PostgreSQL is a strong choice. It's built from the ground up to provide strong consistency guarantees through ACID transactions."

PostgreSQL helps achieve ACID-consistency by use of row-locking or setting an appropriate isolation level. How does that guarantee strong consistency in CAP theorem context?

The page summary also says "Be prepared to discuss key trade-offs like ACID vs eventual consistency" implying there's a tradeoff between data integrity type consistency afforded by ACID properties and read-latest-write type consistency. Can you please explain this some more?

Show more

2

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm60ztwb2004wmy5kulu6ppan)

> PostgreSQL helps achieve ACID-consistency by use of row-locking or setting an appropriate isolation level. How does that guarantee strong consistency in CAP theorem context?

Postgres isn't going to attain consistency in a CAP context if you're dealing with async replicas, but with serializable transaction isolation and writing to a single partition leader (or using sync replication), you do achieve CAP consistency.

Show more

3

Reply

B

BiologicalMoccasinTahr305

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm6147o15017e4i6ql66jgj0u)

Serializable transaction isolation -> ACID consistency Synchronous replication -> CAP consistency

Is there any relation between ACID consistency supporting CAP consistency?

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm614luzz017l4i6qt3xv93k1)

Even though synchronous replication ensures all replicas have written the data before considering a transaction complete, read uncommitted is not ACID and means that other transactions can see data before it's committed. This creates a scenario where:

-   Transaction A modifies data but hasn't committed
-   The modification is replicated synchronously to all nodes
-   Transaction B on one node could see the uncommitted data
-   Transaction C on another node might not see the data (if it gets rolled back)

This breaks the "C" (Consistency) guarantee in CAP because different nodes can return different values for the same query at the same time.

Realistically, these are different concepts, I wouldn't get too gigabrain trying to draw associations between them.

Show more

2

Reply

![Dnyaneshwar Chavan](https://lh3.googleusercontent.com/a/ACg8ocKbq9RnysBY8g-RE0zb2XyfIzFPZ1QHzXfOOhokbrfXmU_D3L3EDQ=s96-c)

Dnyaneshwar Chavan

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm63534is029wcf2a70n87jgp)

@Evan Articles about OLAP DB internal and Spark internals would really help

Show more

0

Reply

S

SoleBlushSilverfish368

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm6glnivw078esn4ht4abj05h)

Where does MySQL stand in all this? MySQL is also used prevalently in the industry.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm6gotnpz07f26hpqna9wn4sk)

This is a deep dive on Postgres! MySQL is great. If you know more about MySQL, go for it. If you don't, Postgres has a wide feature set and we recommend it.

Show more

0

Reply

C

CautiousPurpleParrotfish718

[• 6 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm6k0ecxo01s3lhfltwwgf35e)

I'm wondering if we could get another explanation for Postgres and Global Multi-Region Requirements. That was the one area I felt very unclear on at the end of the article. Maybe an example?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm9t9thsu00jdad08wikxwywt)

Let me explain with a concrete example: Imagine you have a social media app with users in both SF and London. With Postgres, you'd need to pick one region as the "primary" - let's say SF. When a London user likes a post, that write has to travel to SF before being confirmed (if using synchronous replication) or the London user might not see their own like for a few seconds (if async). Even worse, if SF goes down, London can't write at all. The alternative would be to let both regions accept writes, but then you need complex conflict resolution - what happens if two users edit the same post simultaneously in different regions? Postgres wasn't built for this. That's why for true multi-region active-active setups, folks use either eventual consistency databases like Cassandra (where conflicts are resolved by last-write-wins) or databases specifically built for global distribution like CockroachDB (which handles the conflict resolution for you while maintaining ACID).

Show more

1

Reply

![Oleg Orlov](https://lh3.googleusercontent.com/a/ACg8ocKSBSFWro9mlr-ZsL1iZvGfhAYkXXT5ZWA2JvoKKmBlc21LIuY=s96-c)

Oleg Orlov

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm6wzmzdv0294dp6hj2z0u67b)

Nice article, but I was surprised not to see any mention of BRIN and hash indexes, connection pooling, prepared statements, vacuum, etc. Also, I expected to find here a bit more information on sharding. Talking about extensions: it's nice that you've mentioned PostGIS, but there are plenty of useful extensions, I think at least TimescaleDB is worth mentioning

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm6x62y3100118xn74go8go40)

You're right, those are all important topics in PostgreSQL. The article focuses on core concepts most relevant to system design interviews. BRIN and hash indexes are more niche, while connection pooling and prepared statements are implementation details. Vacuum is crucial for performance but rarely comes up in high-level design discussions. Sharding deserved more coverage, agreed. TimescaleDB is great for time-series data but less general-purpose than PostGIS. The goal was to cover the most interview-relevant topics without going too deep into PostgreSQL internals.

Show more

0

Reply

![Oleg Orlov](https://lh3.googleusercontent.com/a/ACg8ocKSBSFWro9mlr-ZsL1iZvGfhAYkXXT5ZWA2JvoKKmBlc21LIuY=s96-c)

Oleg Orlov

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm6xrl7nx00gjun62iki75y9g)

Hi Evan, thanks for your reply. I agree that most of the concepts I've mentioned are niche or they are just implementation details .. except connection pooling. Because PG uses multiprocessing, number of connections is very limited. That is why you can not efficiently scale on the application level without a pooler sitting in between. So it's something you have to keep in mind during the system design interview

Show more

2

Reply

![Dmitry Grigorenko](https://lh3.googleusercontent.com/a/ACg8ocKWISD4A7ZGdhKGbgPoOHdMPuzpoX6CD_cK-cO1JJDXH3mKrZOeuA=s96-c)

Dmitry Grigorenko

[• 6 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm7a4qz9e0534r9w57c1p357w)

"NoSQL databases (like Cassandra) for event streaming" - mind elaborating? Your deep dive on cassandra does not cover event streaming and for event streaming you already have Kafka or Redis. Why would one go with cassandra?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm7njp0nc00q2hqff36bzmllh)

Not the stream itself, that is Kafka (et al.)'s wheelhouse. But to handle the large volume of writes that comes with event streaming services, Cassandra excels.

Show more

0

Reply

![Rahul Grover](https://lh3.googleusercontent.com/a/ACg8ocIwEaCGSf2dBHruTo57bRgTwWsQV0etaiq7vXrxE-Z2Cfw5idzYiA=s96-c)

Rahul Grover

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm7o32mov00bq11ywbz6xxcxs)

I had a quick question about the Practical Performance Limits.

I notice you mentioned 5000 writes per second per core, while the reads (or Simple indexed lookups, to be precise) mention thousands per second per core.

Does it mean that writes throughput is almost equal to read throughtput? Am I misinterpreting this?

Show more

1

Reply

D

DominantGrayGayal505

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm7te5c4e02qbe08kbkfd30j2)

Thank you for the awesome write up! I have a quick question about concurrency control.

In the auction bidding system you mentioned, could the problem be addressed using application-level optimistic locking (e.g: using a version column)?

Additionally, on a broader level, how does application-level locking (both optimistic and pessimistic) in ORMs differ from database-level concurrency control (such as row locking and higher transaction isolation levels)? Am I correct in understanding that application-level locking is implemented using database-level locks?

Show more

1

Reply

![khader zatari](https://lh3.googleusercontent.com/a/ACg8ocKUcVV5BpyZCAJITAEc99Q4q-ssHI7PNCCcQjDPpxx4JpYBO5MP3w=s96-c)

khader zatari

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm7vmi2i300afdimn9rnnhtfr)

how to answer this question, how to scale horizontally?

Show more

0

Reply

Y

ytfordinesh

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm7xdez3h00fke77fm3ksifu9)

Fabulous content. It will be useful if we can have a quiz at the end of each topic.

Show more

0

Reply

M

MassJadeBoar807

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm852xh6s00v5j2eyzwi88qmt)

I am aware this is just an example and it also says that you are designing a growing social media company. Personally I feel that postgres (PG) is a terrible choice for a Social media company. I would rather stick with DynamoDB (DDB) as it scales way better. I have worked at Amazon and the largest payment processor in the US. Both these companies have rejected PG for their solutions as it cannot scale like DDB. At Amazon any tier 0 or tier 1 service is strictly advised not to use Aurora as it just cannot scale (Even though Aurora is from Amazon). Below are my reasons why PG is bad.

1.  Storage

Aurora PG has a 128 TB limit. While this might seem a lot but in reality it can fill really fast for a social media use case. Social media is heavily reliant on text for DM's, long captions and long comments. If you just take the case of DM's then you can have long text messages with multiple of your friends. Any growing social media site will easily hit this limit within months / weeks if it gains traction. You can try sharding to overcome this. However, Aurora doesn’t support sharding outside of the box. This means you need to build your own layer to keep a track of where the data is in which shard (non trivial). Aurora recently released sharding support but it's pretty new and might not be battle tested yet.

Whereas DDB storage is limitless. There is really no upper bound. It handles sharding automatically and resizes partitions automatically. DDB is just superior in this case.

1.  Social media needs a multi region and a multi active database

Any respectable social media platform will have users all over the world. It will never be just one country. The user experience in terms of latency should be good for all the users across the world. If we use the primary and replica architecture mentioned over here then the primary has to be in some country. For simplicity let's say the primary is present in the US and the replicas are scattered over europe,US and asia. If I am posting an image from Thailand then my request has to travel all over the world to the US as my primary is over there. This can be really slow. To tackle this we need multi region and multi active databases where writes can happen in any region and replicated to other regions with last write wins conflict resolution. This is not possible in Aurora PG. This configuration is kind of possible using Aurora global DB but it has a lot of limitations which I will not go into.

Whereas DDB supports global tables where one can write from any region and it gets replicated to other region \[0\]. In the above example we can have DDB global tables in Asia, US and Europe where all are in active mode. User requests from a country goes to the nearest data center and writes over there. DDB global table guarantees strong consistency in the same region of write and eventual consistency in other regions. This provides a nice read your own write consistency for the users in that country and eventual consistency for others. This is perfectly fine for a social media use case. In Fact the replication is done within a second. Most of the social media users will just be in one country for the majority of their lifetime so this model just works seamlessly. When a user is travelling they can contact the nearest data center for writes and this just works beautifully. All these come with DDB with minimal configuration.

1.  Disaster recovery

When a region outage happens in PG one of the secondaries is promoted to primary. The unplanned failover can take around 1 minute according to AWS. In the 1 minute downtime you won't be able to process writes. And also when a failover happens all the secondaries are rebuilt to match the new primary which can take hours \[1\]. During this time the secondaries will not serve read requests they will be hard down. So your applications relying on secondaries for eventual reads will be down as well.

The chance of DDB having an regional outage is very slim as it's a tier 0 service in AWS \[2\]. This means other AWS services rely on DDB for their functionality. Even if a regional outage happens you can reroute requests to the nearest region quickly. This is non trivial but definitely possible and doesn’t have the downsides of Aurora PG.

1.  ACID transactions

Many people have a misconception that DDB doesn’t support ACID. In reality it does and supports transactions too \[3\]

What are the downsides of using DDB ?

You will not have referential integrity checks. This should be handled by the application. However, with the help of transactions you can guarantee atomic writes of related entities.

You will not get cascading deletes for related data like in PG. I feel this is totally fine as many companies dont really hard delete data for auditing purposes. They will just soft delete. Even if you want to hard delete something this can be done by some offline jobs periodically.

All these points make DDB really superior. This doesn’t mean PG is useless. DDB provides a lot of features with very minimal configurations whereas PG needs a lot of effort and work required to have the same features as DDB. That's why I feel it's just better to stick with DDB for a social media use case.

\[0\] - https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-global-table-design.prescriptive-guidance.writemodes.html

\[1\] - https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-global-database-disaster-recovery.html#aurora-global-database-failover

\[2\] - https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/V2globaltables\_reqs\_bestpractices.html#outage

\[3\] - https://aws.amazon.com/blogs/aws/new-amazon-dynamodb-transactions/

Show more

15

Reply

E

ExtraAmaranthDuck389

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmauli2yo02k7ad081ef9zj7g)

Thank you!

Show more

0

Reply

R

risers.bodkin1o

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm8hlqgrn01k3p0hf5b5jn6c2)

I love this write up! It helps you not just for interviews but also will make you look like a Chad in read design discussions at work.

Show more

0

Reply

P

PeacefulAmaranthTarantula827

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm8kpn7gb00kgvgc7zvzlu3d2)

Hey, I am not sure if it has already been covered in some other comment(s). Performance Limits mentions a lot of numbers, example complex joins have a limit of 100 per second, and so on.

For interview perspective, what exactly should I remember to weigh in the performance limitations of Postgresql? There is a note "In your interview, showing knowledge of these practical limits helps demonstrate that you understand not just how to use PostgreSQL, but when you might need to consider alternatives or additional optimization strategies.", but there are lot of numbers to remember there. Anything you'd recommend?

Show more

0

Reply

A

AtomicIvoryUrial668

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm8upljpc01dxad08uknkofx0)

In the auction system where we're reading rows and locking them for update, wouldn't we potentially still write the $95 after writing the $100 if they were both read as $90 initially? regardless of the transactions interlocking?

Or do you mean only to update the max bid with a condition? because in either case you're updating the max bid

Show more

0

Reply

A

AtomicIvoryUrial668

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm8upqo0d01e9ad08dmvma6am)

Ah maybe you meant to add more details in the "-- Place new bid if it's higher" section but chose to omit the conditions?

Show more

0

Reply

H

HandsomeIvoryCrow799

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm917r1hq00cbad08cat79gec)

> The practical implication of this design is that write performance is typically bounded by how fast you can write to the WAL (disk I/O), how many indexes need to be updated, and how much memory is available for the buffer cache.

Isn't buffer cache the step after WAL? And we consider the write to be successful if the WAL steps succeeds right? If so, how is buffer cache relevant to the write performance?

From the official document, it seems the write is first written to WAL buffer and then gets flushed to WAL segment on disk, which is the reason why we'd need to increase RAM if we'd like to increase the write performance. Yet, this piece of information is not covered in the article.

Link to the official document: https://www.postgresql.org/docs/current/wal-configuration.html

Show more

1

Reply

![Xuezi Zhang (Manfred)](https://lh3.googleusercontent.com/a/ACg8ocLZc0llYTr9-j46ZqLfVndE3SQgD8QaLCWC9p7Ikh3TchFMzKR8=s96-c)

Xuezi Zhang (Manfred)

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm93qb5u800cmad08ktkopte0)

For the batch processing process, batching writes for the same row(in your example) definitely helps a lot. What if we're batching writes on different rows? Would it help with writes performance as well? e.g

INSERT INTO likes (post\_id, user\_id) VALUES 
  (1, 101), (2, 102), ..., (N, 1000);

Show more

0

Reply

S

SuperbMoccasinTurkey766

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm9aflmlz000sad070pzkz5x3)

so read performance is 'thousands' per second per core and write is 5000 per second per core? That doesn't look right. Unless by thousands you mean 10's of thousands of read

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm9ahti68003iad077mh2z5o1)

yes, 10s of thousands! Good catch, updating next release.

Show more

0

Reply

R

RacialTurquoiseRabbit968

[• 4 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm9d4wpoy009iad08m8wrgxg8)

> PostgreSQL might not be the best fit, such as cases requiring extreme write scaling

both Postgres and Cassandra first write onto WAL and then write to buffercache and MemTable resp (both in memory) so what makes Postgres writes slower to Cassandra's?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cma55qbg6005vad077gonmfs0)

Couple things. Cassandra's write path is optimized for append-only operations. It just dumps data to the commit log and memtable without worrying about indexes or MVCC. Postgres has to maintain ACID guarantees, update indexes, and handle MVCC versioning for each write. Plus, Cassandra's distributed architecture means writes can be spread across nodes without coordination (eventual consistency), while Postgres needs synchronous replication for consistency.

Show more

2

Reply

![Aiwei Zheng](https://lh3.googleusercontent.com/a/ACg8ocJznVspMNiSLNKxG4QDo7lwVPd3HAdQzMc21V7YWUIv5_b5XdM=s96-c)

Aiwei Zheng

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm9de652i0006ad08iv754er2)

Great deep dive! Thank you so much!

Show more

1

Reply

![Prantik Bhowmick](https://lh3.googleusercontent.com/a/ACg8ocIo5VCYVDyo-VBazeuOiArbFppJSD9OC_ycqy3lHvpAO3ay5E8=s96-c)

Prantik Bhowmick

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm9orju7s01k0ad080urbr4ty)

If Postgress is recommended for Content management systems & E-Commerce systems but again for extremely high write throughput systems , it's not recommended. Even content management systems can have a higher write throughput , same goes with E-commerce systems which will have substantial write throughput during sales like Black Friday. Why can't databases like MongoDB be used in such cases which also provides rich querying and transaction support and an easy to use sharding capabilities.

Show more

0

Reply

D

DecentGreenFox390

[• 4 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cm9t3o22w003cad08wiz75btl)

I don't think you ever check if the new bid is actually higher than the maxbid, you just overwrite the maxbid.

\-- Update the max bid UPDATE Auction SET maxBid = 100 WHERE id = 123; COMMIT;

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cma55slof0061ad0708danfiu)

True! Just trying to show a simplified version to teach transactions, but you are technically correct to call this out :)

Show more

0

Reply

![Arun prasath](https://lh3.googleusercontent.com/a/ACg8ocJwk6FK5NTjnp9SpWHD0wUa_6v0qHw3KQTdSOT5UZqZMUU85Q=s96-c)

Arun prasath

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cma4whlx700i2ad08cwr4i5sc)

Thanks Evan. This is great, I particularly like the how to choose DB section. This is the kind of thing that is most useful in desiging systems.

Show more

1

Reply

![sidhant chadda](https://lh3.googleusercontent.com/a/ACg8ocLfwBmZRhgVozeaUUCkuxN2BLfouk80UIlYNNcfTAjREXKRxlrn_Q=s96-c)

sidhant chadda

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmafu5uz90170ad0801e8v67l)

Can Postgres support optimistic concurrency control? Should we evaluate this choice when write content is low over row locking?

Show more

0

Reply

O

OutdoorFuchsiaGerbil417

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmazt4j2p0091ad07n14vt8to)

Would we need to know more about optimistic / pessimistic concurrency control? Does it come up in other system design questions, and what else would you recommend knowing about it?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmb026tnk00o7ad0707ojh691)

Definitely good to know. Comes up in online auction, yelp, ticketmaster, just to name a couple

Show more

0

Reply

![SovietTanPanda828](https://lh3.googleusercontent.com/a/ACg8ocJbu4FdnAnKhBiPfHpWeAbPMWe9GXfUZMEqaL6xz01i2V-1ZQ=s96-c)

SovietTanPanda828

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmb78r43m0085ad08g3nxt8xa)

Can you explain what do you mean by this?

Consider using Elasticsearch over postgresql for full-text search

Real-time index updates

Usually updates to elasticsearch are propagated by CDC, so why is it that more preferable or more real-time compared to postgresql?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmb7cjegh00k0ad08skzox3lg)

Yah you're right that this is misleading. Both indexes can be updated in "real-time".

Show more

1

Reply

S

SovietTanPanda828

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmb8isxeg00gtad07xjlhs3x0)

Are there any performance benchmarks that you can link? Just for me to feel more confident when making claims about performance of postgres

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmb8j7hm200j3ad080gv61dd8)

I bet you could find some good benchmarks with a quick search :)

Show more

0

Reply

![SovietTanPanda828](https://lh3.googleusercontent.com/a/ACg8ocJbu4FdnAnKhBiPfHpWeAbPMWe9GXfUZMEqaL6xz01i2V-1ZQ=s96-c)

SovietTanPanda828

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmb8jrlr200hpad07zny2zjz3)

These numbers assume PostgreSQL's default transaction isolation level (Read Committed), where transactions only see data that was committed before their query began. If you change the default isolation level these numbers can go up or down.

I guess they won't go up, given other forms of isolation are more strict which will have a performance penalty? I am being pedantic here, sorry!

Show more

0

Reply

S

SovietTanPanda828

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmb9yba0j000iad0831e8tb86)

There seems to be a typo:

There's one key caveat with read replicas: replication lag. If a user makes a change and immediately tries to read it back, they might not see their change if they hit a replica that hasn't caught up yet. This is known as "read-your-writes" consistency.

This seems like a violation of read your writes consistency. I know in Cassandra and I guess even Dynamo, this can be addressed by having a quorum, R+W > N, but do you know what is the standard way to do this in postgres?

Show more

0

Reply

E

ExactAmethystLark466

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmbn18bb000ef08ad246qlbwx)

"Extreme Write Throughput" suggests we should consider Cassandra but doesnt that have the same Write throughput issues?

Show more

0

Reply

A

altal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmc7xjcy60h71ad08dh11r3x7)

Thanks for the writeup. Really informative and helpful.

It would make the post even better if the concepts of node, instance, partition, core etc. were formally defined (similar to the Kafka post). Especially once you combine partitioning, sharding, replication etc together, it is no longer clear how data is organized and distributed, and what is a logical and what is a physical separation. Thanks!

Show more

0

Reply

![Jockey Cheng](https://lh3.googleusercontent.com/a/ACg8ocIa7Je2TNuBexsMfQtkEez5_32-hDHAmXSg62f4pPvBmXXlX94=s96-c)

Jockey Cheng

[• 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmcnzcuev00igad08cemwy2gz)

I don't quite understand why we can't use postgres under Global Multi-Region Requirements. In system design interview, I think most design are global service does that mean we can't use postgres?

Show more

1

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmcqldt7w003jad07rmslkmkk)

The section on Table Partitioning emphasizes how the social media posts database can handle more writes as a result of partitioning. I don't see that at all. If the table is partitioned by post creation time, then all of the newly-created posts will have the same date of creation and they will go to the same partition.

I think that partitioning and archiving does help scale the read QPS.

Google's Spanner database supports inter-leaving rows from different table, which means you can store comments on the same partition as posts. In that case, we can imagine that users will add new comments for a few days after the post is created. Even then, I don't think that the partitioing by creation date will alleviate the write QPS concentration by a huge degree. PostGres does not support inter-leaving rows AFAICT.

Show more

0

Reply

![ThePabs213](https://lh3.googleusercontent.com/a/ACg8ocK3ykP-c_Ppi0z1Q4ceM_gtXtUFKSIxYtNlHXt8tOae2CiH7A=s96-c)

ThePabs213

[• 12 days ago• edited 12 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cme9mfkci0abiad08uxwhehy5)

I think the benefit comes from allowing parititions of more recent dates to be stored in faster memory. If you don't partition you will find yourself writing to cold storage more often than not.

Show more

0

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmcqsiiln00edad071opo971t)

I think it's worth flagging that row-level locking can cause deadlocks if two separate queries lock the same rows in different order: https://dba.stackexchange.com/questions/323040/avoiding-deadlocks-when-locking-multiple-rows-without-using-nowait

Show more

1

Reply

![Akshay Mani Aggarwal](https://lh3.googleusercontent.com/a/ACg8ocJHJ4N0S4qtNPd983BL1uLiwshHYD7smGXVATVC4AC42gNECeHh=s96-c)

Akshay Mani Aggarwal

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmcwqlggq02gfad08n45t0liz)

"Unlike DynamoDB, PostgreSQL doesn't have a built-in sharding solution. You'll need to implement sharding manually, which can be a bit of a challenge. Alternatively, you can use managed services like Citus which handles many of the sharding complexities for you." Wanted to understand how multi DC setup for postgres will work with sharding. Lets say i have a payment system construct which requires strong consistency and the application serves world wide users. Are we going to setup this topology where each shard basis geography lies in separate regions and replicas will have data across regions (given users might move across geographies they would like to see there past payments across regions) and has anyone actually setup something like this ?

Evan - can we have a write up on cockroach db as well ?

Show more

0

Reply

D

DevotedPlumReindeer651

[• 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmdfa0e7u02cgad08i7znd3mm)

Awesome content, thank you. Noticed a few small typos:

Remember, were talking about a single node here! (were->we’re) For example, we may consider sharing our posts by user\_id (sharing->sharding)

Show more

0

Reply

E

EconomicBeigeGuineafowl908

[• 23 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmdu4zqvv0652ad08m3loetq6)

Changes are made to data pages in PostgreSQL’s shared buffer cache, which holds **in-memory copies of parts of tables and indexes** currently in use

Show more

0

Reply

A

AvailableJadeCattle348

[• 10 days ago• edited 10 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmec32lw10184ad08s1jgs68h)

For scaling/sharding, shouldn't something like PgPool-II be discussed as it handles HA features like failover of traffic to secondary nodes if a primary goes down (instead of the example listed: Repointing applications to new primary)? It does a lot more than that as well.

Same with this point - PgPool-II should help with this:

> Even with sharding, coordinating writes across many PostgreSQL nodes adds complexity and latency

You can also set up PgPool-II in a multi-node format for redundancy at ingress level and have Active-Passive PgPool instances.

Ref: https://www.postgresql.fastware.com/postgresql-insider-ha-pgpool-ii

Show more

0

Reply

O

OrganicYellowGerbil909

[• 9 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmeddhiiy0315ad08dsqn024i)

In this example (copied from above), where are we checking new bid amount with maxBid ?

`BEGIN; -- Get current max bid for item 123 SELECT maxBid from Auction where id = 123; -- Place new bid if it's higher INSERT INTO bids (item_id, user_id, amount)  VALUES (123, 456, 100); ...`

Show more

0

Reply

A

AvailableJadeCattle348

[• 9 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmedlazqj056wad08uw7qh2xd)

What about a section on distributed locks using something like Postgres Advisory Locks?

Show more

0

Reply

![unicorn ray](https://lh3.googleusercontent.com/a/ACg8ocImoVXm2bIN8iLNkqRNbxn3JnyGZbglKkX3MVN2hSgoap1CKg=s96-c)

unicorn ray

[• 7 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/postgres#comment-cmeglpbae038tad08lt8gtvye)

Learned a lot from the DynamoDB video! Thank you! Since you mentioned PostgreSQL and DynamoDB are two DBs we should learn about, can we have a video version of this as well please (:

Show more

0

Reply
 2025 Optick Labs Inc. All rights reserved.