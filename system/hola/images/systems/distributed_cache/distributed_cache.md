# Design a Distributed Cache

Scaling Reads

[![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.66fdc8bf.png&w=96&q=75&dpl=e097d75362416d314ca97da7e72db8953ccb9c4d)

Evan King

Ex-Meta Staff Engineer

](https://www.linkedin.com/in/evan-king-40072280/)

hard

Published Jul 11, 2024

* * *

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

## Understanding the Problem

**💾 What is a Distributed Cache?** A distributed cache is a system that stores data as key-value pairs in memory across multiple machines in a network. Unlike single-node caches that are limited by the resources of one machine, distributed caches scale horizontally across many nodes to handle massive workloads. The cache cluster works together to partition and replicate data, ensuring high availability and fault tolerance when individual nodes fail.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements-1)

**Core Requirements**

1.  Users should be able to set, get, and delete key-value pairs.
    
2.  Users should be able to configure the expiration time for key-value pairs.
    
3.  Data should be evicted according to Least Recently Used (LRU) policy.
    

**Below the line (out of scope)**

-   Users should be able to configure the cache size.
    

We opted for an LRU eviction policy, but you'll want to ask your interviewer what they're looking for if they weren't explicitly upfront. There are, of course, other eviction policies you could implement, like LFU, FIFO, and custom policies.

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements-1)

At this point in the interview, you should ask the interviewer what sort of scale we are expecting. This will have a big impact on your design, starting with how you define the non-functional requirements.

If I were your interviewer, I would say we need to store up to 1TB of data and expect to handle a peak of up to 100k requests per second.

**Core Requirements**

1.  The system should be highly available. Eventual consistency is acceptable.
    
2.  The system should support low latency operations (< 10ms for get and set requests).
    
3.  The system should be scalable to support the expected 1TB of data and 100k requests per second.
    

**Below the line (out of scope)**

-   Durability (data persistence across restarts)
    
-   Strong consistency guarantees
    
-   Complex querying capabilities
    
-   Transaction support
    

Note that I'm making quite a few strong assumptions about what we care about here. Make sure you're confirming this with your interviewer. Chances are you've used a cache before, so you know the plethora of potential trade-offs. Some interviewers might care about durability, for example, just ask.

## The Set Up

### Planning the Approach

For a problem like this, you need to show flexibility when choosing the right path through the [Hello Interview Delivery Framework.](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery) In fact, this is a famous question that is asked very differently by different interviewers at different companies. Some are looking for more low-level design, even code in some instances. Others are more focused on how the system should be architected and scaled.

In this breakdown, we'll follow the most common path (and the one I take when I ask this question) where we balance low-level design with a high-level design that is scalable and handles the expected load.

I'm going to opt for documenting the core entities and the API, but in an actual interview I would not spend a lot of time on these two sections, recognizing they're largely trivial. I'd either skip them, or rush through them to spend time on my high-level design and deep dives.

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

The core entities are right there in front of our face! We're building a cache that stores key-value pairs, so our entities are keys and values.

In other words, the data we need to persist (in memory) are the keys and their associated values.

### [The API](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#4-api-or-system-interface)

According to our functional requirements, we have three key operations that we'll need to expose via an API: set, get, and delete.

These should each be rather straightforward, so we can breeze through this.

Identifying where to spend more and less time in a system design interview based on where the challenges are is an important skill to have. Spending time on simple, auxiliary, or otherwise less important parts of the system is a recipe for running out of time.

Setting a key-value pair:

`POST /:key {   "value": "..." }`

Getting a key-value pair:

`GET /:key -> {value: "..."}`

Deleting a key-value pair:

`DELETE /:key`

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes-1)

We start by building an MVP that works to satisfy the core functional requirements. This doesn't need to scale or be perfect. It's just a foundation for us to build upon later. We will walk through each functional requirement, making sure each is satisfied by the high-level design.

### 1) Users should be able to set, get, and delete key-value pairs

Let's start with the absolute basics. No fancy distributed systems yet - just a working cache that handles gets, sets, and deletes.

At its core, a cache is just a hash table. Every programming language has one: Python's dict, Java's HashMap, Go's map. They're perfect for this because they give us O(1) lookups and inserts.

Your interviewer may ask you to sketch out some pseudocode. Here is what that might look like:

`class Cache:     data = {}  # Simple hash table     get(key):         return self.data[key]     set(key, value):         self.data[key] = value     delete(key):         delete self.data[key]`

When asked to write pseudocode in a design interview, don't worry about the syntax. Just focus on the logic. You'll note that I don't handle corner cases like key not found or key already exists. No worries, that's not the point here.

That said, as always, it's worth confirming this with your interviewer. "I'm going to write some pseudocode, is it ok that it's not syntactically correct or missing some details?"

That's it. Three operations, all O(1). The hash table handles collision resolution and dynamic resizing for us.

We can host this code on a single server. When a user makes a API request, we'll parse the request, and then call the appropriate method on our Cache instance, returning the appropriate response.

Simple Cache

Now, this isn't production-ready. It'll fall over if we try to handle real traffic or store too much data. But it's a solid foundation that meets our core functional requirement: storing and retrieving key-value pairs.

### 2) Users should be able to configure the expiration time for key-value pairs

Let's add expiration functionality to our cache. We'll need to store a timestamp alongside each value and check it during reads. We'll also need a way to clean up expired entries.

The pseudocode below shows how we can modify our cache to support TTLs (Time To Live). The key changes are:

1.  Instead of storing just values, we now store tuples of (value, expiry\_timestamp)
    
2.  The get() method checks if the entry has expired before returning it
    
3.  The set() method takes an optional TTL parameter and calculates the expiry timestamp
    

This gives us the ability to automatically expire cache entries after a specified time period, which is necessary for keeping the cache fresh and preventing stale data from being served.

`# Check the expiry time of the key on get get(key):     (value, expiry) = data[key]     if expiry and currentTime() > expiry:         # Key has expired, remove it         delete data[key]         return null             return value # Set the expiry time of the key on set set(key, value, ttl):     expiry = currentTime() + ttl if ttl else null    data[key] = (value, expiry)`

This handles the basic TTL functionality, but there's a problem: expired keys only get cleaned up when they're accessed. This means our cache could fill up with expired entries that nobody is requesting anymore.

To fix this, we need a background process (often called a "janitor") that periodically scans for and removes expired entries:

`cleanup():         # Find all expired keys and delete     for key, value in data:         if value.expiry and current_time > value.expiry:             delete data[key]`

Simple Cache with TTLs

This cleanup process can run on a schedule (say every minute) or when memory pressure hits certain thresholds. The trade-off here is between CPU usage (checking entries) and memory efficiency (removing expired data promptly).

Now we have TTL support, but we still need to handle memory constraints. That's where our next requirement comes in: LRU eviction.

### 3) Data should be evicted according to LRU policy

Now we need to handle what happens when our cache gets full. We'll use the Least Recently Used (LRU) policy, which removes the entries that haven't been accessed for the longest time.

Make sure you've confirmed with your interviewer that you're implementing an LRU cache. They may have a different eviction policy in mind, or even ask that your implementation is configurable.

We have a challenge: we need to find items quickly AND track which items were used least recently. If we just used a hash table, we could find items fast but wouldn't know their access order. If we just used a list, we could track order but would be slow to find items.

The solution is to combine two data structures:

1.  A hash table for O(1) lookups - This gives us instant access to any item in the cache
    
2.  A doubly linked list to track access order - This lets us maintain a perfect history of which items were used when
    

Together, these structures let us build an efficient LRU cache where both get and set operations take O(1) time.

When we add or access an item:

1.  We create a Node object containing the key, value, and expiry time
    
2.  We add an entry to our hash table mapping the key to this Node
    
3.  We insert the Node at the front of our doubly-linked list (right after the dummy head)
    
4.  If we're at capacity, we remove the least recently used item from both the hash table and the linked list
    

The doubly-linked list maintains the exact order of access - the head of the list contains the most recently used items, and the tail contains the least recently used. When we need to evict an item, we simply remove the node right before our dummy tail.

For example, if we add items A, B, C to a cache with capacity 2:

1.  Add A: \[A\]
    
2.  Add B: \[B -> A\]
    
3.  Add C: Cache is full, evict A (least recently used), resulting in \[C -> B\]
    

When we access an existing item, we:

1.  Look up its Node in our hash table (O(1))
    
2.  Remove the Node from its current position in the list (O(1))
    
3.  Move it to the front of the list (O(1))
    

This way, frequently accessed items stay near the head and are safe from eviction, while rarely used items drift towards the tail where they'll eventually be removed when space is needed.

The hash table gives us O(1) lookups, while the doubly-linked list gives us O(1) updates to our access order. By combining them, we get the best of both worlds - fast access and efficient LRU tracking.

Here is some pseudocode for the implementation:

`class Node # Node in our doubly-linked list storing values and expiry times class Cache:     get(key):         # get the node from the hash table         node = data[key]                  # Check if the entry has expired         if node.expiry and currentTime() > node.expiry:             # Remove expired node from both hash table and linked list             delete data[key]             delete node             return null                     # Move node to front of list         move_to_front(node)         return node.value     set(key, value, ttl):         # Calculate expiry timestamp if TTL is provided         expiry = currentTime() + ttl if ttl else null                 if key in data:             # Update existing entry             node = data[key]             node.value = value            node.expiry = expiry            move_to_front(node)         else:             # Add new entry             node = Node(key, value, expiry)             data[key] = node            add_node(node)                          # If over capacity, remove least recently used item             if size > capacity:                 lru = tail.prev                 delete lru                 delete data[lru.key]     cleanup():         expired_keys = []                  # Scan from LRU end towards head         current = tail.prev         while current != head:             if current.expiry and currentTime() > current.expiry:                 expired_keys.add(current.key)             current = current.prev                      # Remove expired entries         for key in expired_keys:             node = data[key]             delete data[key]             delete node`

The code examples provided here are meant to illustrate core concepts and are intentionally simplified for clarity. A production implementation would need to handle many additional concerns including thread safety, error handling, monitoring, and performance optimizations.

Simple LRU Cache

The clever part about this implementation is that all operations (get, set, and even eviction) remain O(1). When we access or add an item, we move it to the front of the list. When we need to evict, we remove from the back.

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes-1)

Ok, truth be told, up until this point this has been more of a low level design interview than pure system design, but the deep dives is where that changes and we discuss how we turn our single, in-memory cache instance into a distributed system that can meet our non-functional requirements.

For these types of deeper infra questions, the same pattern applies where you should try to lead the conversation toward deep dives in order to satisfy your non-functional requirements. However, it's also the case that your interviewer will likely jump in and hit you with probing questions, so be prepared to be flexible.

Here are some of the questions I'll usually ask (or a candidate could do this proactively).

### 1) How do we ensure our cache is highly available and fault tolerant?

Our high-level design works great on a single node, but what happens when that node fails? In a production environment, we need our cache to stay available even when things go wrong. Let's look at how we can make that happen.

The key challenge here is data replication - we need multiple copies of our data spread across different nodes. But this opens up a whole new set of questions:

1.  How many copies should we keep?
    
2.  Which nodes should store the copies?
    
3.  How do we keep the copies in sync?
    
4.  What happens when nodes fail or can't communicate?
    

There are several well-established patterns for handling this. Each has its own trade-offs between consistency, availability, and complexity.

These patterns extend far beyond distributed caches - they're fundamental building blocks for achieving high availability and fault tolerance in any distributed system, from databases to message queues.

### 

Bad Solution: Synchronous Replication

###### Approach

The simplest approach is to maintain a few replicas of each shard. When a write comes in, we update all replicas synchronously and only respond once all replicas have acknowledged the write. While straightforward, this approach can impact write latency and availability if any replica is slow or down.

Synchronous replication is preferred for caches that need strong consistency guarantees (which ours does not).

Synchronous Replication

###### Challenges

The main drawbacks of this stem from its synchronous nature. Write operations must wait for confirmation from all replicas before completing, which can significantly impact latency. If any replica becomes unavailable, the entire system's availability suffers since writes cannot proceed without full consensus. Additionally, as you add more replicas to the system, these problems compound - each new replica increases the likelihood of delays or failures, making it difficult to scale the system horizontally.

For our use case at least, this is not a good fit. We need to prioritize availability and performance over consistency.

### 

Good Solution: Asynchronous Replication

###### Approach

Another option is to update one primary copy immediately and then propagate changes to replicas asynchronously -- confirming the write once only the primary has acknowledged the change. This aligns well with the eventual consistency model that most caches adopt (and is a non-functional requirement for us), making it more suitable for a cache with our requirements than synchronous replication.

The asynchronous nature provides several key advantages. First, it enables better write performance since we don't need to wait for replica acknowledgement. It also offers higher availability, as writes can proceed even when replicas are down. The system scales better with additional replicas since they don't impact write latency. Finally, it's a natural fit for cache use cases where some staleness is acceptable.

Asynchronous Replication

###### Challenges

The main trade-offs come from the asynchronous nature. Replicas may temporarily have stale data until changes fully propagate through the system. Since all writes go through a single primary node, there's no need for complex conflict resolution - the primary node determines the order of all updates. However, failure recovery becomes more complex since we need to track which updates may have been missed while replicas were down and ensure they get properly synchronized when they come back online. Additionally, if the primary node fails, we need a mechanism to promote one of the replicas to become the new primary, which can introduce complexity and potential downtime during the failover process.

### 

Good Solution: Peer-to-Peer Replication

###### Approach

In peer-to-peer replication, each node is equal and can accept both reads and writes. Changes are propagated to other nodes using [gossip protocols](https://en.wikipedia.org/wiki/Gossip_protocol), where nodes periodically exchange information about their state with randomly selected peers.

This provides scalability and availability since there's no single point of failure. When a node receives a write, it can process it immediately and then asynchronously propagate the change to its peers. The gossip protocol ensures that changes eventually reach all nodes in the cluster.

Peer-to-Peer Replication

###### Challenges

While peer-to-peer replication offers great scalability and availability, it comes with some significant challenges. The implementation is more complex than other approaches since each node needs to maintain connections with multiple peers and handle conflict resolution. The eventual consistency model means that different nodes may temporarily have different values for the same key. Additionally, careful consideration must be given to conflict resolution strategies when concurrent updates occur at different nodes.

All of the above approaches could be good depending on your specific use case and constraints. Redis, for example, uses asynchronous replication by default, though it does provide a WAIT command that allows clients to wait for replica acknowledgement if needed (sometimes called "semi-synchronous" behavior). Other distributed systems allow you to configure the replication type in order to choose the right trade-offs for your workload.

In your interview, I'd discuss the options with your interviewer. It's likely they care less about you making a decision here and more about seeing that you're aware of the options and trade-offs.

Given our requirements, I'd discuss two options with my interviewer:

1.  Asynchronous replication for a good balance of availability and simplicity
    
2.  Peer-to-peer for maximum scalability
    

The reality is, there is no single right answer here.

### 2) How do we ensure our cache is scalable?

Designing a reliable, single-node cache is a good start, but it won't meet the lofty non-functional requirements of handling 1TB of data or sustaining 100k requests per second. We need to spread the load across multiple machines to maintain low latency and high availability as we grow. This is where scaling strategies come into play, and one of the core tactics is to distribute data and requests intelligently.

We can't fit 1TB of data efficiently on a single node without hitting memory and performance limits. We need to split—or **shard**—our key-value pairs across multiple nodes. If done well, each node will hold a manageable portion of the dataset, allowing us to scale horizontally by simply adding more machines.

What is the difference between partitioning and sharding? Technically, partitioning refers to splitting data within a single database/system, while sharding specifically means splitting data across multiple machines/nodes. However, you'll hear these terms used interchangeably in practice - when someone mentions "partitioning" in a distributed systems context, they very likely mean distributing data across different nodes (sharding). The key thing is to understand the core concept of splitting data, regardless of which term is used.

We can estimate the number of nodes needed by considering both our throughput and storage requirements. Let's start with throughput. Let's assume we've benchmarked a single host to be able to perform 20,000 requests per second - we would need at least 100,000 / 20,000 = 5 nodes to handle our throughput requirements. Adding some buffer for traffic spikes and potential node failures, we should plan for around 8 nodes minimum to handle our throughput needs.

For storage requirements, we need to consider that a typical AWS instance with 32GB RAM can reliably use about 24GB for cache data after accounting for system overhead. Since we need to store 1TB (1024GB) of data, we would need approximately 1024GB / 24GB = 43 nodes just for storage. Adding some buffer for data growth and operational overhead, we should plan for about 50 nodes.

Note that the throughput (20k requests/second) and memory capacity (24GB usable from 32GB RAM) estimates used above are rough approximations for illustration. Actual performance will vary significantly based on hardware specs, cache configuration, access patterns, and the nature of your cached data. Always benchmark with your specific workload to determine true capacity needs.

Since our storage-based calculation (50 nodes) exceeds our throughput-based calculation (8 nodes), we should provision based on the storage needs. This higher node count actually works in our favor, as it gives us plenty of headroom for handling our throughput requirements. Of course, these are rough estimates and the actual numbers would vary based on several factors including our choice of instance types, the average size of our cached values, read/write ratio, replication requirements, and expected growth rate. But, for our purposes, we are going to say we'll have 50 nodes in our cluster!

So we have 50 nodes, but how do we know which node a given key-value pair should be stored on?

### 3) How can we ensure an even distribution of keys across our nodes?

The answer? [Consistent hashing](https://www.hellointerview.com/learn/system-design/deep-dives/consistent-hashing).

I'm sure you've read about consistent hashing before, but let's quickly review the core concept.

Without consistent hashing, the naive solution would be to use a simple modulo operation to determine which node a key-value pair should be stored on. For example, if you had 4 nodes, you could use hash(key) % 4 and the result would be the node number that that key-value pair should be stored on.

This works great when you have a fixed number of nodes, but what happens when you add or remove a node?

Consistent hashing is a technique that helps us distribute keys across our cache nodes while minimizing the number of keys that need to be remapped when nodes are added or removed. Instead of using simple modulo arithmetic (which would require remapping most keys when the number of nodes changes), consistent hashing arranges both nodes and keys in a circular keyspace.

So, instead of using hash(key) % 4, we use a consistent hashing function like [MurmurHash](https://en.wikipedia.org/wiki/MurmurHash) to get a position on the circle. Given that position, we move clockwise around the circle until we find the first node. That is the node that should store the key-value pair on!

Consistent Hashing

Why go through all this trouble?

Because now if a node is added or removed, only the keys that fall within that node's range need to be remapped, rather than having to redistribute all keys across all nodes as would be the case if our modulo operation went from hash(key) % 4 to hash(key) % 5 or hash(key) % 3.

To read more about consistent hashing, I'd recommend checking out the [Cassandra deep dive](https://www.hellointerview.com/learn/system-design/deep-dives/cassandra#partitioning).

### 4) What happens if you have a hot key that is being read from a lot?

You're bound to get the classic interview follow up question: What about hot keys?

To be honest, this is usually a property of the client or user of the cache, not the cache itself. Still, it comes up frequently in interviews, so it's best to be prepared with a thoughtful answer.

Hot keys are a common challenge in distributed caching systems. They occur when certain keys receive disproportionately high traffic compared to others - imagine a viral tweet's data or a flash sale product's inventory count. When too many requests concentrate on a single shard holding these popular keys, it creates a hotspot that can degrade performance for that entire shard.

There are two distinct types of hot key problems we need to handle:

1.  **Hot reads**: Keys that receive an extremely high volume of read requests, like a viral tweet's data that millions of users are trying to view simultaneously
    
2.  **Hot writes**: Keys that receive many concurrent write requests, like a counter tracking real-time votes.
    

###### Pattern: Scaling Reads

Hot key scenarios in distributed caches perfectly demonstrate **scaling reads** challenges. When millions of users simultaneously request the same viral content, traditional caching assumptions break down.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/scaling-reads)

Let's first explore strategies for handling hot reads, then we'll discuss approaches for hot writes afterwards.

### 

Bad Solution: Vertical Scaling all Nodes

###### Approach

The simplest approach is to scale up the hardware of all nodes in the cluster to handle hot keys. This means upgrading CPU, memory, and network capacity across the board. While straightforward, this treats all nodes equally regardless of their actual load.

###### Challenges

This is an expensive and inefficient solution since we're upgrading resources for all nodes when only some are experiencing high load. It also doesn't address the root cause of hot keys and can still lead to bottlenecks if keys become extremely hot. The cost scales linearly with node count, making it unsustainable for large clusters.

### 

Good Solution: Dedicated Hot Key Cache

###### Approach

This strategy involves creating a separate caching layer specifically for handling hot keys. When a key is identified as "hot" through monitoring, it gets promoted to this specialized tier that uses more powerful hardware and is optimized for high throughput. This tier can be geographically distributed and placed closer to users for better performance.

Dedicated Hot Key Cache

###### Challenges

Managing consistency between the main cache and hot key cache adds complexity. We need sophisticated monitoring to identify hot keys and mechanisms to promote/demote keys between tiers. There's also additional operational overhead in maintaining a separate caching infrastructure. The system needs careful tuning to determine when keys should be promoted or demoted.

### 

Good Solution: Read Replicas

###### Approach

We talked about this in high availability with replication. As a refresher, this is where we create multiple copies of the same data across different nodes. It makes it easy to distribute read requests across the replicas.

###### Challenges

While read replicas can effectively distribute read load, they come with significant overhead in terms of storage and network bandwidth since entire nodes need to be replicated. This approach also requires careful management of replication lag and consistency between primary and replica nodes. Additionally, the operational complexity increases as you need to maintain and monitor multiple copies of the same data, handle failover scenarios, and ensure proper synchronization across all replicas. This solution may be overkill if only a small subset of keys are actually experiencing high load.

### 

Great Solution: Copies of Hot Keys

###### Approach

Unlike read replicas which copy entire nodes, this approach selectively copies only the specific keys that are experiencing high read traffic. The system creates multiple copies of hot keys across different nodes to distribute read load, making it a more targeted solution for handling specific traffic hotspots.

Here's how it works:

1.  First, the system monitors key access patterns to detect hot keys that are frequently read
    
2.  When a key becomes "hot", instead of having just one copy as user:123, the system creates multiple copies with different suffixes:
    
    -   user:123#1 -> Node A stores a copy
        
    -   user:123#2 -> Node B stores a copy
        
    -   user:123#3 -> Node C stores a copy
        
    
3.  These copies get distributed to different nodes via consistent hashing
    
4.  For reads, clients randomly choose one of the suffixed keys, spreading read load across multiple nodes
    
5.  For writes, the system must update all copies to maintain consistency
    

For example, if "product:iphone" becomes a hot key during a flash sale with 100,000 reads/second but only 1 write/second:

-   Without copies: A single node handles all 100,000 reads/second
    
-   With 4 copies: Each copy handles ~25,000 reads/second, but writes need to update all 4 copies
    

This approach is specifically designed for read-heavy hot keys. If you have a key that's hot for both reads and writes, this approach can actually make things worse due to the overhead of maintaining consistency across copies.

Copies of Hot Keys

###### Challenges

The main challenge is keeping all copies in sync when data changes. When updating a hot key, we need to update all copies of that key across different nodes. While we could try to update all copies simultaneously (atomic updates), this adds significant complexity. However, most distributed caching systems, including ours, are designed to be eventually consistent - meaning it's acceptable if copies are briefly out of sync as long as they converge to the same value. This makes the consistency challenge much more manageable since we don't need perfect synchronization.

There's also overhead in monitoring to detect hot keys and managing the lifecycle of copies - when to create them and when to remove them if a key is no longer hot. The approach works best when hot keys are primarily read-heavy with minimal writes.

The key takeaway for your interviewer is demonstrating that you understand hot keys are a real operational concern that can't be ignored. By proactively discussing these mitigation strategies, you show that you've thought through the practical challenges of running a distributed cache at scale and have concrete approaches for addressing them. This kind of operational awareness is what separates theoretical knowledge from practical system design experience.

### 5) What happens if you have a hot key that is being written to a lot?

Hot writes are a similar problem to hot reads, but they're a bit more complex and have a different set of trade-offs.

### 

Great Solution: Write Batching

###### Approach

Write batching addresses hot writes by collecting multiple write operations over a short time window and applying them as a single atomic update. Instead of processing each write individually as it arrives, the client buffers writes for a brief period (typically 50-100ms) and then consolidates them into a single operation. This approach is particularly effective for counters, metrics, and other scenarios where the final state matters more than tracking each individual update.

Consider a viral video receiving 10,000 view updates per second. Rather than executing 10,000 separate operations to set new values (e.g. views=1, views=2, views=3, etc), write batching might collect these updates for 100ms, then execute a single operation to set the final value 1,000 higher. This reduces the write pressure on the cache node by an order of magnitude while still maintaining reasonable accuracy. The trade-off is a small delay in write visibility, but for many use cases, this delay is acceptable given the substantial performance benefits.

###### Challenges

The main challenge with write batching is managing the trade-off between batching delay and write visibility. Longer batching windows reduce system load but increase the time until writes are visible. There's also the complexity of handling failures during the batching window - if the batch processor fails, you need mechanisms to recover or replay the buffered writes. Additionally, batching introduces slight inconsistency in read operations, as there's always some amount of pending writes in the buffer. This approach works best for metrics and counters where eventual consistency is acceptable, but may not be suitable for scenarios requiring immediate write visibility.

### 

Great Solution: Sharding Hot Key With Suffixes

###### Approach

Key sharding takes a different approach to handling hot writes by splitting a single hot key into multiple sub-keys distributed across different nodes. This is the one we talk about in many other Hello Interview breakdowns.

Instead of having a single counter or value that receives all writes, the system spreads writes across multiple shards using a suffix strategy. For example, a hot counter key "views:video123" might be split into 10 shards: "views:video123:1" through "views:video123:10". When a write arrives, the system randomly selects one of these shards to update.

This approach effectively distributes write load across multiple nodes in the cluster. For our viral video example with 10,000 writes per second, using 10 shards would reduce the per-shard write load to roughly 1,000 operations per second. When reading the total value, the system simply needs to sum the values from all shards. This technique is particularly powerful because it works with any operation that can be decomposed and recomposed, not just simple counters.

###### Challenges

The primary challenge with key sharding is the increased complexity of read operations, which now need to aggregate data from multiple shards. This can increase read latency and resource usage, effectively trading write performance for read performance. There's also the challenge of maintaining consistency across shards during failure scenarios or rebalancing operations. The number of shards needs to be carefully chosen - too few won't adequately distribute the load, while too many will make reads unnecessarily complex. Finally, this approach may not be suitable for operations that can't be easily decomposed, such as operations that need to maintain strict ordering or atomicity across all updates.

### 6) How do we ensure our cache is performant?

Our single-node design started off simple and snappy—lookups and writes were O(1) and served straight out of memory. But as we grow our system into a large, distributed cache spanning dozens or even hundreds of nodes, the problem changes. Suddenly, performance isn’t just about how fast a hash table runs in memory. It’s about how quickly clients can find the right node, how efficiently multiple requests are bundled together, and how we avoid unnecessary network chatter.

Once we start scaling out, we can no longer rely solely on local performance optimizations. Even if each individual node is blazing fast, the interaction between nodes and clients, and the overhead of network hops, can introduce significant latency. It’s not unusual for a request that was once served in microseconds to slow down when it has to traverse the network multiple times, deal with connection setups, or handle many small requests individually.

The good news is, we already solved this problem with each of our previous solutions. Request batching, which helped us with our hot writes, is also an effective general technique for reducing latency since it reduces the number of round trips between the client and server.

Consistent hashing, which we talked about in our sharding solution, is also an effective general technique for reducing latency since it means we don't need to query a central routing service to find out which node has our data -- saving us a round trip.

The only other thing I may consider mentioning in an interview if asking this question is connection pooling. Constantly tearing down and re-establishing network connections between the client and servers is a recipe for wasted time. Instead of spinning up a fresh connection for every request, clients should maintain a pool of open, persistent connections. This ensures there’s always a ready-to-use channel for requests, removing expensive round-trip handshakes and drastically reducing tail latencies (like those p95 and p99 response times that can make or break user experience).

## Tying it all together

Ok, tying it all together, on each node, we'll have two data structures:

1.  A hash table for storing our key-value pairs
    
2.  A linked list for our LRU eviction policy
    

When it comes to scaling, it depends on which of the above approaches you choose, as you can end up with a slightly different design. Assuming you opted for:

1.  Asynchronous replication for high availability and handling hot key reads.
    
2.  Consistent hashing for sharding and routing.
    
3.  Random suffixes for distributing hot key writes across nodes.
    
4.  Write batching and connection pooling for decreasing network latency/overhead.
    

Your final design might look something like this:

Final Design

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

### Mid-level

For mid-level candidates, we'll typically spend more time on the high-level design section (aka the low-level design) and less time on the scale section. I expect that they can reason about the right combination of data structures to achieve our latency requirements and that they can problem-solve their way to basic answers about scalability and availability. For example, understanding the need for replication and figuring out the different ways that could be implemented.

### Senior

For senior candidates, I expect that the low-level design portion is relatively straightforward and that we instead spend most of our time talking about scale. I want to see their ability to weigh tradeoffs and make decisions given the requirements of our system. They should know basic strategies for handling hot reads/writes and be able to talk about things like consistent hashing as if they are second nature.

### Staff

I don't typically ask this of staff engineers because, by the time you are staff, these concepts likely come naturally, which means it isn't the best question to evaluate a staff engineer.

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

![Tejas Venky](https://lh3.googleusercontent.com/a/ACg8ocJjjTUKhBAfgG0KUqnba_vBhBl8Nf4yr_d_1xIpVV4EqNHpnJE=s96-c)

Tejas Venky

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4ktzpg6004f5en629qnr6vv)

damn I rlly thought this problem was super hard, legit just scaling the leetcode problem lru cache lolzzz thx for the article big boss

Show more

21

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4kwhz1o008112o4hf374te3)

Ez as 1,2,3!

Show more

19

Reply

S

SecureTomatoLynx933

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4kyd7jj009p5en6snr8o7ye)

Thanks for the write-up! In this example, it appears that the client seems to know a lot of things:

1.  It knows about the multiple hot key copies (with random suffixes)
2.  It knows to batch writes
3.  It also has the a map of the consistent hashing scheme and thus sends a request to the appropriate shard.

But how does the client come to know all of this information? Is there a master broker in our Cache that is relaying all of this information to the client?

Show more

9

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4kygeqr00a714hjn28kx53h)

The client here is a bit different than the client we're used to seeing in other designs. It's not an end user, it's a service within your application. Typically, this would be a client library (like redis-py for Redis or memcached for Memcached) that handles all this complexity.

Some managed systems like Redis Enterprise or AWS ElastiCache abstract this away completely, but for a custom implementation, you'd just use a client library.

Show more

6

Reply

S

SecureTomatoLynx933

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4kymc5k00bbv0vvyt9z3u26)

Thanks for the reply. So can we assume that the client library is communicating with one (or more) of the Cache nodes to get all this information? Similar to how (I believe) Kafka Consumers stay up to date on the configurations of Kafka Brokers.

Show more

0

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4thggdt00efqchtxc39s755)

SecureTomatoLynx933 asked what I was about to! My thought on this is that the client that did the write could receive the sharded keys to use for future requests, but other clients will have to make an initial request to the primary key server before coming to know about the existence of sharded keys. Same logic applies if the client application is restarted.

Is this what you had in mind? Perhaps this point should be clarified a little more in the deep dives for hot reads and writes, since it's an important question.

Show more

4

Reply

M

ManagerialCyanSawfish581

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmb8lgbpc00u3ad08ua18cd7o)

Great Article, Can we use the approach for answering other questions like system design for memcached?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmb8lsko600soad085yrojbk4)

Of course! That's the point!

Show more

1

Reply

M

ManagerialCyanSawfish581

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbebwx7x035jad07azi2uhvi)

Thank you, reason being I had been asked in an interview to do system design for a single node memcached as I do not have much expertise with distributed systems but I had this article in my mind as I had practiced this article so many times, I answered exactly what you had given here in this article hoping it suffices. Thank you

Show more

0

Reply

D

Danny

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm976g4r600jvad084kx76zi1)

The batch writes section is ambiguous about the client being involved. I think that deserves updating. As for the connections maintained between the client and nodes, that is a lot more detail that would add a lot of complexity to the interview. Might require a separate write up.

Show more

1

Reply

![Shreeharsha Voonna](https://lh3.googleusercontent.com/a/ACg8ocJFOfqMA6ZDt2EamTjU3FJJThz35r5s7qZyqePh9qAYYlJq_K8p=s96-c)

Shreeharsha Voonna

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4nv392e00x0wlufan4mryc2)

Evan, thanks for the great content once again. I have 2 inputs :

1.  Being a premium user, it would be great if you notify me for every new content thats getting published so that I stay up to date with it.
2.  I have felt lack of depth in certain articles such as Whatsapp, Gopuff. Ideally, irrespective of the author, depth of the content should follow same pattern and standards.

Let me know your thoughts on the same.

Show more

24

Reply

![Shreeharsha Voonna](https://lh3.googleusercontent.com/a/ACg8ocJFOfqMA6ZDt2EamTjU3FJJThz35r5s7qZyqePh9qAYYlJq_K8p=s96-c)

Shreeharsha Voonna

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4nv7oz100xzlkt9npdh2li3)

If not notification, least we could have is publishedOn/updatedOn date for each of the article. Thanks

Show more

7

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4oh5kit018kjs6sse6n51ss)

Hey Shreeharsha - We'll be sending periodic updates to premium subscribers whenever new content is published so you can keep up to date.

We have a new revision for Whatsapp going out later this week. If you have questions or want clarity on Gopuff, go ahead and drop a comment there and we'll try to accommodate.

Show more

3

Reply

![Will](https://lh3.googleusercontent.com/a/ACg8ocL1s_xaZXzcDVbVslekgcHNZM3C__oW_Q-dar2pISsA2modSg=s96-c)

Will

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4qapvj300dsxwp6mbw6dusc)

Hey Stefan,

Thanks for all the great content. Do you have a specific date for when the new revision for Whatsapp will be released?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4qb4io400eaxwp6wo6f5g5h)

Live today.

Show more

1

Reply

T

TechnologicalBlueWhippet396

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4un3zrm001zu2qa2pkn07go)

Hi Stefan, Could you include a changelog for article you guys updated? The reason is I take notes from your articles, and I would like to revisit if something new added

Show more

3

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4thb5b400dpe4jkcj3jazny)

Shreeharsha spoke my mind! A last updated date would be really helpful, since it'd indicate a revision had been made, as well as help weed out comments that indicated problems addressed by later updates. I've now seen a handful of articles where the comments are referring to things that don't exist, and it's really confusing.

Show more

1

Reply

F

FlyingTomatoHeron745

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4s4n2al00wdrpk75ys7qkum)

Great write up! for premium write-ups does the team plan to make videos as well?

Show more

9

Reply

S

StaticJadeLemur903

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4t1zz2c0054w579v28siy6k)

Can you add a portion of single node deep dive such as sharded lock within a map?

Show more

2

Reply

VB

vishal bajoria

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4ta9ism006vtgj5pqj0qrkx)

Great explanation. make it sounds super easy.

Show more

1

Reply

![lloyd lasrado](https://lh3.googleusercontent.com/a/ACg8ocKvBIGFW31V3eIZmmqFiX_LZ8HKwu7sb2S7AG_FUYIbvHaxWCwc=s96-c)

lloyd lasrado

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4uwyva200629pjq4a35d678)

Thanks for documenting most common problems. Is it possible for you to upload videos for all the common problems to paid members. I understand more well watching your videos and the way you explain in depth.

Show more

9

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4vjrkn2007011622l1ns3sp)

Noted. Balancing videos with more content. But will keep this in mind :)

Show more

3

Reply

![lloyd lasrado](https://lh3.googleusercontent.com/a/ACg8ocKvBIGFW31V3eIZmmqFiX_LZ8HKwu7sb2S7AG_FUYIbvHaxWCwc=s96-c)

lloyd lasrado

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4vjuvv1007ty3n1zya9obxt)

Thank you for replying. One suggestion I have if you can do videos at least for Infra design as they are more complicated. Also may be pick unique ones like Distributed cache or Notification service etc. Product design interviews are mostly similar in many ways.

Show more

2

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5ahb04h028jc4lgfvjz2s4c)

There aren’t separate videos for premium members yet, right? Could you please confirm, Evan? Additionally, does your team have any plans for this in the near future? If so, I’d love to hear an update.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5al7grw02gyv808worbyf4m)

Nope, not yet. And not sure. Still weighing the tradeoffs between videos on existing premium content vs net new content.

Show more

0

Reply

![Katie McCorkell](https://lh3.googleusercontent.com/a/ACg8ocJhhgdomrgC_-4WZgadOwN2X8drRXio1OaPECBGUAB1SRYpPV4UHw=s96-c)

Katie McCorkell

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7wgujyj016rjmejmqcfkote)

Chiming in with one vote from me preferring blogs over videos!

Show more

0

Reply

G

GleamingChocolateChinchilla112

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4xbf62b01fuhjxb7elj3fwz)

In this article, you assumed the system just knows which key is hot, but in reality, the system should query a data store whether a key is hot or not. How would you store this data? Hard coding hot keys might be simple, but for dynamic solution it might get very complicated, isn't it?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm51ssycu02fiwbx3suizolu7)

That's up to the client to figure out which key is hot

Show more

0

Reply

I

IndividualGrayToucan573

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5fpms8j008rrkl807ure99l)

Evan, does the following sound right? first, client just queries the given key. If not available, client queries one of the random key. If available, return. If not, it means no this entry in the cache.

another approach is for every query, first try a random key, but I feel most keys are not hot keys, so no random key.

Show more

0

Reply

![ashish pant](https://lh3.googleusercontent.com/a/ACg8ocK2AJJYf_XqecbVLt16PD1svpuKQ2Y6lvbaaz9T4-ueqjeFXhez=s96-c)

ashish pant

[• 12 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cme7ue53w02vwad0862cez5u9)

One way to solve this is to leave the primary key also as a potential read key. Some additional bits are added to the value about how many times it is sharded. Next time the same service queries this key it can use that information. If the service has n replicas then there will be at-least n separate queries to the primary read key and then subsequent request are spread out. This works because we are using this only for hot keys.

Show more

0

Reply

R

RealisticHarlequinKangaroo309

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4yip27o022specvnybjb3bc)

How do we handle LRU evictions at scale with the distributed architecture?

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm51ssc8102bs13vusc8ys2z3)

You usually wouldn't. Just handle at the node level.

Show more

1

Reply

P

PrimeMoccasinAsp131

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm52xuj5o032yub4z3xedeate)

Can you elaborate on this more? Even if we handled LRU evictions at the node level (but for scale), wouldn't we need some form of data persistence for the keys that are evicted? Or are you saying we don't need to worry about any LRU evictions at scale since we did the estimations assuming all of the data can reside in memory and there will be no evictions ?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm535k67t03anub4zkw42p63v)

Why do you want to persist the evicted keys? They're evicted, get rid of them! Each node handles it's own eviction via LRU with the method documented above.

Show more

3

Reply

![Anirudh Gubba](https://lh3.googleusercontent.com/a/ACg8ocJ0UTZ3bqk5VyuPBCEvYy9XNl2oKgfbbkhVZKllzkZEyyxxKLpd=s96-c)

Anirudh Gubba

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8uos8pf01cfad08yxtj0b94)

But would we not wrongly evict keys that are super hot but may be seeing low access counts at a node level because we've split up traffic to that key?

Show more

1

Reply

![indavarapu aneesh](https://lh3.googleusercontent.com/a/ACg8ocLx77-thRGA5bldZDZhNF8MbwtxB4dZmFZ3zHzbk_Xu4IB-og=s96-c)

indavarapu aneesh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbtnef6804a208adi409eoi2)

the general assumption with sharding is that data requested by a query only touches a single node (that's why chosing the right sharding strategy is important). Hence the LRU policy pertaining to that node is aware of the most recent access for the hot key.

Show more

0

Reply

F

FormalBlushMastodon190

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmd7afiyc02cead08j82zgo96)

A key might be evicted at a node even though the size of the global distributed cache is not full. Is this acceptable?

Show more

0

Reply

F

FreshCrimsonTick387

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm4yzmgx000bsmqq6s8sd4iig)

Hello,

Quick clarification/follow up on my understanding:

For handling the hot read issue, the third option of read replicas, are we saying

1.  Since we anyways went with Leader/Replica architecture for replication, idea is to put more hosts behind each set of shards. So, automatically the reads will be distributed across more hosts?
    
2.  And I guess the concern there is we are unnecessarily increasing more hosts for all the data rather than fixing the root which is just hot keys?
    

Show more

1

Reply

S

SelectiveMaroonTapir734

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm51n0gen02a7ub4zc601oak6)

"Remove the Node from its current position in the list (O(1))" -- do you mean the space complexity? the runtime to search for a given node is O(N)

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm51sqc0002asr5q61vyz0vch)

We dont need to scan the list for it. We have the node from the hashmap lookup.

Show more

0

Reply

S

SelectiveMaroonTapir734

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm51vlzpz02hmub4zhu9viiav)

Oh I see it now, the hash map holds the reference to the actual node. Thank you ❤️

Show more

1

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm58vayil0109i96o09eaveal)

Are these low-level infra questions generally asked for a typical SWE position?

Show more

2

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm58ve83l00xfc4lg9pzbxqz1)

-   what should be our response if the interviewer wants us to code stuff up, since it may take considerable time away from the actual design?

Show more

0

Reply

V

VocalTealMandrill585

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm58xtu2h0145i96oaqwyzyvs)

If we append a suffix for hot key reads, should we use an external metadata service (e.g., ZooKeeper) to track these suffixes, or would that be overkill for this use case?

This question makes me wonder if anytime we need to handle special bookkeeping (e.g., tracking modifications like suffixes or random values appended for collision resolution in bit.ly-style systems), we should default to using an external service or database that specializes in managing such metadata.

Show more

0

Reply

![indavarapu aneesh](https://lh3.googleusercontent.com/a/ACg8ocLx77-thRGA5bldZDZhNF8MbwtxB4dZmFZ3zHzbk_Xu4IB-og=s96-c)

indavarapu aneesh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbtngusp04at08ads57lmnee)

there are two ways to handle this

1.  Make the client (the lib you're providing to the end user) handle the suffixed hot key reads aggregation.
2.  Or have a query router which fans out for suffixed reads and does the job for you.

Show more

0

Reply

A

aniu.reg

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm59p1u4c01h1c4lgsf8zlp8g)

For deep dive 2 and 3, you just briefly mentioned consistent hashing. A general question regarding consistent hashing (not only this design problem) do we always need a zookeeper like component to store "key range" -> "(v)node" mapping.

It looks like in this write up you imply the cache client plays the zookeeper role. This is why I have the aforementioned question.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5alcwab02h6v808svzuu9ze)

You could also have a few designated "coordinator" nodes in the cache cluster maintain and serve the mapping

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8l7eni300grve357e5s8lof)

@Evan Is there a decentralized way that no coordinator required? Does gossip protocol work?

Show more

0

Reply

![indavarapu aneesh](https://lh3.googleusercontent.com/a/ACg8ocLx77-thRGA5bldZDZhNF8MbwtxB4dZmFZ3zHzbk_Xu4IB-og=s96-c)

indavarapu aneesh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbtnnc4204c108adxi1moory)

the way cassandra routes the data is that there are designated coordinators which find the right node and relay the connection. So the gossip protocol helps the nodes keep track of metadata.

Show more

0

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5ahdfa402cue1nmasb7djzl)

Evan, request for clarification on the approaches outlined for "What happens if you have a hot key that is being written to a lot?". Here the key is assumed as some sort of counter. I don't think of any use case, however what if the key is not a counter. It's value is getting changed very frequently. How to handle such problem?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5alany802h3v808n0f3jngn)

You'd probably use write batching here. It's not very common that you'd have so many writes to a single value that an in-memory cache (even a single node) can't handle it. I'd probably ask why we have a single value changing so much in the first place and is there a different (maybe event based?) abstraction.

Show more

0

Reply

![Liliiia Rafikova](https://lh3.googleusercontent.com/a/ACg8ocLhKI61yr7h7HM0rSPvHT8QjFtJteOZbg84lT2Kk6f-YRWWSw=s96-c)

Liliiia Rafikova

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5e6j31100f56zza3f34frn7)

concurrency aspect for single node implementation would have been nice to cover

-   How we are going to handle read and write concurrently
-   Clean up operation will it pause all operations on Cache?

Show more

4

Reply

N

NeutralAmethystUnicorn745

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5exyig4005cf933gsx8zmi8)

+1 Evan please answer this question!

Show more

0

Reply

Y

youssef.antwan

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5egavj500p069iz686foiow)

Enjoying the premium content so far! It would be nice if you add a highlight/note feature.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5ejy2ap0050cjftn9wg23q0)

I like this idea! There's no guarantee of when we can get to it, but I will add it to our feature wishlist.

Show more

1

Reply

C

CuddlyCopperWildebeest704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5g0wpkr003uepsuugm51u3k)

"We can't fit 1TB of data efficiently on a single node without hitting memory and performance limits. "

Why not use https://aws.amazon.com/ec2/instance-types/high-memory/ ? They seem to have enough memory and bandwidth to satisfy the requirements

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5kj98cj04kl1osrl072fvgp)

Sheesh, that's a lot of mem... If vertical scaling solves the problem, amazing. In an interview, I'd appreciate this from a candidate, then just raise the number to force a distributed system (since that's what's being evaluated).

Show more

6

Reply

C

CuddlyCopperWildebeest704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5kjddfs04l61osrdgvsh0it)

Haha kind of expected that answer :))

Show more

2

Reply

T

Tom

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmcdvcfah09ndad08tgg0143o)

LOL, good to see this thread. Had the same thought when doing Guided Practice.

Show more

0

Reply

![prakhar](https://lh3.googleusercontent.com/a/ACg8ocKDXXoGteJDNOueUSTQRS4w_FcP_2LM2rmDgu5vmw2SK28XSQ=s96-c)

prakhar

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5p28h7b00syiuxnxvx9pchg)

gossip protocol deep dive when?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5pjfcj4010s114y6pivxjve)

Add it to https://www.hellointerview.com/learn/system-design/in-a-hurry/vote !

Show more

1

Reply

![prakhar](https://lh3.googleusercontent.com/a/ACg8ocKDXXoGteJDNOueUSTQRS4w_FcP_2LM2rmDgu5vmw2SK28XSQ=s96-c)

prakhar

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5pl43re011z114y6fbc9isx)

Added :D

Show more

0

Reply

![prakhar](https://lh3.googleusercontent.com/a/ACg8ocKDXXoGteJDNOueUSTQRS4w_FcP_2LM2rmDgu5vmw2SK28XSQ=s96-c)

prakhar

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5pl5j730140wu5h6p46mif0)

Btw, the like button seems broken, When I like some comment, I am not able to like other's unless I reload

Show more

2

Reply

I

ImmenseIndigoMoose446

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5qkdsho0250114ylh9yfb01)

Thanks for the post! I’m curious about how LRU eviction is handled in a distributed cache with a consistent hash ring. Does it rely on a local LRU policy with each server having a fixed capacity, or does it implement a global LRU mechanism across distributed cache servers?

If local LRU, then I can also see a potential issue with "thrashing," where frequent evictions and reinsertions occur on the same node. How is this scenario typically addressed?

Show more

0

Reply

C

ColouredChocolateMockingbird464

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5urd0j50040a62znl1urkbo)

What does the timeline look like for low-level design content (object-oriented)? HLD content has helped me immensely.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5vsetm5016ba62zlaglv5gu)

Just two of us so planning is not that sophisticated. Looking outside the 2 month window at the current moment.

Show more

1

Reply

M

MagnificentPeachBoa251

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5vs2wjx015qa62zny97ldrs)

Can you explain this scenario about handling evictions?

-   We have two nodes: one handling keys A-M and the other handling keys N-Z
-   both nodes' caches are full
-   a new key in the A-M range is set so we need to evict a key. A-M and N-Z node have their own LRU key-value pair, but the N-Z LRU key-value pair is actually the overall LRU.

If we evict the A-M LRU, won't our cache state technically be "incorrect" ?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5zkokg000b9vw0mntwo4ug1)

Yeah, it's okay :) In practice, distributed caches like Redis simply maintain independent LRU policies per node and rely on random distribution of keys (via consistent hashing) to achieve a "good enough" approximation of global LRU behavior. If you had a use case that, for some reason, required perfect global eviction, then you'd need synchronization. But the overhead is usually not worth it.

Show more

1

Reply

R

RadicalAquaBedbug232

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5w43zyi03mbp38vt8t3hkoy)

Thank you for the very good practice! Could you please help me to understand the approach with hot keys copies?

Assume that system detected that there are frequent requests to some\_key and I decided to create 4 copies: some\_key#1; some\_key#2; some\_key#3; some\_key#4

What I don't understand is: how client will request value? if client randomly select number and request some\_key#3 then it means that client magically knows that the key is hot - correct? Should we avoid that because client should not be aware about the implementation details?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5zklopl00a12078vghsv4lx)

Depends on the cache, but yes, this is typically done at the application layer. So the client needs to know who/what is hot. That said, note that "client" is not a user here. its some internal service that interfaces with your cache.

Show more

2

Reply

R

RadicalAquaBedbug232

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5zl2tzk00bvvw0m8h64n2rt)

Thank you for your reply, Evan!

> That said, note that "client" is not a user here. its some internal service that interfaces with your cache.

Yes, exactly! I think that in the design it will be good to have a router layer. This layer will have several nodes and each node will have a list of hot keys and will be responsible for routing the requests. Does it make sense?

More details: we have a layer, let us name it RoutingLayer. The primary function of this layer is to route requests from the client to the appropriate node that contains the desired key.

Show more

0

Reply

F

FancyTanGoldfish347

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7eyire4015qc411v8wrtba3)

Would it be better if the cache has a layer that handles the hot keys in a company with many services relying on it? It seems like we would offload a lot of hot key tracking responsibility to the calling services to know what keys are really hot when they may just be product teams that don't get too involved with the infra side of things.

Show more

0

Reply

F

FancyTanGoldfish347

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7eyttbm014fqlvzlkcvdb2u)

In general, why do we prefer to track hot keys at the application layer? I assume it is more localized and internal services may set a lower QPS limit to prevent throttling by the distributed cache when tracking hot keys.

Show more

0

Reply

S

surajrider

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7leit1w01fh8gbtzc1kzt9z)

Hi @Evan,

I disagree with making client being aware of the hot keys. This is a problem when you access caching service via multiple clients. How would client identify if a specific key is hot or not without any coordination ?

Show more

0

Reply

C

chris11josephite

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5x6ub8a00dvja3xsrklvvd9)

Question, at step 4, we have use read replicas to handle hot keys, Since we are already handled replication in Step 1 of deep dive.

Another round of read replicas is just to handle the hot keys, so we are talking about two replicas possibily for the hot keys? or i'm i missing ssomething here ?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5zkhsvi00a8rruodiy1fzwh)

No no, just same solution. read replicas help for both hot reads and fault tolerance

Show more

1

Reply

Q

QuietTomatoEchidna500

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5zhkb5p007xrruobft2p04w)

In the "Simple LRU Cache" picture, Node object should contain Key as well, right? Otherwise how you look up for least recently used one in the Hash table?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm5zkh1oc00aq12uxrd3ho06n)

yah :)

Show more

0

Reply

Y

yingdi1111

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm63z2mi700oky43n9c5d28c2)

While read replicas can effectively distribute read load, they come with significant overhead in terms of storage and network bandwidth since entire nodes need to be replicated. This approach also requires careful management of replication lag and consistency between primary and replica nodes. Additionally, the operational complexity increases as you need to maintain and monitor multiple copies of the same data, handle failover scenarios, and ensure proper synchronization across all replicas. This solution may be overkill if only a small subset of keys are actually experiencing high load.

I believe some of it is not accurate since we are doing replication no matter if need to address hot key or not. Since we are doing it no matter what. This should not be the challenges here.

Evan, what do you think?

Show more

0

Reply

![Vishwanath Seshagiri](https://lh3.googleusercontent.com/a/ACg8ocJPMnw34pexYIfo4msMIjXffaJx6mW4KOIfvTK9MyY6I2lDXYbe=s96-c)

Vishwanath Seshagiri

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm68sp5q801dyyoq9xd7ukowl)

One thing to consider in this article is recent work on applicability of LRU for these use cases. While conventionally LRU has been used for eviction, algorithms like S3FIFO and SIEVE have shown much better performance on web workloads which typically follow a power law distribution.

Also, with LRU when you're trying to access it via multiple threads, every "read" is also a write since you're updating the list and that leads to lock contention problems leading to reduced throughput.

Show more

0

Reply

Y

yingdi1111

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm69pvivz02axe7bz8getyy4v)

In order to address hot key read, the great solution is copies hot keys. However, how do we know which key is hot and how many copies do we have for each of the hot key?

We can make a simple assumption that each hot key have the same copies. However, we still need to know if a key is hot or not. How do we find out this information?

Option 1.

We always send a query to the first copy of the key (same for hot key and not hot key key instead of key-2 or key-3) and the value will indicate if it is hot {value: 123, ishot: 'false'} or {value: 234, ishot: 'true'} and the server that sends the request will have a map so it knows later if the key is hot or not.

Upside, it only need to query to first copy once. Downsize, it has to access to first copy for the first time and it could become hot for the first access, also each access server need to keep a map of the key and the hotness it cost extra memory usage.

Option 2.

We have another redis cache just to cache if the key is hot or not.

Upside. It solve the problem Downside. We are introducing a new hot key space. (But the value will be smaller(a boolean value) to show if it is hot or not. And every time we query, we need to query 2 keys (if it is hot) and another to the actual copy.

Is there another option?

Which makes me feel the read replica may not be a bad option since we need to have replicas either way and have read replica to increase read throughput makes sense to me.

And we can monitor the cache and if we have more traffic for certain parititons, we can assign more read replica to it.

What do you think of this?

Thanks, Di

Show more

1

Reply

S

surajrider

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7lemqoy01dtu3lmz8tpruol)

Great idea yingdi. Evan can you please explain how would client/backend get the mapping of hot key ? Do we maintain a separate cache

Show more

0

Reply

I

IntenseSapphireMastodon393

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm6daaakf03czsagc08ljsh96)

Are "Copies of hot keys" and "Sharding hot key with suffixes" basically the same idea? If we want to improve hot read, then we need to keep the copies in sync; if we just want to improve hot write then we don't need to keep them in sync and we can just aggregate the copies when read?

Show more

0

Reply

P

PersistentCoffeeClam140

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm6stwvlm02so50yfr14p8gg6)

Hi team, thanks for the great material! I'm new to system design, and I'm preparing for persistent kv store. I went through this mateiral, but wasn't sure what can be added/changed to support durability?

Show more

1

Reply

![Birzhan Auganov](https://lh3.googleusercontent.com/a/ACg8ocLFLhzbuYWICoDtJRRYc_CQjJowncNBmRNPD6ozyP_V1-1s2aY=s96-c)

Birzhan Auganov

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm726v5sb00gpihpgior94ga0)

From my understanding, we store frequency for each key and when it exceed hot key threshold, it will be replicated to other nodes. If it is true, I have 2 questions regarding this:

1.  how do we lower the frequency over time? do we periodically reduce hot keys frequency by 10% to make sure if they are still hot or not?
2.  what if one of our replicas gets hot? (for example user123#3)

Show more

0

Reply

![Donya ahmadi](https://lh3.googleusercontent.com/a/ACg8ocL34CCm9UXun8DZ78kow0OtIKmPfplNrNew6YHGUlOxpPRhsQ=s96-c)

Donya ahmadi

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm76m8o5803j188mxbkc6smln)

in step 3, why removing an item from doubly linked list is O(1) and not O(n)? 'Remove the Node from its current position in the list (O(1))'

Show more

0

Reply

S

SecretBronzeCardinal926

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmaj2ufu701fsad081fbuel32)

In the hashmap you keep the reference to the node. So when you access the key..you get the node itself as value.

Show more

0

Reply

![Birzhan Auganov](https://lh3.googleusercontent.com/a/ACg8ocLFLhzbuYWICoDtJRRYc_CQjJowncNBmRNPD6ozyP_V1-1s2aY=s96-c)

Birzhan Auganov

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm77gjjy9012fr9w5rp6rwvcl)

Hello, are the common problems section sorted by popularity? if I don't have time to solve all problems but want to prepare to my interview, what would be right order of solving the problems?

Show more

0

Reply

S

surajrider

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7lerg9v01ft8gbtvluahnsx)

Hi Evan,

Thanks for the great article! Can you please shed some light how is client made aware of hot keys ?

-   Leaving this to client means that we probably need synchronization among all clients so that we take aggregate numbers to identify correct hot keys
-   Leaving it to backend means, we need some sort of map which says whether a key is hot or not. And if a key is hot, we need the corresponding values (e.g. UserABC -> UserABC#1, UserABC#2 etc). How does backend maintain the state ? Can you please elaborate more on this ?

Show more

0

Reply

G

genre-tankers-1c

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7nej3yo00jrs6asybygpdo9)

5: This reduces the write pressure on the cache node by an order of magnitude while still maintaining reasonable accuracy.

It's 3 orders of magnitude

10000 per second -> 1 per 100ms = 10 per second so 1000x less

Show more

0

Reply

![Seulgi Kim](https://lh3.googleusercontent.com/a/ACg8ocLQHgYw_51XVN7bP30J81jKwwrUtpR3x4lixctpfKzEfY8dNuDC=s96-c)

Seulgi Kim

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7nqy63w00465g7ok44hwcui)

Super insightful and helpful, as always. Thank you!

Show more

0

Reply

![Katie McCorkell](https://lh3.googleusercontent.com/a/ACg8ocJhhgdomrgC_-4WZgadOwN2X8drRXio1OaPECBGUAB1SRYpPV4UHw=s96-c)

Katie McCorkell

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm7wk1fsa01dw133rze1qlder)

TLDR: In your section "Another option is to update one primary copy immediately and then propagate changes to replicas asynchronously" How is that done?

* * *

Is pub/sub also an option for Peer-to-Peer replication, instead of gossip protocol?

I had a question recently about scaling a "deny list" cache for IP addresses / malicious internet content. There was a black box that would say good/bad, and the challenge was to scale that system by building a cache, esp. one that worked across continents and datacenters. Eventual consistency is fine. An ip address once good could change to bad, but once bad it was ok to consider it bad forever.

The recruiter specified the solution should include a sort of leader-follower situation with near real-time eventual consistency.

It seems like having a cache in each data center and a pub/sub system to push to the other datacenters would work well. Is that correct, or is it better to use gossip protocol like you state here?

I think the main difference between pub/sub and gossip protocol is the existence of a central broker in pub/sub. But wouldn't the central broker just be the datacenter that is receiving the write? How would that be different than pub/sub?

Thanks!

Edit to add: Perhaps my suggestion is the "asynchronous replication" you have outlined here. The distinction of leader/follower is less important because the only conflict is that you don't want to write duplicates, basically you just merge the lists of keys, rather than needing to update any key-value pairs.

TLDR: In your section "Another option is to update one primary copy immediately and then propagate changes to replicas asynchronously" How is that done?

Show more

0

Reply

W

WiseAzureMacaw681

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm9xdy3q300gjad08gnux4yyr)

The primary/leader node acknowledges the write to the client immediately after persisting it in memory. It then spawns of a thread to asynchronously invoke all the follower nodes to write the same key/value, ensuring that the client write to the leader is not blocked. The leader is aware of all the followers and can connect to each of them.

As to whether Pub/Sub can be used as a way to do this async replication between leader and followers - I suppose it could be done. However, pub/sub systems usually have at-most-once delivery and are "fire and forget". Meaning if a subscriber aka follower node loses connectivity, it won't get the published change. In the case where the leader is explicitly calling the follower, it can do retries and keep a record of messages which were not sent to a follower. Unless you're pub/sub system is specially built to handle these edge-cases, you may have follower nodes being permanently out of sync.

Show more

0

Reply

![Brendan Robert](https://lh3.googleusercontent.com/a/ACg8ocJ4uBlrnNiooDQdc--fClet5qOjzegdHRirW4qXsm7rM5t-XMA=s96-c)

Brendan Robert

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm86e61km001mlecvvvz6edpj)

For hot reads if we have our keys on multiple nodes what system tracks what nodes have a copy both for distributing reads as well as for updating all copies on write, would it be stored in zookeeper or some kind of gossip protocol where the main node gets a read and distributes it? I would imagine it would have to be at the zookeeper/orchestrator level?

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm89jz9qe00d4hur6gsk4yhek)

Is the client actually doing the coordinate work and act like zookeeper? How to scale the client then?

Show more

0

Reply

![Wang lei](https://lh3.googleusercontent.com/a/ACg8ocIBwcYDZiesH-WGea9evEz-VtPcpOiYrYCCdqZM0uHbfMdpWw=s96-c)

Wang lei

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8e2sza1011w40wjnbbey930)

in deep dive 3) How can we ensure an even distribution of keys across our nodes, the solution is consistent hashing, but actually consistent hashing is not used for even distribution, instead, it is for minimizing data redistribution during scale. Also Redis does not use consistent hashing, it uses 16384 hash slots.

Show more

0

Reply

![Priyanshu Agarwal](https://lh3.googleusercontent.com/a/ACg8ocJgTv2BcGcmp9oDzLYhZbfujMo69WP2U2krPt6fwAhv0MI5lKU=s96-c)

Priyanshu Agarwal

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8oobprm00icfmoca4uvms3b)

Can you cover some ground on how we will be handling the concurrency of the system for parallel operation requests while maintaining availability? If you could suggest resources that I can look into, that too would be helpful. Thanks

Show more

0

Reply

![Garrett Mac](https://lh3.googleusercontent.com/a/ACg8ocKvc-WsjvNfRoWnWXLh2G95DQ31dmFv83g1P2q-nlplgPZOnQ=s96-c)

Garrett Mac

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8ry45wj00mxad0875e6jrnx)

I could have sworn there was a video for this a few weeks ago? Did it get removed?

Also small but where selecting "sort by" in the comments it goes to the top of the page and prevents scrolling (so i dont think its sorting)

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8rz1r3p00qrad08plvzwuu7)

Nope, no video for this one just yet. Soon though :)

As for sort by, seems to be working for me. You're talking about the "new", "old", "popular" dropdown, yah?

Show more

0

Reply

![Garrett Mac](https://lh3.googleusercontent.com/a/ACg8ocKvc-WsjvNfRoWnWXLh2G95DQ31dmFv83g1P2q-nlplgPZOnQ=s96-c)

Garrett Mac

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8rz448500gcad08j5n42d4v)

Oh my mistake then. Thanks for clarifying

oh works now must have been a "me issue"

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8rzb21d00r3ad08cvwddt3x)

No worries :)

Show more

0

Reply

![Ayush Shah](https://lh3.googleusercontent.com/a/ACg8ocLXl2Ir_8gOzUxfPa-vYAih41uxvDNAhx3-l_J13YMJIdcPtA=s96-c)

Ayush Shah

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm8w7h2ll0099ad07o1ujyyiy)

Is quorum read an acceptable solution for "highly available and fault tolerant?"

Show more

0

Reply

![Vishal Thakur](https://lh3.googleusercontent.com/a/ACg8ocLfTHpRgqbBZovPKQ9-Mas5BC81GUnvRicrR1vngsZ7fnrZSg=s96-c)

Vishal Thakur

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm90wmmcl006oad08zx8txsf0)

Should we also talk about durability here ? In my understanding, writes should also be written to a Write Ahead Log apart from in-memory hashmap.

Show more

0

Reply

L

LexicalAquamarineJaguar532

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm921noml00hpad08ufc0oiuf)

Question about the reliability of the cache, if we go with the async replication, do we need to talk about quorum or do we not care about that for this problem?

Show more

0

Reply

W

WiseAzureMacaw681

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm9xad0g1005iad09rhv4qiai)

I'm always interested in how we calculate "the number of instances needed to handle the required throughput." But is the specific calculation method less important than recognizing when you need multiple instances?

Fwiw, when I was doing this problem in the simulator - my estimate was CPU bound rather than memory: Assumption: 1 node has an 8 core CPU, 64 GB

1.  CPU: Each operation takes 5 ms, i.e one core can do 200 operations/sec. 8 core machine can do 1600 operations/sec. So, we'd need ~65 nodes.
2.  Memory: Need 1TB/64 GB == 15 nodes

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm9xdzfxs00gwad084p4vwwe2)

Much more important to realize where you need to scale vs exact numbers.

You're probably by off by an order of magnitude for CPU time of an operation, Redis will do tens to hundreds of thousands of ops on a single core. And if the operation on the CPU is taking longer than a millisecond it has to be doing something fancier than a memory fetch.

Good rule of thumb: caches are bottlenecked by memory!

Show more

1

Reply

W

WiseAzureMacaw681

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cm9xecqca00bcad09xzowe50y)

Thanks - I grossly overestimated the CPU time and confused it with the requirement to have total latency < 5ms (which would include other things). You're right - a memory access is in the order of nanoseconds, implying a much larger throughput.

Show more

0

Reply

![Sudhanshu Bansal](https://lh3.googleusercontent.com/a/ACg8ocISLhjg7YR3Nd9WTWPrYqJ6vgpMifWAM9QvTQK0xIfPPkDSDGr0=s96-c)

Sudhanshu Bansal

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cma8jdzjz010lad071afxkn9y)

LRU Cache with TTL Implementation looks wrong. If cache is full instead of removing least recently used shouldn't we first check if any key is expired and if yes we should remove it.

Show more

1

Reply

I

InquisitiveMoccasinBeetle684

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmac04ilx002lad078r8tqzha)

You say consistent hashing allows for us to skip a trip to some centralized server... so this means the client executes the actual hash function?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmac18nwl003gad08ad1brq4o)

Yep! Very common pattern used by (e.g.) Redis.

Show more

0

Reply

![Rahul Garg](https://lh3.googleusercontent.com/a/ACg8ocLg1LAwUvjgYc05Syhag0OzUatbBN9Bcxatf3DTThrFQU77k7I=s96-c)

Rahul Garg

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmaei8z7y006pad08ug4uweah)

Hi, can we introduce some kind of short notes for every writeup, or some Anki flashcards. help to collect the thoughts at the end and also read while travelling.

Show more

2

Reply

S

SmallCoffeeCamel827

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cman46czi01f8ad083qljl83f)

I noticed you used 32GB for RAM here but in your https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know section you mentioned caching hold up to 1TB and said "Gone are the days of 32-64GB Redis instances." Which number do you prefer to use for interviews?

Show more

0

Reply

![abhinay natti](https://lh3.googleusercontent.com/a/ACg8ocJsCbVr1a1xmAmO4lGu17xxmA5Q-6JXuFdEKGSo1dBpjO03sg=s96-c)

abhinay natti

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmaslgyzb00afad08ufopani9)

The 32GB ram is for a single node. Here we are designing a distributed cache by combining 40-50 nodes which make up 1TB. The whole system together holds 1TB, the user of the cache if he considers everything inside as a blackbox, sees 1TB cache.

Show more

0

Reply

S

SmallCoffeeCamel827

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmasm93n900hmad081toajjqo)

In the "numbers to know" section it mentions one node can store up to 1TB. So question is: why not use that here?

Show more

0

Reply

I

IncreasedGoldGayal497

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmanukkfy023jad073j2qc2zi)

Thanks for this article, it's awesome! Just had a follow-up question; how would the replication be implemented? And is that something that would be asked to expand on in the interview?

For example, I am thinking you would need some endpoint on your follower nodes that recieves updates from the leader. I.e. if the leader accepts a write, it can immediately respond to the client, and then maybe send a post request to all of the followers? Or keep an open stream to all of the followers and continuously send data with each write?

Show more

1

Reply

V

VeryEmeraldBird248

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmasre5sc00n4ad07ceqa1car)

This article lacks too many details and is very hypothetical. I don't have too much depth into Cache but would not pass anyone who presents these 10k feet level details about a system.

1.  It explains sharding to distribute load but doesn't explain how multiple read instances for each shard will work together in case of failure and writes.
2.  Hot key: doesn't explain how writes will work and which component will do what.
3.  Doesn't explain how all nodes will have consistent values for keys when key is suffixed to distribute across nodes.
4.  Doesn't explain how write batching will work and which layer or component it'll exist on.
5.  Hot key writes: you are distributing writes but lacking details in how those writes will be consolidated across multiple nodes.

I hope a paid version of this article is re-written as I'd like to pay and learn depth rather than jargon.

Show more

1

Reply

M

ManagerialCyanSawfish581

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbfkasm900bkad08yufclvyn)

I had been asked in an interview to do system design for a single node memcached as I do not have much expertise with distributed systems but I had this article in my mind as I had practiced this article so many times, I answered exactly what had been given here in this article hoping it suffices?? Thank you

Show more

0

Reply

J

jyotitriyar

[• 3 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmatkmy0t00wmad086fu0ixbj)

Curious how do host determine and maintain if the key is hot or not and append the suffix? Also when does it know it is no longer hot?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbflbjqa0022ad08k2f8v5r6)

We track hot keys using a combination of monitoring and thresholds. Each node maintains a counter for key access frequency (usually in a sliding window, like last 5 minutes). When a key's access rate exceeds a configured threshold (e.g. >1000 requests/sec), we mark it as "hot" and start creating suffixed copies. To determine it's no longer hot, we use a lower threshold (to prevent thrashing). if requests drop below this for a sustained period (e.g. <100 requests/sec for 5+ minutes), we remove the copies and revert to single-key mode. The actual thresholds depend on your workload and hardware capabilities.

Typically this is out of scope for an interview. But cool to know

Show more

1

Reply

H

HelpfulIvoryBandicoot743

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmb1viegv000vad08pkfyyt94)

Redis is generally considered a CP system(whereas the article is to design a AP system). Maybe we should change the title to something else?

Show more

1

Reply

M

ManagerialCyanSawfish581

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbfgyem5003fad08f4v7vgs4)

I had been asked in an interview to do system design for a single node memcached as I do not have much expertise with distributed systems but I had this article in my mind as I had practiced this article so many times, I answered exactly what you had given here in this article hoping it suffices?? Thank you

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbfl9r5x001uad08nq6ky9ug)

I mean, this is a distributed cache. Not a single node.

Show more

0

Reply

G

GrandPlumSilkworm276

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbpynskx00cx08adnj2v1ln5)

Since we're using a LRU eviction policy here for the LLD. How would this differ if the interviewer asks us to use LFU? Only the LLD pseudo-code changes right?

Show more

0

Reply

![indavarapu aneesh](https://lh3.googleusercontent.com/a/ACg8ocLx77-thRGA5bldZDZhNF8MbwtxB4dZmFZ3zHzbk_Xu4IB-og=s96-c)

indavarapu aneesh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmbtnsuc704cr08advl1i934t)

One minor improvement for expiring the keys

you can start a different thread which is consuming entries of a priority key where keys with ttl are added. So consider this workflow

1.  for a PUT request with TTL, insert the key into the priority queue along with the hash table.
2.  the background thread would always be peeling of the top of the priority queue where the current time is greater than the top of the queue. This way we wouldn't have to scan the hash table and take read locks.

Show more

0

Reply

![Jack Copland](https://lh3.googleusercontent.com/a/ACg8ocIuc_0acp8OBG__bZ3WuVYaqssUDrx7kEywyatLki56KL2Nhw=s96-c)

Jack Copland

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmc0dnmvf05te08ad6tf2opep)

This problem to me is way more interesting in a low level interview, or a distributed systems interview which focuses on the implementation details beyond linking components.

For the former, talking about contention, concurency and replacement algorithms/lru approximations is revealing, and I think the same for discussing tradeoffs between leader based consensus vs quorum vs reconfiguration.

If that's removed...

Show more

0

Reply

X

XerothermicAzureCat469

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmc40bqea0154ad08jz7a8h76)

The node calculation that rounded off to 50 should consider replication as well. Replication factor will increase that number. If replication is 2, we are looking at 100 nodes.

Show more

0

Reply

![Bhashkarjya Nath](https://lh3.googleusercontent.com/a/ACg8ocIPuNE7wW4Pec9OVRhCB90zYo8sSDdoIecT3rAXfErExHqPUA=s96-c)

Bhashkarjya Nath

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmc9qz9pv01psad08l42hi2jf)

In the distributed system we have sharded the keys, so each hash-table and the doubly linked list in one instance would only store a portion of the keys. If we reach the cache limit and we have to evict any one key, we would have to collect the data from all the hash tables and doubly linked lists stored in each of the instances. So, for that we would need to make multiple network hops to each of the sharded nodes. I believe this would increase the write/read latency. I believe we can solve this issue in two ways -

1.  We could make sure that eviction happens in the background asynchronously.
2.  One more solution could be to share the cache size equally among all the nodes and if any node reaches the defined limit, it would evict a key from it's own hash-table and double linked list only.

Please provide some thoughts on which approach would be better.

Show more

0

Reply

![Phouthasak Douanglee](https://lh3.googleusercontent.com/a/ACg8ocJ70qFarzzd3tX2UL3grQL-dYPHB6a3KrCWJRP7619z65AsoQ=s96-c)

Phouthasak Douanglee

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmcbn3tny06jcad080w50jw8o)

At the end, how come there is no cleanup process tied back into the final design? Does each node have their own respective janitor services (each shared node and each read replica node separately)?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmcu1wtcz00o1ad08xbzu925u)

Each node runs its own janitor process to clean up expired entries in its local cache. This is more efficient than having a centralized cleanup service since each node can independently manage its own memory. The read replicas also run their own janitor processes, but they don't need to coordinate with the primary , if they expire something early or late it's not a big deal since we're eventually consistent anyway.

Show more

1

Reply

E

ElectricalYellowAnaconda994

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmcu1st9g00n7ad07fsrc7r1j)

Hi, the supporing images are from different write up, i seems, can you help to rectify the same?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmcu1vlbv00n2ad083o45g015)

which?

Show more

0

Reply

E

ElectricalYellowAnaconda994

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmcu347wb012zad07dfy45f85)

Hey Evan, All of the diagrams seem to reflect this issue — they resemble charts from camelcamelcamel based on their content. Interestingly, when I zoom into the images, the diagrams display correctly, although sometimes they appear cut off. However, when I reduce the browser zoom level to around 80%, the diagrams become fully visible and properly formatted.

Just wanted to flag this in case it's helpful. I am not sure if others are seeing this problem.

PS: Refreshing the page again this time has solved the problem, not sure if you have made some amends in backend or it has automatically happened

Show more

0

Reply

D

DistinctAmberHarrier825

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmcucqt5j02y2ad08wrc3l1xu)

Thank you for the very informative write up. in interviews can there be questions as to how 1TB data can be accommodated in-memory. also should we discuss about data persistence. TIA.

Show more

0

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmd7ym5il0056ad08xqhvbpq0)

I think this solution hand-waived a lot of implementation details, like:

1.  How do clients know the ring hashing function
2.  How does the system detect the hot keys and split them across replicas?
3.  How do you tell the clients that this specific key is split across multiple shards and they need to read all of them to calculate the number of views?

In an interview, I would expect the candidate to explain them and include them in the system diagram.

I think that Zookeeper's notification feature solves (1) & (3) well, but I'm not sure if we're allowed to use it here.

Show more

1

Reply

![Jose](https://lh3.googleusercontent.com/a/ACg8ocI0U4FzBIZeE_jngEBfFE3NF4Tj7WyqSOZo_DC7kBBEDA=s96-c)

Jose

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmdefvdj801nyad08mlv77tx3)

Agree. Things I'd have liked to see:

1.  Solve using a client wrapper of the cache maybe. It's not only about knowing the shard, but what about Membership ? Either Zookeeper here as you mentioned aftewards, or via Gossip.
2.  My hunch is we could say hot are those keys that are at the tail of the doubly linked list.
3.  +1, maybe each node could have a background worker putting those hot keys on a sorted set which gets periodically propagated to the client wrapper, then it becomes transparent to the user the need to fan-out.

Show more

0

Reply

J

jyotitriyar

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmd9if7rp01abad08mx0aar2g)

Few questions.

1.  How do you determine if a key is getting hot? Any system that will determine a key is getting hot will itself start getting hot as you will need to maintain some counter?
2.  how do we arrive and maintain optimal cache hit rate? For e.g. what is best TTL to use that can reduce load on backend but also not lead to data staleness?

Show more

1

Reply

![Niranjhan Kantharaj](https://lh3.googleusercontent.com/a/ACg8ocISPQAOL90hPbJ5ilQZyBLjxhaLIFqJu_r3HTDm2rz8q-Wq=s96-c)

Niranjhan Kantharaj

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmdcvzj7t03f4ad08808tfbrf)

how do we maintain Doubly linked list (DLL) to get LRU as we shard to multiple instance?

With DLL in each instance, we cannot really say which is least recently used. This seems to be a gap in the overall solution

Show more

0

Reply

![Jose](https://lh3.googleusercontent.com/a/ACg8ocI0U4FzBIZeE_jngEBfFE3NF4Tj7WyqSOZo_DC7kBBEDA=s96-c)

Jose

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmdefhygr01lvad08ed45n8kf)

LRU is locally handled at the node level, there is no global LRU.

Show more

0

Reply

![Niranjhan Kantharaj](https://lh3.googleusercontent.com/a/ACg8ocISPQAOL90hPbJ5ilQZyBLjxhaLIFqJu_r3HTDm2rz8q-Wq=s96-c)

Niranjhan Kantharaj

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmdf15sv107nsad080x86zqyr)

In that case, LRU might not evict the least recently used key of the entire cache right ? Isnt that a bug ?

Show more

0

Reply

![Jose](https://lh3.googleusercontent.com/a/ACg8ocI0U4FzBIZeE_jngEBfFE3NF4Tj7WyqSOZo_DC7kBBEDA=s96-c)

Jose

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmdfmrr6301vsad08xhsxdyfd)

Assuming load is evenly spread out, on average you are evicting the least used keys on the whole cache indeed. Not a bug, if global strictly ordered LRU is a requirement then you might need to build additional complexity for it. I think it's just an overkill.

Show more

0

Reply

![Kunal](https://lh3.googleusercontent.com/a/ACg8ocLfQd6Z2fUmnSVYXdCYf1mflLoPdnn-xeLSj6cOn5CT5ch_fw=s96-c)

Kunal

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmdfv9ls10338ad076ms6a30v)

can we get design video as well for premium folks

Show more

1

Reply

N

nsghumman

[• 23 days ago• edited 23 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmds950bh05l3ad08a8tkldff)

I'm wondering about the LRU constraint after we shard the cache. Seems like each instance can still be implementing LRU locally, but not globally across the key space.

Show more

0

Reply

R

red-D

[• 22 days ago• edited 22 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmdtgz0nj012pad083gelleh1)

This write-up seems to be trivial and easy to understand. The fact that you wont ask Staff Eng this ques should categorize it to be medium !

Show more

0

Reply

RM

Reem Madkour

[• 17 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cme181gjs0796ad08o1nvifcp)

Is there a video?

Show more

0

Reply

I

IncreasingYellowVulture891

[• 18 hours ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmeomt8no05uwad08pjrzvvsk)

+1

Show more

0

Reply

![Prerana K](https://lh3.googleusercontent.com/a/ACg8ocJshNL6ReiMywhZOBnFdecLMYxpWu9zg4kqNK_tC46kIZVFoYLK0g=s96-c)

Prerana K

[• 4 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmeiow9uz01eead07r467t4ee)

What if Cache gets restarted, we can also talk abut dumping data periodically into a S3 blob for persistence.

Show more

0

Reply

![Kaushal Shah](https://lh3.googleusercontent.com/a/ACg8ocId-63hz5DmLeFnhcm-rnp2UeXraTh7tc0QtVF0dFcVm27i3A=s96-c)

Kaushal Shah

[• 4 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmejgl82z02kwad084f7jlwei)

But how does consistent hashing work in real life? Where is it hosted? How are the nodes assigned to consistent hashing?

I feel like all this information should be a part of this explanation. I've interviewed for 4 months now and none of the interviewer are happy with sprinkling buzzwords here and there. They need actual data or understanding of the tool/tech/algorithm.

Show more

0

Reply

A

AncientTealEarthworm578

[• 1 day ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmeo9bsfv01ldad09htgeoage)

Evan, amazing write up! I just have one quick question on this - Considering the client runs within some application embedded, how does it keep itself updated with current state of CH ring? It has to know the latest state and key distribution, to be able to send the request to correct node in the ring. What are some good ways to achieve this?

Show more

0

Reply

I

IncreasingYellowVulture891

[• 18 hours ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-cache#comment-cmeomssvb05unad08br06z6mu)

would be great if we have a video for this!

Show more

0

Reply
