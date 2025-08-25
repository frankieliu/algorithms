# Design Online Auction

Dealing with Contention

Real-time Updates

[![Evan King](/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75&dpl=2c00fec7d898fc5204eb9b0f9b4dae7f4e34d61b)

Evan King

Ex-Meta Staff Engineer

](https://www.linkedin.com/in/evan-king-40072280/)

medium

Published Jul 10, 2024

---

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

0:00

Play

Mute

0%

0:00

/

48:52

Premium Content

Closed-Captions On

Chapters

Settings

AirPlay

Google Cast

Enter PiP

Enter Fullscreen

## Understanding the Problem

**🛍️ What is an online auction?** An online auction service lets users list items for sale while others compete to purchase them by placing increasingly higher bids until the auction ends, with the highest bidder winning the item.

As is the case with all of our common question breakdowns, we'll walk through this problem step by step, using the [Hello Interview System Design Framework](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery) as our guide. Note that I go into more detail here than would be required or possible in an interview, but I think the added detail is helpful for teaching concepts and deepening understanding.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements)

**Core Requirements**

1. Users should be able to post an item for auction with a starting price and end date.
    
2. Users should be able to bid on an item. Where bids are accepted if they are higher than the current highest bid.
    
3. Users should be able to view an auction, including the current highest bid.
    

**Below the line (out of scope):**

- Users should be able to search for items.
    
- Users should be able to filter items by category.
    
- Users should be able to sort items by price.
    
- Users should be able to view the auction history of an item.
    

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements)

Before diving into the non-functional requirements, ask your interviewer about the expected scale of the system. Understanding the scale requirements early will help inform key architectural decisions throughout your design.

**Core Requirements**

1. The system should maintain strong consistency for bids to ensure all users see the same highest bid.
    
2. The system should be fault tolerant and durable. We can't drop any bids.
    
3. The system should display the current highest bid in real-time so users know what they are bidding against.
    
4. The system should scale to support 10M concurrent auctions.
    

**Below the line (out of scope):**

- The system should have proper observability and monitoring.
    
- The system should be secure and protect user data.
    
- The system should be well tested and easy to deploy (CI/CD pipelines)
    

On the whiteboard, this could be short hand like this:

Requirements

## The Set Up

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

Let's start by identifying the core entities of the system. We'll keep the details light for now and focus on specific fields later. Having these key entities defined will help structure our thinking as we design the API.

To satisfy our key functional requirements, we'll need the following entities:

1. **Auction**: This represents an auction for an item. It would include information like the starting price, end date, and the item being auctioned.
    
2. **Item**: This represents an item being auctioned. It would include information like the name, description, and image.
    
3. **Bid**: This represents a bid on an auction. It would include information like the amount bid, the user placing the bid, and the auction being bid on.
    
4. **User**: This represents a user of the system who either starts an auction or bids on an auction.
    

While we could embed the Item details directly on the Auction entity, normalizing them into separate entities has some advantages:

1. Items can be reused across multiple auctions (e.g. if a seller wants to relist an unsold item)
    
2. Item details can be updated independently of auction details
    
3. We can more easily add item-specific features like categories or search
    

In an interview, I'm fine with either option - the key is to explain your reasoning.

Here is how it could look on the whiteboard:

Core Entities

### [API or System Interface](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#api-or-system-interface-5-minutes)

Let's define our API before getting into the high-level design, as it establishes the contract between our client and system. We'll go through each functional requirement and define the necessary APIs.

For creating auctions, we need a POST endpoint that takes the auction details and returns the created auction:

`POST /auctions -> Auction & Item {     item: Item,     startDate: Date,     endDate: Date,     startingPrice: number,     }`

For placing bids, we need a POST endpoint that takes the bid details and returns the created bid:

`POST /auctions/:auctionId/bids -> Bid {     Bid }`

For viewing auctions, we need a GET endpoint that takes an auctionId and returns the auction and item details:

`GET /auctions/:auctionId -> Auction & Item`

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes)

### 1) Users should be able to post an item for auction with a starting price and end date.

First things first, users need a way to start a new auction. They'll do this by POSTing to the /auctions endpoint with the auction details, including information about the item they are selling.

We start by laying out the core components for communicating between the client and our microservices. We add our first service, "Auction Service," which connects to a database that stores the auction and item data outlined in the Core Entities above. This service will handle the reading/viewing of auctions.

Create an Auction

1. **Client** Users will interact with the system through the clients website or app. All client requests will be routed to the system's backend through an API Gateway.
    
2. **API Gateway** The API Gateway handles requests from the client, including authentication, rate limiting, and, most importantly, routing to the appropriate service.
    
3. **Auction Service** The Auction Service, at this point, is just a thin wrapper around the database. It takes the auction details from the request, validates them, and stores them in the database.
    
4. **Database** The Database stores tables for auctions and items.
    

This part of the design is just a basic CRUD application. But let's still be explicit about the data-flow, walking through exactly what happens when a user posts a new auction.

1. **Client** sends a POST request to /auctions with the auction details.
    
2. **API Gateway** routes the request to the Auction Service.
    
3. **Auction Service** validates the request and stores the auction and item data in the database.
    
4. **Database** stores the auction and item data in tables.
    

Note that I don't mention anything about images here. This was intentional; it's not the most interesting part of the problem, so I don't want to waste time on it. In reality, we would store images in blob storage and reference them by URL in our items table. You can call this out in your interview to align on this not being the focus.

### 2) Users should be able to bid on an item. Where bids are accepted if they are higher than the current highest bid.

Bidding is the most interesting part of this problem, and where we will spend the most time in the interview. We'll start high-level and then dig into details on scale and consistency in our deep dives.

To handle the bidding functionality, we'll introduce a dedicated "Bidding Service" separate from our Auction Service. This new service will:

1. Validate incoming bids (e.g. check bid amount is higher than current max)
    
2. Update the auction with new highest bid
    
3. Store bid history in the database
    
4. Notify relevant parties of bid updates
    

We choose to separate bidding into its own service for several key reasons:

1. **Independent Scaling**: Bidding traffic is typically much higher volume than auction creation - we expect ~100x more bids than auctions. Having a separate service allows us to scale the bidding infrastructure independently.
    
2. **Isolation of Concerns**: Bidding has its own complex business logic around validation, race conditions, and real-time updates. Keeping it separate helps maintain clean service boundaries.
    
3. **Performance Optimization**: We can optimize the bidding service specifically for high-throughput write operations, while the auction service can be optimized for read-heavy workloads.
    

Place Bids

In the simple case, when a bid is placed, the following happens:

1. **Client** sends a POST request to /auctions/:auctionId/bids with the bid details.
    
2. **API Gateway** routes the request to the Bidding Service.
    
3. **Bidding Service** queries the database for the highest current bid on this auction. It stores the bid in the database with a status of either "accepted" (if the bid is higher than current highest) or "rejected" (if not). The service then returns the bid status to the client.
    
4. **Database** bids are stored in a new bids table, linked to the auction by auctionId.
    

When I ask this question, many candidates suggest storing just a maxBidPrice field on the auction object instead of maintaining a bids table. While simpler, this violates a core principle of data integrity: avoid destroying historical data.

Overwriting the maximum bid with each update means permanently losing critical information. This makes it impossible to audit the bidding process, investigate disputes, or analyze patterns of behavior. Inevitably, a user is going to complain that their bid was not recorded and that they should have won the auction. Without a complete audit trail, you have no way to prove them wrong.

### 3) Users should be able to view an auction, including the current highest bid.

Users need to be able to view an auction for two reasons:

1. They want to learn about the item in case they decide they're interested in buying it.
    
2. They want to place a bid, so they need to know the current highest bid that they need to beat.
    

These two are similar, but have different requirements. The first is a read-only operation. The second requires real-time consistency.

We'll offload the depth of discussion here to our deep dives, but let's outline the basic approach which ensures that the current highest bid is at least reasonably up-to-date.

View Auction

When a user first goes to view an auction, they'll make a GET request to /auctions/:auctionId which will return the relevant auction and item details to be displayed on the page. Great.

What happens next is more interesting. If we never refresh the maximum bid price, then the user will bid based on a stale amount and be confused (and frustrated) when they are told their bid was not accepted. Especially in an auction with a lot of activity, this is a problem. To solve this, we can simply poll for the latest maximum bid price every few seconds. While imperfect, this ensures at least some degree of consistency and reduces the likelihood of a user being told their bid was not accepted when it actually was.

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes)

With the high-level design in place, it's time to go deep. How much you lead the conversation here is a function of your seniority. We'll go into a handful of the deep dives I like to cover when I ask this question but keep in mind, this is not exhaustive.

### 1) How can we ensure strong consistency for bids?

Ensuring the consistency of bids is the most critical aspect of designing our online auction system. Let's look at an example that shows why proper synchronization is essential.

###### Pattern: Dealing with Contention

Online auctions are a classic example of the dealing with contention pattern. When multiple users bid on the same auction simultaneously, we face race conditions and need to ensure only one bid wins. This requires careful coordination using techniques like optimistic concurrency control, row locking, or caching strategies to manage competition for the same resource.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention)

Example:

- The current highest bid is **$10**.
    
- **User A** decides to place a bid of **$100**.
    
- **User B** decides to place a bid of **$20** shortly after.
    

Without proper concurrency control, the sequence of operations might unfold as follows:

1. **User A's Read**: User A reads the current highest bid, which is $10.
    
2. **User A's Write**: User A writes their bid of $100 to the database. The system accepts this bid because $100 is higher than $10.
    
3. **User B's Read**: User B reads the current highest bid. Due to a delay in data propagation or read consistency issues, they still see the highest bid as $10 instead of $100.
    
4. **User B's Write**: User B writes their bid of $20 to the database. The system accepts this bid because $20 is higher than $10 (the stale value they read earlier).
    

As a result, **both bids are accepted**, and the auction now has two users who think they have the highest bid.

- A bid of $100 from User A (the actual highest bid).
    
- A bid of $20 from User B (which should have been rejected).
    

This inconsistency occurs because User B's bid of $20 was accepted based on outdated information. Ideally, User B's bid should have been compared against the updated highest bid of $100 and subsequently rejected.

There is an answer to this question which asserts that strong consistency is actually not necessary. I see this argument periodically from staff candidates. Their argument is that it doesn't matter if we accept both bids. We just need to rectify the client side by later telling User B that a bid came in higher than theirs and they're no longer winning. The reality is, whether User A's or User B's bid came in first is not important (unless they were for the same amount). The end result is the same; User A should win the auction.

This argument is valid and requires careful consideration of client-side rendering and a process that waits for eventual consistency to settle before notifying any users of the ultimate outcome.

While this is an interesting discussion, it largely dodges the complexity of the problem, so most interviewers will still ask that you solve for strong consistency.

Show More

Great, we understand the problem, but how do we solve it?

### 

Bad Solution: Row Locking with Bids Query

##### Approach

One initial approach might be to use **row-level locking** with a query to get the current maximum bid. When a user tries to place a bid, we need to ensure that no other bids are being processed for the same auction simultaneously to avoid race conditions. Here's how we could implement this:

1. **Begin a transaction**: Start a database transaction to maintain atomicity.
    
2. **Lock all bid rows for the auction using SELECT ... FOR UPDATE**: This locks all existing bids for the auction, preventing other transactions from modifying them until the current transaction is complete.
    
3. **Query the current maximum bid from the bids table**: Retrieve the highest bid currently placed on the auction.
    
4. **Compare the new bid against it**: Check if the incoming bid amount is higher than the current maximum bid.
    
5. **Write the new bid if accepted**: Insert the new bid into the bids table.
    
6. **Commit the transaction**: Finalize the transaction to persist changes.
    

Here's what the SQL query could look like:

`BEGIN; WITH current_max AS (     SELECT MAX(amount) AS max_bid    FROM bids    WHERE auction_id = :auction_id    FOR UPDATE ) INSERT INTO bids (auction_id, user_id, amount, bid_time) SELECT :auction_id, :user_id, :amount, NOW() FROM current_max WHERE :amount > COALESCE(max_bid, 0) RETURNING id; COMMIT;`

##### Challenges

While this approach ensures that no two bids are processed concurrently for the same auction, it has significant drawbacks:

1. **Performance and Scalability Issues**: Locking all bid rows for an auction serializes bid processing, creating a major bottleneck. As the number of concurrent auctions and bids increases, the contention for locks intensifies, leading to increased latency and poor user experience. The database may even escalate to table-level locks under heavy load, blocking all operations on the bids table.
    
2. **Poor User Experience**: The delays introduced by lock contention and serialized processing result in slow response times or timeouts when placing bids. This is unacceptable in a competitive bidding environment where users expect real-time responsiveness and consistent performance.
    

In general, this is a big no-no. You want to use row locking sparingly and with optimizations to ensure that a) you lock as few rows as possible and b) you lock rows for as short of a duration as possible. We don't respect either of these principles here.

### 

Good Solution: Cache max bid externally

##### Approach

The next thing most candidates think of is to cache the max bid in memory. They realize that the main issue with the above approach is that we are holding the lock on a large number of rows for a reasonable duration, since we need to query the bid table to calculate the max bid each time.

To avoid this, we can cache the max bid in memory in something like Redis. Now, the data flow when a bid comes in looks very different:

1. **Read cache**: Read the max bid for the given auction from Redis.
    
2. **Update cache**: If the new bid is higher, update the cache with the new max bid.
    
3. **Write bid**: Write the bid to the database with a status of accepted or rejected.
    

Cache max bid

##### Challenges

We solved one problem, but created another. Now we've moved this from a consistency problem within the database to a consistency problem between the database and the cache. We need a couple things to be true:

1. We need the cache read and cache write to happen atomically so we don't have a race condition like in our initial scenario. Fortunately, this is easy enough as Redis is single threaded and supports atomic operations. We can use either Redis transactions or Lua scripting to ensure our read-modify-write sequence happens as one operation:
    
    `-- Lua script to atomically compare and set max bid local current_max_bid = tonumber(redis.call('GET', auction_key) or '0') local proposed_bid_amount = tonumber(proposed_bid) if proposed_bid_amount > current_max_bid then     redis.call('SET', auction_key, proposed_bid_amount)     return true else     return false end`
    
2. We need to make sure that our cache is strongly consistent with the database. Otherwise we could find ourselves in a place where the cache says the max bid is one thing but that bid is not in our database (because of failure or any other issue). To solve this, we have a few options:
    
    - Use distributed transactions (two-phase commit) to ensure atomicity across Redis and the database. This adds significant complexity and performance overhead.
        
    - Accept Redis as the source of truth during the auction and write to the database asynchronously. This trades consistency for performance.
        
    - Use optimistic concurrency with retry logic: update the cache atomically first, then write to the database. If the database write fails, roll back the cache update.
        
    

Most importantly, if you find yourself in an interview where a distributed transaction is needed, consider whether you can restructure your system to avoid it.

### 

Great Solution: Cache max bid in database

##### Approach

We solved the issue with locking a lot of rows for a long time by moving the max bid to a cache, but that introduced a new issue with consistency between the cache and the database. We can solve both problems by storing the max bid in the database. Effectively using the Auction table as our cache.

Now, our flow looks like this:

1. Lock the auction row for the given auction (just one row)
    
2. Read the max bid for the given auction from the Auction table.
    
3. Write the bid to the database with a status of accepted or rejected.
    
4. Update the max bid in the Auction table if the new bid is higher.
    
5. Commit the transaction.
    

We now only lock a single row and for a short duration. If we want to avoid pessimistic locking altogether, we can use [optimistic concurrency control (OCC)](https://en.wikipedia.org/wiki/Optimistic_concurrency_control).

OCC is ideal for our system because bid conflicts are relatively rare (most bids won't happen simultaneously on the same auction). Here's how it works:

1. Read the auction row and get the current max bid (this is what is referred to as the 'version' with OCC)
    
2. Validate that the new bid is higher than the max bid
    
3. Try to update the auction row, but only if the max bid hasn't changed:
    
    `UPDATE auctions  SET max_bid = :new_bid  WHERE id = :auction_id AND max_bid = :original_max_bid`
    
4. If the update succeeds, write the bid record. If it fails, retry from step 1.
    

This approach avoids locks entirely while still maintaining consistency, at the cost of occasional retries when conflicts do occur.

Cache max bid in DB

### 2) How can we ensure that the system is fault tolerant and durable?

Dropping a bid is a non-starter. Imagine telling a user that they lost an auction because their winning bid was "lost" - that would be catastrophic for trust in the platform. We need to guarantee durability and ensure that all bids are recorded and processed, even in the face of system failures.

The best approach here is to introduce a durable message queue and get bids into it as soon as possible. This offers several benefits:

1. **Durable Storage**: When a bid comes in, we immediately write it to the message queue. Even if the entire Bid Service crashes, the bid is safely stored and won't be lost. Think of it like writing your name on a waiting list at a restaurant - even if the host takes a break, your place in line is secured.
    
2. **Buffering Against Load Spikes**: Imagine a popular auction entering its final minutes. We might suddenly get thousands of bids per second - far more than our Bid Service can handle. Without a message queue, we'd have to either:
    
    - Drop bids (unacceptable)
        
    - Crash under the load (also unacceptable)
        
    - Massively over-provision our servers (expensive)
        
    
    With a message queue, these surge periods become manageable. The queue acts like a buffer, storing bids temporarily until the Bid Service can process them. It's like having an infinitely large waiting room for your bids.
    
3. **Guaranteed Ordering**: Message queues (particularly Kafka) can guarantee that messages are processed in the order they were received. This is important for fairness - if two users bid the same amount, we want to ensure the first bid wins. The queue gives us this ordering for free.
    

Fault tolerant system

For implementation, we'll use Kafka as our message queue. While other solutions like RabbitMQ or AWS SQS would work, Kafka is well-suited for our needs because:

1. **High Throughput**: Kafka can handle millions of messages per second, perfect for high-volume bidding periods.
    
2. **Durability**: Kafka persists all messages to disk and can be configured for replication, ensuring we never lose a bid.
    
3. **Partitioning**: We can partition our queue by auctionId, ensuring that all bids for the same auction are processed in order while allowing parallel processing of bids for different auctions.
    

Here's how the flow works:

1. User submits a bid through our API
    
2. API Gateway routes to a producer which immediately writes the bid to Kafka
    
3. Kafka acknowledges the write, and we can tell the user their bid was received
    
4. The Bid Service consumes the message from Kafka at its own pace
    
5. If the bid is valid, it's written to the database
    
6. If the Bid Service fails at any point, the bid remains in Kafka and can be retried
    

While message queues do add some latency (typically 2-10ms under normal conditions), this tradeoff for durability is usually worth it. There are also multiple patterns for maintaining responsive user experiences while using queues, though each comes with different consistency/latency tradeoffs. For fear of going too deep here, we'll leave it at that. But I will discuss bid broadcasting later on when we discuss scale, which solves the asynchronous notification problem.

### 3) How can we ensure that the system displays the current highest bid in real-time?

Going back to the functional requirement of 'Users should be able to view an auction, including the current highest bid,' or current solution using polling, which has a few key issues:

1. **Too slow**: We're updating the highest bid every few seconds, but for a hot auction, this may not be fast enough.
    
2. **Inefficient**: Every client is hitting the database on every request, and in the overwhelming majority of cases, the max bid hasn't changed. This is wasteful.
    

###### Pattern: Real-time Updates

Keeping clients updated in real-time about the current highest bid is a perfect example of how to apply the **real-time updates pattern**.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates)

We now need to expand the solution to satisfy the non-functional requirement of 'The system displays the current highest bid in real-time'.

Here is how we can do it:

### 

Good Solution: Long polling for max bid

##### Approach

Long polling offers a significant improvement over regular polling by maintaining an open connection to the server until new data is available or a timeout occurs. When a client wants to monitor an auction's current bid, they establish a connection that remains open until either the maximum bid changes or a timeout is reached (typically 30-60 seconds).

The server holds the request open instead of responding immediately. When a new bid is accepted, the server responds to all waiting requests with the updated maximum bid. The clients then immediately initiate a new long-polling request, maintaining a near-continuous connection.

The client side implementation typically looks something like this:

``async function pollMaxBid(auctionId) {   try {     const response = await fetch(`/api/auctions/${auctionId}/max-bid`, {       timeout: 30000 // 30 second timeout     });          if (response.ok) {       const { maxBid } = await response.json();       updateUI(maxBid);     }   } catch (error) {     // Handle timeout or error   }      // Immediately start the next long poll   pollMaxBid(auctionId); }``

On the server side, we maintain a map of pending requests for each auction. When a new bid is accepted, we respond to all waiting requests for that auction with the new maximum bid.

##### Challenges

While long polling is better than regular polling, it still has some limitations. The server must maintain open connections for all clients watching an auction, which can consume significant resources when dealing with popular auctions. Additionally, if many clients are watching the same auction, we might face a "thundering herd" problem where all clients attempt to reconnect simultaneously after receiving an update.

The timeout mechanism also introduces a small delay - if a bid comes in just after a client's previous long poll times out, they won't see the update until their next request completes. This creates a tradeoff between resource usage (shorter timeouts mean more requests) and latency (longer timeouts mean potentially delayed updates).

Lastly, scaling becomes a challenge since each server needs to maintain its own set of open connections. If we want to scale horizontally by adding more servers, we need additional infrastructure like Redis or a message queue to coordinate bid updates across all servers. This adds complexity and potential points of failure to the system. More on this when we discuss scaling.

### 

Great Solution: Server-Sent Events (SSE)

##### Approach

[Server-Sent Events (SSE)](https://en.wikipedia.org/wiki/Server-sent_events) provides a more elegant solution for real-time bid updates. SSE establishes a unidirectional channel from server to client, allowing the server to push updates whenever they occur without requiring the client to poll or maintain multiple connections.

When a user views an auction, their browser establishes a single SSE connection. The server can then push new maximum bid values through this connection whenever they change. This creates a true real-time experience while being more efficient than both regular polling and long polling.

The client-side implementation is remarkably simple:

``const eventSource = new EventSource(`/api/auctions/${auctionId}/bid-stream`); eventSource.onmessage = (event) => {   const { maxBid } = JSON.parse(event.data);   updateUI(maxBid); };``

On the server side, we maintain a set of active SSE connections for each auction. When a new bid is accepted, we push the updated maximum bid to all connected clients for that auction. The server implementation might look something like this:

``class AuctionEventManager {   private connections: Map<string, Set<Response>> = new Map();      addConnection(auctionId: string, connection: Response) {     if (!this.connections.has(auctionId)) {       this.connections.set(auctionId, new Set());     }     this.connections.get(auctionId).add(connection);   }      broadcastNewBid(auctionId: string, maxBid: number) {     const connections = this.connections.get(auctionId);     if (connections) {       const event = `data: ${JSON.stringify({ maxBid })}\n\n`;       connections.forEach(response => response.write(event));     }   } }``

You could use websockets here as well, but SSE is arguably a better fit given the communication is unidirectional and SSE is typically lighter weight and easier to implement.

SSE

##### Challenges

The main challenge is scaling the real-time updates across multiple servers. As our user base grows, we'll need multiple servers to handle all the SSE connections. However, this creates a coordination problem: when a new bid comes in to Server A, there may be users watching that same auction who are connected to Server B. Without additional infrastructure, Server A has no way to notify those users about the new bid, since it only knows about its own connections.

For example, imagine User 1 and User 2 are both watching Auction X, but User 1 is connected to Server A while User 2 is connected to Server B. If User 3 places a bid that goes to Server A, only User 1 would receive the update - User 2 would be left with stale data since Server B doesn't know about the new bid.

We'll talk about the solution this this problem next as we get into scaling.

### 4) How can we ensure that the system scales to support 10M concurrent auctions?

When it comes to discussing scale, you'll typically want to follow a similar process for every system design question. Working left to right, evaluate each component in your design asking the following questions:

1. What are the resource requirements at peak? Consider storage, compute, and bandwidth.
    
2. Does the current design satisfy the requirement at peak?
    
3. If not, how can we scale the component to meet the new requirement?
    

We can start with some basic throughput assumptions. We have 10M concurrent auctions, and each auction has ~100 bids. That's 1B bids per day. 1B / 100,000 (rounded seconds in day) = 10K bids per second.

**Message Queue**

Let's start with our message queue for handling bids. At 10,000 requests per second, and decent hardware can handle this without issue. So no need to even partition the queue.

**Bid Service**

Next, we consider both the Bid Service (consumer of the message queue) and Auction Service. As is the case with almost all stateless services, we can horizontally scale these by adding more servers. By enabling auto-scaling, we can ensure that we're always running the right number of servers to handle the current load based on memory or CPU usage.

**Database**

Let's consider our persistence layer. Starting with storage, we can round up and say each Auction is 1kb. We'll say each Bid is 500 bytes. If the average auction runs for a week, we'd have 10M \* 52 = 520M auctions per year. That's 520M \* (1kb + (0.5kb \* 100 bids per auction)) = 25 TB of storage per year.

That is a decent amount for sure, but nothing crazy. Modern SSDs can handle 100+ TBs of storage. While we'd want to ensure the basics with regards to some hot replication, we're not going to run out of storage any time soon. We'd be wise to shard, but the more pressing constraint is with regards to write throughput.

10k writes per second is at the limit of a well-provisioned single Postgres instance. If we want to stick to our current solution for handling consistency, we'll need to shard our database, again by auction ID, so that we can handle the write load. We don't have any worries about scatter-gathers since all reads/writes for a single auction are on the same shard.

**SSE**

Lastly, we talked about how our SSE solution for broadcasting new bids to users would not scale well. To recap, the problem is that when a new bid comes in, we need to broadcast it to all the clients that are watching that auction. If we have 10M auctions and 100M users, we could have 100M connections. That's a lot of connections and they wont fit on the same server. So we need a way for Bid Service servers to coordinate with one another.

The solution is Pub/Sub. Whether using a technology like Redis Pub/Sub, Kafka, or even a custom solution, we need to broadcast the new bid to all relevant servers so they can push the update to their connected clients. The way this works is simple: when Server A receives a new bid, it publishes that bid to the pub/sub system, essentially putting it on a queue. All other instances of the Bid Service are subscribed to the same pub/sub channel and receive the message. If the bid is for an auction that one of their current client connections is subscribed to, they send that connection the new bid data.

If you want to learn more about Pub/Sub checkout the breakdown of [FB Live Comments](https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-live-comments) where I go into more detail.

Scale

### Some additional deep dives you might consider

There are a lot of possible directions and expansions to this question. Here are a few of the most popular.

1. **Dynamic auction end times**: How would you end an auction dynamically such that it should be ended when an hour has passed since the last bid was placed? For this, there is a simple, imprecise solution that is likely good enough, where you simply update the auction end time on the auction table with each new bid. A cron job can then run periodically to look for auctions that have ended. If the interviewer wants something more precise, you can add each new bid to a queue like SQS with a visibility timeout of an hour. A worker will process each item after an hour and check if that bid is still winning. If yes, end the auction.
    
2. **Purchasing**: Send an email to the winner at the end of the auction asking them to pay for the item. If they fail to pay within N minutes/hours/days, go to the next highest bidder.
    
3. **Bid history**: How would you enable a user to see the history of bids for a given auction in real-time? This is basically the same problem as real-time updates for bids, so you can refer back to the solution for that.
    

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

### Mid-level

This is a more challenging question for a mid-level candidate, and to be honest, I don't typically ask it in mid-level interviews. That said, I'd be looking for a candidate who can create a high-level design and then reasonably respond to inquiries about consistency, scaling, and other non-functional requirements. It's likely I'm driving the conversation here, but I'd be evaluating a candidate's ability to problem-solve, think on their feet, and come up with reasonable solutions despite not having hands-on experience. For example, they may propose the caching solution without understanding the downsides of distributed transactions.

### Senior

For senior candidates, I expect that they recognize that consistency and real-time broadcasting on bids are foundational to the problem and that they lead the conversation toward a solution that meets the requirements. They may not go into detail in all places, but they should be able to adequately arrive at a solution for the consistency problem and explain how bids will be kept up to date on the client. While they may not have time to discuss scale in depth, I expect they recognize many of the problems their system introduced.

### Staff

For staff engineers, I expect them to demonstrate mastery of the core challenges around consistency and real-time updates while also proactively surfacing additional complex considerations. A strong candidate might, unprompted, discuss how ending auctions presents unique distributed systems challenges. They'd explain that while fixed end times seem straightforward, implementing dynamic endings (where auctions extend after late bids) requires careful orchestration to handle clock drift, concurrent termination attempts, and delayed valid bids. They might propose a dedicated scheduling service using tools like Apache Airflow or a custom queue-based solution to manage auction completions reliably. This kind of unprompted deep dive into adjacent problems—whether it's auction completion, fraud prevention, or system failure handling—demonstrates the technical breadth and leadership thinking expected at the staff level. Ultimately, the most important thing is that they lead the conversation, go deep, and show technical accuracy and expertise.

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

T

ThoughtfulEmeraldPheasant808

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm39zuy9a00a97sv2fqfuuf62)

This solution does not address synchronous responses to the user when they submit a bid. If we put bids in Kafka they will be processed at their own pace and the response to the user ( if their bid is the winning bid or not) will most likely be not in in the same session making it a bad user experience. I think a better solution is to have an in memory hash table for the highest bid on the bid service in the critical path and then write the bid to Kafka (for record keeping). The user should respond to the user only after writing to Kafka so we do not lose any bids in case the service goes down.

Show more

12

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3d9b5h901gjmcj73gds3ywx)

We have an open SSE connection to the user! We're broadcasting all incoming bids to connected clients. We can also ensure we broadcast rejected bids back to the creator. Then the client can update accordingly.

Show more

5

Reply

T

ThoughtfulEmeraldPheasant808

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3dbk85h01i2mcj7z678j0e7)

This will still make this asynchronous. When they are placing the bid that is not over SSE so it will not be in the same session. My understanding is that SSE is for checking the highest bid which is an independent operation. I think the user should be informed in the same session when they place the bid.

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3dbnb6801iamcj7p0tscjpa)

Correct, this is asynchronous, technically. But the user should be none the wiser. See the comment about latency added by a message queue in the callout above. We can still show a "processing bid" message and then a result. While it was asynchronous, to the user, it felt like a synchronous response to their action.

You can response synchronously, you're just losing the message queue, so need another option for ordering, scaling, and durability that it offers.

Show more

11

Reply

W

WoodenAquamarineSnipe931

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8lyq9hh00dx4wukoo55uclr)

So a queue effectively is the last resort within the context of this problem. I read in another question breakdown (can’t recall which one at the moment) that queues usually mask performance problems of your service; is there a principle or criteria that you use to decide when it is okay to use a queue as a tradeoff?

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8ymwocj00vtad07xyvw59cg)

Yah, generally it depends on the priority on fault tolerance. If dropping a request is tolerable (like adding a like, if it fails, we can tell the client to try again), then the message queue may not be a good idea. On the other hand, for financial systems where fault tolerance is critical (we can't drop a bid!) then it's worth the trade off.

Show more

8

Reply

W

WoodenAquamarineSnipe931

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8yuonaf0150ad0772wsgm46)

Ah. That makes sense. Thanks!

Show more

0

Reply

J

JollyYellowGrouse728

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9nexfh8003nad07zkuz5zkb)

Isn't the result of not being able to write to Kafka the same as not being able to write to a database in this case?

If we accept the request but cannot write to Kafka because of a hardware fault on our server, we will have still dropped the bid and expect the client to retry. I don't understand why that's fundamentally different than failing to write it to our database with the exception that there's a smaller probability of that happening.

Show more

3

Reply

![Garvit Chugh](https://lh3.googleusercontent.com/a/ACg8ocK9V05XNzgNi6hE_IJ6uHiiL1o6LyshT2Cz3gHPpxufoXTc=s96-c)

Garvit Chugh

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmancnhjw01m0ad08wzu4ruxo)

Accepting a bid ideally means that the write operation to kafka succeeded. Think of the POST API call we are making to write post a bid. The response will be 200 OK only when the bid gets written onto kafka. If it fails before that, then we would return proper error via that API itself to the client and they can retry if they wish to. This is different from failing while writing to the DB because once a bid is accepted, it is the system's responsibility to persist it and have high fault tolerance. Now, if anything goes wrong after accepting the bid(i.e, writing to kafka), we need to have a way to be able to retain that bid which is where kafka becomes important. It helps with two things - one is to retain the bid well, the other is that will have it in the order it was supposed to be and provides capability to replay the bids in that order.

Show more

1

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcd2x04z049fad08zipntkva)

I have query on the ordering part, though we have queue partitioned by the auctionId but there are multiplt Bid Service instances running, so IMO this order would have been maintained if there was a single Bid Service instance but with multiple consumers how we are managing the ordering or are we saying all the Bid Service instances are part of a consumer group where only one of the instances will read the bid for a particular auction?

Show more

0

Reply

![Garvit Chugh](https://lh3.googleusercontent.com/a/ACg8ocK9V05XNzgNi6hE_IJ6uHiiL1o6LyshT2Cz3gHPpxufoXTc=s96-c)

Garvit Chugh

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmancwg9n01mlad087hfew5bc)

The operation would be async in only the literal sense. If you think about most of the online auction or bidding systems, they have a separate tab where you can view your bids, that is what this will support. The idea being that once you write it to kafka, assuming it gets written onto the DB and gets communicated to your client via the SSE, the client knows your userID and can easily handle the responsibility to segregate that bid from the others. So, once you place a bid, you get a sync response in the same session, which says Bid initiated. And then you can head over to the My Bids section and see the response. Kafka scales extremely well with such a small throughput and you should get the result of your bid to your client in virtually no time and then you could place another bid. This would seem real time when thought of from the perspective of human perception. And that is very important in this system because it will only be the human (user) of the app who will initiate the bid again.

Show more

1

Reply

![Alfred Gui](https://lh3.googleusercontent.com/a/ACg8ocJrrl0mvI_wtYdm8wn7lF6lcgdWZ1TOO3l91WPFN0UZSrFoROc=s96-c)

Alfred Gui

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4su61n801ee5owsrfn4yuyk)

The bid qps is just 10k per sec, so we don't need further optimizations to the architecture in this article. But if the throughput is unrealistically high (like 1 billion per sec), we will need more extensive caching of the highest bid, including the bid cache on each bid server. Early rejections could reduce downstream load. (We will still need to push all bids to the message queue for book keeping, but that's not the critical path).

Show more

6

Reply

E

ElectricRedOwl923

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6uekxtg01707oi9zhuyaldk)

why not just maintain a in-memory copy of maxPrice? this way we do not need to hit distant redis or database

Show more

0

Reply

W

WorthyApricotHarrier535

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm79x67rl04myl7b17uka1exz)

Bid Service is horizontally scaled and each instance will have it's own maxPrice in memore for a given auction

Show more

1

Reply

![Neeraj jain](https://lh3.googleusercontent.com/a/ACg8ocJAzrMQriHY2NYkdaOuS5_-r2zdrYIXz-3bABYqasyyHNOvtik=s96-c)

Neeraj jain

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm932mrqz005dad08thxch8v6)

What are you planning to reject even if your bid is not the winning bid you need to store the bid entry into the DB right ?

Show more

0

Reply

F

FancyTanGoldfish347

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9eogcyk012qad08htjf0rnz)

With 1 billion bids per second that means we are assuming 100 trillion concurrent auctions at peak time. Sure this is an interesting though experiment but this seems very unrealistic.

Show more

0

Reply

A

AppropriateGreenHoverfly664

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3a9dgjr00cj8of9hmw3eayy)

Thanks a lot Evan.. We all are waiting for this..Right on time.. Really appreciate it

Show more

0

Reply

Y

YouthfulBlueVole468

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3bzhwy800h4mcj7mgvwk9v5)

How to handle partial failures well?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3d9bfn8007r16s9g5r8nl1w)

In what part of the system? :)

Show more

0

Reply

![nurad 23](https://lh3.googleusercontent.com/a/ACg8ocIA08JF0pIs48BfDkpXk2lMrsbrQ4faHBH_LrrrJSzPfoW9i57Mlw=s96-c)

nurad 23

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3ce882400f1n4w8urlqvinj)

Can we have a write through redis cache that gets updated with any new bids and then the auction/bid service can refer this to get real time updates of max bid?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3d9dgzm01grmcj7g39sbx6j)

There is a design here that would work. Likely using Redis sorted sets. Same issue with regards to a distributed transaction though to keep cache and DB consistent (totally possible, just introduces more complexity)

Show more

0

Reply

R

RadicalBlackBeetle554

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3du6h1v00rv16s9kcmo5n0b)

Online-auction: The wait is over!! Thanks Evan :) For sharding the database, could we use extension like Citus for sharding Postgres DB tables: auction & item? Users data would be comparatively lot less. What's your thought in replicating the Users table data to all the Postgres DB instances? That way, for any auction - read/write queries goes to the same shard and related Users data is also available in the same shard. Eager to hear your thoughts, Evan. TIA!

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3dynha901msn4w8ndhznouh)

TBH, I wasn't familiar with Citus, but just skimming the landing page really quickly, I'd say yes. It just looks like a managed or self-hosted distributed Postgres instance.

I wouldn't replicate the user data personally just because colocation isn't that important with the query pattern we have. Having user data in its own DB instance or a table on a single shared instance should be fine. Not worth the cost.

Show more

0

Reply

R

RadicalBlackBeetle554

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3ey0cmv009qsh985ld1tvvu)

Thanks for the clarification, that makes sense. Are you considering to callout this in appropriate use cases to use such extension with Postgres? (use cases, such as an auction system that requires strong consistency, transactional guarantees, and sharding)

Show more

0

Reply

![Rohit Sai Chowdary Potluri](https://lh3.googleusercontent.com/a/ACg8ocIUHt2rl1_c0gR5ONj7flXHJsYXm-Vw57gwQzgVQtLfEPmetg=s96-c)

Rohit Sai Chowdary Potluri

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3eing1h0040130rx96e4ho9)

Any reason why the Bid service is maintaining the SSE connections instead of the Auction service? We create an SSE connection whenever a user tries to view the auction details right?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3enbs8p007mmv20o2w16e4c)

Because we are broadcasting bids :) When a user views an auction, we can fetch the auction details from the Auction Service and connect them to an SSE connection via the bid service at the same time.

Show more

1

Reply

S

ScientificSapphireGull777

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm62vtid401vahqkjpou2kv6o)

So if the request comes to the Auction Service and the SSE connection is created via the Bid Service, should the diagram have an arrow from the Auction Service to the Bid Service so the request can flow down to the Bid Service where the SSE connection can be made?

Show more

0

Reply

M

MinorAquamarineHare569

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5bo09ws03eze1nmj6psx4lh)

To move SSE connections to Auctions Service we could consume updates to Auction DB with a DynamoDB stream and write to all SSE connections with that auctionId. This would add some ms of latency but maybe still within our limits

Show more

0

Reply

![Luiz Felipe Mendes](https://lh3.googleusercontent.com/a/ACg8ocLiaxrQj_iBKa0tyykfUuU5Jx14UBC9Sf_7rf2bEHaxFQBwLX9dNg=s96-c)

Luiz Felipe Mendes

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmao2ol4h02jvad08glvf3jyd)

Could I split bid service in 2? Like, one would handle the bidding per se and the other just to update users. One would send the message to the pub/sub and the other would be listening and sending to the user?

Show more

2

Reply

O

ObedientSapphireCardinal930

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3ez1oks000kq5z957o9isr0)

i tried to parse this and was trying to find out which kind of DB you used here. i know you have said it in the past - SQL/ noSQL doesnt matter but here is it sql ? postgres will ensure strong cosistency for maxBid amount

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3fid2ya005op0v4zgu2mdpo)

Yah postgres in this write up :)

Show more

0

Reply

S

SupportingScarletLadybug162

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3f9p6c700dkeve8ra520u11)

Why is writing to db and cache not great here due to distributed transactions but is a valid solution for LeetCode design?

Show more

3

Reply

P

ParliamentaryOlivePelican346

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3g9bni700jmp0v4nu47lwq0)

Thanks, Evan, this is very helpful. I have one question:

I've noticed that in some designs, multiple services connect to the same database (at least in the diagram). I always thought that in a microservices architecture, a clear separation of databases is the preferred pattern. However, I don’t see that here—could this be a red flag for some interviewers? I’d love to hear your thoughts on this.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3gbgz3g000z132jvul6wgn7)

Definitely have some thoughts there! Check out the YouTube short Stefan created which answers this question :) https://www.youtube.com/shorts/jg-Dy0GNPvQ

Show more

0

Reply

M

ModerateBeigeCattle366

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6wrb1jm01rzdp6hew8w0ob2)

IMO, sharing tables makes sense only for small products. As your business grows, different teams will own different domains or subdomains. Reading from or writing directly to another team's table bypasses their business logic, concurrency controls, and other safeguards.

That said, I think sharing a table is reasonable if both services are owned by the same team and there are no plans to split them into different teams.

Show more

0

Reply

G

GleamingChocolateChinchilla112

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3ggkufo002g13qyr0bzdmqb)

Isn't pessimistic locking better than optimistic locking for use cases with many collisions? I think there might be many collisions, especially in the auction's last few minutes/seconds. If it's true, you can improve the performance by combining pessimistic locking and caching the max bid. Do I miss something?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3ggn9pb002i13qys6fw9woq)

Doubtful that there will be many collisions, actually. Two main reasons:

1. The more bids, the higher the price, and the more people are priced out. So "hot" items price people out quickly.
2. Users need to react to bids. They see an increase, type a number, and hit enter. As a result, we are not talking about thousands of concurrent bids. Write throughput is low (per auction), making collisions less probable.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3ggxe9x002d73ydhw1jnidq)

eBay "snipers" try to get all of their bids in at the final moments of the auction to prevent a bidding war. The bid distribution is very heavy in the final moments!

Source: used to sell Pokemon cards online in the early 2000's.

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3ggyaw3003foqb0rftgalk0)

You made some money I hope?!

Show more

2

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3ggzudl003ioqb0vp666ayr)

Made a killing before Paypal locked my account because I was under 18. They held on to like $300 (millions for a teenager!) for 3-4 years. I'll never trust them again.

Show more

7

Reply

C

ChristianPlumTakin500

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7a3x1wq050f2azpg03g4n96)

I'm sure there are many of us curious to know what your favorite pokemon was and is.

Show more

0

Reply

F

FastJadeGrouse442

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3m69bbu03bkn2u2gof5vjbx)

Is this a good framework to think about detecting an Auction has ended?

- Cron Jobs: while simple, it is suffered from delayed by N (interval) and expensive, as they query the entire table.
- Better Approach, Queue with Visibility timeout on expired datetime, now task workers can process only relevant items, avoiding full-table scans.

If we were to use NoSQL like Cassandra or Mongo then we can just use TTL on a column then capture via Changefeed?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm41j5rhn00kqfio1khx0ld9d)

Yup this is great. Note that keeping a bunch of records in SQS with a vis timeout might bet a little expensive, but most interviewers wont press you on that.

Another option is a workflow engine like temporal.io or airflow

Show more

0

Reply

C

ChristianPlumTakin500

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7a5lcvg054qr9w520kwyu1r)

> Better Approach, Queue with Visibility timeout on expired datetime, now task workers can process only relevant items, avoiding full-table scans.

If this is about SQS, the max visibility timeout that can be set is just 12 hours. Auction can last for days! I guess when they're visible, you can push by 12 more hours, but it will end up in a state where we have to process auctions at a max rate of total auctions for a 12 hour period.

Show more

1

Reply

C

CapitalistBlackChickadee849

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm3s71pya01eyahfgghgfqn0x)

This link "Hello Interview System Design Framework " in the doc is broken. I think it should be this https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm41j2tdl00kj130qyeq6faam)

Fixing!

Show more

0

Reply

![CollectiveAmberGorilla135](https://lh3.googleusercontent.com/a/ACg8ocK97eUQSMUZOAyak7LbAyd7N3KA69IXOcl51N1P_0NQZipHAQ=s96-c)

CollectiveAmberGorilla135

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm40riwr3003o130qeu6bj751)

I feel there is a catch with Kafka route. If we publish the bid in Kakfa , we cannot confirm the customer that their Bid is received as Bid will be accepted only when it more than the max bid for auction currently. THis validation would be possible only when Kafka message is consumed and processed by Bid Service.

Ideally this will choke the bidding system as due to the Kafka introduced latency, Bidder would not know the maximum price (I know later we introduced Web socket.

Bidder requested is submitted but he/she would not know if it is accepted or not and they might not Bid again . It will keep them guessing. Any thoughts ?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm41j7ek300j78uaj1ekul1h3)

You'd have that bidirectional connection. Either SSE or websocket. Show the client "processing bid" until they get the confirmation.

Show more

0

Reply

C

CapableScarletFly802

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm80g5199048xtvyjcmbp133i)

The SSE would update the global maximum bid for that auction, but how does the client know that their particular bid was accepted?

Show more

1

Reply

L

LogicalBlackCrane246

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm44endi400bd2trsboiyswt8)

Can we use Layer 7 Load balancing on auctionId instead of pub sub for broadcasting updates to auction watchers? Clients connect to same auction server via load balancer and messages are routed to same server via load balancer.

Or would we run into limits of current connections on the same host because of this (really popular auction, more active connections than one host can handle)?

Show more

0

Reply

G

GiganticApricotSwordfish902

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm44ll0gx00n9siamg3ddau1e)

Have the same question here.

Also since we have kafka partitioned by auctionID, wouldn't this mean all bids for an auction would be consumed by one Bid server anyway, so couldn't we hash all users connected to an auction to that same server?

In this case where we were routing users to any bid server and using pubsub, wouldn't this mean each server makes the bid updates for a subset of auctions (based on the kafka partitions) but also sending the max bid updates for any auction they have users connected to? is this a cases where we may want to separate out the read and write path?

Show more

2

Reply

A

aniu.reg

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm58cdsh700f8ke2rka8pbaa0)

To GiganticApricotSwordfish902 +1 on the point of "wouldn't this mean all bids for an auction would be consumed by one Bid server anyway". If this is the case, it defeats the scalability claim.

Can Evan or Stephan shed some lights here? Thank you!

Show more

0

Reply

C

ChristianPlumTakin500

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7a6bz1u052d4nd52fpmmjhe)

Kafka provides high publish throughput compared to autoscaling for handling spikes in events. Since we can't afford to lose bids, using a queue is essential. Without it, there's a high chance that autoscaling won't be able to keep up with sudden surges, and over-provisioning would be extremely expensive.

Kafka consumer can still optimize processing by reading in batches and performing batch writes for both rejected and successful bids per batch. However, this may not be necessary, as bid spikes occur only briefly toward the end of an auction, and the worker should eventually catch up.

Show more

0

Reply

A

aniu.reg

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm58c6yqo00ewke2rgi31bxjk)

To LogicalBlackCrane246, IMO, there are two ways of communications here. You can use L7 LB to always route same users request to the same backend server, but when the server side needs to send update to the client, L7 LB is not helpful here.

Show more

0

Reply

L

LogicalBlackCrane246

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm58e65qv00hyi96o7ogj27ua)

Could you expand more on why it wouldn’t be helpful for server sent updates?

I figured if both client requests and server updates use the same load balancer than they would arrive at the same host.

Show more

0

Reply

A

aniu.reg

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm58osyqi00sci96o70b0vfgi)

client 1 connects to bid service1, client 2 connects to bid service 2, both clients are bidding in the same auction. When client1 becomes the highest bid, bid service1 can definitely send update to client1, but bid service1 does not know where client2 connects to. (This discussion makes sense when an auction is hosted across bid service hosts. But from this thread's discussion, I am not sure if this assumption holds or not.)

Show more

0

Reply

L

LogicalBlackCrane246

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm58q061i00rpc4lgsjj0m6tz)

Layer 7 load balancing would balance the load across hosts by the auction ID. So both clients would be connected to the same host for the same auction. Every client watching the same auction would be on the same host. I think this would work but the only time it wouldn’t is if one host can’t handle the number of clients watching one auction ID

Show more

0

Reply

![Michael Zhu](https://lh3.googleusercontent.com/a/ACg8ocLOpBA3quICF9uMJffMPlWU7ldyHY2aF0sta8EziVT5X1PBjLD0=s96-c)

Michael Zhu

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm46779oj0144hco9vllyakj8)

The timeout mechanism also introduces a small delay - if a bid comes in just After a client's previous long poll times out, they won't see the update until their next request completes. This creates a tradeoff between resource usage (shorter timeouts mean more requests) and latency (longer timeouts mean potentially delayed updates).

Can you explain it in more details? "if a bid comes in just After a client's previous long poll times out, they won't see the update until their next request completes" Should it be once the second connection is build, the server would return the bid instantly? The delay here is caused by network latency between one request timeout, and second request reach server. I don't see it is related to the timeout setting.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm46m2lzq01p6hco9hg757k68)

You're absolutely right - I should be more precise here. The delay isn't related to the timeout duration itself, but rather to the network round-trip time needed to establish a new connection after a timeout occurs. When a bid arrives during this brief reconnection window, the client won't receive it until the new connection is established. While the timeout setting doesn't directly cause the delay, it determines how frequently we might encounter this brief gap in coverage, which is one reason why persistent connections like SSE or WebSockets can provide more consistent real-time updates.

Show more

2

Reply

![Paul England](https://lh3.googleusercontent.com/a/ACg8ocI9EL7D8ZUYgaO3tz6A9B9lvnN4vq3pCnDPV5l0kizwLwta78bH=s96-c)

Paul England

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4ddmfbz01z2j6nb7413qx95)

I think the constraint for bids being strongly consistent is artificial and not reflective of how most auction systems work. It also impacts the design. Let's consider what Stephan said above about the last seconds of an auction being what truly matters. (Source: I sold mountains of vintage games in the late 90s!! Oof!). Ebay has evolved a lot since then but the bidding is basically the same. In fact, I think they still tell you "you're winning, but you can still be outbid!"

Assume an auction currently has a winning bid at $5, and that we concurrently receive bids for $6, $7, and $8. Here are the following permutations of processing order

- $6, $7, $8: All bids are accepted, but the users who made bids for $6, and $7 are eventually notified they were outbid.
- $7, $6, $8: Bids for $7 and $8 are accepted, but $7 was outbid. $6 was rejected, and the response to the bid said as such.
- $7, $8, $6: Same as above
- $8, $6, $7: Only $8 bid is accepted. The other two are rejected, and the response says as such.
- $8, $7, $6: Same as above

That's not to say very loose consistency is acceptable. If we process one of those bids much longer than say the average user latency of a request, it will result in bad user experience. However, handling them concurrently within a reasonable window will result in the same thing to the user: highest bid wins. The only difference is what will show up in the bid history and what won't, which I would argue doesn't matter much in this case. Other thoughts:

- In a system where bids can be cancelled (by either side) this has implications. Ebay used to allow this, and still might.
- Potential brownie points. We're assuming auctions end just like they do on Ebay as they're the dominant site in the west. Real auctions do not end like this, which includes some electronic auctions that interface to real auctions (LiveAuctioneers.com, for example). Any nerds that have bought items from Yahoo Japan knows that if a bid is received at any point within the remaining 5 minutes of the auction, the end time is extended for 5 minutes. Slow motion bid wars. They're maddening, but it does save them from this technical hurdle.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4hjt3gv01fu5gbo4vhgb7ye)

Yeah, this is correct. I think I allude to this with a callout in the article as well. Great staff candidates tend to call this out, and they're right. The reality is, when this question is asked in interviews, the interviewer is typically wanting to evaluate how you handle consistency. So even if you (correctly) raise this, it would be appreciated, but they'll likely still ask how you would handle strong consistency if you needed to.

Show more

0

Reply

![Paul England](https://lh3.googleusercontent.com/a/ACg8ocI9EL7D8ZUYgaO3tz6A9B9lvnN4vq3pCnDPV5l0kizwLwta78bH=s96-c)

Paul England

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4i0nnae0242bb1krvin1eh1)

Good point. Thanks for the explanation.

Show more

0

Reply

C

ChemicalOrangeBaboon761

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4epinkg03zl1i3rsre7jlsn)

RE "Partitioning: We can partition our queue by auctionId, ensuring that all bids for the same auction are processed in order while allowing parallel processing of bids for different auctions.", does this suggest that we can have ONLY one consumer (one bid server) if we want to preserve the order? In this case, do we still need the strong consistency given we can process bids of each auction sequentially on the only bid server?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4hjreah01dv11jtf35iyhl2)

Astute! If you force serialization like this (and don't kick off concurrent requests from the consumer) then yes, you would not need strong consistency in the DB.

Show more

0

Reply

F

FlexibleSapphireEchidna890

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7fodsnw00giozj8i9ct3mqd)

Won't this create a problem similar to "hot key" issue i.e. if the auction is wildly popular, bids will be processed quite slowly?

Show more

1

Reply

H

HandsomeIvoryCrow799

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmahosrve006mad07hdvv7uby)

Kafka can only handle hundreds of thousands of topics and we have 10 million concurrent auctions to handle. How do we come around this limitation? One potential solution I can think of is to use consistent hashing so that each Kafka partition would handle several auction IDs. Any thoughts?

Show more

1

Reply

G

GrandEmeraldParrotfish500

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4f5zlu804l51i3rwkn1bk2l)

Typo in “The best approach here is to introduce a dirable message queue and get bids into it as soon as possible”

Show more

1

Reply

![Alonza Huang](https://lh3.googleusercontent.com/a/ACg8ocLWXptO6UbmQwUQJDjYE2DDP_XWsT7dUFo9CcLOuH70JZddbEM=s96-c)

Alonza Huang

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4g1kiqv04aourisjbndqpxk)

Hi Even, for the last deep dive, since servers are sharded by auctionID, why is pub/sub still needed? Wouldn't all users for a given auctionID already connect to the same server? Or is this to address a hot key issue where a single server is unable to handle SSE for a hot auction?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4hjoux401g1bb1kpx30vibm)

Possible! You could go this route like in Google Docs. But you can have some popular auctions that have millions of viewers (think Michael Jordan Game 7 jersey or something). In this case, not all connections fit on a single server.

Show more

1

Reply

B

BiologicalMoccasinTahr305

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm930h4bf003fad08y9edovbw)

Hi Evan - thanks for the video and write-up. You're the best!

I suppose we will have a Zookeeper instance to tell the client which bid server to open an SSE connection to. One thing I don't understand is -- what is creating/updating this mapping in Zookeeper? I don't think Zookeeper on its own can handle the allocation logic, can it? And there is some controller service in front of Zookeeper that does this. But at that point, is Zookeeper just like a distributed cache and if so, can we also use Redis? I had the same question in FB live comments.

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4hav0ft015w5gbowm52e41g)

The system should maintain strong consistency for bids to ensure all users see the same highest bid.

i think this requirement is only when someone queries the DB? cuz obv there could be some delay for latest bid to be reflected on users browser monitoring it

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4hjnmkk01e38jlrr2t1btgi)

Yah that's correct. This is for determining whether a bid should be accepted or rejected in the backend.

Show more

0

Reply

![Anurag Pande](https://lh3.googleusercontent.com/a/ACg8ocKwI_Wz-hkDTPidsfr7cQ7IX-pcTgTwy6O1BwYnvhEYw8lGHG4T=s96-c)

Anurag Pande

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4jd7z0800r3q7aibr8au73l)

dirable -> dirgible

Show more

0

Reply

B

bhavsi

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4pcnidr028cjs6sn1c0hroo)

Thanks for the amazing post, covers a lot of topics - concurrency control vs distributed locking, long polling vs SSE vs websockets.

I had a question about partitioning the new bids Kafka queue by auctionId. I did think about that but then if we have 10M concurrent auctions, does it mean we will have 10M concurrent partitions? Is that a bit too much? There would be a significant number of auctions that are not popular. Is there a way to dynamically add and remove queues for auctions if they become popular, or is that over-engineering?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4rukp8i00lurpk7a7iwouvq)

You wont have a partition for each auction id! You're partitioning based on hash(auctionId) % N where N is the desired number of partitions.

This means all bids from a given auction are guaranteed to be in the same partition (for order), but a single partition can have multiple auctions.

Show more

1

Reply

I

IncrediblePeachCheetah570

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5m6i5se00o811w5fqqzirl8)

Should we worry about hot partitions from a popular auction? Also you mentioned elsewhere that serial processing for bids of the same auction would be too slow and lead to bad customer experience, but a single Kafka partition can only be processed by a single consumer.

Show more

0

Reply

![Shreeharsha Voonna](https://lh3.googleusercontent.com/a/ACg8ocJFOfqMA6ZDt2EamTjU3FJJThz35r5s7qZyqePh9qAYYlJq_K8p=s96-c)

Shreeharsha Voonna

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4pzfqu502nklkt9b8gf3r08)

Great content Evan. I have few questions wrt your final design:

1. Isn't Bid service doing way too many things? Reading incoming bid requests from Kafka, updating MaxBidPrice in Auctions Table in DB, inserting current bid details in Bids table in DB, pushing MaxDBPrice to Redis Pub-Sub channel for corresponding auctionId, and maintaining SSE connection with user device.
2. How is this design taking care of user's incoming bid request's response to be synchronous? I see that you mentioned you will show "In-Progress" but when it does get evaluated in the end, how does user get the right response to that request? Through SSE's he is able to see latest price of the auction afaik. Are you suggesting user can short poll for his bid request with bid Id?

Show more

0

Reply

![Jagrit](https://lh3.googleusercontent.com/a/ACg8ocLVQF2_4KCJMgtZ0FF2xVqiW2qacI3u57liReHgnXzKXfc-iOwX=s96-c)

Jagrit

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4ruh8sg00nruek61a1vh6lx)

Amazing guide and design editorial. While following the article, In the start I was wondering why didn't we have the maxBidPrice in the auction table as that would make things simple. But later, you added it in the first deep dive. Glad I could think of it as well. I have gone through almost all of your free designs multiple times and I guess it's the result of that, now I am able to think of some things on my own as well. Thanks for doing the amazing work!

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm4rui26k00n95owsj7eqgdbk)

Great stuff! Sounds like this is starting to become second nature for you :)

Show more

1

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm50ts65701ojub4z0r2cki6d)

While it can be argued that the queue producer does less work than the bid service, and thus, holds on to the bid for a shorter time, there's always a chance that it may crash soon after receiving a bid. I don't think we can design a 100% fault-tolerant system, and thus, the queue seems unnecessary.

Show more

1

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm50u06e701nir5q695y08d30)

One thing that can be done is to deliberately send the same bid to multiple bid services randomly, thus making the probabilities of all them crashing at the same time astronomically small. Because of the DB ACID properties, only one will them win.

Show more

0

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm53q6rdz03jb13vuude1j1kg)

Hey Evan, would it be correct to mention that because we are likely handling an auction on specific nodes, that we should also maintain SSE connections for that auction to be on the same service?

In that sense, using something like Zookeeper to maintain node knowledge, so we can use consistent hashing to ensure that users on a specific auction can be routed to the same server. Or is this redundant because we are using pub/sub and thus we can get away with something like round-robin or least connections for SSE connections?

Show more

0

Reply

T

TechnologicalBlueWhippet396

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm54l1nvh043rwbx334pkane5)

I think this solution is mentioned in Facebook Live Comment btw!

Show more

0

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5542viv04i7mqq6bvopcx83)

I must have also missed his explanation, or misread but it was answered in this problem's deep dives!

Show more

0

Reply

T

TechnologicalBlueWhippet396

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm54kfw44042j13vuridutyd6)

Hi Evan thank you the post!

I have a question for Deep Dive 3 challenge part "If User 3 places a bid that goes to Server A". My understanding is that maxBidPrice is store in database, so all server should have the maxBidPrice with SQL query, how could a bid only available on a single service?

Also, if we create a bid go through server A then go to DB, how do we notify each of bid service to get the latest maxBidPrice, do we do via a SQL query or some db trigger/push notification. Thanks!

Show more

0

Reply

C

ConfidentOrangeGull774

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm59v7df501pui96o8epa2i6t)

Hi Evan, thanks for the article!! I have question about the "producer" that is introduced to put message into the queue. (1) This would be a service called by API gateway to place the bids? right? (2) If we think about not losing any bids, how it is different than API gateway calling Bid service directly since the call to the producer or from producer to the queue might still fail resulting in not recording the bid? (3) Can we have the logic in api gateway to put bids to the queue directly? What are the tradeoffs?

Show more

0

Reply

W

WelcomeAmberWhitefish906

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmddtu3pa04h8ad09l9nlg6fd)

yeah same. I also have the same questions. what's the use of producer, it can also crash. @Evan can you please provide the answer of this.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmddwd9f7056pad08aajbqhx0)

It's a lightweight producer that does nothing but put bids on a queue. This way, there are fewer failure modes as it does not do much, and we can run it on small, cheaper hardware, which allows us to scale up horizontally preemptively in a cost-effective manner.

Show more

0

Reply

N

NewFuchsiaSilkworm545

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5a3bxcb01uwc4lgvalj63at)

Hey Evan,

Why not use consistent hashing on Bid Service and partition by auction ID? We can route all SSE connections to the same Bid Server. Why Pub/Sub solution is better? And would you specifically use Kafka instead of Redis for Bid Service pub/sub?

Show more

1

Reply

I

InclinedSapphireScallop143

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5cwafei00ndkz0h9fzc0snf)

a single modern server can handle around 100k SSE connections, if the auction is watched by 1 million people, I guess routing all SSE connections of the same auction to the same server is not a good idea, but Pub/Sub solution doesn't have this constraint and can be scaled infinitely, I think it's a matter of scale and should just mention this to the interviewer.

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5cwnfx400k1gzymsrzflezr)

Indeed

Show more

0

Reply

R

ResponsibleIvoryAlbatross601

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5czkemg00ndm9c3ppjjzpjo)

Hi Evan, thanks for the great content. I have a question. In Deep Dive #4, it has been said "we consider both the Bid Service (consumer of the message queue) and Auction Service. As is the case with almost all stateless services, we can horizontally scale these by adding more servers." Is the Bid Service stateless in this design? Doesn't it have to keep SSE connections mapping info?

Show more

0

Reply

![Jiatang Dong](https://lh3.googleusercontent.com/a/ACg8ocKfQgaYilpR7RBKGa8_AXqyhuDM2GA6B29pLwiJomT1-dI5c0tTlg=s96-c)

Jiatang Dong

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5eom3ln00bpshbyhse6y8e7)

A lot of topics have talked about using locks. But I am still not very clear on some things:

1. We have talked about database row locking, distributed lock over cache, distributed lock over zookeeper, optimistic locking (we don't talk explicitly about pessimistic lock that often though). I want to see if we can't have a separate talk for this topic. I think a lot of concepts needs to be clarified.
2. Besides locking, I have seen another strategy call Try-Confirm/Cancel to handle high concurrency issue on the data update. Can we talk about when is a good scenario to use this stratergy? It's pros and cons against locking.

Show more

0

Reply

M

MedicalCoralBarracuda145

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5fswmnz00f7dubfhghiqwfs)

Hi, would it be possible to separate the questions from product design and infra style?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5fsyubw00eedi0dw2vv6d4n)

It's not. For Meta, there is considerable overlap in the questions that are asked. And most companies don't have the distinction.

Show more

0

Reply

I

IndividualGrayToucan573

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5fzen5h001knji00xhteqwe)

Hello Evan, I feel there is a trade-off after having message queue. It introduced another layer, thus more complexity. without using the message queue, if one bidding service host is dead, the client side can retry since the bidding request information is still in the client side. So, no bid is dropped. Comments?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5fzkjbn001nbfl0mrnrhmop)

Yeah, ultimately it comes down to how much your bid service is doing. No message queue, but write directly to the database first, then even a different "realtime" service to have the active SSE connections works great. Maybe even preferred.

Show more

3

Reply

S

satishm752

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5igryqw01u710u89q5n1rf6)

"Try to update the auction row, but only if the max bid hasn't changed:"

Needs to be changed to

"Try to update the auction row, but only if the max bid has changed:"

Show more

0

Reply

I

IncrediblePeachCheetah570

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5m5oe5q00np11w5mk0p3vys)

For the bad solution of database transaction with row level locking: "Lock all bid rows for the auction using SELECT ... FOR UPDATE: This locks all existing bids for the auction, preventing other transactions from modifying them until the current transaction is complete."

We have a status field in the bidding table. If we ensure that there is at most one bidding with status accepted for an auction, we can begin the transaction and just lock that single row with accepted status, right? Does this address the performance concern?

Show more

0

Reply

Z

ZippyCoffeeMastodon872

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmasb8p8u011mad09elp4xa06)

I've been thinking about this a lot. My intuition was exactly what you said. However maybe the "status" is really the "status at the time of the bid"

Show more

0

Reply

![Taishan corp](https://lh3.googleusercontent.com/a/ACg8ocLN9rxqC17hbi3qMITfhNUJgncLk3bNgKIhgReyhjP9w49Rwg=s96-c)

Taishan corp

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5pp5lvh018niuxn7iyk2t8g)

For the solution "Cache maximum bid in a database". We are updating maximum bid in the Auctions table from the Bidding service, but reading it from the Auction services. Does it mean that we have shared tables between Auctions and Bidding services ? Given these are separate scalable services, it could result in operational challenges and coupling between services.

Also we need to update both Bids table and Auctions table at the same time. Doesn't this require a transaction , so how does optimistic concurrency control address that ?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5pp6mc7018gkzmh1w5br64a)

Yes, bids for a given auction need to be collocated with the auction itself. Just shard both tables by auctionid

Show more

0

Reply

![Taishan corp](https://lh3.googleusercontent.com/a/ACg8ocLN9rxqC17hbi3qMITfhNUJgncLk3bNgKIhgReyhjP9w49Rwg=s96-c)

Taishan corp

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5pphgav0191iuxnmum7h0oa)

It's not so much a scalability issue (what sharding addresses) as it's an operational issue. It means we we have raw table accesses to same table from two separate services.

I haven't seen this in practice for non-monoliths. Also, not sure how you would update both auctions table and the bids table without a transaction if there's a fault between writing to both tables (e.g. write to bids table succeeds and system fails). I suppose a further kafka retry would not be harmful on bids even when it succeeded originally, but it can still lead to things being out of sync temporarily.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5pqt5jt01cdqsjk1i94crch)

Wait, we may not be on the same page. The point of putting these tables in the same database is so that we can use a transaction. You'd need to update the maxBid and the bids table via a transaction, absolutely.

Having multiple services reading/writing to the same DB is totally fine and happens all over. The whole "every microservice needs its own DB" is a bit antiquated. Check out Stefan's youtube short. https://www.youtube.com/shorts/jg-Dy0GNPvQ

Show more

1

Reply

![Taishan corp](https://lh3.googleusercontent.com/a/ACg8ocLN9rxqC17hbi3qMITfhNUJgncLk3bNgKIhgReyhjP9w49Rwg=s96-c)

Taishan corp

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5uk0r8j01w0p38v3m7jiz41)

How about the same table being read/written by multiple services? The Bid service will update the max bid value in the Auction table , so effectively the Auction table will be written by the Bid service and read by the Auction service

Show more

0

Reply

C

ComfortableMaroonCrow982

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5ql42q0025i114yft8ynggp)

Hi, Should we use max heap to keep track of highest bid? We will still need a queue but this way we have a way to keep track of highest bid which should help avoid any conflicts.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5xhfpu100zb123dco8yscki)

Sounds like overkill to me. There aren't that many bids, and grabbing the highest with an index is easy, especially if cached like in the breakdown.

Show more

0

Reply

![zec hong](https://lh3.googleusercontent.com/a/ACg8ocIWHPHeaNbvLoPqhfuB9L5HVPdtPN0uZqbS0RqzHDzSgOy4hw=s96-c)

zec hong

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5xhdg7600yqja3xkgsmk53l)

I feel like we are talking about strong consistency on place bid here but not about viewing the max bid. Even with the current architecture, it's still possible for the user to see stale data from auction table in DB when you try to read and update at same time. Unless you block the read request while updating the max bid for a specific auction. Hence, the non-functional requirement seems to be: Eventually consistency for reading the max bid for an auction and strong consistency for placing and winning a bid. Let me know if this make sense.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm5xheuqn00yuja3x5h4t15hp)

This is correct. We do not require strong consistency for reading the max bid on the client for the reason you mentioned. It's best effort there.

Show more

1

Reply

S

ScientificSapphireGull777

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm62uhr1u01sqhqkjumi05way)

For Deep Dive 1, isn't it also possible to just do a row lock on the auction entry without having the max bid column? It'd just lock on the auction row instead of the bidding rows whenever a bid for an auction comes in and the bidding service will check the latest bid of that auction and compare. Then, it will insert the new bid if it is higher. Even with having that max bid column in the auction table, we still need to insert a new row if the bid is accepted. So I think the only difference between the bad solution and the great solution of Deep Dive 1 is whether the max bid is also stored in the auction table or not. Other than that, I'm thinking both solutions can just lock on the auction row. Or am I missing something?

Show more

2

Reply

F

FavourablePlumBird366

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmd0ifvml00liad080r378aht)

I have actually used this in a production scenario (mysql InnoDB on repeatable read Isolation). What you are describing is correct. It does not matter whether the auction row has the maxBid column or not.

It works because now both the threads in the transactions are asking the database to lock a particular row whichever thread is able to get the lock will be successful in reading the row. Only when this thread commits its transaction the other thread will get unblocked and read the auction row and by that time the bids row(s) would have been updated.

All in all this works pretty great.

Show more

1

Reply

![Sae Young Kim](https://lh3.googleusercontent.com/a/ACg8ocLaE5vPe7nu5FZWbjNK8U9ydbsIkm8lC39-bsRNLvRjGmqjwA=s96-c)

Sae Young Kim

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm62usxid01whcf2a0nnjy2k0)

For Deep Dive 2, we are putting the kafka queue to make sure the bid request is not lost when there is a system failure. However, that's only applicable if the system that failed is after the kafka queue. If anything that is behind the kafka queue (ie. the producer itself), the bid request will still get lost, won't it? Is this still considered as meeting the requirements?

Show more

0

Reply

C

ChristianPlumTakin500

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7a826zy053s4nd5suv7wco3)

Until the enqueue of message, the call is synchronous from the user's perspective, and failure will result in immediate feedback with a retry error message. The message is not lost as it was never successfully submitted. The message loss typically means user submitted something successfully, and we lost it.

Show more

0

Reply

S

ScientificSapphireGull777

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7arwl8f0267ngojcaesfzop)

If the system didn't have the kafka queue in the first place, the user request will be synchronous. In this case, then, when is the request/message considered lost? Any system failure before the bid is save to DB will result in "immediate feedback with a retry error message". After saving it to DB, the message won't get lost. I'm not sure why the kafka queue is added to ensure the request is not lost in this case, then.

Show more

0

Reply

Y

YouthfulBlueVole468

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm63e4xiq008iufnhilr51gwt)

"While message queues do add some latency (typically 2-10ms under normal conditions)"

I was doing guided practice and judge said that using message queue is not good as it'll cause latency. What do you think if we used streams and flink to process it, and flush data from streams to s3 after retention period for durability

Show more

0

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm68ehifo00sse7bzgald72a0)

Evan, could you please clarify on sharding: "shard both the tables on the auctionId". For Bid table, auctionId is already generated..so we are good over there in partitioning. When a new auction is created, is the auctionId going to be DB server generated or application generated? If it is DB server generated, how would it know which DB node should host that record? If application generated, how to ensure uniqueness on auctionId?

Show more

1

Reply

C

CloudyGrayAnt409

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6ddqpd103gbsagcsub3ynx9)

In some other guides you've mentioned how we don't want to lock the database row because if the service locking it goes down, we'll be stuck in that state. Instead it was recommended to use a distributed cache like Redis that allows for TTL. Is this not a concern here? Or did you not consider it mainly because OCC was a possibility in this case.

Another thought - do we need to update the bid in a transaction? We could make the write to the bid DB async possibly using the same queue as the SSE queue. It might add some delay for the bid to reach the bids DB but I don't think we care about what's in the bids DB since we're not using it to read.

Show more

6

Reply

O

OlympicBlackLynx843

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6fsfhmv06l3iq3ncbvbnath)

"Partition Kafka by AuctionID" - but we need to scale this to 10M concurrent auctions. Does Kafka really handle 10M partitions, on one topic? Google search and chatGPT seem to suggest that this is excessive - is there something I am missing?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6il1ozh006ylhflvabg6gln)

It would not be a partition per auction; you'd group based on hash ranges. hash(auctionid) % N where N is the number of partitions.

Show more

2

Reply

Q

QuixoticMoccasinAnteater839

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6ikxn09007911hvi60pmz0l)

Hey thanks for the article, in your "numbers to know" articule you said queues can handle upto a million requests per second. but in this article you said 10k is too much and we should partition by auction id (which i think is a good idea but not sure on capacity limits)

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6il0wty006rlhflwy76bzu9)

Yah this is a good call out. Let me update this. 10k is pretty small. If you're on any reasonable hardware, no need to partition.

Show more

0

Reply

![Aman Thakkar](https://lh3.googleusercontent.com/a/ACg8ocLwwtnCae2aEApD8yZrf0xacaCP13PFYMjrKui7yq7amQ=s96-c)

Aman Thakkar

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7csv0d702jy19rhhzlz1wpj)

if queues have workers processing messages serially, there's no need for concurrency when dealing with cache updates or DB writes! however in the scenario where may have multiple workers just consuming messages from a kafka partition and working simultaneously, we would still need some sort of concurrency protection.

Show more

0

Reply

Z

ztan5362

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6kaqt9s02b4lhfl58crkzz3)

Doesn't Deep Dive 3(How can we ensure that the system displays the current highest bid in real-time?) automatically solve the issue of Deep Dive 1(How can we ensure strong consistency for bids)?

Users will automatically get realtime bidding info (via websockets/sse) therefore will not bid upon stale/delayed data.

Show more

0

Reply

![Aman Thakkar](https://lh3.googleusercontent.com/a/ACg8ocLwwtnCae2aEApD8yZrf0xacaCP13PFYMjrKui7yq7amQ=s96-c)

Aman Thakkar

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7cstca602ju19rhtx2a3oge)

yeah if a queue has workers processing stuff only serially then we have no worries about concurrent updates! but as to your 2nd line --> 2 users could still simultaneously bid when looking at a number on their screen. they will be processed sequentially but the events would be generated almost in parallel.

Show more

0

Reply

![Ruizhen Mai](https://lh3.googleusercontent.com/a/ACg8ocJaHmFr0IrNZOea1MIo1snLo5Mo52SwALt5AXl_DOyF78QBkD_h=s96-c)

Ruizhen Mai

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6oeeobj01mo124qy8nehz3y)

I'm not sure if others mentioned this. But if we have an ordered messages within each auctionId (partition key), we shouldn't need row lock since we're always updating with the highest price?

1. on BidService / BidProducer, we always send the message, let frontend guards bids < highestBid, this has latency, no problem
2. on BidConsumer, we update the highestBid = bidPrice (in auction table) iff bidPrice > highest bid; this is not concurrent in terms of auctionId bc of ordered messages is partitioned by auctionId
3. notify users on bid's success/failure based on the condition check on step2

Show more

5

Reply

S

SecondaryIvoryAntelope292

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6t3niwx035xsaswlflz3giz)

If placing a bid in the MQ, how to notify the users when the bid has been successfully post? if it is via SSE, the user will only knows the highest price but they will not know if this was their bids.

Show more

0

Reply

E

ElectricRedOwl923

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm6uehpwr011sp7h8oqahyius)

for deep dive 1, why nod use consistent hashing to partition on auctionId, so bid coming for same auction will always land on the same server, so that we can maintain a in-memory copy of maxPrice? this way we do not need to hit distant redis or database

Show more

0

Reply

H

HumanEmeraldGiraffe139

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm71jl10600xrgavh1n75numi)

I was asked this question in Meta Interview (“Instagram auction post”): I was asked - How to handle a case of “celebrity sells his guitar”? I think that in this case the writer ratio will be really big, especially in the last hour of the auction. I suggested in the interview using queue for processing the bids, and caching for reading the current bid and the auction details, but the feedback mentioned it was not enough. What else can we do in this case? Maybe using a DB which is better for writing like Cassandra? But Cassandra doesn’t support strong consistency which is really important. Maybe using throttling? Maybe accept only really higher bids than the current bid to reduce load? Maybe adding a “waiting queue” like we did in TicketMaster question?

Thank you!

Show more

0

Reply

C

ChristianPlumTakin500

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7a7knqa055g2azps6vcjnwz)

> but the feedback mentioned it was not enough

Did they say why it wasn't enough?

Show more

0

Reply

H

HumanEmeraldGiraffe139

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7cxas7c02qq19rhzmczk7fh)

No, just mentioning I didn’t handle this case good enough. I think they wanted me to suggest something creative like the virtual waiting queue in TicketMaster and/or using priority queue to prioritize high bids but this is just a guess

Show more

0

Reply

![Aman Thakkar](https://lh3.googleusercontent.com/a/ACg8ocLwwtnCae2aEApD8yZrf0xacaCP13PFYMjrKui7yq7amQ=s96-c)

Aman Thakkar

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7csqfsx02j3xkf7vsmr33qd)

damn. which level was this for? don't tell me E4

Show more

0

Reply

H

HumanEmeraldGiraffe139

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7cx8rcv02rhxkf7ri6tygsy)

For E4/E5

Show more

0

Reply

P

PleasantBronzeCougar977

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cma6b7c2z00buad08e9pu9r28)

They might have been looking for you to special case celebrities (similar to Twitter celebrity problem).

One issue with the queue approach here is head of line blocking on your writes. Suppose you have some ridiculous number of bids like 1M for a single item. Even if you scale, you need to process those 1M writes _sequentially_. Napkin math: 1e+6 ms -> ~16.67 minutes to process the whole thing. Bids on a different item in the same kafka partition would also get delayed.

Possible solutions: we special case and weaken requirements for celebrities/items with large bid spikes. Specifically, we won't show the winning bid in realtime, and instead compute this offline. Write to a special queue -> use that queue for a total ordering of bids -> consume and compute offline with something like Spark.

If they don't like weakening the requirement, you could consider using the cache as a way to know whether to use a transaction. If the bid is < cache amount, we know that bid is not the winner. So there is no need to lock the Auction table in this case, which could speed things up significantly

Show more

0

Reply

![E Z](https://lh3.googleusercontent.com/a/ACg8ocIvzX0SuEb-25SaNWiD-Ye0PAgT4B_Bjg2gbwo6kHyq995G5U8=s96-c)

E Z

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7406lya00vq88mxcr3qncso)

For whoever does not understand the visibility timeout approach for dealing with 1 hour auction auto-ending.

one example: 1 am: a new bid 1 is sent to sqs, and processed by a worker but the worker does not acknowledge sqs. this bid will disappear in the sqs at 1AM and will re-appear in the sqs at 2 am.

2 AM: worker sees bid 1 again and compares the request time of bid 1 (which is 1 am), and the latest bid time in the database (which is also 1 am) , if they are the same, the worker will end the auction.

challenges: how do we handle bid 1 everytime a worker processes it? For the first time, the worker will not acknowledge sqs, but the 2nd time? If we do not acknowledge, the bid 1 will re-appear over and over again in the sqs, resulting in unnecessary resource waste. If we acknowledge for the 2nd time, how does the worker know it is the 2nd time that the worker sees this message? Do we have a counter in database to track the count?

Show more

0

Reply

![Akshay Joshi](https://lh3.googleusercontent.com/a/ACg8ocJf1MkU7xawrrM4uCUeYeLXMJzfl0jvK5o1D6u1_wdsls57iw=s96-c)

Akshay Joshi

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7aql7f705lcl7b11v5vew88)

I have many doubts

1. If update of max\_bid failes in Aution table, the retry mechanism has to be initiated again by the bidder right? The system will not retry on behalf.
    
2. The statement "we need to broadcast it to all the clients that are watching that auction", Why do we have to update the clients for all the new bids, should'nt we just update the max\_bid to all the clients? Can you please elaborate more on the flow where redis pub/sub will be used in this scenario where we have to first check if the bid is the maximum value bid from DB and then use the redis pub/sub to notify all.
    
3. What should be DB choices because this is asked very frequently in interviews.
    

Show more

0

Reply

![Aman](https://lh3.googleusercontent.com/a/ACg8ocIL1sf1GqL1Y0Cdy-1dChmUWHTiLSbcMpqcT_x3eMdwF93fQ3ev=s96-c)

Aman

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7cds5em00euab605inms0h5)

Small query we introduce Kafka as fault tolerant step in our system but API gate way again relying on the producer to send the message to Kafka, in a hypothetical scenario what will the behavior if producer went down and message doesn't even went to Kafka?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7cexxt802006mflmetob8k1)

We'd send an error back to the client and not register the bid. Generally, you make those services as lightweight as possible and scale aggressively to prevent this.

Show more

1

Reply

![Aman Thakkar](https://lh3.googleusercontent.com/a/ACg8ocLwwtnCae2aEApD8yZrf0xacaCP13PFYMjrKui7yq7amQ=s96-c)

Aman Thakkar

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7cru61f02hxxkf71ucgek96)

Hey Evan - can't we store current winner in cache along with maxBid? Insert into DB for each bid that comes in. Simultaneously write to cache a hashmap using HSET. Use a LUA script to write {maxBid, currentWinnerId } only when currentBid > maxBid. userIds here could be maybe an alias that is set for this particular auction to avoid poor practices for storing userIds in cache.

The various servers can be subscribed to this event and then use SSE to inform to all users on the UI. UI can then show the winner to the logged in user based on information about logged in user and winnerId stored in the cache, sent via SSE. Redis can have AOF for durability.

There is an edge case where there could be another user with the same bid as maxBid but not the earlier one. There is a case to be made where UI can show the user that he was late, and should bid again. I think that scenario isn't really addressed in the write up as well though. We could also, for bids that do not update redis, directly reject them and show failures to UI.

Show more

1

Reply

N

nimish.jindal

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7dtmsvu00ukv31cvhbqvaru)

A question around Server Sent events (SSE). It seems most API gateways (like the ones managed by aws and azure) don't support SSE (they will timeout the connection). Even your diagram shows that the SSE connection bypasses the Gateway. How does this work? How does the client know which service to connect to? How is this SSE connection secured if it bypasses the gateway?

Show more

0

Reply

D

drover\_bookish3s

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7ehs4w900ywu2r1odg4jv4g)

If the message queue, partitioned by AuctionId, is already consumed by the bid service, why can't this consumer in the Bid service do both - 1) update the DB, and 2) send SSE? Does partitioning the queue by AuctionId mean that _all_ the messages for a given auction Id will be consumed by a single server?

Show more

0

Reply

![Mert Akozcan](https://lh3.googleusercontent.com/a/ACg8ocKSxI0C2qovG213b5d3uEKDzt1coTLC7Kbz11UVUf1Ea9ouDC1W=s96-c)

Mert Akozcan

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7i5gs8m030eimqn7phm22wi)

> Does partitioning the queue by AuctionId mean that all the messages for a given auction Id will be consumed by a single server?

Yes, that's how Kafka works. In a given consumer group, only one consumer app/process/thread will be assigned to a partition. In other words, there can't be more than one consumer app/process/thread consuming from the same partition. (However, one consumer app/process/thread can be responsible for multiple partitions.)

Show more

0

Reply

![Aditya Jain](https://lh3.googleusercontent.com/a/ACg8ocJjHh-eky22nfbV7YSOwPct6ROk615alDtDMGafjdNJV1gQcQ=s96-c)

Aditya Jain

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7fhj26c006y10fsg8cdvyde)

To maintain strong consistency while bidding, we can lock the row (with OCC) in DynamoDb and update the max\_bid attribute for an auction. Using DynamoDB streams with some batch window and time period (can be analysed via traffic pattern) , we trigger the lambda to record the bids in the bid table.

Show more

0

Reply

![Leonardo Zhu](https://lh3.googleusercontent.com/a/ACg8ocJipESw5zoYb9E7EzMeLU8FhlKBXd2kcO_DIDcqc3r43yMOqxhv=s96-c)

Leonardo Zhu

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7g1g8tl0053imqnyf0e8503)

I don't quite understand that why Kafka improve the durability. I know that Kafka immediately write down things into the disk, but what if Kafka itself goes down? We still need replication for Kafka to prevent this happen, right? And also the producer next to the API gateway, does it also need replication to ensure durability? By adding those two components and replications, why don't we just set up some replica for the Bid Service? Considering other benifits like Buffering Against Load Spikes and Guaranteed Ordering, I understand it is better to have Kafka but does it acutally provide more fault tolerance?

Show more

0

Reply

![Mert Akozcan](https://lh3.googleusercontent.com/a/ACg8ocKSxI0C2qovG213b5d3uEKDzt1coTLC7Kbz11UVUf1Ea9ouDC1W=s96-c)

Mert Akozcan

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7i522zv032gmbkqhq7iym1a)

I think using Kafka somewhat contradicts with what you said in the strong consistency discussion. After writing the bid to Kafka, we are telling user that their bid is received. However, we don't know whether that bid is actually valid, and we can only understand this after consuming the bid from Kafka and trying to update the database.

In the end, I think we will have an eventually consistent system and it's not much different than accepting all bids and figuring out which one is the highest asynchronously.

And maybe one more thing: When we go with a Kafka-based approach, we probably won't need OCC, because there will only be one consumer responsible for a given Kafka partition, and since all bids for a given auction are flowing through the same partition, we won't have any race conditions while updating the database with highest bid price.

Show more

7

Reply

![Hikaru](https://lh3.googleusercontent.com/a/ACg8ocLgfTAEX9Goi4MmPpHJNLgXN8IJaz3onPTgcj3EhJG2ECWu6w=s96-c)

Hikaru

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8xcxaem003zad08p7l5s90b)

It seems to me there must be some tradeoff for this problem.

If we don't use Kafka (or if we only use Kafka to record bids history, and use a Redis to record highest bids without going to Kafka), there is risk of dropping bids, especially for popular auctions. The the fault tolerant and durable requirement might not be satisfied.

Also, if we only use Redis, Redis will still need to propagate the new data to other replicas, which takes time (especially Redis replicas are deployed in different data centers). Therefore, in the meantime, some read requests will still get stale highest bid sometimes if they read from those read replicas. Unless we force all read requests to go to the leader node, which breaks our scalability requirement.

What do you think? Any idea how to solve the issue?

Show more

0

Reply

L

LogicalAmethystToad166

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9u9jg3s00oead08w4dts3lz)

This is a valid point actually.. I would be interested in hearing what Evan has to say here as it may help many readers. Thank you!

Show more

0

Reply

![Dolly Singh](https://lh3.googleusercontent.com/a/ACg8ocJStoNwfe3CfMDYfZ-xYTyHqA4kNOmkYyVo0C-SsIrCeggbE7sF=s96-c)

Dolly Singh

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmckrtrd700s7ad08658fogid)

Thats correct that this is not Stongly consistent, but i beleive this is Sequentially consistent meaning relaxed and preserving predictable behavior. Link: https://hazelcast.com/blog/navigating-consistency-in-distributed-systems-choosing-the-right-trade-offs/ Evan's thought would be helpful.

Show more

0

Reply

![Hikaru](https://lh3.googleusercontent.com/a/ACg8ocLgfTAEX9Goi4MmPpHJNLgXN8IJaz3onPTgcj3EhJG2ECWu6w=s96-c)

Hikaru

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7pss8sp019gpqpvhvftzh57)

Hi Evan,

I think the OCC solution for the first deep dive issue, 'How can we ensure strong consistency for bids', mainly addresses the 'isolation' aspect of ACID. It ensures that multiple concurrent transactions are properly isolated to prevent race conditions like lost updates, but this is a 'single-node issue'. However, it doesn’t really solve the consistency problem as defined in the CAP theorem, which is more about stale data in DB read replicas (a 'multi-node consistency issue'). So while OCC can prevent race conditions, it doesn’t stop users from seeing an outdated max bid if they happen to read from a stale replica.

Therefore, I think the second approach (using Redis Lua script) to this issue is a better solution because it solves both the race condition (by atomicity of Lua script and single threaded nature of Redis) as well as the stale data issue.

Did I misunderstand something? Let me know what you think.

Thanks

Show more

0

Reply

![Hikaru](https://lh3.googleusercontent.com/a/ACg8ocLgfTAEX9Goi4MmPpHJNLgXN8IJaz3onPTgcj3EhJG2ECWu6w=s96-c)

Hikaru

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7smeckv01z2ciytwhgvpxp7)

Update: After careful consideration, I believe the 2nd approach (using Redis Lua script) has a similar issue since Redis also needs to replicate the data from write replica to read replicas so there will be lag.

Also, the 2nd approach has consistency issue between Redis and DB. (I originally thought we update Redis first then asynchronously update to DB).

It seems like there is no simple approach to fix the read inconsistency issue, unless we force all reads for max bid go to the write replica, which is not scalable.

Show more

0

Reply

B

BriefBeigeWoodpecker393

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7pw6g6o01d4joxcwjnc2mw9)

"For implementation, we'll use Kafka as our message queue. While other solutions like RabbitMQ or AWS SQS would work..."

This is not wrong. A minor note, standard AWS SQS does not guarantee order. You want to use FIFO queue or AWS Kinesis Stream (~Kafka).

Show more

0

Reply

![Hikaru](https://lh3.googleusercontent.com/a/ACg8ocLgfTAEX9Goi4MmPpHJNLgXN8IJaz3onPTgcj3EhJG2ECWu6w=s96-c)

Hikaru

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7sjomwp01vz10vd2hmssrj9)

How can we prevent hot partition issues in Kafka? If we use auction ID as the partition key and an auction gets a lot of traffic, we might run into a hot partition problem.

One possible solution is using a compound or salted key to distribute the load, but that could break bid ordering for the auction.

Are there any other ways to handle this, or is it reasonable to accept slight ordering inconsistencies across bidders?

Show more

0

Reply

![Erick Takeshi Mine Rezende](https://lh3.googleusercontent.com/a/ACg8ocK2cjXUIj2mB69_Hc5ZVesE8etdmqFpSR6rfbYMnbwPzc6oy_UH=s96-c)

Erick Takeshi Mine Rezende

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm7wtuwl501tkstkwa1m8sg79)

I would say that this assumption "OCC is ideal for our system because bid conflicts are relatively rare (most bids won't happen simultaneously on the same auction)" is incorrect. I’m not stating this based on ebay auctions, but for Tibia character auctions, there is a TTL for the auction. Players would place their bids exactly 1 minute before the auction ends, usually a lot of players would do that, at the same time.

I would expect that an Ebay auction would have the effect, since almost all buyer would like to buy the item for the cheapest price.

So, we would need to handle a large amount of bids in the same auction, and using a DB level lock in a SQL database, would not be ideal and, potentially, would not handle the problem well.

I would say that redlock or zookeeper lock is a better strategy for implementing this locking feature, but I don't know.

Any thoughts on that?

Show more

1

Reply

![Hikaru](https://lh3.googleusercontent.com/a/ACg8ocLgfTAEX9Goi4MmPpHJNLgXN8IJaz3onPTgcj3EhJG2ECWu6w=s96-c)

Hikaru

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8wdyfhk00nqad08dlip9pe0)

I agree that sometimes there are bid conflicts. However, I think OCC handles the problem without row level locks. Also, by using Kafka, bids come into Kafka in order. And a Kafka consumer that is responsible for a Kafka partition picks up the bids for an auction in order (assuming we partition the bids based on auction IDs), so we should be able to avoid race condition when writing to DB.

Show more

0

Reply

![Thomas Liu](https://lh3.googleusercontent.com/a/ACg8ocKAXuTs1Ry2D5WVpO1yBNyNs65ZQbhNLB76l9xVzEZVaLesN8atEw=s96-c)

Thomas Liu

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm80wrm8v055ktvyjflibxj4u)

For durability of the DB, can we also use replication? Replaying from the task queue is probably more complex than restoring a failed DB master. in this case, we would need strong consistency among replications. (still i think the bidding queue is still useful for event driven and peak traffic handling)

Show more

0

Reply

E

EvolutionaryChocolateEchidna983

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm82gs6bx01yh444oftu4qydq)

How can we tell when an auction is over and notify the owner/bidder of that auction?

Show more

0

Reply

I

InjuredYellowTick667

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm833ss8p00ky148t09zq7afj)

great article, thankyou so much!

Show more

1

Reply

![he she](https://lh3.googleusercontent.com/a/ACg8ocKr0F309X3jNe30WX_3FRRvD3Scq7ohWH5P844nbLDtThZC=s96-c)

he she

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm83nbt1601os148tl2n3wrur)

why do we want to save each bids in the db table? If the purpose is just to keep the records, should it be a simple one line log in some files?

Show more

0

Reply

F

FinancialGreenTick391

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm84kpyr400ftn83rd9mohzz3)

Regarding deep dive 1 and the "bad solution". I'm not sure why this is a solution at all. If we lock the rows for writing, two concurrent transaction could still read all of these rows and write two new separated "accepted" rows. Could you please elaborate? Thank you.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8xms66r00ltad085cgh267h)

The row locking approach actually would prevent the scenario you described. When you do SELECT ... FOR UPDATE, it locks those rows for both reading and writing until the transaction commits. So if Transaction A locks the rows, Transaction B has to wait until Transaction A commits before it can even read the rows, let alone write a new bid.

Show more

0

Reply

S

SmoothSilverHippopotamus562

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm88gxzym000yzfxjsh9l7tgf)

Thanks for the amazing post Evan! Maybe a dumb question, regarding this statement - "All other instances of the Bid Service are subscribed to the same pub/sub channel and receive the message. ", why don't we have clients subscribe to the pub/sub channel directly?

Show more

0

Reply

![Brendan Robert](https://lh3.googleusercontent.com/a/ACg8ocJ4uBlrnNiooDQdc--fClet5qOjzegdHRirW4qXsm7rM5t-XMA=s96-c)

Brendan Robert

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8ag4ojs00p9o5ih27cq5rg5)

Why is a distributed lock not sufficient here but is for other strong consistency problems like uber driver match or ticket master? If we used a db that didnt support ACID and used a distributed lock, locking on the item id until the bid was processed, we would lose functionality if the lock went down but we could just fail all bids in that case to maintain consistency? I get using postgres and db level atomicity features works great i just wasnt sure what was fundamentally different here that would shy away from distributed locks

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8xmqhy400lmad08nzwkmvns)

No need to use a distributed lock when all you need to do is lock for the duration of a transaction, this is what RDBMS are built to do! Let em do it.

Show more

0

Reply

L

LogicalAmethystToad166

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9u9e2o900nwad08olwspjw0)

We wanted to use distributed-locks for TicketMaster/uber, because the lock itself is around 10 mins which is quite a lot..., but really good for that use-case.

In this example, the locking is milliseconds, and shouldn't be more than that.. Database row locking is perfect for such scenario where you need strong consistency and the locking is sub-milliseconds.

Show more

0

Reply

S

shashanksinghal1992

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8d5azvl01m811orfnbe7zi5)

Hi Evan, Great content . Just curious on how to handle popular auctions which might have 10M bidders. How will the load balancer make sure that all the 10M bidders connections are not going on a single SSE Server & is distributed? Also on a side note , should the bid service be responsible for also maintaining SSE connections with user or should there be other service with solely responsibility of passing messages via SSE connections?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8xmp1x200ldad084gw1skrx)

We need consistent hashing at the load balancer to distribute SSE connections across servers. Each client connection gets hashed to a specific server. When that server is at capacity, we add a new one and redistribute some of the connections. As for your second point - separating SSE connections into their own service adds complexity without much benefit. The bid service already needs to maintain state about bids, so having it handle the connections directly is simpler. If scale becomes an issue, we can split the bid service into read/write components, but that's premature until we hit those limits.

Show more

1

Reply

![Wang lei](https://lh3.googleusercontent.com/a/ACg8ocIBwcYDZiesH-WGea9evEz-VtPcpOiYrYCCdqZM0uHbfMdpWw=s96-c)

Wang lei

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm90bcmj801idad07chq6eszy)

> We need consistent hashing at the load balancer to distribute SSE connections across servers. Each client connection gets hashed to a specific server.

Hi Evan, what is used to hash client connection? if it is auction id, then all connections to the same auction will land to the same server, if there are 10M bidder, they will connect to single server and it is too much and not working.

Is that hinted-off an option, that is used in Cassandra? it uses consistent hashing to route the request to server, and also aware of workload of server, can route to the next healthy node if the hashed node is busy?

Show more

0

Reply

L

LogicalAmethystToad166

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9u9amed00nfad08148idn0w)

I believe the connections will be hashed by 'auction-id' and the hashed partitions will NOT just map to a single server.. it would be a set of servers (e.g., ZooKeeper would hold that info and API gateway could potentially refer to that for distributing), and we can use round-robin and/or least-connections type balancing there.. So, to your concern, a single auction-id would probably NOT go to a single server, but rather a deterministic set of servers when needed.

Show more

0

Reply

![Hershel Wathore](https://lh3.googleusercontent.com/a/ACg8ocKSXe0Rio5bMtCJ2TRYRwqggI9-V4DTjuMyRhGHUJ1sTw3saGFn=s96-c)

Hershel Wathore

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8hygryu00cftqednpmb8v72)

For dynamic auction times handled through the SQS method. I think we should be using DelaySeconds rather than visibilityTimeout. Visibility timeout prevents multiple consumers from reading the same message by making it invisible while a consumer processes. We want to delay when any consumer can view this message.

Show more

2

Reply

![Fred Boundou](https://lh3.googleusercontent.com/a/ACg8ocLoYV_4NV-vhXepYBJ2VKUy6SvRrjumdoKWCJdC9Lc7rPhhKvha=s96-c)

Fred Boundou

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8izptk9003kuxg8ezz8qljq)

Hi Evan! Thanks for your fantastic job. Is there a way to get access to all excalidraw URLs for all videos that are not on Youtube? Like that: Excalidraw used in the video: https://link.excalidraw.com/l/...

Show more

0

Reply

![Will Zhang](https://lh3.googleusercontent.com/a/ACg8ocK6IAnbqwWoPdtbBq_MFlyrS9vWTFxacrpqGnuCdQlk2r8HANk=s96-c)

Will Zhang

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8pki6ry00j2s2abo2kl3pr3)

Hey Evan, Thanks for having the amazing article, one question here: Since we already have message queue (potentially kafka) over there in front of bid service, why would we introduce another pub/sub system? All users in the same auction can be placed on the same server, and each server only subscribes the partition (partitioned by auctionId) it cares, get the max bid price and send the bid to the all of the connections in this server. Seems the existing message queue can handle that already. No?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8xmnyxl00l1ad08xh9f1ktw)

Different steps in the flow. The queue is all unconfirmed bids, the pub/sub is broadcasting confirmed bids

Show more

0

Reply

![prakhar](https://lh3.googleusercontent.com/a/ACg8ocKDXXoGteJDNOueUSTQRS4w_FcP_2LM2rmDgu5vmw2SK28XSQ=s96-c)

prakhar

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8s1hjbs001oad08q08xphq4)

I appreciate how you've included the code snippets in dropdowns. It makes the approach feel more concrete and practical rather than just high-level.

Show more

2

Reply

![prakhar](https://lh3.googleusercontent.com/a/ACg8ocKDXXoGteJDNOueUSTQRS4w_FcP_2LM2rmDgu5vmw2SK28XSQ=s96-c)

prakhar

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8s1inac001qad085nkdw8d1)

Would recommend adding more code in different languages and also adding it at more places

Show more

0

Reply

N

NormalAmethystAnglerfish726

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm8ze5fj000c2ad08eamxcgqj)

Hi Evan,

Could you please help me providing Kafka message sample here ? What I'm looking for is, in the Kafka message, we need some kind of information that would help the Bid Service to establish the SSE to the client. I'm not clear through the Kafka message, how that would look like.

Show more

0

Reply

![Wang lei](https://lh3.googleusercontent.com/a/ACg8ocIBwcYDZiesH-WGea9evEz-VtPcpOiYrYCCdqZM0uHbfMdpWw=s96-c)

Wang lei

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm90ak8lu01ghad0762ym8hkq)

For Long polling for max bid, is it assumed to have single server in that deep dive? if there are multiple servers, we also need broadcast mechanism like pub/sub to broadcast higher bid from one server to other relevant servers. Is that correct?

Show more

0

Reply

W

WilyPeachLocust414

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm90n8oys01p4ad08pmjiu9uc)

How will the "bid service" be able to send SSE events to the client if the client establishes an HTTP connection with the "producer" service, not with the "bid service"?

Show more

1

Reply

L

LogicalAmethystToad166

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9romlwy006wad086p1efp7w)

I was thinking the same! I think it should be SSE connection to the "Auction Creation/Viewing" Service! Basically, the publishing to Redis Pub/Sub should still be through the "Bidding Service", but the subscriber itself would be the "Auction Service" basically!

Show more

0

Reply

H

HandsomeIvoryCrow799

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmahmr7hp004zad07cbh97z3a)

Exactly, even if we can somehow establish the SSE connection with the bid service directly, it'd defeat the purpose of setting up the Kafka queue for fault-tolerance in the first place: How can we secure enough bid service to establish SSE connection during traffic spikes?

I guess that the SSE discussion was done considering only the getBids path where it's perfectly acceptable to set up a direct SSE connection with the bid service. But on the createBid path, we'd need to setup the SSE connection with the producer service instead.

Considering that we'd need to setup the SSE connection and Redis subscription on the producer servers, I wonder if doing so still make the producer servers lightweight and cheap comparing to bid service?

Show more

0

Reply

M

MassAmaranthCarp180

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm91xlmi700fcad07qvqnepkt)

@Evan The video was not detailed looked like you were in a rush, I have followed all your youtube videos and the amount of depth you go in them is definitely much better than this. Not sure about the reason but just a little feedback

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9219m2k00hbad08c4gm68yv)

Shoot! Bummed you felt that way. Definitely not the intention. Were there places in particular you wanted more depth? Ended up being the same length as all the others.

Show more

0

Reply

![Neeraj jain](https://lh3.googleusercontent.com/a/ACg8ocJAzrMQriHY2NYkdaOuS5_-r2zdrYIXz-3bABYqasyyHNOvtik=s96-c)

Neeraj jain

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm92ogkbp00i7ad0855h14b79)

For the bad solution I think the correct row locking with bids query is

BEGIN;

-- Step 1: Lock all rows for this auction to prevent race conditions
SELECT MAX(amount) AS max\_bid 
FROM bids 
WHERE auction\_id = :auction\_id 
FOR UPDATE;

-- Step 2: Insert the bid only if it is higher than the max bid
INSERT INTO bids (auction\_id, user\_id, amount, bid\_time, status)
VALUES (:auction\_id, :user\_id, :amount, NOW(), 
        CASE 
            WHEN :amount > COALESCE((SELECT MAX(amount) FROM bids WHERE auction\_id = :auction\_id), 0) 
            THEN 'accepted' 
            ELSE 'rejected' 
        END)
RETURNING id, status;

COMMIT;

Show more

0

Reply

![K D](https://lh3.googleusercontent.com/a/ACg8ocKzqLcyqOapAuxHbMpsYTo_-zD0HTvshGEtrXS76m0x20rmHA=s96-c)

K D

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm94t6bi901ipad08dd1z9hyj)

How postgres is a good option? If we have 10M auctions and 1000 bids on average for each auction where each bid requires 1KB, it will be like 10 TB data just for bids!

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm94t920o01ldad087zll28zp)

1000 bids per auction sounds way too high to me. So too 1kb. It’s a couple hundred bytes per bid and maybe 100 per auction (even that’s likely high). That and Postgres can handle 100+ TB in a single instance on the right hardware. You’d still want to shard, but this isn’t a ton of data.

Show more

0

Reply

![Jagrit](https://lh3.googleusercontent.com/a/ACg8ocLVQF2_4KCJMgtZ0FF2xVqiW2qacI3u57liReHgnXzKXfc-iOwX=s96-c)

Jagrit

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm962sild02ucad08aesai1b5)

Good to see video on premium tab! 2 years down the line, we may need a video on HOW TO DESIGN HELLO INTERVIEW - as it has video, comments, drawing all the functionality. Huge respect to you!

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm962v6aw031nad08to6wuwid)

Haha it’s true, much of what we talk about we use directly to build Hello Interview and it’s a tone of fun! Real-time updates, peer-to-peer, chat, workflow orchestration, etc!

Show more

1

Reply

C

ComparableOliveFox759

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm96lodq3002kad0885emg2y5)

Evan, thank you for your write-up. I have a few suggestions.

I noticed that in most of your post, you present row locking as a bad pattern. However, OCC is not magical. As someone who has worked with databases for years, I can say that most systems start with OCC and eventually add support for PCC. Bidding systems can experience significant contention, and in this case, a PCC approach would likely be more effective. OCC rollbacks can be quite expensive under high contention.

Another suggestion: I believe you meant "serializability" when referring to strong consistency. In this context, snapshot isolation might be a better fit. It’s acceptable to see slightly stale bids with some delay, as long as the final state is correct.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9a7ggzx00m7ad08q2deu2sv)

Oh wait, to be clear, pessimistic locking is great! It is NOT great when you are locking for a period of time longer than a transaction or if you are locking lots of rows. In this design, pessimistic locking all bid rows for an auction is not great. But Pessimistic locking only the auction row (with a cached maxbid) is!

Show more

0

Reply

![Akash Anand](https://lh3.googleusercontent.com/a/ACg8ocIY2mQcDvJIl1uVdt9I1WYQFRhCJ9yI74HDgzKoovEGIqbY0ry-=s96-c)

Akash Anand

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm97gqqvm00apad08pfd9c9nv)

You brought up pessimistic locking during the initial deep dive, where you were updating the maxBid column of a single auction row within a single transaction. However, I believe that when updating the maxBid column, we might also need to update the data in the Bids table. If the write to the Bids table fails but the write to the Auction table succeeds, we could end up with an inconsistency. In that case, I think we may need to implement two-phase locking to handle this situation.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9a7h0ek00mbad08gq44yntq)

No need for two-phase locking here. The transaction already handles this atomically. We update both the Auction and Bids tables within a single transaction - if either write fails, the entire transaction rolls back. Two-phase locking is for distributed transactions across different databases/systems, which isn't what we're doing here. We're just using a standard ACID transaction within a single database.

Show more

0

Reply

L

LogicalAmethystToad166

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9rofj31006dad08wd5kv9bs)

As long as both tables' partitions (e.g., sharded by auction id) are located in the same Postgres instance, it will be a single atomic transaction (handled by the same transaction manager). I believe Evan mentioned that they are indeed partitioned by the same auction id, solving that issue.

If they were not under the same partition, then it wouldn't be a single transaction right Evan?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9roo4ic003tad08indmjhm5)

That's right, ya

Show more

1

Reply

F

FancyTanGoldfish347

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9ewxcam00ciad08w2i0ti8e)

At first I did think 2PL is required (we definitely want to avoid it as much as possible) to avoid missing writes since transactions don't guarantee a stronger degree of isolation (e.g. only one transaction can act on a row at a time). However, in the deep dive, Evan mentioned we use SELECT ... FOR UPDATE to update the max bid for the Auction which already locks other transactions out from updating the same row (the query returns the max so it locks all existing rows that fit in the SELECT MAX() query). However, I do think we need to update the isolation level to repeatable-read instead of read-commited to avoid the non-repeatable read issue.

Show more

0

Reply

V

VastAquamarineMeerkat851

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm99pthlq0075ad08n8n6f10u)

How to deal with SSE listeners in practice when redis pub/sub only supports at most once semantics. This means some of the bid notifications will never reach the clients who care about them.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9a7hosc00mfad080vox7671)

Redis at-most-once delivery isn't actually an issue here. The bid itself is durably stored in the database, and Redis pub/sub is just for real-time notifications. If a client misses a notification, they'll still get the correct state on page load or through their normal polling fallback. Plus, Redis pub/sub is actually pretty reliable in practice - packet loss is rare in a properly configured datacenter network. If you really need stronger guarantees, you could use Kafka instead, but that's probably overkill for what's essentially just a UI enhancement.

Show more

1

Reply

![Yevgeny Yurgenson](https://lh3.googleusercontent.com/a/ACg8ocJZqbSRbP7C3OcOtui-35AaLpuBZ3TTEBgb8dkWRk36iT3qQQ=s96-c)

Yevgeny Yurgenson

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9c0gcev003bad08rcfyqr1c)

Seems like you would just be able to SELECT max(bid) FROM bids FOR UPDATE to only lock on the last row, instead of caching it on the auction table?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9c0o4520036ad07albbnl7o)

I'm pretty sure SELECT MAX(bid) FOR UPDATE would lock all rows used to compute the aggregate, so same issue.

Show more

2

Reply

![Tirth Nileshbhai Patel](https://lh3.googleusercontent.com/a/ACg8ocLy7a2IQOQi_xL1qdkcseRXYnGyjTf79DBD5QjWrerIOIqzaQ=s96-c)

Tirth Nileshbhai Patel

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9f5ou4200lead08z8rms0ub)

Not sure if it’s a good question but can we use separate database for bids ? So we can scale it separately? I guess it will break the foreign key relationship. But if not here when do you want to use seperate database for writes generally? I guess we can DynamoDB as it’s managed and sharding is done at table level?

Show more

1

Reply

![Nisaanth Natarajan](https://lh3.googleusercontent.com/a/ACg8ocIQTJ_-xGKS8IDqsbLmuo3eaByhH-Ejcz4LYak0Lk_Go-uO_Q=s96-c)

Nisaanth Natarajan

[• 1 day ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmelyi9ef03rxad08a1aub8ol)

+1 on this question @Evan. I had this exact dilemma during my interview.

Show more

0

Reply

V

VoluntaryBronzeMarmot509

[• 1 day ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmelyk4cx03sqad08tjt9zo0u)

+1 on this @Evan. I had this exact dilemma during my interview.

Show more

0

Reply

![Dewan](https://lh3.googleusercontent.com/a/ACg8ocL4JI5POtYZ6kS5WwPv3AzL__vRZUOpIvv2Mj7lNSwaOIIGUA=s96-c)

Dewan

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9oqc5tp01pmad080npud18c)

Not sure if I missed it, but don't we also need to handle scaling the producer? And if we scale the producer service horizontally, it could break ordering of the bids. For example:

- User-1 bid on auction-1 at timestamp 1 and user-2 bid on auction-1 at timestamp 2, both for $10.
- User-1 request is send to host 1 of producer service and user2 request is send to host2.
- Host1 had many requests so it took longer to process user1 bid, and host2 put user2 bid in Kafka first.
- In Kafka partition user-2's bid will be ordered first to consume?

Show more

0

Reply

![Rajan Jha](https://lh3.googleusercontent.com/a/ACg8ocJY3LlnnTwEvPjYFbGrj4257FEjvezoWp37LBcyGfE30QqNvA-T=s96-c)

Rajan Jha

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9rijauq00d3ad08gqudnn7p)

Initially, it was suggested that introducing a message queue like SQS purely for serialization was a bad design choice, and that an ACID-compliant database like Postgres should handle concurrency instead. However, we eventually introduced Kafka for fault tolerance and partitioned the stream by auctionId to ensure bids for the same auction are processed in order — allowing us to deterministically pick the first among equal bids.

To me, this seems equivalent to achieving serialization, just through Kafka instead of locks. So now, with ordering handled by Kafka and atomicity ensured via DB transactions, do we still need explicit serialization mechanisms like locks? Am I missing something?

Show more

0

Reply

R

RacialTurquoiseRabbit968

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cm9s49k6h00f9ad08kly02tfi)

Hi Evan, In the Practice Round, there is a question around 'how can you improve row level locking on writes to prevent deadlocks during a lot of writes' and the suggestion is 'sharding'. As the row level lock will only happen for a particular action row, how will sharding help in that case?

Show more

0

Reply

M

MildOliveAngelfish918

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cma151wqp01hlad08fq83dc5m)

Dumb question regarding the redis lock lasting 1s: can't the service just release the redis lock when it's done (presumably less than TTL)?

And how does serializing the results in a queue before the service help, if we shard the service out to scale? It seems like you just have the same issues.

Just thinking of how I'd answer this if I wanted to go that direction.

Show more

0

Reply

W

WiseAzureMacaw681

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cma1vwm0e023lad08mmxe0uzk)

You can avoid having separate read and then update statements in the optimistic locking solution by combining them into one atomic statement:

UPDATE auctions
SET max\_bid = :new\_bid
WHERE id = :auction\_id AND max\_bid < :new\_bid;

Then check the number of affected rows. If affected rows = 1, the bid is a success otherwise rejected

Show more

0

Reply

P

ParallelIvorySwallow280

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cma38wgpx00jxad08xgoby22t)

Can we use gossip protocol for the broadcasting of new bids?

Show more

0

Reply

![Arun prasath](https://lh3.googleusercontent.com/a/ACg8ocJwk6FK5NTjnp9SpWHD0wUa_6v0qHw3KQTdSOT5UZqZMUU85Q=s96-c)

Arun prasath

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cma3l3yoa013sad0882dwnz61)

@22:22 mins the -15 bid should be rejected(Evan says is correctly) the diagram is missing it.

Show more

0

Reply

![Khuong Nguyen (Kay)](https://lh3.googleusercontent.com/a/ACg8ocJwJNHhKXakqQF1r7VCNdIf87lhhubDXxF7gtJbzuDrpuELn64=s96-c)

Khuong Nguyen (Kay)

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cma6pv3f800mrad08lq05vnid)

OCC is great. But doesn't the message queue already handle the race condition issues, since we are partitioning our queue by auctionId?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmah8dq6s02moad081dzz4rod)

The message queue doesn't solve the race condition by itself. Even with partitioning by auctionId, if you have multiple consumers reading from the same partition, you can still get race conditions when they try to update the max bid in the database. You'd need either OCC, locks, or a single serial consumer.

Show more

1

Reply

P

PastLimeBug305

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmb4fx5oj02bgad077s3ztqvk)

In Kafka to read from the same partition using multiple consumers you will have to use low-level consumer API without consumer groups and therefore manage and coordinate offsets between your consumers yourself which is going to be a non-trivial task.

Without such coordination every bid in a partition will be duplicated by every consumer of the partition. While CC will guardrail from persisting duplicates, it is unnecessary write amplification and increased load on the database. Not good.

Ok, now let's say we want to avoid that and don't want to mess with offsets coordination and instead rely on Kafka consumer groups to do that for us automatically. Then this statement "So no need to even partition the queue" is wrong as without partitioning only one consumer will be consuming at a time and the rest will be in a standby mode effectively preventing the system from scaling up.

am I wrong?

Show more

0

Reply

![Ivan Atroshchenko](https://lh3.googleusercontent.com/a/ACg8ocKs2RdLYX_h3iuQWXlqtX5XXhV9XOGR24Ytb7Tq4fmMDCBRj68b=s96-c)

Ivan Atroshchenko

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cma9cl8zl01wuad0705tz53lu)

What is the purpose of splitting services if we don't split the database?) I does not make any sense, we can do it as monolith in this case, can't we?

Show more

0

Reply

![Jerry Lee](https://lh3.googleusercontent.com/a/ACg8ocIbEQKYtKTRVdMYDjQglbBaKfAONGWVkRVFBsM-p5kbIbjywg=s96-c)

Jerry Lee

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmacrm9ms00m5ad0801gwq1d8)

different read write patterns, u may have way more bid service calls than auction service. and after doing the math he know they can all fit in a single db. for simplicity and easier join query looks up his design opt to keep them in same table. There is always a tradeoff

Show more

0

Reply

S

sumsha18

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmaapr2g000bvad09umddyzp5)

I have a question regarding the need for pub/sub here. If we are partitioning based on the AuctionID, then all the bids from all the clients for that Auction should be handled by the same "Bid Server" who is the only consumer for that partition. So, in this case, all the clients for a particular auction would be connected over SSE to only that "Bid Server" and hence we don't need pub/sub. Could you please clarify it a bit more?

Show more

0

Reply

![Jerry Lee](https://lh3.googleusercontent.com/a/ACg8ocIbEQKYtKTRVdMYDjQglbBaKfAONGWVkRVFBsM-p5kbIbjywg=s96-c)

Jerry Lee

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmac2xsnn006cad09x7p9utrp)

i was thinking about the same thing. But the more i think about it, i think the problem lies in when you need to scale up. If bid server needs to increase, say its controlled by aws, currently there is no sharding strategy on it. That means, the number of the bit server will go up and new users may be connected to the new bid server. If so, your message still could be picked up by different bid servers not by only that "Bid server". Unless your system garantees that your bid server will not scale but this goes against of the designs. The system needs to scale to handle high throughput and hot servers. Hope this clarifies

Show more

0

Reply

S

sumsha18

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmac4snit009bad08186xb8r7)

Thanks for sharing your views. However, for scaling up the system, you still may not need to spread the bids for the same AuctionID across > 1 partition as this is usually done when we have hot partition problem due to massive number of requests for that partition (in this case AuctionID as was clarified by even Evan in the video). So, we may not have a case wherein there are millions of Bid requests for the same AuctionID (as every new Bid means price goes up) in which case we don't need to split the traffic, hence you may still be good using the same Bid server. You may still want to provide scalability to this design for increased number of auctions (i.e. > 10M concurrent auctions), however. in that case, we may add new Bid servers to handle those new auctions, however, for a particular auction, the story remains same.

Show more

0

Reply

![Jerry Lee](https://lh3.googleusercontent.com/a/ACg8ocIbEQKYtKTRVdMYDjQglbBaKfAONGWVkRVFBsM-p5kbIbjywg=s96-c)

Jerry Lee

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmacrd83100lpad08cx6pzjwq)

" we may add new Bid servers to handle those new auctions" afaik your api gateway or aws can not automatically be smart enough to route bids to the new bid server but keep existing bids(bidders) to the old server. What happens to a user log off and log back in and now they are very likely to be at the new server now since our api gateway has no logics around it. Thats why i think adding a pub/sub there can solve this. yeah its more overheads but we dont have to deal with customized routing.

Show more

0

Reply

![Jerry Lee](https://lh3.googleusercontent.com/a/ACg8ocIbEQKYtKTRVdMYDjQglbBaKfAONGWVkRVFBsM-p5kbIbjywg=s96-c)

Jerry Lee

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmac1g9az0039ad08xy9xk9cx)

PLease correct me if im wrong, adding kafka in front of the bid service not only make it asynchronous but also add delay to the bid cycle. So the user will have a bad experience for the last 10 - 15 seoncds of the auction because of this delay introduced. I believe a more lightweight queue like sqs is a better option here? In reality dealing with kafka, a 1-2 seoncds of delay is inevitable from my previous expereience. please advise

Show more

1

Reply

A

AddedLimeCarp593

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmagc15z201gdad08iqdovksw)

Instead of setting Postgres's default isolation level to SERIALIZABLE couldn't you just set the single transaction to SERIALIZABLE? I think this is possible with this syntax: BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

Show more

0

Reply

A

abrar.a.hussain

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmatvenq201anad0960rbu2um)

Yes you can also set the individual transaction as serializable but there's actually a tradeoff.

Explicit FOR...UPDATE: simple predictable behaviour, slight blocking during higher load SERIALIZABLE: You actually have to handle the retries yourself as the client (more annoying).

Show more

0

Reply

W

WorthwhileCrimsonOrangutan479

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmah3w37f01x5ad0851ylrjhf)

Hey Evan I dont know I got something wrong, but as we have a redis pub/sub, why we don't save the max bid number in it

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmah8cic702mkad082rmtukp5)

Redis pub/sub isn't storage, it's just a message bus for broadcasting events

Show more

0

Reply

![Yan Zhang](https://lh3.googleusercontent.com/a/ACg8ocLOOWVxeLjV4KNAwDV2ubDW7UiOsagbqu8Lv7K3uCUIoI4Am-h5=s96-c)

Yan Zhang

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmahgey1j02luad07auhprvwu)

Oh I mean the redis cache DB could be used as both pub/sub and database to store the max bid.

Show more

0

Reply

I

ImprovedMagentaKoala455

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmamu130d00u3ad081iszg0ic)

"As is the case with almost all stateless services, we can horizontally scale these by adding more servers." But isn't Bid service stateful? because of SSE connections

Show more

1

Reply

![Shubham Goel](https://lh3.googleusercontent.com/a/ACg8ocJl-K7saVxE1eD3fVQYVU74W_RKrwUdc68C0mB0QZt90xkkcHgc=s96-c)

Shubham Goel

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmany63ni0232ad08myn5j3gq)

BEGIN;

WITH current\_max AS (
    SELECT MAX(amount) AS max\_bid
    FROM bids
    WHERE auction\_id = :auction\_id
    FOR UPDATE
)
INSERT INTO bids (auction\_id, user\_id, amount, bid\_time)
SELECT :auction\_id, :user\_id, :amount, NOW()
FROM current\_max
WHERE :amount > COALESCE(max\_bid, 0)
RETURNING id;

COMMIT;

This can lead to two concurent inserts

Show more

0

Reply

A

abrar.a.hussain

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmaqub6h200abad08o3ew9goz)

I'm trying to understand why using Redis would be worse here when comparing to the Tinder problem. So for Tinder we use Redis to handle the high write throughput + serialization requirements we had for matching. Don't we have something similar for this question? I'm thinking the difference is specifically then from scale?

Because for Tinder you did also have the option to use a distributed SQL store where you used info like

swipe\_id user\_1 user\_2 user\_1\_likes\_user2\_ user\_2\_likes\_user1

Then we used row-level locks to make sure we never lost a matching pair. Is avoiding Redis specifically because of concerns for DB write failures? In Tinder we had matching and then the entries for a specific match is done and partial failures with writing to disk doesn't affect future matches. Here in this case with the auction if you had auction\_id:max\_bid in Redis then writing to the bid table after the fact and having a failure leaves everything in some temporary inconsistent state. But future writes to look into max\_bid on a specific auction in Redis should still see the relevant correct data shouldn't they? So if you have max\_bid (10) -> writes to DB -> max\_bid(15) -> fails write to DB -> max\_bid(20) update is still functioning.

Show more

0

Reply

![Sudhanshu](https://lh3.googleusercontent.com/a/ACg8ocLNhPeK2pYoUwd2DcMaipO6CWYcGvCOODBZnu3Nh2h6RlFS7qfwSA=s96-c)

Sudhanshu

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmb1cypbi0288ad08zll6sfqx)

How do we handle strong consistency for cases where 2 bids are put in at same value around same time?

The problem I find over here is that the person who puts in the bid first should get the priority and the other one should receive reject. How will we modify our system to handle this? The only approach I can think of is the put in queue and have a single bid service instance read from that queue. Let me know if there's a better way.

Show more

0

Reply

![Cathy Liu](https://lh3.googleusercontent.com/a/ACg8ocJmCvdH_IsRWWU27J-ArQYMkSqqhNX8V_Mv8kIlMZvqDcNPBofk=s96-c)

Cathy Liu

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmb2ltlki00fdad08gmc4rlkm)

yes, there is a better way, for example, Google uses the concept of true time https://cloud.google.com/spanner/docs/true-time-external-consistency If strong consistency is that important to the business, invest in an atomic clock.

Show more

0

Reply

![Sudhanshu](https://lh3.googleusercontent.com/a/ACg8ocLNhPeK2pYoUwd2DcMaipO6CWYcGvCOODBZnu3Nh2h6RlFS7qfwSA=s96-c)

Sudhanshu

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmb3xqudq01wdad08upliawzu)

How will we leverage this and incorporate into our current design? Let's say two bid landed of $20 landed on two different bid service. The bid service which received a bid with an earlier timestamp got in GC or stuck in other process whereas the other bid went through. In this case, ordering is screwed. How are we suppose to leverage spanner in this case?

Show more

0

Reply

W

WilfulAzureFox189

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmb63fene007yad08v6zlejfs)

For ensuring Auction and Bid tables are updated simultaneously, won't POSTGRES REPEATABLE READ isolation level be sufficient?

Show more

0

Reply

W

WilfulAzureFox189

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmb642xuk0085ad0888528qrk)

Comment on video , you can set transaction level of single transaction as well BEGIN ISOLATION LEVEL REPEATABLE READ statement ...

Show more

2

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbb11vwq006pad076wy7d35c)

hey , you havent covered how we ill notify user when a bid goes to message queue , that your bid request has been accepted

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbb1w1c0006oad08zo4r630r)

SSE

Show more

0

Reply

![Purusharth Chandrayan](https://lh3.googleusercontent.com/a/ACg8ocJI4bLi2r1yWLhJ44n7pqs4Am23YHXyziuXXGQ-GmJzWto9kEn2=s96-c)

Purusharth Chandrayan

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbbj6c4r00ydad07484t4hp4)

In context to ensuring strong consistency for bids OCC solution, what if auction max\_bid updates successfully but writing bid operation fails?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbbj86wn00u1ad08qf6utwhk)

It's wrapped in a transaction. So it either all succeeds or all fails

Show more

0

Reply

![crimemaster gogo](https://lh3.googleusercontent.com/a/ACg8ocILbitJXS2jDtPHQCmqE4LpahlX_XZfKGFBr1jSDwSIBu1qzOtC=s96-c)

crimemaster gogo

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbeukm1703xbad07xsdax6xk)

How does SSE flow works now that there's an event stream in-between? now, the LB cannot create a symmetric connection to the bid service?

Show more

1

Reply

![Abhishek Khelge](https://lh3.googleusercontent.com/a/ACg8ocJbSgOwlcslOOSQu7MeEt9ayzSaXshpH9wRqOleYId6AnU8sQutrA=s96-c)

Abhishek Khelge

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbqrjk9o00xj08adk38xdb4m)

All CRUD applications will have some APIs that have more reads or more writes, we don't split there, why here ?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbqsew1300t408adri4w6at2)

Here is my take. The bidding service needs to handle 100x more traffic, requires strong consistency guarantees, and needs real-time updates to connected clients. The auction service is basically just a catalog. Trying to handle both in the same service means you can't optimize for either use case, and you end up with a service that's either over-provisioned for basic CRUD or under-provisioned for high-throughput bidding. Plus, the failure modes are different, if the auction service goes down, users can't create new auctions (annoying but ok). If the bidding service goes down, active auctions are frozen (catastrophic).

Show more

2

Reply

H

HorizontalCopperMosquito931

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbuncn9a060j08adtnim8wfb)

Why is the SSE connection maintained with the Bid service instead of the Auction service? It feels weird that initial read request for an auction goes to the auction service but we establish an SSE connection with the Bid service at the same time for future updates?

Is it because we introduced Kafka on the bid write path, so intend to use the same SSE connection to inform client whether bid was accepted/rejected after async processing of the write; and then re-use the same connection for any future updates to highest bid for the auction?

Show more

1

Reply

![Avisha Manik](https://lh3.googleusercontent.com/a/ACg8ocJHS9guvigl3EtMqgOWl_aiPAcon5XdoCnIxRwbPT00SgaK1ILO=s96-c)

Avisha Manik

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmbvso2va032b08ad6fuey1sm)

In the video, you talk about maintaining an in-memory config in BidService for all the persistent SSE connections to scale. I'm wondering if that'd make the system less fault tolerant, e.g. if one of the service instances was unavailable, would the client connectios not receive any bid updates? Or would the SSE connections get re-established with another service somehow? Would it be wise to introduce Zookeeper in addition to the Pub/Sub technology?

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcs3845q0abbad07z51tqklj)

if the server is up but the connection is broken, it can reestablished. if the server itself goes down, all clients are disconnected and shud reach out to LB with auction id as the key they are interested in, and LB shud check with ZK to redirect them to the correct server that should maintain those sse connections moving forward

Show more

0

Reply

D

dpschhina

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmc3ycj0y00ktad08n2y22syu)

How would the SSE work if the user refreshes the page for example?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmc3yfcbo00lcad08w898i07t)

SSE connections automatically reconnect on page refresh. it's built into the browser. No special handling needed on our end.

Show more

1

Reply

![Sourav Samanta](https://lh3.googleusercontent.com/a/ACg8ocL-c6eO_oqVkFPIYJf8rhtJ9gMgOqZopCaFCWKBeDzJBwNayg=s96-c)

Sourav Samanta

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmc5o86jl02dead0881od8vsa)

Would it make sense to consider storing the SSE connections in something like a Redis cache instead of using Pub/Sub? That way, any instance of the Bid Service receiving a bid could broadcast changes in the maxBid to all connected clients. This approach might also help handle scenarios where the server managing the SSE connections for a particular auction goes down.

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcs33dhj0a9had07vqd0hse8)

sse connections are not being stored in pub sub at all.

Show more

0

Reply

P

PhysicalTanLouse139

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcdpv5m8084mad08ia5k2qbz)

If an auction goes viral and there are thousands of bids happening concurrently, what is the mechanism to process all these bids with low latency and high throughput ? Would something like Apache Spark work in this case?

Do we also need to talk about Bid Administration framework and how bid start and end time will be enforced ?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcdq2b6208bdad089axkkc7w)

Apache Spark is probably overkill here. The article covers handling high throughput with Kafka + sharded databases. Also, thousands is not that much fwiw

Start and end times are just on the auction table and get checked via any transaction to add a new bid

Show more

0

Reply

P

PhysicalTanLouse139

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcdrq6ip08rtad08emj5diwz)

Thanks Evan

Show more

0

Reply

S

singerb

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmciv34cz04whad08ew6ooodt)

On the consistency topic, I'm just noting that in a staff interview I did I started with the non-functional requirement of strong consistency; we deep dived and talked about the scaling issues there, and the interviewer eventually pushed me to acknowledge that strong consistency is not actually needed; since we already need a mechanism to notify users when they get outbid, it doesn't actually matter that much if we get the consistency strictly right as long as we tell people that they're outbid, whether this happens almost instantly or later in the auction. So definitely be prepared to discuss this tradeoff, particularly from the user perspective.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcj8o94o06o1ad088bsaec7t)

Yah i do the same thing in Staff interviews. But always just push the other side. If they started without eventual consistency I would ask howd they'd do that and to weigh the tradeoffs.

See the callout in the above doc about this:

> There is an answer to this question which asserts that strong consistency is actually not necessary. I see this argument periodically from staff candidates. Their argument is that it doesn't matter if we accept both bids. We just need to rectify the client side by later telling User B that a bid came in higher than theirs and they're no longer winning. The reality is, whether User A's or User B's bid came in first is not important (unless they were for the same amount). The end result is the same; User A should win the auction. This argument is valid and requires careful consideration of client-side rendering and a process that waits for eventual consistency to settle before notifying any users of the ultimate outcome. While this is an interesting discussion, it largely dodges the complexity of the problem, so most interviewers will still ask that you solve for strong consistency.

Show more

0

Reply

![Huijing](https://lh3.googleusercontent.com/a/ACg8ocJeKPCg6fcYsUcK3OlW4168BbIQdT-dRlXO_2PE59-oWcZmgjU=s96-c)

Huijing

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcoh38rv03zzad08qnt92fkc)

How should an online auction system handle bid statuses as new, higher bids are placed?

For example, suppose the current highest bid is $10. A second user places a bid of $15, which is accepted and written to the database as "accepted." Then a third bid comes in at $20. Should the system update the status of the second bid to or "rejected"? what are best practices for managing bid statuses throughout the auction process?

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcs30xj60a83ad07eheq18cb)

it depends on how u want to design it u could keep the old bid row that said "accepted" unchanged even if higher bids came in later. in this case, your interpretation of that column would be "did this bid win (better all previous bids) when it was made

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcrb4izz03qhad07m3v1is5e)

my comment is off of video. moments before u introduce a queue, u said "mid level candidates introduce queue cuz they dont know a simpler/cleaner solution with DB locking already exists". but then u anyway pivot to this solution to support scale. and then, come the end of auction time when there could be flurry of requests, u not onlly use queues but needlessly lock rows in db?

edit: after adding kafka, whether locking rows in DB is needed or not depends upon how u partition kafka. if u partition kafka by auction\_id, then locking database is unnecessary. if u partition by bid\_id, then you definitely need to lock rows like u do

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcrbluwb03v1ad07uqp6oekr)

towards the end of ur video, i do appreciate the SSE logic you put but I dont like that Bid Services are doing too many unrelated things. maintaining open connections to clients and also acting as consumers of kafka. i would prefer to just create different servers for maintaining connections. same reasons as u give when creating different microservices. if one function goes down, u don't want it to affect the other. in this case, bid processing is way more important than communicating to clients.

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcs2vkoi0a6tad074xiel447)

i think polling is completely fine here for majority of auctions that dont have a lot of activity.

Show more

0

Reply

![Ronak Gupta](https://lh3.googleusercontent.com/a/ACg8ocLYnvLkE5NmslZUDNgizN_GfnHa-26NcpfsMZI67LVu_4tRyA=s96-c)

Ronak Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmcwzc6db0012ad0811eiw3cq)

Hi Evan, Can you please shed some info on this topic? I am confused reading this online. When you take a lock on the auction table for a row and issue read and write in a transaction, does it really prevent the other transaction from reading this row. What I have read is you are only taking a write lock however other isolated transaction can still read this row. If we use isolation level serializable, does it enforce that any locked row will not be available to even read.

Show more

0

Reply

L

LikeAmberDove772

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmczgbgr10ddkad08mssz5k8y)

Hi Evan,

I have a question for "cache max price" in auction table approach in deep dive section 1). My understanding is that the OCC and transaction approach will only work when auction and bid table are within same database. For the case where auction and bid table are in separate database, we might need distributed lock, right? Any other alternatives?

Show more

0

Reply

![Bhargav Mohanty](https://lh3.googleusercontent.com/a/ACg8ocIrb0lXjyX4Ccz6JcUwQPcRVo5Eb2HyvoUKyLHla6UNByMsE9Dc=s96-c)

Bhargav Mohanty

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmd3qt4ej01gpad088efdc2e4)

NIT: I think for your line " If the interviewer wants something more precise, you can add each new bid to a queue like SQS with a visibility timeout of an hour. "

1. i think you meant [delivery delay](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-delay-queues.html) here instead of [visibility timeout](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html)
2. The max delivery delay for SQS is [15 mins](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-delay-queues.html), so the hour delay might not work. (run every 5 mins instead?)

The problem might just be closer to a distributed job scheduler like airflow in your other write-ups.

Show more

1

Reply

F

FlyingTomatoHeron745

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmd40x7jc0l3cad08a0hsf6uc)

Nice write-up! shouldn't this also include/expand on "real time updates" pattern?

Show more

0

Reply

![Priyankar Raj gupta](https://lh3.googleusercontent.com/a/ACg8ocLj4znexnJYoaFwdkTmM26gju9vXeJeZHeGkBO0YPITob8d3Rsl=s96-c)

Priyankar Raj gupta

[• 29 days ago• edited 29 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmdj3gk5y011uad087px3wku0)

Not able to understand how concurrency control is needed when later we introduced the queue for sequential processing?

Show more

1

Reply

![Priyankar Raj gupta](https://lh3.googleusercontent.com/a/ACg8ocLj4znexnJYoaFwdkTmM26gju9vXeJeZHeGkBO0YPITob8d3Rsl=s96-c)

Priyankar Raj gupta

[• 27 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmdltns6h03luad08jepufrl1)

If user submitted a bid we immediately enqueue it in kafka. Now if immediately user wants to know status of bid (which should be processing or in queue), he can not do that because we have not persistent anything in DB.

User at this point is lost not knowing what happened to his bid -->reject/accepted/processing/waiting...nothing

This gives very bad user experience is in it ?

why do we care about OCC or locks when we later introduced queue ?

Show more

0

Reply

![Constantin Dogaru](https://lh3.googleusercontent.com/a/ACg8ocJBoc2tfi0rcNMv1v8ppAeqnvRbCNHSsXaUc35d0HJy0_2HLxvN=s96-c)

Constantin Dogaru

[• 26 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmdm166yn00lfad08rd4tvish)

First of all, thank you for this amazing walkthrough!

However, I don't get how the SSE connections would scale.

We have around 10M concurrent auctions and let's say around 100 concurrent users per auction. That's 1B concurrent users. How are we going to keep alive 1B concurrent connections in our server? In "numbers to know" section (https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know), you said a server can handle around 100,000 concurrent connections, which would imply that we need 10,000 servers to support this system. That sounds unfeasible to me and it's not something addressed in the design.

Show more

0

Reply

![Rishab Goyal](https://lh3.googleusercontent.com/a/ACg8ocIZDX-_NQiMW4loniXnt7qw4-H4Kqr7K-tBqbpUoSoECNmcmgqOEA=s96-c)

Rishab Goyal

[• 22 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmdssldok09e1ad08fkrxv68h)

Hi, I have question why not use DDB with strong consistency for TicketMaster and Auction bid systems, considering it's auto scale provides ACID with batch 25 or transactBatch across 10 tables in 1 go? Won't it more durable and scalable 5 years down the line, if our access patterns are served by elastic search for search and our tickets and ticket booking handled via redis locks? thoughts? I am inclined towards both , with SQL there we need to manage sharding and partitioning manually, operationally heavy, there is limit how many instance can be there in Amazon RDS cluster and also limit to max instance size. With DDB, we just need to correct modeling for access patterns and OCC will handle ACID. It's trade off decision for E5 Interview I have in 2 weeks, I am super confused with my own thoughts.

Evan or stefan or other folks thoughts on this.

Show more

0

Reply

![Sulagna Bal](https://lh3.googleusercontent.com/a/ACg8ocLA1uZR3CXjziYXHp_MkcKdvzuInD5Jq6frp2hPSVO2KGP-tQ=s96-c)

Sulagna Bal

[• 18 days ago• edited 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmdxwgdeh017rad086xpdl5sj)

"We can start with some basic throughput assumptions. We have 10M concurrent auctions, and each auction has ~100 bids. That's 1B bids per day. 1B / 100,000 (rounded seconds in day) = 10K bids per second" Its a lot that an auction receives 100bids per day. Do we have any real-time system like this?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme200p940067ad08njfb7epm)

It's definitely a lot for an average, but not for a max. Popular items could receive thousands of bids. Best to be on the high end when proving scale though.

Show more

0

Reply

R

RuralTanOx934

[• 16 days ago• edited 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme1eqs6e08g0ad08i0o09plk)

In optimistic concurrency control, is the operation to read current value , compare and update it atomic? What is that atomic operation called, is it a database version of test-and-set instruction?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme20k2w70001it9jy13m4eba)

The “read” isn’t part of the atomic step in OCC. The atomic bit is a single conditional UPDATE that says “only update if the version/old value still equals X.” The database guarantees that statement is atomic (via MVCC/ACID), so either it updates and bumps the version or it affects 0 rows and you retry. Conceptually this is compare-and-swap, not test-and-set.

Show more

0

Reply

![AMY](https://lh3.googleusercontent.com/a/ACg8ocI5pEy2ZmxEShspapPFQ6zVHaZyIakms7LgnuFEQjC9edtD-A=s96-c)

AMY

[• 15 days ago• edited 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme1zpjel001sad08o21ca1h5)

The system looks incomplete. It does not implement when auction ends, the highest bid wins. If there is no winner in an auction, what's the point of this bid system?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme1zywdt005ead083v6h91cf)

Thats why we have functional requirements. To scope a design to something that can be done in 35-45 minutes. If you compare this to a real auction system, then I agree, its incredibly incomplete. For more reasons than just not handling auction ending! :)

Show more

1

Reply

![AMY](https://lh3.googleusercontent.com/a/ACg8ocI5pEy2ZmxEShspapPFQ6zVHaZyIakms7LgnuFEQjC9edtD-A=s96-c)

AMY

[• 15 days ago• edited 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme20895b007rad08816z9r0b)

I'm thinking that in a real interview, what if the interviewer ask how to determine the winner. I just realize there is no good way to know when the auction ends. My proposal is every time there is a new auction, we can can schedule a workflow that will run at the end of the auction and then the system will look at the highest bid in the database and resolve tie by time. It is just my naive solution not sure if it the industry standard.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme209ci7006jad080to8k43f)

Yah! Great use case for workflow orchestration. Something like temporal.io https://www.hellointerview.com/learn/system-design/patterns/multi-step-processes

Show more

1

Reply

S

smishra

[• 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme22yglx011ead087hqput1b)

I feel Kafka's primary role is to ensure is fairness. Bids will be processed in the order that they were placed in the queue, which is better than the service picking randomly winners and losers based on which connections it managed to attend to first. If the service fails, all bids rejected due to a missed deadline will at least be rejected in the correct order. On ebay for popular items it is not uncommon to have a dozen bids arrive in the last second.

Another option is to have the producer be aware of auction deadlines. If the producer is aware of auction deadlines it can reject bids that are past that deadline. The bid service can then on its own time process any bids in the queue. The producer doesn't need to know all deadlines, only those coming up in say the next five seconds which might be a manageable number. Thoughts?

Show more

0

Reply

![Sid Khanna](https://lh3.googleusercontent.com/a/ACg8ocL0O_4eCgAsC4QmzlKqDVd9miKuiQx8-LaSM1xEHX7ObrBPqz_1=s96-c)

Sid Khanna

[• 12 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme61y7fu0o2tad07xs2ovg6y)

If say in the last 30 seconds we get 1000s of bid for an auction

- all the bids are going to be passed to to MQ
- then worker will pick up the bid and process it to our DB (Say worst case all bids are valid)
- after 30 seconds of the auction ending - the MQ bid service is still processing the bids

How is that aspect handled ? i.e. bid is coming to an end. and the MQ bids are still going on.

Would they not expire ? Or are we simply also checking the timestamp of when the bids are made and will just announce the winner of the auction with a slight delay ?

Show more

0

Reply

N

NetEmeraldLimpet205

[• 11 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cme86qq6m0171ad079rj9sgv3)

Great write-up! One thing I noticed about the number estimation, it appears that when estimating number of requests, the calculation is based on 100 bids per day per auction.

> We have 10M concurrent auctions, and each auction has ~100 bids. That's 1B bids per day.

However, when estimating storage, it becomes 100 bids per auction over a week.

> Let's consider our persistence layer. Starting with storage, we can round up and say each Auction is 1kb. We'll say each Bid is 500 bytes. If the average auction runs for a week, we'd have 10M \* 52 = 520M auctions per year. That's 520M \* (1kb + (0.5kb \* 100 bids per auction)) = 25 TB of storage per year.

Not sure if I missed anything, could someone please shed some light on this?

Show more

0

Reply

![Prakhar](https://lh3.googleusercontent.com/a/ACg8ocIBeDbPs_I9nVtL1gPF9L9vGtwzArmAaghIDiyzTKpY0CX-LoZtpQ=s96-c)

Prakhar

[• 8 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/online-auction#comment-cmec3iime01iqad08m5plb9rh)

What a phenomenal video and write up. Loved the deep dive into various concurrency controls offered by Postgres.

Show more

0

Reply
