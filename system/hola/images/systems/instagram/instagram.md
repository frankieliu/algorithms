# Design Instagram

Scaling Reads

Managing Long Running Tasks

Handling Large Blobs

[![Evan King](/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.66fdc8bf.png&w=96&q=75&dpl=e097d75362416d314ca97da7e72db8953ccb9c4d)

Evan King

Ex-Meta Staff Engineer

](https://www.linkedin.com/in/evan-king-40072280/)

hard

Published Feb 25, 2025

---

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

Hey, folks. Let's go ahead and walk through how we would design

0:00

Play

Mute

0%

0:00

/

1:05:10

Premium Content

Closed-Captions On

Chapters

Settings

AirPlay

Google Cast

Enter PiP

Enter Fullscreen

## Understanding the Problem

**📸 What is [Instagram](https://www.instagram.com/)?** Instagram is a social media platform primarily focused on visual content, allowing users to share photos and videos with their followers.

Designing Instagram is one of the most common system design interview questions asked not just at Meta, but across all FAANG and FAANG-adjacent companies. It has a lot of similarities with our breakdowns of [FB News Feed](https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-news-feed) and [Dropbox](https://www.hellointerview.com/learn/system-design/problem-breakdowns/dropbox), but given the popularity and demand, we decided this warranted its own breakdown.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements)

**Core Requirements**

1. Users should be able to create posts featuring photos, videos, and a simple caption.
    
2. Users should be able to follow other users.
    
3. Users should be able to see a chronological feed of posts from the users they follow.
    

**Below the line (out of scope):**

- Users should be able to like and comment on posts.
    
- Users should be able to search for users, hashtags, or locations.
    
- Users should be able to create and view stories (ephemeral content).
    
- Users should be able to go live (real-time video streaming).
    

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements)

If you're someone who often struggles to come up with your non-functional requirements, take a look at this list of [common non-functional requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements) that should be considered. Just remember, most systems are all these things (fault tolerant, scalable, etc) but your goal is to identify the unique characteristics that make this system challenging or unique.

Before defining your non-functional requirements in an interview, it's wise to inquire about the scale of the system as this will have a meaningful impact on your design. In this case, we'll be looking at a system with 500M DAU with 100M posts per day.

**Core Requirements**

1. The system should be [highly available](https://www.hellointerview.com/learn/system-design/deep-dives/cap-theorem), prioritizing availability of photos/videos over consistency (eventual consistency is fine, up to 2 minutes).
    
2. The system should deliver feed content with low latency (< 500ms end-to-end response time for feed requests).
    
3. The system should render photos and videos instantly (low latency media delivery).
    
4. The system should be scalable to support 500M DAU.
    

**Below the line (out of scope):**

- The system should be secure and protect user data (authentication, authorization, encryption).
    
- The system should be fault-tolerant and highly reliable (no data loss).
    
- The system should provide analytics on user behavior and engagement.
    

Here's how it might look on your whiteboard:

IG Requirements

Adding features that are out of scope is a "nice to have". It shows product thinking and gives your interviewer a chance to help you reprioritize based on what they want to see in the interview. That said, it's very much a nice to have. If additional features are not coming to you quickly, don't waste your time and move on.

## The Set Up

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

Let's start by identifying the core entities we'll need. I prefer to begin with a high-level overview before diving into specifics - this helps establish the foundation we'll build upon. That said, if you're someone who finds value in outlining the complete schema upfront with all columns and relationships defined, that's perfectly fine too! There's no single "right" approach - do what works best for you. The key is to have a clear understanding of the main building blocks we'll need.

To satisfy our key functional requirements, we'll need the following entities:

1. **User:** This entity will store user information like username, profile details, etc. (We'll keep it minimal for now).
    
2. **Post:** This will include a reference to the media file, a caption, and the user who created it. Crucially, a single Post entity will handle BOTH photos and videos.
    
3. **Media:** Represents the actual bytes of the media file. We'll use S3 for this.
    
4. **Follow:** Represents the relationship between two users, where one user (the follower) is following another user (the followee). This captures the uni-directional "follow" relationship.
    

In the actual interview, this can be as simple as a short list like this. Just make sure you talk through the entities with your interviewer to ensure you are on the same page.

IG Entities

### [API or System Interface](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#api-or-system-interface-5-minutes)

The API is the main way users will interact with Instagram. Defining it early helps us structure the rest of our design. We'll start simple and, as always, we can add more detail as we go. I'll just create one endpoint for each of our core requirements.

First, users need to create posts. We'll use a POST request for that:

`POST /posts -> postId {   "media": {photo or video bytes},   "caption": "My cool photo!", }`

We're going to handle the actual upload of the photo or video separately, using a pre-signed URL. I'll get into the details of why we do that later, when we talk about handling large files. For now, I'm just focusing on the basic API to create the post metadata. We can assume the client gets a postId back, which it'll use later.

Next, users need to follow other users. We'll model this as a relationship resource:

`POST /follows {   "followedId": "123" }`

The follower's ID will be extracted from the authentication token (JWT or session), so we don't need to specify it in the request body. This is both more secure and follows the principle of least privilege. In practice, it's fine to include here, you'll just need to compare it to the session before following.

Finally, users need to see their feed. This is a GET request:

`GET /feed?cursor={cursor}&limit={page_size} -> Post[]`

We'll use a cursor for pagination, and a limit to control the page size. I'm keeping the details of the posts array vague for now – we can fill that in later. The important thing is that we return an array of posts, and a next\_cursor for getting the next page.

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes)

For our high-level design, we're simply going to work one-by-one through our functional requirements.

### 1) Users should be able to create posts featuring photos, videos, and a simple caption

When a user is active on instagram, they should see a big button somewhere to create a post. Clicking it will take them to a new page or modal that asks for the media they want to upload and a caption which will trigger our POST /posts endpoint.

We can lay out the core components necessary to handle this request while making the initial assumption that our media is small enough to be uploaded directly via a single request.

Create Post

1. **Clients:** Users will interact with the system through the clients website or app. All client requests will be routed to the system's backend through an API Gateway.
    
2. **API Gateway:** This serves as an entry point for clients to access the different microservices of the system. It's primarily responsible for routing requests to the appropriate services but can also be configured to handle cross-cutting concerns like authentication, rate limiting, and logging.
    
3. **Post Service:** Our first microservice is responsible for handling post creation requests. It will receive the media and caption, store the post metadata in the database, the actual bytes on the media in a blob store, and return a postId to the client.
    
4. **Posts DB:** Stores metadata about each post including who created it, when it was created, the caption, and a link to the media in the blob store.
    
5. **Blob Store:** Stores the actual bytes of the media. We'll use S3 in our case as it's the most popular solution.
    

Quickly, let's go over what happens when a user uploads a post.

1. The client makes a POST request to the API Gateway with the media and caption.
    
2. The API Gateway routes the request to the Post Service.
    
3. The Post Service receives the media and caption, stores the post metadata in the database, and the actual bytes on the media in a blob store.
    
4. The Post Service returns a postId to the client.
    

Easy enough!

### 2) Users should be able to follow other users

Instagram is social! In order to see the posts of the people we follow, we need to be able to follow other users in the first place.

Importantly, this is a unidirectional relationship. For example, I follow @leomessi, but he (sadly) doesn't follow me back. Thus, I'll see his posts in my feed, but he won't see mine.

We can model this relationship with just a Followers table in our database that stores the followerId and followedId. Each time we receive a new POST /follows request, we'll insert a single new row into our table.

Followers

We've added a dedicated Follow Service to handle follow/unfollow operations separately from the Post Service. Since following users happens less frequently than posting and viewing content, this separation lets us optimize and scale each service based on its specific needs.

We get a lot of comments about people concerned about two services writing to the same database. While some claim that each service should own its data exclusively, this is arguably more academic than practical. Many production systems at FAANG and other large tech companies do have multiple services writing to the same database.

The tradeoffs are worth understanding:

Separate databases per service: ✅ Stronger encapsulation and isolation ✅ Independent scaling of database resources ✅ Freedom to choose different database technologies per service ❌ Requires distributed transactions or eventual consistency for cross-service operations ❌ Increases operational complexity (more databases to maintain) ❌ Potentially higher infrastructure costs

Shared database: ✅ Simpler transactions across domain boundaries ✅ Easier to maintain referential integrity ✅ Often simpler operationally and developmentally ❌ Tighter coupling between services ❌ Risk of one service's queries affecting another's performance ❌ Schema changes require more coordination

In practice, many organizations start with shared databases and evolve toward more isolated data stores as specific needs arise. For our Instagram design, having both the Post Service and Follow Service access the same database is a perfectly valid approach, especially since they're dealing with related domain concepts and the coupling between them is natural.

Show More

### 3) Users should be able to see a chronological feed of posts from the users they follow

Last up from our core functional requirements is the need to view a chronological feed of posts from the users we follow. Let's start with the simplest approach, and then we'll identify its limitations and improve it in our deep dives. Initially, we could:

1. Get followees: Query the Follow table to get a list of user\_ids that the current user follows.
    
2. Get Posts: For _each_ of those followed users, query the Post table to get their recent posts.
    
3. Merge and Sort: Combine all the retrieved posts and sort them chronologically (by timestamp or postId).
    
4. Return: Return the sorted posts to the client.
    

I'm going to opt to have the post service handle this for now rather than creating a new service, but there is no right or wrong answer. You're trading off between the complexity of the system and the cost of the operations.

Note that these queries would be incredibly slow if we needed to look through every single Followers row for every user and then search through every Post row to find the ones we want. To avoid these full table scans, we can add a few [indexes to our database](https://www.hellointerview.com/learn/system-design/deep-dives/db-indexing).

Since we're about to discuss indexing, it's an appropriate time to also choose a database technology. Given the scale, limited number of relationships, and the fact that eventual consistency is acceptable, I'm going to go with [DynamoDB](https://www.hellointerview.com/learn/system-design/deep-dives/dynamodb). That said, [PostgreSQL](https://www.hellointerview.com/learn/system-design/deep-dives/postgres) or most other databases would be equally valid choices, you'll just want to be able to justify your decision.

Fun fact: Instagram uses PostgreSQL as its main Post metadata database. This is interesting because if you were following the SQL vs NoSQL debates of yesteryear you may be convinced that a SQL DB could not scale to support our 500M DAU. Clearly not the case!

Back to indexing, given we're opting for DynamoDB, we'll need to add a few indexes to our tables to avoid full table scans. For the Follower table, we'll make the partition key the followerId and the sort key the followedId. This allows us to efficiently query for all users that a given user follows.

For the Post table, we'll make the partition key the userId since most of our queries will be to get the posts of a given user. We can make the sort key a composite of createdAt and postId to ensure chronological ordering while maintaining uniqueness.

Feed Generation

So here is what happens with our simple approach.

1. The client makes a GET request to the API Gateway with the cursor and limit.
    
2. The API Gateway routes the request to the Post Service.
    
3. The Post Service queries the Follow table to get the followed users of the current user.
    
4. The Post Service queries the Post table for each followed user to get their recent posts.
    
5. The Post Service combines and sorts the posts and returns them to the client, limited by the cursor and limit.
    

You're probably thinking, "wait a minute, what if I'm following thousands of people? This will be crazy slow, even with indexes, right?" And you're right! This is exactly the kind of problem we'll solve in our deep dives.

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes)

At this point, we have a basic, functioning system that satisfies the core functional requirements of Instagram - users can upload photos/videos, follow other users, and view a chronological feed. However, our current design has significant limitations, particularly around feed generation performance and media delivery at scale. Let's look back at our non-functional requirements and explore how we can improve our system to handle 500M DAU with low latency and global distribution.

### 1) The system should deliver feed content with low latency (< 500ms )

We'll start our deep dives right where we left off with our high-level design -- feed generation. Our current "fan-out on read" approach could work for a small app, but it's not going to scale to 500M DAU.

Let's first understand why in the context of the database we chose.

Our first concern would be that for users following 1,000+ accounts, we need 1,000+ queries to the Posts table to get posts from each of their followed accounts.

DynamoDB does offer batch operations, but each batch is limited to 100 items and 16MB of data. For a user following 1,000 accounts, we'd need at least 10 separate batch operations to query recent posts. While these can operate in parallel, it still creates several major problems.

1. Read Amplification: Every time a user refreshes their feed, we generate a large number of database reads. With 500M daily active users refreshing their feeds multiple times per day, this quickly becomes unsustainable. This is going to get expensive fast.
    
2. Repeated Work: If two users follow many of the same accounts (which is common), we're repeatedly querying for the same posts. At Instagram's scale, popular posts might be retrieved millions of times.
    
3. Unpredictable Performance: The latency of a feed request would vary dramatically based on how many accounts a user follows and how active those accounts are. This makes consistent performance nearly impossible.
    

Let's put this in perspective with some numbers:

- Each feed refresh might need to process 10,000 posts (1,000 followed accounts × 10 posts/day)
    
- With 500M DAU, if each user refreshes their feed 5 times daily, that's 2.5 billion feed generations per day
    
- During peak usage (e.g., evenings, major events), we might see 150,000+ feed requests per second
    

With these numbers, it's evident that even with a well-architected DynamoDB implementation, we'd struggle to maintain our 500ms latency target. We'd either:

1. Need to massively overprovision our database capacity, making the system prohibitively expensive, or
    
2. Accept higher latencies during peak usage, violating our requirements
    

This inefficiency is inherent to the fan-out on read model. The core issue is that we're postponing the computational work until read time, which is precisely when users expect immediate results.

###### Pattern: Scaling Reads

Instagram showcases the perfect **scaling reads** scenario - users view hundreds of posts daily but post only occasionally. This extreme read-to-write ratio demands sharding by user ID for posts, vertical partitioning for different data types (profiles, posts, analytics), and hierarchical storage for older content.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/scaling-reads)

Let's analyze some alternatives.

### 

Bad Solution: Simple Caching (Improvement on Fan-out on Read)

###### Approach

The most obvious improvement to the fan-out on read approach is adding a cache in front of the Posts table to cache each users recent posts. We can use Redis for this. The idea is simple: before querying the database for a user's followed users' posts, we check the cache. If the posts are in the cache, we return them. If not, we query the database and then store the results in the cache for future requests.

We can key the cache by a combination of the user\_id and a cursor (or timestamp). This lets us get a specific "page" of the feed from the cache. An example key would be: feed:{user\_id}:{cursor}. The value would be a list of postIds.

Feed Generation with Simple Cache

###### Challenges

Caching helps but doesn't fundamentally solve our scalability challenges. The cache needs to be massive to achieve a meaningful hit rate at Instagram's scale, and we still perform expensive fan-out reads to aggregate posts from all followed users for every feed request. While caching is a useful optimization, it's treating the symptom rather than addressing the root problem of our fan-out read architecture.

### 

Good Solution: Precompute Feeds (Fan-out on Write)

###### Approach

A much better approach is to precompute the feeds. Instead of generating the feed when the user requests it (fan-out on read), we generate it when a user posts (fan-out on write).

Here's how it works: When a user creates a new post, we query the Follows table to get all users who follow the posting user. For this to be efficient, we need a Global Secondary Index (GSI) on the Follows table with followedId as the partition key and followerId as the sort key. This lets us quickly find all followers of a user who just posted. For each follower, we prepend the new postId to their precomputed feed. This precomputed feed can be stored in a dedicated Feeds table (in DynamoDB, for example) or in a cache like Redis.

Let's use Redis. It provides very fast read and write access, perfect for feeds. We can store each user's feed as a sorted set (ZSET in Redis). The members are postIds, and the scores are timestamps (or postIds, if they are chronologically sortable).

Precomputed Feeds

**Data Model (Redis):**

- Key: feed:{user\_id}
    
- Type: Sorted Set (ZSET)
    
- Members: postId
    
- Scores: timestamp (or postId)
    

When a user requests their feed, we read the top N posts from their sorted set in Redis, which is a single, fast operation. However, we still need to hydrate the posts based on these postIds. To do this we have 3 options:

1. For each postId in the cache, go fetch the metadata from the Posts table in DynamoDB. This is simple but requires an additional database query for every feed request.
    
2. Rather than caching the postId, we could cache the entire post metadata in Redis. This way we don't have to make a second query to the Posts table to get the post metadata. This is faster but uses more memory and introduces data consistency challenges.
    
3. Use a hybrid approach with two Redis data structures: one for the feed's postIds (ZSET) and another for a short-lived post metadata cache (HASH). When we need to hydrate posts:
    
    - First, try to get all post metadata from the Redis post cache
        
    - For any cache misses, batch fetch from DynamoDB using BatchGetItem
        
    - Update the Redis post cache with the fetched metadata
        
    - Return the combined results
        
    

The hybrid approach (option 3) gives us the best balance of performance and resource usage. We can tune the TTL of the post metadata cache based on our memory constraints and consistency requirements. For example, if a post were to get updated, like a change to the caption, we would just invalidate the cache for that post and it would be updated in the next feed request.

To recap, here is what happens when a new post is created.

1. We store the post metadata in the Posts table and the media in the blob store like before.
    
2. We put the postId onto a queue to be asynchronously processed by the Feed Fan-out service.
    
3. The feed fan-out service will query the Follows table to get all users who follow the posting user.
    
4. For each follower, it will prepend the new postId to their precomputed feed.
    
5. The feed fan-out service will store the new feed in Redis.
    

Then, when a user requests their feed, we:

1. Read the top N posts from their sorted set in Redis.
    
2. Hydrate the posts based on these postIds. We first check if the post metadata is in the Redis post cache. If it is, we use that. If it's not, we batch fetch from DynamoDB using BatchGetItem.
    
3. Combine the results and return them to the user.
    

###### Challenges

This 'fan-out on write' approach significantly improves read performance, making feed retrieval a fast, single Redis query (plus metadata lookups). However, we've traded read-time complexity for write-time complexity and increased storage. The primary challenge is write amplification, especially the 'celebrity problem'. A single post by a user with millions of followers triggers millions of writes to Redis, potentially overwhelming our system and increasing write latency for popular users. This is because we need to update the feed cache for the millions of people following them.

### 

Great Solution: Hybrid Approach (Precompute + Real-time)

###### Approach

We can address the main limitation of the precompute approach, that celebrities will result in massive write amplification, by using a hybrid approach. We combine fanout-on-write for most users with fanout-on-read for popular accounts. This provides a good balance: fast reads for most users and manageable write amplification.

Here's how it works: We define a threshold for the number of followers. Let's say, 100,000 followers. For users with fewer than 100,000 followers, we precompute their followers' feeds just like in the "good" approach above. For users with more than 100,000 followers (the "celebrities"), we DON'T precompute their posts into their followers' feeds.

Instead:

1. When a "celebrity" posts, we add the post to the Posts table and do not trigger an asynchronous feed update for their followers.
    
2. When a user requests their feed: We fetch the precomputed portion of their feed from Redis (posts from users with < 100,000 followers). Then, we also query the Posts table for recent posts from the "celebrities" they follow. We then merge the precomputed feed with the recent posts from celebrities, chronologically and return the merged feed.
    

Thus, we end up with an effective mix between pre-computation and real-time merging.

1. Fanout-on-write for the majority of users (follower count < 100,000)
    
2. Fanout-on-read for the few "celebrity" users (follower count > 100,000)
    

###### Challenges

As is always the case, more complexity comes with its own tradeoffs. The 100,000 follower threshold needs to be carefully tuned - set it too low and we don't solve the write amplification problem, set it too high and we impact read performance for too many users.

Users following many celebrities will experience slower feed loads than those who don't, creating an inconsistent user experience. We'll need to set clear SLAs for different user segments and potentially implement additional caching specifically for celebrity posts. We also introduce storage complexity by maintaining two separate systems - the precomputed feeds in Redis for regular users and the real-time query system for celebrity posts. This increases operational complexity and requires careful monitoring of both systems.

Despite these challenges, this hybrid approach has proven effective in practice. Instagram actually uses a similar approach in production, demonstrating that the benefits outweigh the added complexity.

###### Pattern: Managing Long Running Tasks

The fanout on write is a good example of the pattern **managing long-running tasks**. When a user posts, we need to update the feeds of all their followers. This is a long-running task that can take a while to complete and need to happen asynchronously.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/long-running-tasks)

When proposing Redis as a feed storage solution in your system design, make sure to proactively address durability concerns. Some interviewers could rightfully question what happens if your Redis instances go down. Without proper persistence configuration, you could lose all cached feed data, causing service degradation and poor user experience. In your interview,you could explain that you'd implement Redis Sentinel for high availability with automatic failover, potentially use Redis Cluster for data sharding across multiple nodes, and configure persistence options like AOF (Append-Only File) or RDB snapshots to recover data after failures. A thoughtful statement like "We'd implement Redis with AOF persistence and Sentinel for failover to handle node failures with minimal data loss" demonstrates you've considered potential failure modes and aren't treating Redis as a magical black box.

### 2) The system should render photos and videos instantly, supporting photos up to 8mb and videos up to 4GB

There are two key challenges to large media files.

1. Upload efficiency.
    
2. Download/viewing latency.
    

Lets take these one at a time.

First, as we discuss in both [DropBox](https://www.hellointerview.com/learn/system-design/problem-breakdowns/dropbox) and [YouTube](https://www.hellointerview.com/learn/system-design/problem-breakdowns/youtube), uploading large media files requires chunking. This is because a single HTTP request is usually constrained to a max payload size < 2GB. Thus, in order to upload a 4GB video, we would need to send at least 2 (in practice, more given metadata) requests.

A common solution is to use AWS S3's multipart upload API. At a high-level, here is how it works:

1. First, we will call POST /posts to create the post metadata and get a postId as before, but now it will also return a pre-signed URL that can be used to upload the media. This URL is valid for a limited time (e.g., 1 hour) and allows the user to upload directly to S3 without having to go through our servers.
    
2. Client side, we use the multipart upload API to upload the file in chunks to the pre-signed URL.
    
3. S3 will automatically handle reassembling the chunks and store the final file in S3.
    

###### Pattern: Handling Large Blobs

Instagram's approach to handling large media files demonstrates the **large-blobs pattern** used across many interviews. The combination of presigned URLs for direct uploads, chunked uploads for reliability, and CDN distribution for global access appears in many systems handling substantial file transfers, from video platforms to file storage services.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/large-blobs)

When we initially uploaded the post metadata, we would include a media upload status field originally set to pending. Once the media is uploaded, we update the post metadata with the S3 object\_key and update the media upload status to complete.

There are two main ways to handle the post metadata update after upload completion:

1. Client-driven approach: The client sends a PATCH request to update the post metadata with the S3 object\_key and sets the media upload status to complete once the multipart upload finishes.
    
2. Server-driven approach: We configure S3 event notifications to trigger a Lambda function or background job when the multipart upload completes. This job then updates the post metadata automatically.
    

The client-driven approach (Option 1) is simpler to implement but less reliable since we have to trust clients to properly report upload completion. The server-driven approach (Option 2) is more complex but provides better data consistency guarantees since our backend maintains full control over the metadata updates. Most production systems opt for Option 2 despite the added complexity.

Upload

Now that we have all our media uploaded and in S3, let's talk about how we get it to render quickly when a user views it.

### 

Bad Solution: Direct S3 Serving

###### Approach

The simplest approach would be to serve media directly from S3. When a user views a post, we return the S3 URL for the media file, and their client downloads it directly from S3. This is straightforward to implement and works fine for small applications, but it doesn't scale well for a global application like Instagram.

Direct S3 Serving

###### Challenges

This approach falls short in several ways. First, users far from our S3 bucket's region will experience high latency when loading media. A user in Singapore accessing media stored in us-east-1 might wait seconds for images to load. Second, every request hits S3 directly, which becomes expensive at scale and doesn't take advantage of any caching. Finally, we're serving the same high-resolution files to all users regardless of their device or network conditions, wasting bandwidth and slowing down the user experience.

### 

Good Solution: Global CDN Distribution

###### Approach

A significant improvement is to put a CDN like CloudFront in front of our S3 bucket. The CDN maintains a global network of edge locations that cache our media files closer to users. When a user requests media, they're automatically routed to the nearest edge location. If the media isn't cached there, the CDN fetches it from S3 and caches it for future requests.

This way, instead of our Singaporean user waiting seconds to load an image originally stored on the US East Coast, they can load it from a server in the region closest to them.

We can configure the CDN to cache files based on their type and popularity. For example, we might cache images for 24 hours at edge locations since they rarely change once uploaded. This dramatically reduces latency for users and load on our origin S3 bucket. We can also use the CDN's built-in compression to reduce file sizes during transfer.

Global CDN Distribution

###### Challenges

While this solves our global distribution problem, we're still serving the same files to all users. A user on a mobile device with a slow connection receives the same high-resolution image as someone on a fast desktop connection. We also have limited control over the caching strategy, which means popular media might get evicted from edge locations due to cache size limits, leading to unnecessary origin fetches.

### 

Great Solution: CDN with Dynamic Media Optimization

###### Approach

The most effective solution combines CDN distribution with dynamic media optimization. When media is uploaded, we use a media processing service (like Cloudinary or Imgix) to generate multiple variants optimized for different devices and network conditions. For images, this includes different resolutions and formats like WebP for supported browsers. For videos, we create multiple quality levels and use adaptive streaming (more details in the [YouTube](https://www.hellointerview.com/learn/system-design/problem-breakdowns/youtube) breakdown).

The CDN then serves these optimized variants based on the requesting device and network conditions. Mobile users automatically receive appropriately sized images, while desktop users get higher resolution versions. The client includes device information and network conditions in its requests, allowing the CDN to serve the most appropriate variant.

We also implement intelligent caching strategies. Popular media is cached more aggressively at edge locations, while less accessed content might have shorter TTLs. For viral content that suddenly gains popularity, we can proactively warm caches in regions where we predict high viewership.

CDN with Dynamic Media Optimization

###### Challenges

This approach significantly improves user experience but introduces more complexity in our system. We need to manage multiple versions of each media file, increasing storage costs and complexity. The media processing pipeline adds latency to uploads and requires careful monitoring. Despite these challenges, this is the approach most large-scale media platforms use as it provides the best balance of performance and manageability.

### 3) The system should be scalable to support 500M DAU

We've been keeping scale in mind throughout our deep dives, but let's summarize the key design choices that enable us to efficiently serve 500M daily active users while maintaining performance and reliability.

1. Precomputed Feeds (Hybrid Approach): The cornerstone of our scalable feed generation is the hybrid approach. By precomputing feeds for the vast majority of users (those following accounts with fewer than our defined threshold of followers), we drastically reduce read-time load. The real-time merging for "celebrity" posts is a carefully considered trade-off to manage write amplification.
    
2. Content Delivery Network (CDN): Using a CDN like CloudFront for media delivery is essential for global scalability and low-latency access. Dynamic media optimization further improves performance for diverse devices and network conditions.
    
3. Chunked Uploads: For large files (especially videos), chunking uploads improves reliability and user experience, and allows for parallel uploads.
    
4. Database Choice and Indexing: Our choice of DynamoDB (or a similarly scalable NoSQL database) provides horizontal scalability for metadata storage. The careful use of partition keys and sort keys ensured efficient queries.
    

Let's do some math and determine whether our existing scaling strategies are sufficient.

For the media files themselves, we can assume an average media size of ~2MB, so 100M \* 2MB = 200TB of binary data each day.

Starting with the media, this is about 750 PB of data over 10 years. S3 can handle this, but it's not cheap. If we worry about cost, we can always move the media that has not been accessed in a long time to cheaper storage like [Glacier](https://aws.amazon.com/pm/s3-glacier).

For the metadata, we're looking at 100M \* 1KB = 100GB of new (non-binary) data created each day. Similarly, if we grow concerned about the price of DynamoDB, we can move infrequently accessed data over to S3.

This is a common pattern. The warmer the storage, the more expensive it is. Working back from CDN -> Cache/Memory -> SSD -> HDD -> Tape, we can always take data that is infrequently accessed and move it "down a level" to save money.

When it comes to throughput, as always, we can dynamically horizontally scale our microservices to handle the load. This happens automatically with most cloud providers and is triggered based on either CPU or memory usage thresholds. Each set of horizontally scaled microservices thus implicitly has a load balancer in front of it to distribute the traffic.

After all is said and done, you might have a design that looks something like this:

Final Design

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

So, what am I looking for at each level?

### Mid-level

At mid-level, I'm looking for a candidate's ability to create a working, high-level design that addresses the core requirements of Instagram. They should understand the basic components needed for photo/video upload, user follows, and feed generation. I expect them to propose a simple database schema and explain how posts would be stored and retrieved. While they might initially suggest a naive fan-out on read approach, with some hints they should be able to recognize its limitations and arrive at a fan-out on write solution. I want to see them problem-solve through performance bottlenecks when prompted. They should also grasp basic media storage concepts like using S3 and understand why a CDN might be necessary, even if they don't dive into the details of media optimization.

### Senior

For senior candidates, I expect you to nail the majority of the deep dives, particularly around feed generation optimization and media delivery. You should understand the tradeoffs between fan-out on read, fan-out on write, and hybrid approaches, and be able to articulate why Instagram would choose a hybrid model to solve the "celebrity problem." I expect knowledge of efficient media upload handling for large files, and strong justifications for your technology choices. You should be able to discuss database indexing strategies in detail and understand how to optimize for our read-heavy workload. You likely would not have time to cover all the deep dives we did here, but if asked, you should be able to articulate the key tradeoffs and arrive at a reasonable solution.

### Staff+

For staff candidates, I'm evaluating your ability to identify the true scalability bottlenecks and propose elegant solutions that balance complexity against real needs. You should quickly recognize that feed generation and media delivery are the key challenges and focus your design accordingly. I expect you to discuss system evolution over time - how would we handle growing from 1M to 500M users? Rather than immediately jumping to complex distributed systems, you should be able to articulate where simpler solutions suffice and precisely when we'd need to evolve to more sophisticated approaches. Staff candidates demonstrate a keen understanding of operational concerns, anticipate failure modes, and propose robust solutions that prioritize user experience above all.

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

S

singhlrah

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7ltm045007314o178dhuw6s)

This is a good post Evan ,and i compared my recent interview response(which was instagram) to this.I am able to find gap in my design and response .1. handling of fetch and rendering -> is it client vs server , your proposal of CDN with dynamic media optimisation is effective. 2. Good distinction on when to do Fan out on write vs fan out to read. Additional question will be : how the user will be notified if there are new feed available for them ? what is the most optimal way to be notified i.e. we cant notify them for all the feed ?

Show more

2

Reply

![Sean](https://lh3.googleusercontent.com/a/ACg8ocL2qrLyiLoMEzphe-VPoPmaI8640yKpWva4i7jXb-1QouH5xw=s96-c)

Sean

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7y0992901c19n2kr3nzywmr)

> how the user will be notified if there are new feed available for them ?

maybe a simple poll on the client using the cursor original /feed request, so if any newer items pop up we know we can fetch for the new items via the cursor

> what is the most optimal way to be notified i.e. we cant notify them for all the feed ?

seems like infrequent, ~60s polling, on the feed( if thats OK UX) solves this? generally people will be scrolling downwards for older items, anyone hanging on the top of their feed for a while can just send a poll every N seconds if there was an update. can adjust this based on # user is following, or how often those they follow generally post.

pls correct me if i am wrong / offbase!

Show more

0

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cma542l2i003qad08j1cpiden)

with regards to being notified, I think either using polling or SSE could work - although SSE might be overkill here? Depending on the infra, might be trickier to set up than polling, and we can set up the poll rate to be every 5-10s to offload some work from the backend.

Could also have a range where people with fewer follows could have a much higher poll rate, and people with many more followings could have a lower interval

Show more

0

Reply

I

InnerHarlequinFirefly879

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7mmdoqa00kblel6ug2jyi1j)

This is great! Thank you! Please upload a video for it :)

Show more

18

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8nkru1r00xmvmfk9003i58n)

Just added!

Show more

8

Reply

W

WoodenAquamarineSnipe931

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8pnihq600p76xhmu1lj00j1)

Anyway to be notified when 'Premium Only' content is added or changes?

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8q253no00x96xhmt44fokw5)

Working on that right now! Hopefully end of week you'll start seeing emails :)

Show more

7

Reply

W

WoodenAquamarineSnipe931

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8q5llr3013luai32990t8qw)

yay!

Show more

0

Reply

G

ghanekar.omkar

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9we49fi00vqad089pt1zrz0)

Also, is it possible to add auto-generated subtitles for premium videos? I'm sure some off-the-shelf tool would be able to do that in our "AI powered" world.

Show more

0

Reply

![Will Shang](https://lh3.googleusercontent.com/a/ACg8ocLL-_zI633rdwKuZxrzSInrilUdONavtZFCWy3x-mtPY2VPdA=s96-c)

Will Shang

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7nphkzr001m14cnofx31qwh)

For the facebook news feed breakdown, there is a feed table introduced in the dynamoDB vs here the feed is stored in the redis cache. How could we weight the tradeoffs?

Show more

6

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7nubjro008mdd6ln1tcc969)

TLDR: no right or wrong answer.

Redis gives you blazing-fast in-memory feed access but is volatile and memory-constrained, while DynamoDB offers persistence and unlimited storage at slightly higher latency. Your choice depends on whether real-time performance (Redis) or data durability (DynamoDB) matters more for your feed experience.

Show more

4

Reply

![Amit Kumar](https://lh3.googleusercontent.com/a/ACg8ocKdR-tkIC5chKLk0xi9-74XdtORqQguBnHAHzMM8r8qYbofOB2NgQ=s96-c)

Amit Kumar

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8ccz9hv00o71ipglia2unkc)

Can we not use both? DynamoDB to persist feeds and redis as an in-memory cache on top of it? For Cache miss, we can enrich the redis with DynamoDB?

Is it okay or an overkill?

Show more

7

Reply

L

LikeAmberDove772

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmbjx2dxm003bad08c1zgld24)

redis + DynamoDB (or any disk based persistent storage) is a more solid solution here I would say. And It's not overkill. Considering the scale of instagram, the redis cluster storage soon becomes "Not Enough". Solely using redis as storage without backing by a persistent storage layer could be a red flag in the interview. Though redis introduces fancy feature to make it behave like a "persistent storage", for most of the cases, it's still an layer for optimizing the reads.

Show more

0

Reply

M

MarriedMoccasinSpoonbill355

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8819syx008y1ats029n4dxf)

I just had an on-site with Meta and there was a reverse shadow. I proposed the Redis solution for feeds and I used Postgres for the user table and the post table. The interviewer had a hard time understanding how the feeds can be generated in this way. After I explained it 3 times, he asked with a confused face "So no feed table? You are only going to store it in a cache? I ... am not sure if it works." I think when proposing solutions, we not only need to consider tradeoffs, but also the interviewer's competency, especially when a reverse shadow is present.

Show more

8

Reply

Z

ZealousFuchsiaLeopon345

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9cjyxlp00epad08c5n2waoc)

Lets assume the interviewers are competent, and you need to work on your communication / influencing skill.

If someone tells me they are going to store the feed only in the cache, without clearly explaining the durability considerations and fault modes, I'd count it against them.

Its on you, not them.

Show more

3

Reply

Y

YummyTurquoiseNightingale482

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9e6g73700utad08rcmos2a1)

yeah, have to agree with you on this. The redis can also offer durablity like AOF or RDB.

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9eesqgh00zfad08ii9801st)

+1 to the above. There isn't a right or wrong answer. As always, it's about trade offs. Redis can be plenty durable for this use case. Worth me adding a little callout to this in the write up though. Will do.

Show more

4

Reply

M

ModernAzureOx557

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7nqknrp003r5g7oevj4vky3)

While the post metadata and content are written to the database and distributed to followers' feeds, the media upload may still be in progress, pending, or could fail depending on its size. How can we ensure a seamless user experience so that neither the poster nor their followers encounter broken media when viewing the post?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7nu8uft008paakwg6abs5lc)

Filter in memory! The post table has an upload status, so don't send any posts to the client where the status is still "pending"

Show more

3

Reply

M

ModernAzureOx557

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7nww2rw00d65g7oaz35k5be)

Could this be another option, for post service to only write post to the database when media upload is complete util then client manages/stores the partial post(without the media) in internal device storage and can keep trying/re-trying in case of failures/delays/time-outs?

Show more

0

Reply

![Jeff](https://lh3.googleusercontent.com/a/ACg8ocLJ9FOEMCAn7UaWkdSd-PEnAlAru404ExShynnQVbMJzAJbMQ=s96-c)

Jeff

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9irvt7l01a2ad08zedo62f6)

I'm not an expert at all here, but a concern I would have with this approach is we should treat the client as unreliable. Similar to the write up where it discusses updating the media upload status.

Assuming the UI is giving the user instant feedback, the user would see that their upload was submitted, but the post would not actually exist in any persistence layer. Additionally, it could happen that the upload to S3 succeeds, but the subsequent post to save to DB fails, which leaves us with a set of orphaned data in S3.

There is another concern about idempotency I think with retries in this state, but that might be acceptable since this use case isn't something like financial transactions. I think the main concern might be that this approach puts a lot of responsibility on the client when we should be letting the backend own persistence and media upload status.

Show more

0

Reply

![Jingyu Yao](https://lh3.googleusercontent.com/a/ACg8ocJ_EEXI7nAD89LVOc99VTFCOXtl7LgtH55G1zWZ6xFCXLo2vHU=s96-c)

Jingyu Yao

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7nu6k5q0098lzf3iqbyszga)

This is not medium at all lol, this is like the final boss after you cleared

1. Facebook live feeds
2. Youtube
3. Dropbox

GG.

Show more

38

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7nu98rd008edd6ldmn79hpk)

Lol! Maybe you're right, if this gets some upvotes I'll update it to hard ;)

Show more

14

Reply

![Ivan Atroshchenko](https://lh3.googleusercontent.com/a/ACg8ocKs2RdLYX_h3iuQWXlqtX5XXhV9XOGR24Ytb7Tq4fmMDCBRj68b=s96-c)

Ivan Atroshchenko

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9n4c11a00yrad08ryaawxt5)

Basically it's medium it's just twitter + dropbox + media convertions

Show more

2

Reply

M

mattchiou001

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7o1o9vu007emhvmpa6hv8jk)

Great content! Really love it.

One small callout (I assume you omitted this for simplicity) is that you actually need multiple pre-signed URLs to do multipart uploads (https://aws.amazon.com/blogs/compute/uploading-large-objects-to-amazon-s3-using-multipart-upload-and-transfer-acceleration/). The logic is still conceptually correct but might be worth explaining it in the actual interviews to showcase the depth.

Show more

16

Reply

![Aman Thakkar](https://lh3.googleusercontent.com/a/ACg8ocLwwtnCae2aEApD8yZrf0xacaCP13PFYMjrKui7yq7amQ=s96-c)

Aman Thakkar

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qmnn4z01yqpqpv1lsbk6b0)

I believe you could also decide on the client side about how many parts/chunks you want to have, then repeatedly query the service for the url for "next part to be uploaded". Pls correct me if I am wrong.

Show more

0

Reply

M

mattchiou001

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qnx55t0289jtnqhrovyp09)

The number of chunks is determined by how you want to split the file (e.g. the max size per chunk). This number is going to affect how efficient your client upload is. While app like instagram does have control on the client, you don't have control on "when" your client decide to update the app. This means if you get the chunk size wrong (or simply just want to adjust it due to xyz reasons), there's going to be some delay. I think this type of trade-off usually only makes sense if the logic would have been pretty complex on the server side and pretty simple on client side (such as client side metrics) but I don't think this would be one of those cases.

I would also argue that the implementing some pagination logic (or do it on the fly as requested) is pretty overkill and without a lot of benefit. I think just having the client send you the file size and return list of pre-signed url probably makes the most sense.

Show more

1

Reply

M

mattchiou001

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7tpo3k0000ppso33b7c8w9b)

I think you most likely would want to avoid doing that even if you could.

Here are two reasons that I can think of:

1. Latency: you will have to do unnecessary round trips (between your client and server) to retrieve pre-signed urls and you actually have to generate the pre-signed url from the server by calling S3 as well. Instead, it is much more efficient to just have your client tell you the file size, server decides the number of segments, call S3 in parallel for each segment pre-sign url generation, return the list of pre-signed urls. The size of the urls should be small enough that it won't make any meaningful difference in the latency.
2. While apps like instagram has control over the client, you don't have control over how frequently your users update the app. If you have a bug on core logic like this, you would want to patch them immediately and it would be much faster to do it on the sever side.

Generally, I would avoid doing client side logic unless it's much simpler to do (and way harder to do the same thing on server side).

Show more

2

Reply

F

FormidableRoseMeerkat174

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7o4imuc00c4mhvm1lvgwo9q)

"For example, if a post were to get updated, like a change to the caption, we would just invalidate the cache for that post and it would be updated in the next feed request." How is this change detected and reconciled if we store all post metadata in cache instead of PK?

How does cursor work when reading directly from cache?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7s0niwl00s710vd067zis8v)

For caption updates - we'd still write both to the database (source of truth) AND invalidate the cache. The system would detect changes through standard application logic when users edit posts. Even with metadata in cache, we maintain the post ID as the identifier in both systems. Regarding cursors with Redis - we'd typically implement this using ZRANGE with LIMIT/OFFSET on our sorted sets. The cursor would be either a timestamp or post ID that marks where the previous page ended. When the user requests the next page, we'd use that cursor value in our Redis query to fetch the next batch of posts.

Show more

0

Reply

![Ge Xu](https://lh3.googleusercontent.com/a/ACg8ocIUwlvTlvr8XDYlJr0mawAAAJrlIz-ROily9Zg_FOKgpjV_7mqN=s96-c)

Ge Xu

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7ohoxdt000upqpvie0m29w2)

This is very similar to FB News Feed, in which a feed table is used. May I ask why is redis feed cache used here without saving a copy in feed table in DB? What if redis fails, how to handle failover?

Show more

0

Reply

![Eddie Tsai](https://lh3.googleusercontent.com/a/ACg8ocJnAMB6Hi0ZNagSvzFi8Aw_ojEqyzNYxXt7OqEdkytPniN2rg=s96-c)

Eddie Tsai

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7pw2r6s01g5jtnqbnp2pv8a)

> I'm going to opt to have the **feed** service handle this for now rather than creating a new service

Should the feed service be **post** service?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qigusy01vmjoxcqh3ij077)

Correct, updating!

Show more

0

Reply

P

pandey.aman.61

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qief3201rxpqpvjz0ds1oy)

> **The system should render photos and videos instantly, supporting photos up to 8mb and videos up to 4GB**

- In the system design diagram for Instagram, there appears to be an inconsistency in how the media upload flow is depicted. The diagram shows the 'Upload via pre-signed URL' arrow going from the Client to the Database, but shouldn't this arrow instead be drawn between the Client and S3 since pre-signed URLs allow direct upload from clients to S3 storage? Is my understanding correct?

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qigpk001xarpj4h6sqvzag)

Shoot, good catch. You're right, just a bad arrow :) Will update!

Show more

0

Reply

E

ElegantMoccasinSwan892

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm84y9o5u012u11rg5aiztyg6)

I can barely see the images for the final design. It seems there are two images at the top left and bottom right corners; when I enlarge the image, I cannot see them well. Please can you update to make them more visible? Thanks!

Show more

0

Reply

![Aman Thakkar](https://lh3.googleusercontent.com/a/ACg8ocLwwtnCae2aEApD8yZrf0xacaCP13PFYMjrKui7yq7amQ=s96-c)

Aman Thakkar

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qmlwm5025zjtnqbkr7fbd3)

postponing my Amazon loop just so I can get through all these

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7s0h8mu00rr10vdsi3r5chv)

Godspeed 🫡

Show more

1

Reply

M

MassAmaranthCarp180

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9bshb3g0069ad075k89yem5)

Did you finish @Aman ?

Show more

0

Reply

![Aman Thakkar](https://lh3.googleusercontent.com/a/ACg8ocLwwtnCae2aEApD8yZrf0xacaCP13PFYMjrKui7yq7amQ=s96-c)

Aman Thakkar

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9bwf9yz001ead087eaaj6om)

I did and I got in!

Show more

7

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9bwvvs00016ad080jidr1q4)

Let's go!!! Major congrats Aman!

Show more

2

Reply

![Aman Thakkar](https://lh3.googleusercontent.com/a/ACg8ocLwwtnCae2aEApD8yZrf0xacaCP13PFYMjrKui7yq7amQ=s96-c)

Aman Thakkar

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9bx2z16002fad08s2w1252y)

Thanks Evan!!

Show more

1

Reply

V

VisibleVioletIguana133

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qnxdbj0260rpj4xxnam9jc)

Good post Evan. When I read this post, I was thinking about the difference of design a news feed system and Instagram. Can you list out some biggest differences please?

Show more

0

Reply

T

TechnologicalFuchsiaMammal664

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qyeehg02rnjoxcj44q1ows)

Nit: links to dynamoDb and postgresDb are broken.

Great content keep it up :)

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7s0how500q1ciytkhp24gef)

Thanks! Updating

Show more

0

Reply

B

BeneficialBrownBarnacle904

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7qzxph502o8pqpv0fm88680)

Thanks for the post!

Couple of questions:

- Is this question asked for Staff candidates? If so, could you please provide pointers on what additional focus areas should be discussed?
- Should we also discuss scaling for the Redis feed cache? Assuming about 10 KB of feed data per user Id, the size would be 500M \* 10 KB = 5 TB.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7sej0ro01jr2d76wdng0mqf)

Yes, we've seen it asked at many companies at all levels. Scaling the cache is a great potential topic.

Show more

2

Reply

M

ManagingAquaFlea863

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7rpdg9x008se08kjtfzk0t5)

Thank you for the article! This is a great opportunity to revise the approaches from the previous topics and how they work together. My question is not related to this content. Have you ever considered adding a dark mode on the website and switching to the dark mode on Excalidraw in your videos? According to statistics, most software engineers prefer the dark mode. Just a suggestion:) Thank you!

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7tpqycx007lj56aj6420nux)

Yah, just haven't prioritized it :) Opting for more features and more content before implementing dark mode.

Show more

1

Reply

F

FederalHarlequinLamprey277

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7skb1at01v7ciyt4icn5xsg)

Great article, thanks Evan! Questions:

1. We provided Follower\_Followee table, do we also need Followee\_Follower table for efficient lookup for fan-out on write ?
    
2. For a follower, following lots of other users. How do we figure out the celebrities at real time on read: is it via querying each of their followees to determine if their followers exceeded the set threshold? If yes, does dynamodb have an efficient way to crafting a query to return the celebrities or is it via multiple calls?
    
3. It seems s3 does not return object-key so the client is just updating the status?
    

This is super helpful. Thanks in advance for providing clarity.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7tpq2z7005tncsem9hi6fer)

No need for a Followee\_Follower table. The Follower\_Followee table is sufficient for both read and write operations if indexed correctly. For celebrities, we'd maintain a separate "celebrity" flag or threshold in the user metadata, not query followers every time. For S3 we dont want to trust the client anyway, so we'd rely on s3 notifications instead.

Show more

1

Reply

F

FederalHarlequinLamprey277

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7tvh2r200g8ncsewkzpssjp)

Thx!

Show more

0

Reply

![Preet](https://lh3.googleusercontent.com/a/ACg8ocIzJi7-W4mfNfe1nBMIkxFc9OPRcBw0fXSCbkJIFjdOkxtApdV9rw=s96-c)

Preet

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm7zg5zd8037y5693mkl9ib6j)

The post links cuid to github repository of cuid2. FWIU, cuid2 ids are not monotonically increasing. I think we can use a composite of createdAt and postId as the sort key. As a consequence of this, the cursor object need to be appropriately encoded

Show more

1

Reply

![Sean](https://lh3.googleusercontent.com/a/ACg8ocL2qrLyiLoMEzphe-VPoPmaI8640yKpWva4i7jXb-1QouH5xw=s96-c)

Sean

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm80mywi504ff5693uarsovnv)

I think I’ve discovered an inconsistency in the Follow indices proposed.

With the sort key as FollowedID and partition key as FollowerID, that makes the query to “get all accounts that a single user is following” efficient, NOT to “get all of the followers of a single user”, right?

Given this setup, when building the feeds using fan out on write, it would be very inefficient to find the followers of a single user. Again, we need to find “all followers of the user who made this post”, but the indices optimize “all the users the poster is following”. In other words, we would be querying against the sort key (FollowedID and not the partition key(FollowerID), Dynamo would have to go through the entire table.

as both relationships might be useful, dynamo supports Global Secondary Indices - trade-off being more rights, and storage to maintain.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm822x6rf00xuimsr0mjy64u0)

Yup, you're right. We'd want FollowedID as the partition key to efficiently get all followers of a posting user. The current setup (FollowerID partition key) only helps with "who am I following?" queries. We'd need a GSI with FollowedID as the partition key to make fan-out on write.

Show more

0

Reply

S

sumsha18

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm81mkeb800a14ki91ec45ky9)

It seems that the following statement should have the reverse claims:

This: "As is always the case, more complexity comes with its own tradeoffs. The 100,000 follower threshold needs to be carefully tuned - set it too low and we don't solve the write amplification problem, set it too high and we impact read performance for too many users."

Should Be: "As is always the case, more complexity comes with its own tradeoffs. The 100,000 follower threshold needs to be carefully tuned - set it too low and we don't solve the read amplification problem, set it too high and we impact write performance for too many users."

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm822w7u500ye444o1qwg1zmj)

No this looks correct. When we set the threshold too low (e.g. 1,000 followers), too many users get treated as "celebrities" and we end up with too many fan-out-on-read operations, defeating the purpose of the hybrid approach. When we set it too high (e.g. 1M followers), we're doing fan-out-on-write for users with large follower counts, causing write amplification.

Show more

1

Reply

M

MobileBrownHyena409

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8wk928000x8ad073zhavrvs)

wait so then shouldn't it be: "As is always the case, more complexity comes with its own tradeoffs. The 100,000 follower threshold needs to be carefully tuned - set it too high and we don't solve the write amplification problem, set it too low and we impact read performance for too many users."

Show more

1

Reply

S

Splash

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmar95tsc0049ad09isapk5rr)

I think @sumsha18 is correct.

As you said,

> "When we set the threshold too low (e.g. 1,000 followers), too many users get treated as "celebrities" and we end up with too many fan-out-on-read operations"

"Too many fan-out-on-read operations" is the problem of read amplifications, not write amplications.

Overall, it's just a minor mistake and we all get the idea.

Show more

0

Reply

W

WidespreadApricotAlligator275

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm82ltvvc0085148th82fqoo3)

Having issues seeing the global cdn with media optimization whiteboard drawing

Show more

1

Reply

E

ElegantMoccasinSwan892

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm84yaviy013311rgb8cbxrcq)

+1 I cannot see the final design diagram as well.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm84yfop600n6j2eyi8377lj6)

what the! Fixing :)

Show more

0

Reply

E

ElegantMoccasinSwan892

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8643lky00dqetkcduabc0ub)

Looks good now. Thanks!

Show more

0

Reply

E

ExtendedPurpleMole175

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm83dr6200107oh5frgqylugh)

Great writeup with attention to detail!! Thank you for creating such insightful content. Question: How many records should we store in the Feed cache per user. Whats our eviction policy for records (feed) in Redis? Also, what happens if hypothetically we store 1K posts per users in feed cache and the client scrolls past the most recent 1K records?

Show more

1

Reply

C

ComparativeAmethystTortoise719

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaoau6fc003bad08rqlrw31z)

Hi Evan, Can you please help answering the above question?

Show more

0

Reply

S

SpecialPlumKrill871

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm841p2bf0008n83rughjxw4o)

Not sure what’d be the best place for this, I wanted to say thank you to HelloInterview team for the high quality system design sessions. I’ve been learning the entire system design section here since Christmas last year, and I got an offer two weeks back. HelloInterview system design is the best resource in the market!

Show more

3

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm841sq16000dn83rtttetx77)

Major congrats on your offer! Glad we could be a small part of it (let's be real, you did the hard work) and thanks for sharing the good news with us :)

Show more

0

Reply

![Liran Jiao](https://lh3.googleusercontent.com/a/ACg8ocL9mVCDc2tZg0cRx_ImcFTZz1ZlRoBMyOV3J4EVbUF3-ZBGBw=s96-c)

Liran Jiao

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm84llrvw00i911rgj9n35rw3)

For the scalability deep dive, as you mentioned

| For the metadata, we're looking at 100M \* 1KB = 100GB of new (non-binary) data created | each day. Similarly, if we grow concerned about the price of DynamoDB, we can move | infrequently accessed data over to S3.

If we moved the infrequently accessed data over to S3, do you mean we could group the DynamoDB rows by the user\_id and store rows with different user\_id into different s3 files, which will be easy to query?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm84ocuet00ld11rg0fkc98jc)

Not quite. When moving data to S3, we typically store it in a columnar format, partitioned by time (e.g. by day or month). This lets us efficiently query historical data when needed while keeping hot data in DynamoDB. The pattern you described would create too many small files and make querying inefficient. Plus, S3 isn't built for random access like that.

Show more

0

Reply

![Shivam Singh](https://lh3.googleusercontent.com/a/ACg8ocKJX3c2VsBr0_uXGqo2dCow5Kl2Cel07o9H8kghj4JvuB2SJm9E=s96-c)

Shivam Singh

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8aspo4s01dbo5ih61pgya49)

For storing posts, comments, and similar data, would it not be more efficient to use a large EBS volume mounted on a low-cost EC2 instance as infrastructure for hosting databases like Cassandra or PostgreSQL, rather than relying on S3 with a columnar format?

With S3 + Parquet/Athena or ORC/Hive, we lack precise knowledge of which partition (by date) needs to be queried. Given our access patterns—either (1) fetching all posts by a user before a specific date or (2) retrieving an exact post by its ID—we would end up scanning large amounts of data, which seems inefficient. Correct me if I am wrong here.

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8cpzeor00dbd7bc18pos6u1)

Why don't we use SSE for this type of product but instead pulling?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8mgph9t00kj9l3jvhzj5xx1)

SSE (Server-Sent Events) would actually be a poor choice here. Instagram's feed is primarily pull-based by design - users explicitly request their feed when they open the app or refresh. SSE is better suited for real-time updates like notifications or live comments, but would be massive overkill for the main feed where maintaining millions of open connections just to push occasional updates would be incredibly resource intensive.

Show more

0

Reply

![Shaik Aziz](https://lh3.googleusercontent.com/a/ACg8ocIjUUX5LWOAytM5JzxTZSDzR0KdeS9fESKy3rF8KyapzJfcvA=s96-c)

Shaik Aziz

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8e552tv0117104dzpj8h4if)

For generating the feed can we add a pub-sub hybrid model where we push the posts to followers? Similar to yt live comments design?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8mgoy9a00kf9l3jnd7dq8be)

Probably overkill tbh. If you want freshness, just refetch when they request it. Like on ig when you pull down on the top to get more.

Show more

0

Reply

W

WoodenAquamarineSnipe931

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8lokd1u00vf9xqy5kqrqf40)

"We put the postId onto a queue to be asynchronously processed by the Feed Fan-out service."

Wouldn't it be better to have postId and UserId (ie., id of the user making the Post)? Otherwise, we'll have to get the post, get the userId and then lookup the followers for the post, right?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8mgo3tg00k39l3j3e3jwsds)

Yah put as much as you need (within reason)

Show more

1

Reply

W

WoodenAquamarineSnipe931

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8mlccg2002hkdyp17xbgn14)

Fair enough. Just wanted to say that you folks are doing a great job! clarity of thought shines through!

Show more

0

Reply

![Divij Gera](https://lh3.googleusercontent.com/a/ACg8ocIvaCla45L2DO5Z88JexhOI8mx7mgYSIlRW5pJsFyLhPIIx0gkv=s96-c)

Divij Gera

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8m7snd8005ufevenlfzmmab)

Why do we partition the post table on the user\_id? Won't this create an imbalance in the partitions for hot users, who post a lot? Since we are already using NoSql, why not a wide column DB to have another table User\_Post table with user\_id -> columns on post\_id, partitioned on the user\_id. You get the post ids and then use the post table with the partition key as post id? Would this be a bad design idea?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8mgno7y00jz9l3j2jhm1kqv)

We partition on user\_id because it's the most common access pattern - getting a user's posts. Hot users aren't really an issue here because even the most active Instagram users post maybe 10 times a day max and we can add limits there. The real hot spots come from read patterns (millions of users trying to read a celebrity's posts), not write patterns and we cache there

Show more

1

Reply

![Sen Wang](https://lh3.googleusercontent.com/a/ACg8ocJAjXPeOB-UDh8xiWQaD-fR6N8m5iZHendUAMZ-mT-RsbGB8c2Q=s96-c)

Sen Wang

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8nu1ei901i4bcat8o83l3du)

You mentioned 500ms end to end (request response ) time for the latency in the non functional requirements, but in the article there is no quantitative calculation to demonstrate this, would it be better to avoid mentioning specific numbers like this in an interview?

Show more

0

Reply

L

LoyalIndigoDove113

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8o4ze4m00f58nv8gl6enqb2)

Thanks for adding video explanation. One small feedback, the audio settings for the video is getting reset to 100% if I pause the video and go to some other tab and come back to play the video.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8oy9zwo017osv4n1u0l94b7)

oooh yah annoying. Let me look into that and fix it. Good flag!

Show more

0

Reply

M

MassAmaranthCarp180

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9bwdt85001rad07abzz79kk)

Also when i change tab the video player is getting restarted from start, pretty annoying

Show more

0

Reply

P

PeacefulAmaranthTarantula827

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8p162o4014p8nv8n74ot9wj)

For the part where you mentioned that the client can upload to S3 using pre-signed URL and once the upload has completed, S3 would use some out of the box lambda function or something to notify DB that the upload for a particular's post has completed. What if the client fails to upload to S3 and there is a stale entry to clean up from the DB & you may need to remove the post from DB if the client never completes uploading? What would be an ideal solution?

- A cron job?

Show more

0

Reply

![Spandan Pathak](https://lh3.googleusercontent.com/a/ACg8ocJ4mZkoGI0NnDWTsqKXBR7-x8i5U27JydTrixG75peiTw2Ma_7yuA=s96-c)

Spandan Pathak

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8p9rfr0009bfg39cw08w4z7)

Hi Even/Stefan, what do you guys think of Jordan's deep dive design using FLink for background feed generation(https://www.youtube.com/watch?v=S2y9\_XYOZsg&lc=UgyA5B5z0aJK\_BBXffN4AaABAg.AFxavLoqeWNAFz7cJq53tA). Would you folks recommend going into that level of deep dive for this/fb news feed questions?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8pcq6fn00806dzqmhy6c7uv)

Jordan loves Flink more than his own mother. I haven't seen the design but there's probably some feasibility to it.

Show more

2

Reply

M

MassAmaranthCarp180

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9bwenko001xad071jeg4ytk)

lol

Show more

0

Reply

H

HomelessAquamarineTrout494

[• 30 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmdipcob4053fad08ou5nrwyy)

Damn, savage but hilarious! xD

Show more

0

Reply

T

tuanl.718

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8qka0k0007vad083kr8jlxj)

This may be an insignificant question in the grand scheme of things, but for the feed fanout service, why is the arrow unidirectionally to the message queue? My intuition is that items from the message queue route to the service meaning that if there is a one-way arrow, it would be message queue --> feed fanout service

Otherwise amazing content! So glad I purchased premium

Show more

0

Reply

![J N](https://lh3.googleusercontent.com/a/ACg8ocIOlyaZ9tXl9uEDEYOtzOXZGQke18-R576Xb1QN3vakRNYHEjw=s96-c)

J N

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8qqjxen0042ad08la3yqvc9)

What we just fan out on read for each user and store the posts in a redis sorted set by ts. Once our sorted set is empty/below a threshold we can repopulate That way we don't need to handle celebrities separately?

Show more

0

Reply

W

WanderingBeigeCrab571

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8rozr9b006gad08f97b3tb4)

Reflection on why not to store binary data in OLTP DBs!

Avoid storing BLOBs (MBs/GBs) in OLTP DB — makes things slow. Why? DB pages are ~4KB, so large rows span many pages. Reading 1 row = multiple disk I/Os

- cache pollution (hot rows evicted). Because if mysql/postgresql cache any hot row containing large binary gets cached in memory, since it is large it will remove many other frequently accessed rows which make things more slow.

S3-like object stores are better — designed for large, sequential reads, cheaper, auto-scalable, and don't impact DB query performance.

Show more

0

Reply

M

MeaningfulSalmonRabbit753

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8t3o8mi00puad08nc2ea7l3)

How does pagination work here ? How do we combine result from the real time api call for feed and cached precomputed feeds.

How do we figure out the next set of posts to be sent in api response.

One way could be to pass on the last seen post timestamp and on the basis of that we can search in redis sorted set and also filter out in the real time for the posts of celebrity they follow.

One issue here could be that there can be few post with same timestamp already shown and few pending in the cache. Maybe sending all the posts with timestamp >= lastSeenTimestamp and then filtering on the client side should work. Is there a better way ?

Show more

0

Reply

S

stef4o

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8td5nwl000cad07msnsqvue)

Image/video digital fingerprint (hashing) as a means to avoid multiple copies was never mentioned. I think it makes sense in this design in order not to just save memory, but to optimize upload ...

Show more

0

Reply

C

CausalPlumTarsier691

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8vx27o302y8ad07cburrchn)

In the high level design, during fanout on read, why would we merge/sort in application code, rather than have the db engine join/order by Followers and Posts ?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8w1m70n000jad080uk971j3)

DynamoDB (which we chose in the design) doesn't support joins, so we have to do the merge in application code. Even if we used PostgreSQL, at Instagram's scale doing joins would be prohibitively expensive - you'd be joining massive tables across different shards. Much better to leverage indexes to quickly fetch the relevant rows and merge in memory, especially since we're only dealing with a small window of recent posts.

Show more

0

Reply

![Hoàng Nguyễn Đình](https://lh3.googleusercontent.com/a/ACg8ocInAQJQbjogIR-qykR97CRJmaPTouU8anqPXi3nbBapyh1YQx76=s96-c)

Hoàng Nguyễn Đình

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm8wt6oz4015mad08683xfg5w)

Hi, for staff+ level, can you provide the guideline to gradually improve the system from 1M users to 500M users? Like what should be focused on and whether this will be incremental changes or we have to rewrite everything.

Show more

0

Reply

W

walnatara2

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm94yx8xc0241ad08su23ab9s)

"Similarly, if we grow concerned about the price of DynamoDB, we can move infrequently accessed data over to S3."

if it is already goes to S3. What happen when the user suddenly request it after long time? what will happen in the data loading? Does it load from S3 to database again?

Show more

0

Reply

P

pricha2811

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9b0wssz0075ad08iwkb5jir)

I had a question, Evan, about CDNs, they charge based on data going in and out of it, so when there is miss on CDN do we allow the data to be fetched and added to CDNs (like a regular cache)? If so do we allow users to rack up the prices or do we choose what to add and what to evict? This issue isn't present on local caches because we are not being charged for data movement in and out. I might be entirely wrong about this (I am a junior level so most of my knowledge is textbook theory-based) .

Evan, your videos made all the other System design videos unwatchable for me. Please keep uploading more blogs and videos. You usually ask in your videos about the length, just wanted to mention, having more info (for beginners like myself) makes even the longest videos worthwhile.

Show more

2

Reply

Z

ZealousFuchsiaLeopon345

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9ck06lo00etad08tlnp7j0e)

How does the client know the CDN urls for the media while downloading? We are only storing the S3 links for the media in the database.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9cxflaa001sad088v3hib9z)

The CDN URLs are generated deterministically from the S3 object key. For example, if your S3 key is posts/123.jpg, your CDN URL might be cdn.instagram.com/posts/123.jpg. The CDN is configured to use S3 as the origin, so when it gets a request for that path, it knows to fetch from the corresponding S3 object.

Show more

1

Reply

Z

ZealousFuchsiaLeopon345

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9cxnl5w0021ad08583te8ta)

I see, so we store the object keys in the s3link column, and the client (or service) deterministically constructs the cdn url.

How would cdn.instagram.com be resolved to the nearest edge location to the end user?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9cy9azr002gad0773srm9uo)

Through DNS. When a client requests cdn.instagram.com, their DNS resolver gets directed to the CDN provider's DNS servers, which use the client's IP address to return the IP of the nearest edge location

Show more

0

Reply

Z

ZealousFuchsiaLeopon345

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9czyakx005had07fyoloswv)

> "which use the client's IP address to return the IP of the nearest edge location"

Any reading resources to read more on this?

Show more

0

Reply

![koushik sarkar](https://lh3.googleusercontent.com/a/ACg8ocKk9TUMqbCgj-dKEp9gXxbZL_s8gCbaYPJNi0msW3CS-lrmFcG6=s96-c)

koushik sarkar

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9h7jr7r0359ad08byx1gj31)

@Evan For a Staff+ Product Architecture interview at Meta, do we need to dive into details like database partitioning or video resolution specifics? From what I understand, these interviews usually cover both system design and product thinking. Given the limited 30–40 minute time frame, what areas should we prioritize or de-emphasize based on the interview type? Specifically, should we also touch on things like observability/metrics collection or the UI/UX experience, or are those better left out at this level?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9h7ly2t035ead086xk6lvq7)

Unfortunately, I can't give you the clear answer you may be looking for here. Generally speaking, observability could be mentioned in passing, but it wouldn't be a focus. Instead, your deep dives should be the intersection of what you know really well and what matters to the problem. If you don't know video encoding, then maybe focus on something else, but ultimately, you need an aggregate amount of depth to prove your technical excellence at staff+

Show more

1

Reply

S

samdreed21

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9kornxz00f1ad08vw28s9se)

Hey Evan,

I've noticed that in most of these designs there is no explicit mention of a load balancer, despite there being mention of horizontal scaling. Is that something we should be mentioning in our designs? Why is it left out?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9koteo000f7ad08hz748lju)

Depends on your level. If mid-level or lower, maybe you should mention it. But it's pretty well implied when you say horizontal scaling that you'll have a load balancer in front. This is largely taken for granted.

Show more

0

Reply

S

samdreed21

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9krjbl700euad08z7kie078)

Got it, thank you!

Show more

0

Reply

M

MusicalCoffeeBedbug291

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9mmt8nr00dead086eqjy8m0)

Hey Evan,

Using shared db vs stand alone for each service, I always do propose a shared db per service initially ( mostly using Aurora in multi-az mode), however mostly the interviewers have always pointed out and termed it as an anti pattern. So is the expectation during the interviews is to always start with an ideal distributed system and not practical ?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9mwhywb00h9ad08u3u9h8wn)

Shared DBs are totally fine in practice and many FAANG systems use them. The key is being able to articulate the tradeoffs. Separate DBs give you better isolation and independent scaling but increase complexity with distributed transactions. Shared DBs are simpler operationally but create tighter coupling. Neither is strictly right/wrong - depends on your specific needs. If an interviewer claims it's always an anti-pattern, they're being too dogmatic imo. BUT its important to call out that you recognize the tradeoffs/tensions.

Show more

1

Reply

![Ivan Atroshchenko](https://lh3.googleusercontent.com/a/ACg8ocKs2RdLYX_h3iuQWXlqtX5XXhV9XOGR24Ytb7Tq4fmMDCBRj68b=s96-c)

Ivan Atroshchenko

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9n3busl00u2ad08w9aeaox2)

Do You know if instagram uses any specific infrastructure provider. Is it AWS?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9n3m00i00x8ad08sypb4eas)

Meta has their own infra teams, they don't use any public clouds.

Show more

1

Reply

![Ivan Atroshchenko](https://lh3.googleusercontent.com/a/ACg8ocKs2RdLYX_h3iuQWXlqtX5XXhV9XOGR24Ytb7Tq4fmMDCBRj68b=s96-c)

Ivan Atroshchenko

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9n49tfk00ylad08qvkns9qx)

is it about privacy or costs ?

Show more

0

Reply

![Neha Gour](https://lh3.googleusercontent.com/a/ACg8ocKMswVmSIqEn_qtLJq0B3K62rPzH4055ngIcDIYIZOrOVf6iVJlaw=s96-c)

Neha Gour

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9qrv0s10029ad079u6k0ims)

How FR is different from Facebook News Feed?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9r7dqj500hoad07xgzbf51f)

They're very similar. Instagram was top voted, so despite FB news feed existing, people still wanted Instagram.

Show more

1

Reply

C

CivilianGreenPartridge214

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9rfmwd900uxad08xzwmmow5)

do we have to care about read-after-write consistency? (users seeing their own posts immediately on their page)

Show more

0

Reply

![Hemanth R](https://lh3.googleusercontent.com/a/ACg8ocICEo7vYYXTwSoi9ffWMxVjNpzYbpifyqVEUjtWWFds9vQ_=s96-c)

Hemanth R

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9wb1aq7011fad08epwf1vmw)

Great video and post — thanks for sharing such insightful content! I had one question I was hoping you could clarify: How does Instagram or any app using a CDN like CloudFront ensure that private content (like images from private accounts) isn’t accessible via a shared or cached CDN link? Wouldn’t that potentially violate the privacy of private account holders, or am I missing something?

Show more

0

Reply

G

ghanekar.omkar

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9we14cl00vmad083kqugw92)

Unrelated to this specific problem but does the "Mark as read" toggle do anything? I expected it to maybe paint a small tick next to the problem in the left pane or maybe a tick at the top of the page for a given problem but I noticed no difference. Is it just me?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cma55985b0059ad07zq5driyx)

It updates the checklist on your dashboard www.hellointerview.com/dashboard.

Will likely add check marks to the left nav one day soon too

Show more

0

Reply

S

shantanur3

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9y68khw00onad073hvbhv9m)

We'll have high read to write ratio. So shouldn't the feed generation logic be separated? Need your thoughts on this.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cma558fm20053ad07sdo6171o)

Yah this makes sense and is a good enough justification.

Show more

0

Reply

I

itbtech06

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cm9yz7w4j01kjad08nhwovhie)

How do we know which s3URL goes to which postId after the upload is done? If we are using the S3 notification system then we need to pass the postId from s3 to the lambda which is going to update the Post table with status and s3Link. Are you saying the postId and the objectKey will be same in that case? More interestingly if user uploads more than one video for one post(I think insta allows upto 20 videos per post) then how are we going to store the video in S3? are we storing at the same place in S3 or different? If different then the schema for S3URLs will be a list right?

On the CDN topic, who is uploading the media content to the CDN and how?

Show more

0

Reply

I

itbtech06

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cma39y56q00j6ad08fi5dxxq1)

Maybe we can prefix the postId along with the fileName while creating the pre-signed URL and when the upload is completed then the S3 notification will return the object key with the postId+filename to Lambda, then Lambda can get the postId from the notification and update the database with upload status as "uploaded" for that postId.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cma5573rs004xad07ykt780hz)

Bingo!

Show more

0

Reply

![Howie Zhou](https://lh3.googleusercontent.com/a/ACg8ocJE7yku1bbWSRWOikGLG1w84ETGND_dgfQbcpnJ1Bq_6gDUxy4=s96-c)

Howie Zhou

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cma7tkn8y008uad0875zttvhs)

"We can key the cache by a combination of the user\_id and a cursor (or timestamp). This lets us get a specific "page" of the feed from the cache. An example key would be: feed:{user\_id}:{cursor}. The value would be a list of postIds."

I'm not sure if using the cursor as the key here is the right approach. We should aggregate the posts first, then apply pagination—not the other way around. Using something like feed:{user\_id} might be a better structure. Sharding by cursor could be quite difficult.

Show more

0

Reply

K

kstarikov

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmae8r82u02guad079p6ruen0)

Now for the most important question: what did Evan do at 36:27?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmae9syil0257ad08v4071h0k)

Oh shoot, bad editing lol :( Let me crop that out. Think I got a text haha

Show more

1

Reply

![GGu T](https://lh3.googleusercontent.com/a/ACg8ocIdslsqsPj4kru3lxTs-5qRxv4nSsxr4X5YU93-rd7bfkNkdg=s96-c)

GGu T

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaewgvgn00dxad086983j6yt)

"We can make the sort key a composite of createdAt and postId to ensure chronological ordering while maintaining uniqueness."

I don't think DynamoDB supports composite sort key? Both sharding key and sort key need to be a single column. Cassandra indeed supports composite clustering keys.

Show more

0

Reply

T

Tom

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmcauyf4t01bkad08804uyqe0)

You can concatenate them.

Show more

1

Reply

P

prats66

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmagv6hi20295ad08pxev0z3n)

Excellent Article!!

Show more

1

Reply

F

FrontYellowPheasant390

[• 3 months ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaj2izog01jtad085f9yrloo)

could you plz add CC to the video, some of us not native speaker hah

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmcauz4n601bdad080waqk69l)

Done :) All premium videos have CC

Show more

0

Reply

![Mohith Raju](https://lh3.googleusercontent.com/a/ACg8ocJsiScf1lwfQzAnIucefXntLEJflcPzttbtqRRRnIiv1sPfMw=s96-c)

Mohith Raju

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmajzqhz702hyad08qjzss5of)

Hi Evan, Is is possible to like a folder of all the excalidraws that you worked on. It would be really helpful to take notes on that and then revise any concepts before the interviews by going through all the excalidraws.

Show more

1

Reply

A

AddedLimeCarp593

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaknvbe903sbad08l3cel7fh)

How does the design ensure that a new post isn't shown in user feeds until all the post's media is successfully uploaded to S3?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaoiiywf00c1ad07a5xlg5ck)

When a post is created, its metadata is stored with a status of 'pending'. Only after the S3 multipart upload completes (and we get a success callback) do we update the status to 'complete' and start fanning out to user feeds. The post won't show up in any feeds during the pending state.

Show more

0

Reply

A

AddedLimeCarp593

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaojaj6300faad08sk5pv62d)

So the machine that handles the S3 upload complete notification is going to 1) update the status to 'complete' and 2) put the post in the message queue so it can be fanned out? The final diagram shows the Post Service putting the post in the queue so that's where I'm confused.

Show more

0

Reply

![nikhil keswaney](https://lh3.googleusercontent.com/a/ACg8ocIiIM9gt3L1x3YC_kLJ_2yhaJkqqA81WIveITkHsJW9rYsSIDrWxw=s96-c)

nikhil keswaney

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaohj0ai00f7ad07xzndc2eg)

any reason why we did not use DDB streams?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaoiigow00bxad07l982i9ah)

Could work to trigger the fanout if your primary story is DDB

Show more

1

Reply

I

itbtech06

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmaq50swm002aad08o61rkxlo)

How does PostService know that the post is from a celebrity? Can we add a column in the User table as isCelebrity=true instead of getting the count of followers? This will be cheaper and faster than back to back queries to DDB.

Show more

1

Reply

U

UrbanAmethystPrimate100

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmb2z0yoj00rbad08z0mf9sq4)

For delete operations, is it necessary to manually remove cached data for all users who have cached a generated post? I understand this may be out of scope since it's not part of the current functional requirements.

However, if we were to support deletion, how should the system design be adapted? Specifically:

Should the Redis cache delete the post ID and also remove references from any users who were following or had cached that post?

Alternatively, should clients simply skip rendering post IDs that no longer exist in the cache, accepting some level of stale data?

What would be the trade-offs between these approaches in terms of consistency, performance, and complexity?

Show more

0

Reply

P

PastLimeBug305

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmb3bsmea0140ad08id99nt8u)

"As is always the case, more complexity comes with its own tradeoffs. The 100,000 follower threshold needs to be carefully tuned - set it too low and we don't solve the write amplification problem, set it too high and we impact read performance for too many users." isn't it mixed up here? If threshold is too low, we need to read recent posts for more users we follow (many "celebrities"); if it is too high, then we need to pre-compute many feeds (only few "celebrities") hence write amplification.

Show more

0

Reply

S

SurprisedTomatoPanda219

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmb4b8ckr02z1ad081pfie4z5)

Why sorted set in Redis for post feed? Can we just use a simple list and prepend the post instead of append since we always get newer posts later?

Show more

0

Reply

![Jack Copland](https://lh3.googleusercontent.com/a/ACg8ocIuc_0acp8OBG__bZ3WuVYaqssUDrx7kEywyatLki56KL2Nhw=s96-c)

Jack Copland

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmb4vzjsb033bad08corggc4y)

In an interview for something similar, I was asked about alternative feed generation strategies, which turned out to be something ML related. I don't know enough to even talk about how this would work, beyond knowing some vague keywords like "embedded" or "recommender system".

Is it okay in that scenario to abstract it into some sort of black box and just say "I don't know enough about this topic to give a meaningful answer", or should I really just watch a couple hours of videos on that topic and hope to bluff my way through?

Show more

0

Reply

E

EasternChocolateLimpet925

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmbg7jgfu00yzad082iy9tann)

Considering the entire feed comes from the cache, what is going to be the approx size of this cache? Could the size of the cache be a bottleneck here?

Show more

0

Reply

![Anthony Kosky](https://lh3.googleusercontent.com/a/ACg8ocIC7Uw0NLB3b9vzWfIlvcYeAd198M42MYIIFd_UFO443WH1nDFl=s96-c)

Anthony Kosky

[• 2 months ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmbuf94xz052807ad4erz59hs)

With regard to the comments about whether microservices should have independent databases, there are some compromises that lie between these two extremes. A single PostgreSQL database, for example, will support multiple "schemas". Terminology seems to vary a lot between databases, but essentially many databases have some sort of namespace concept. Also, one can set up different Roles for different microservices, limiting the tables that they have read or write access to. This sort of approach gives us a lot of the isolation of separate databases, without the inconvenience or overhead of cross-database joins.

(Not sure I'd want to bring this up in my up-coming interview though :-))

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmcav0ykk01ckad08un98klv1)

Sounds like a great thing to bring up to me tbh! Shows practical, hands on experience

Show more

0

Reply

![Prashant Nigam](https://lh3.googleusercontent.com/a/ACg8ocJuxnSuuGVQIDjxhaz20U3M46Ge3w8MVcwcZp_HgrpKx7pYVEF_=s96-c)

Prashant Nigam

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmc1hx81w0dkm08adjq4ykgc4)

> Then, we also query the Posts table for recent posts from the "celebrities" they follow

Why are we reading celeb posts directly from the table and not caching it and reading from there first?

When generating a user's feed:

1. Use fan-out-on-read for celebrity posts
2. Check the Redis cache (keyed by celebrity ID) if available (Cache Hit). redis: {key= celbrity\_id, value = list\[post1, post2, post3...\]}
3. Fall back to the DB in case of cache miss and populate the cache

I understand there will be Hot Keys issue here, which can be solved using

- Redis Clustering with key sharding
- Read replicas

What am I missing?

Show more

0

Reply

C

CurrentSapphireFox662

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmcgg5fc408adad08i3kes0ox)

one qq, for posts, there may be a lot of data (~1PB) to store while there might be just a few data for following, in this case, is it still better to put them in the same database? (like CamelCamelCamel, you seperate dbs because they have different scale of data, why not also do the same here?) thanks

Show more

0

Reply

M

ManyBlackHorse876

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmcmohi9h004vad08xanc2pn3)

There's a little break for the video at around 36:28. Is this intended?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmcmomhe2005dad08mavjyy9b)

no, just bad editing. will fix :)

Show more

0

Reply

JZ

Jon Z

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmd23doii01mcad081c2u9d8d)

Amazing video - one question though - How does the post service know who is a celeb or not. Does the post service need to still get all my followers, check if each one is a celeb or not? Also where should we store that information, on a user table (like follower\_count)? Wouldnt that still require the service to get all followers, and for each one query again to get their count? Other options I can think of is 1. storing it on follower table, but I think would have scaling issues as that count number will change frequently 2. Storing a celeb list on redis which well get and check against for each follower? What is the prefred way to handle this? Afraid of being asked this follow up in an interview and not knowing

Show more

1

Reply

J

Jay

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmd3h8zld048jad08175s1nom)

Hey Evan, greatly apprecaite the walk through! I have two small questions.

1. For the fanout pattern, I was using the database cdc event, so it only happens once the post has successfully been persisted in the DB. But the solution in the walk-through is doing the fanout in parallel. Guess that will benefit the latency, but would have the race condition when it's correctly fanned out but not written in the DB. I think it's a good trade off to disucss in a real interview
    
2. Seems for completing the multi-part upload in S3, the server has to call the S3 with eTags\[\] receveid from the client side, so the clients need to call an API exposed from the server to provide those eTags\[\]. (see more at https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html). Guess that's not the core of this topic but just paste what I found here
    

Show more

0

Reply

![Tharun Reddy](https://lh3.googleusercontent.com/a/ACg8ocIWd4vtgz0bkWJ2r0o61qrayKCq90wJWq2IPgigZ3wdQZoU2g=s96-c)

Tharun Reddy

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmdh9ueiy058uad07m2f6y9yn)

questions

1. for fan out on read. How would the system know ahead that user is following a person that's having more followers so that it need to go and get it from DB instead of entirely relying on cache
2. Could you please explain about the images/video upload. In this process are we storing the image files before sending to Kafka or once the messages are retrieved from Kafka by the fanout service?

Show more

0

Reply

A

abhin8425

[• 24 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmdpzhym002mfad079876vatt)

Premium videos load very slowly

Show more

0

Reply

X

XerothermicBlueGibbon404

[• 22 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmdth8ucb013mad08ct7fgrpn)

I proposed this exact solution with all deep dives mentioned here (except media processing optimization, which I would have done but the interviewer didn’t seem interested in that even a bit.) in meta interview recently and got a no hire for this round. Still unsure what went wrong. I came out of the interview very confident that I will pass this round for at least e5 if not e6. Not doubting the solution, but the way this interviewer assessed me. He was constantly asking to dig deep in the precomputed feed cache flow and even after explaining everything how it’s explained here, he still seemed to have some doubt.

Show more

0

Reply

C

CausalPlumTarsier691

[• 20 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmdvr8y7o0lorad084facui91)

No chunking required for download stage for 4GB videos unlike youtube?

Show more

0

Reply

P

ProperLimePython690

[• 18 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmdysrfx20a0qad08hp68a19l)

I dont think problem covers about read your own writes in this case. Can you elaborate on how we can ensure that?

Show more

0

Reply

![Priyangshu Roy](https://lh3.googleusercontent.com/a/ACg8ocJXi2S6LLHV4HR59WPr_PKRcpuZtBGgrBG7-HsFT24DMocISQ=s96-c)

Priyangshu Roy

[• 17 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cme0wdz4f04dvad08eytzz68f)

Our feed comes entirely from redis, what is the kind of cache size that we are looking at?

Show more

0

Reply

U

UnitedBronzeSawfish390

[• 14 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cme4t230d0bhcad08y88m0doj)

Evan, thanks a lot for yet another great article. Could you please share your Excalidraw file?

Show more

0

Reply

W

wentz.vagner

[• 9 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmebijncn08biad07n2rjhsgp)

I have a question about the minute ~50. For an example, the video told us about multi-part upload, that we can upload directly to s3 and then, it will emit some event from lambda/etc saying that the upload is complete. What is the difference to use the pre-signed url, I noticed it is common to use here in the mock videos but for this situation, why it was not used?

Show more

0

Reply

R

RainyCyanHippopotamus256

[• 8 days ago• edited 8 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram#comment-cmee44sdy01v1ad082tg1qceo)

I have one question regarding populating user feeds for famous and non-famous users. Consider the following scenario:

1. A user is famous and posts a new post. The fan-out architecture will ignore this post because the user is famous, and will not pre-compute the user feed for the user's followers.
2. The user who posted the post now becomes non-famous (due to decreased follower count).
3. One of the user's followers now queries the feed API or page. Since the user was famous before, the post won't be present in pre-computed feed. Also, since the user is non-famous now, we will not query the posts table for his/her posts. As a result, the post which was made when the user was famous would be non-queryable.

How do we efficiently tackle the above scenario/edge-case?

Show more

0

Reply
