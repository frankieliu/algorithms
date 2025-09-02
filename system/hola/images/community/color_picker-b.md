##### Design a Color Picker

by FutureHarlequinMeerkat461• Senior• 11 days ago

Use Cases:

1.  Users can create a favorite colors list and add their favorite colors to it. This is done via the Favorites Service, which receives the request and stores it in the Favorites DB (Dynamo DB).
    
2.  Users can share their lists with other (possibly multiple users). We use a separate color sharing service to handle this. Upon receiving a request, color sharing service makes an entry in the Shares DB and pushes a record into Kafka identify itself as the source of change (key = source user id). A worker pulls messages off of Kafka, gets the list of shared users (target users for notifications) by querying the shares table and instructs a separate notification service to send email notifications about the color list share. Simultaneoulsy, the worker pushes a record to a second queue responsible for feed generation for target users.
    
3.  Users should see a feed with color lists shared with them. Feed generation for users happens asynchronously using another worker reading from the Feed Generation queue. This worker fetches the list of users who have shared their color lists with themselves (using a Global Secondary Index), fetches all the favourite colors for each of these users from the Favorites DB and generates the Color Feed Cache. Feed Service just reads from this Feed Cache to show the color feed.
    
4.  Handling change in favorite color list: If a user who has shared their favorite list updates the list, via CDC we push the user\_id into the Notification Queue. The Worker then sends email notifications to all the target users with whom the list was shared and the second worker would update all of their color feed caches.
    

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

[• 11 days ago](https://www.hellointerview.com/community/submissions/cmecu3su705rrad08ce4ybe8r#comment-cmecu5ptp0075ad3av294aggv)

Nice design! You've done a good job separating concerns with dedicated services for favorites, sharing, feeds, and notifications. Your use of asynchronous processing with Kafka for notifications and feed updates is smart for handling the scale requirements, and using CDC to propagate changes shows thoughtful consideration of data consistency patterns. The GSI on shared\_user\_id for efficient reverse lookups is a particularly clever touch.

1.  **Positives**
    
    -   Using separate services (Favorites, Sharing, Feed, Notification) provides good modularity and allows independent scaling of each component based on load patterns
    -   Asynchronous processing via Kafka queues for notifications and feed generation prevents blocking operations and handles spiky workloads gracefully
    -   CDC (Change Data Capture) for propagating favorites updates ensures you capture all changes without modifying application code, providing eventual consistency
2.  **Potential Issues** **Feed generation fanout explosion**
    
    -   Your worker updates ALL target users' feed caches whenever someone shares or updates their list
    -   If a popular user shares with 1M followers, you're updating 1M cache entries synchronously
    -   This creates O(n\*m) complexity where n=sharers and m=followers, causing massive queue backlogs and stale feeds
    
    **Inefficient DynamoDB schema for favorites**
    
    -   Using color as the sort key means each color is a separate item in DynamoDB
    -   Fetching all colors for a user requires a query operation that might span multiple pages
    -   This increases latency and cost compared to storing the entire color list as a single item or using a more appropriate data structure
    
    **Feed cache scalability**
    
    -   With 1B users, even 10KB per user's feed means 10TB of cache data
    -   No mention of sharding strategy or cache eviction policies
    -   A single feed cache becomes a bottleneck and single point of failure at this scale
3.  **Follow-up Questions**
    
    -   How does feed pagination work when users have thousands of shared lists?
    -   What happens when a celebrity user with 100M followers updates their color list?
    -   How do you handle cache misses and cold starts for users who haven't accessed their feed recently?
    -   Why use CDC instead of having the Favorites Service directly publish change events to Kafka?
    -   How do you ensure read-after-write consistency when a user shares a list and immediately checks if it was shared?
    -   What's your strategy for handling partial failures during the feed update process?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply