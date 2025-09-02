##### Design DoorDash

by Ankit Thakur• Senior• 5 days ago

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

[• 5 days ago](https://www.hellointerview.com/community/submissions/cmel33qlu01clad096a4v1j4y#comment-cmel367ya002kad3a2o4vxaoi)

Nice design! You've built a solid foundation for DoorDash with good architectural choices like location-based sharding and using Redis geospatial indexing for real-time dasher tracking. Your separation of concerns between ordering, search, and dasher matching services shows thoughtful decomposition. However, there are some critical scalability and consistency challenges that need addressing - particularly around handling billions of location updates and ensuring transactional consistency across your sharded architecture.

1.  **Positives**
    
    -   Sharding services by location (Restaurant, Location, Dasher Matching) is smart for geo-distributed workloads and keeps related data close together
    -   Using Redis geospatial index with GADD/GSEARCH for dasher locations provides efficient proximity queries with appropriate TTLs
    -   Separating search into ElasticSearch with location-based indexing will meet your <500ms search latency requirement effectively
2.  **Potential Issues** **Location Update Storm**
    
    -   High frequency location updates from millions of dashers will generate billions of requests per minute to your system
    -   Even with sharding, each location service instance would need to handle millions of updates per second
    -   This will saturate network bandwidth and overwhelm Redis even with geospatial optimizations
    
    **Cross-Shard Transaction Complexity**
    
    -   Orders involve user location, restaurant location, and dasher location which may span multiple shards
    -   No clear strategy for handling orders where user and restaurant are in different location shards
    -   This creates distributed transaction requirements that your current design doesn't address
    
    **DynamoDB for Transactional Workloads**
    
    -   Using DynamoDB for orders limits query flexibility - you'll need to query by user, restaurant, dasher, status, time ranges
    -   Orders require ACID properties during payment and assignment which DynamoDB doesn't naturally provide
    -   The complex access patterns for order history, analytics, and customer service will be difficult to implement efficiently
    
    **Oversimplified Dasher Assignment**
    
    -   "Nearest dasher" doesn't account for dasher's current orders, direction of travel, or capacity
    -   No consideration for dasher preferences, ratings, or historical performance
    -   Missing logic for handling declined requests and reassignment strategies
3.  **Follow-up Questions**
    
    -   How do you handle location updates efficiently when millions of dashers update every few seconds?
    -   What happens when a user in New York orders from a restaurant in New Jersey (cross-shard)?
    -   How do you ensure payment atomicity when the payment provider is external and uses CDC?
    -   What's your strategy for pre-scaling during predictable meal-time rushes vs reactive scaling?
    -   How do you handle dasher assignment when the nearest dasher declines - is there a timeout?
    -   What happens to in-flight orders when a dasher's app crashes or loses connectivity?

_Note: This feedback is based on modern infrastructure capabilities where single database instances can handle 10k+ TPS and message brokers can process 1M+ messages/second._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply