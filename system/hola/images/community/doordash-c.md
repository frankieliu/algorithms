##### Design DoorDash

by SurePeachGayal546• Manager• 4 days ago

Mix of Uber + TicketMaster + GoPuff answers from Hello Interview

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

[• 4 days ago](https://www.hellointerview.com/community/submissions/cmemj3umv0438ad08ysdvrkd3#comment-cmemj5x80004jad3acqf4wzac)

Nice design! You've included several important components for a food delivery system, including CDN caching for static content, distributed locking for dasher matching, and asynchronous processing with SQS. The use of ElasticSearch for restaurant search and Redis for caching shows good understanding of appropriate technology choices. However, there are some scalability concerns around database sharding and geospatial queries that would need addressing before this could handle DoorDash's scale.

1.  **Positives**
    
    -   Using CDN for restaurant menus and static data is smart - reduces load on backend services for frequently accessed, rarely changing content
    -   Implementing distributed locks with Redis for dasher matching prevents race conditions where multiple orders could be assigned to the same dasher
    -   Decoupling order placement from dasher matching via SQS allows orders to be accepted quickly while matching happens asynchronously
2.  **Potential Issues** **Missing database sharding strategy**
    
    -   You're using a single DynamoDB instance for all order, restaurant, and user data
    -   At peak hours, DoorDash processes millions of orders - a single DynamoDB table can only handle ~40k writes/second
    -   This creates an impossible scaling ceiling that will cause write throttling and order failures during dinner rush
    
    **Inefficient dasher location queries**
    
    -   The dasher matching service appears to query all dashers from the location DB to find nearby ones
    -   With thousands of active dashers in a city, scanning all locations for each order match is O(n) and won't scale
    -   You need geospatial indexing (like PostGIS or DynamoDB with geohashing) to efficiently query dashers within a radius
    
    **Confusing order vs ride terminology**
    
    -   Your API shows PATCH /rides/:rideId but this is a food delivery system, not rideshare
    -   This inconsistency suggests the order lifecycle and dasher assignment flow may not be fully thought through
    -   The system needs clear separation between customer orders, restaurant acceptance, and dasher delivery tasks
3.  **Follow-up Questions**
    
    -   How do you efficiently find the 10 closest available dashers to a restaurant without scanning all dasher locations?
    -   What happens to the dasher assignment if a restaurant rejects an order after matching?
    -   How does the system calculate accurate delivery time estimates based on restaurant prep time and dasher location?
    -   What's your strategy for handling 10x load during weekend dinner peaks?
    -   Why do you need CDC between DynamoDB and ElasticSearch when restaurants are relatively static data?
    -   How do you ensure dashers don't get overwhelmed with too many simultaneous delivery requests?

_Remember to explain your data partitioning strategy and geospatial query optimization in your next iteration!_

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply