# Design Yelp

Scaling Reads

[![Evan King](/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75&dpl=1a01e35ef00ef01b910d317b09313e145b78f47f)

Evan King

Ex-Meta Staff Engineer

](https://www.linkedin.com/in/evan-king-40072280/)

medium

Published Jul 15, 2024

---

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

Deep Dives

51:54

Play

Mute

0%

0:00

/

59:15

Premium Content

Closed-Captions On

Chapters

Settings

AirPlay

Google Cast

Enter PiP

Enter Fullscreen

## Understanding the Problem

**🍽️ What is [Yelp](https://www.yelp.com/)?** Yelp is an online platform that allows users to search for and review local businesses, restaurants, and services.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements)

Some interviewers will start the interview by outlining the core functional requirements for you. Other times, you'll be tasked with coming up with them yourself. If you've used the product before, this should be relatively straight forward. However, if you haven't, it's a good idea to ask some questions of your interviewer to better understand the system.

Here is the set of functional requirements we'll focus on in this breakdown (this is also the set of requirements I lead candidates to when asking this question in an interview)

**Core Requirements**

1. Users should be able to search for businesses by name, location (lat/long), and category
    
2. Users should be able to view businesses (and their reviews)
    
3. Users should be able to leave reviews on businesses (mandatory 1-5 star rating and optional text)
    

**Below the line (out of scope):**

- Admins should be able to add, update, and remove businesses (we will focus just on the user)
    
- Users should be able to view businesses on a map
    
- Users should be recommended businesses relevant to them
    

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements)

**Core Requirements**

1. The system should have low latency for search operations (< 500ms)
    
2. The system should be highly available, eventual consistency is fine
    
3. The system should be scalable to handle 100M daily users and 10M businesses
    

**Below the line (out of scope):**

- The system should protect user data and adhere to GDPR
    
- The system should be fault tolerant
    
- The system should protect against spam and abuse
    

If you're someone who often struggles to come up with your non-functional requirements, take a look at this list of [common non-functional requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements) that should be considered. Just remember, most systems are all these things (fault tolerant, scalable, etc) but your goal is to identify the unique characteristics that make this system challenging or unique.

Here is what you might write on the whiteboard:

Yelp Non-Functional Requirements

### Constraints

Depending on the interview, your interviewer may introduce a set of additional constraints. If you're a senior+ candidate, spend some time in the interview to identify these constraints and discuss them with your interviewer.

When I ask yelp, I'll introduce the constraint that **each user can only leave one review per business.**

## The Set Up

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

We recommend that you start with a broad overview of the primary entities. At this stage, it is not necessary to know every specific column or detail. We will focus on the intricacies, such as columns and fields, later when we have a clearer grasp. Initially, establishing these key entities will guide our thought process and lay a solid foundation as we progress towards defining the API.

Just make sure that you let your interviewer know your plan so you're on the same page. I'll often explain that I'm going to start with just a simple list, but as we get to the high-level design, I'll document the data model more thoroughly.

To satisfy our key functional requirements, we'll need the following entities:

1. **Business**: Represents a business or service listed on Yelp. Includes details like name, location, category, and average rating.
    
2. **User**: Represents a Yelp user who can search for businesses and leave reviews.
    
3. **Review**: Represents a review left by a user for a business, including rating and optional text.
    

In the actual interview, this can be as simple as a short list like this. Just make sure you talk through the entities with your interviewer to ensure you are on the same page.

Yelp Entities

### [The API](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#4-api-or-system-interface)

The next step in the framework is to define the APIs of the system. This sets up a contract between the client and the server, and it's the first point of reference for the high-level design.

Your goal is to simply go one-by-one through the core requirements and define the APIs that are necessary to satisfy them. Usually, these map 1:1 to the functional requirements, but there are times when multiple endpoints are needed to satisfy an individual functional requirement.

To search for businesses, we'll need a GET endpoint that takes in a set of search parameters and returns a list of businesses.

`// Search for businesses GET /businesses?query&location&category&page -> Business[]`

Whenever you have an endpoint that can return a large set of results, you should consider adding pagination to it. This minimizes the payload size and makes the system more responsive.

To view a business and its reviews, we'll need a GET endpoint that takes in a business ID and returns the business details and its reviews.

`// View business details and reviews GET /businesses/:businessId -> Business & Review[]`

While this endpoint is enough, you could also split the business and reviews into two separate endpoints. This way we can have pagination on the reviews.

`// View business details GET /businesses/:businessId -> Business  // View reviews for a business GET /businesses/:businessId/reviews?page= -> Review[]`

To leave a review, we'll need a POST endpoint that takes in the business ID, the user ID, the rating, and the optional text, and creates a review.

`// Leave a review POST /businesses/:businessId/reviews {   rating: number,   text?: string }`

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes)

We'll start our design by going one-by-one through our functional requirements and designing a single system to satisfy them. Once we have this in place, we'll layer on depth via our deep dives.

### 1) Users should be able to search for businesses

The first thing users do when they visit a Yelp-like site is search for a business. Search includes any combination of name or term, location, and category like restaurants, bars, coffee shops, etc.

We already laid out our API above of GET /businesses?query&location&category&page, now we just need to draw out a basic architecture that can satisfy this incoming request.

To enable users to search for businesses, we'll start with a basic architecture:

Yelp High-Level Design

1. **Client**: Users interact with the system through a web or mobile application.
    
2. **API Gateway**: Routes incoming requests to the appropriate services. In this case, the Business Service.
    
3. **Business Service**: Handles incoming search requests by processing query parameters and formulating database queries to retrieve relevant business information.
    
4. **Database**: Stores information about businesses such as name, description, location, category, etc.
    

When a user searches for a business:

1. The client sends a GET request to /businesses with the search parameters as optional query params.
    
2. The API Gateway routes this request to the Business Service.
    
3. The Business Service queries the Database based on the search criteria.
    
4. The results are returned to the client via the API Gateway.
    

### 2) Users should be able to view businesses

Once users have submitted their search, they'll be viewing a list of businesses via the search results page. The next user action is to click on a business to view it's details.

Once they do, the client will issue a GET /businesses/:businessId request to the API Gateway.

To handle this, we don't need to introduce any additional services. We can just have the API Gateway route the request to the Business Service. The Business Service will query the Database for the business details and then the reviews. For now, we'll keep reviews in the same database as the businesses, but we'll need to make sure to join the two tables.

###### Pattern: Scaling Reads

Given the massive read:write ratio, reading businesses is an perfect use case for the scaling reads pattern.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/scaling-reads)

Yelp High-Level Design

A common question I receive is when to separate services. There is no hard and fast rule, but the main criteria I will use are (a) whether the functionality is closely related and (b) whether the services need to scale independently due to vastly different read/write patterns.

In this case, viewing a business and searching for a business are closely related, and the read patterns are similar (both read-heavy), so it makes sense to have this logic as part of the same service for now.

When a user views a business:

1. The client sends a GET request to /businesses/:businessId.
    
2. The API Gateway routes this to the Business Service.
    
3. The Business Service retrieves business details and reviews from the Database.
    
4. The combined information is returned to the client.
    

### 3) Users should be able to leave reviews on businesses

Lastly, given this is a review site, users will also want to leave reviews on businesses. This is just a mandatory 1-5 star rating and an optional text field. We won't worry about the constraints that a user can only leave one review per business yet, we'll handle that in our deep dives.

We'll need to introduce one new service, the Review Service. This will handle the creation and management of reviews. We separate this into a different service mainly because the usage pattern is significantly different. Users search/view for businesses a lot, but they hardly ever leave reviews. This insight actually becomes fairly crucial later on in the design, stay tuned.

Yelp High-Level Design

When a user leaves a review:

1. The client sends a POST request to /businesses/:businessId/reviews with the review data.
    
2. The API Gateway routes this to the Review Service.
    
3. The Review Service stores it in the Database.
    
4. A confirmation is sent back to the client.
    

Should we separate the review data into its own database? Aren't all microservices supposed to have their own database?

The answer to this question is a resounding maybe. There are some microservice zealots who will argue this point incessantly, but the reality is that many systems, use the same database for multiple purposes and it's often times the simpler and, arguably, correct answer.

In this case, we have a very tiny amount of data, 10M businesses x 100 reviews each = 1TB. Modern databases can handle this easily in a single instance, so we don't even need to worry about sharding. Additionally, reviews and businesses are tightly coupled and we don't want to have to join across services to get the business details and reviews.

The counter argument is typically related to fault isolation and operational responsibility. We want to make sure that if the review database goes offline, we aren't left unable to search or view businesses. While this is a valid concern, we can mitigate it via other means like simple replication.

At the end of the day, it's a discussion of trade-offs with no single correct answer. I bias toward simplicity unless I can articulate a clear benefit to adding complexity and suggest you do the same if not already strongly principled on the matter.

Show More

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes)

At this point, we have a basic, functioning system that satisfies the functional requirements. However, there are a number of areas we could dive deeper into to improve the system's performance, scalability, and fault tolerance. Depending on your seniority, you'll be expected to drive the conversation toward these deeper topics of interest.

### 1) How would you efficiently calculate and update the average rating for businesses to ensure it's readily available in search results?

When users search for businesses, we don't show the full business details of course in the search results. Instead, we show partial data including things like the business name, location, and category. Importantly, we also want to show users the average rating of the business since this is often the first thing they look at when deciding on a business to visit.

Calculating the average rating on the fly for every search query would be terribly inefficient. Let's dive into a few approaches that can optimize this.

### 

Bad Solution: Naive Approach

###### Approach

As alluded to, the simplest approach is to calculate the average rating on the fly. This is simple and requires no additional data or infrastructure.

We'd simply craft a query that joins the businesses and reviews and then calculates the average rating for each business.

`SELECT   b.business_id,   b.name,   b.location,   b.category,   AVG(r.rating) AS average_rating FROM businesses b JOIN reviews r ON b.business_id = r.business_id GROUP BY b.business_id;`

###### Challenges

While this approach is simple, it has significant scalability issues:

1. Performance degradation: As the number of reviews grows, this query becomes increasingly expensive. The JOIN operation between businesses and reviews tables can be particularly costly, especially for popular businesses with thousands or millions of reviews.
    
2. Unnecessary recalculation: The average rating is recalculated for every search query, even if no new reviews have been added since the last calculation. This is computationally wasteful.
    
3. Impact on read operations: Constantly running this heavy query for every search can significantly slow down other read operations on the database, affecting overall system performance.
    

Given our scale of 100M daily users and 10M businesses, this naive approach would likely lead to unacceptable performance and user experience issues.

### 

Good Solution: Periodic Update with Cron Job

###### Approach

Alternatively, we could pre-compute the average rating for each business and store it in the database. This pre-computation can be done using a cron job that runs periodically (e.g., once a day, once an hour, etc). The cron job would query the reviews table, calculate the average rating for each business, and update a new average\_rating column in the businesses table.

Now, anytime we search or query a business, we can simply read the average\_rating from the database.

Periodic Update with Cron Job

###### Challenges

The main downside with this approach is that it does not account for real-time changes in reviews. You can imagine a business with very few reviews and a currently low average rating. If you give the business a 5-star rating, your expectation would be that the average rating increases to reflect your vote. However, since we're recalculating the average rating periodically, this may not happen for hours or even days, leading to a stale average rating and confused users.

### 

Great Solution: Synchronous Update with Optimistic Locking

###### Approach

Our goal with the great solution is to make sure we can update the average rating as we receive new reviews rather than periodically via a cron job. Doing this is relatively straight forward in that we simply need to both add the new review to the Review table while also updating the average rating for the business in the Business table.

To make that update efficient, we can introduce a new column, num\_reviews, into the Business table. Now, to update an average rating, we simple calculate (old\_rating \* num\_reviews + new\_rating) / (num\_reviews + 1), which is cheap, just a few CPU cycles so it can be done synchronously for each new review.

Synchronous Update with Optimistic Locking

###### Challenges

In doing this, we've actually introduced a new problem. What happens if multiple reviews come in at the same time for the same business? One could overwrite the other, leading to an inconsistent state!

Let's imagine a timeline of events:

1. Business A has 100 reviews with an average rating of 4.0
    
2. User 1 reads the current state: num\_reviews = 100, avg\_rating = 4.0
    
3. User 2 also reads the current state: num\_reviews = 100, avg\_rating = 4.0
    
4. User 1 submits a 5-star review and calculates: new\_avg = (4.0 \* 100 + 5) / 101 ≈ 4.01
    
5. User 2 submits a 3-star review and calculates: new\_avg = (4.0 \* 100 + 3) / 101 ≈ 3.99
    
6. User 1's update completes: num\_reviews = 101, avg\_rating = 4.01
    
7. User 2's update overwrites: num\_reviews = 101, avg\_rating = 3.99
    

The final state (num\_reviews = 101, avg\_rating = 3.99) is incorrect because it doesn't account for User 1's 5-star review. The correct average should be: (4.0 \* 100 + 5 + 3) / 102 ≈ 4.00

To solve this issue, we can use optimistic locking. Optimistic locking is a technique where we first read the current state of the business and then attempt to update it. If the state has changed since we read it, our update fails. We'll often add a version number to the table you want to lock on, but in our case, we can just check if the number of reviews has changed.

Now, when user 2 would have overwritten user 1's review, our update will fail because the number of reviews has changed. User 2 will have to read the state again and recalculate the average rating. This solves the problem of concurrent updates and ensures the average rating is always up to date and consistent.

What about a message queue? Whenever I ask this question, particularly of senior candidates, most will propose we write incoming reviews to a message queue and then have a consumer update the average rating in a separate service. While this is a decent answer, it's important to note that this introduces additional complexity that, it can be argued, is not necessary given the right volume.

As we pointed out early, many people search/review businesses but very few actually leave reviews. We can estimate this read:write ratio at as much as 1000:1. With 100M users, that would mean only 100k writes per day, or 1 write per second. This is tiny. Modern databases can handle thousands of writes per second, so even accounting for surges, this will almost never be a problem.

Calling this out is the hallmark of a staff candidate and is a perfect example of where simplicity actually demonstrates seniority.

### 2) How would you modify your system to ensure that a user can only leave one review per business?

We need to implement a constraint that allows users to leave only one review per business. This constraint serves as a basic measure to prevent spam and abuse. For example, it stops competitors from repeatedly leaving negative reviews (such as 1-star ratings) on their rivals' businesses.

Here are some options.

### 

Bad Solution: Application-level Check

###### Approach

The most basic option is to simply check if the user has already reviewed the business in the application layer, only writing the database if they haven't and returning an error otherwise. This is typically the solution I see mid-level (and often senior) candidates propose first.

Some simple pseudo code:

`def leave_review(user_id, business_id):     reviews_for_business = db.get_reviews_for_business(business_id)     if user_id in reviews_for_business:         return "User has already reviewed this business"     else:         db.leave_review(user_id, business_id)`

###### Challenges

The main issue with the approach is that it's not robust to changes. The reality is that as the company grows there may be other services that also write reviews, Data Engineers running backfills, etc. In any of these cases, these new actors are likely to not be aware of your application-layer constraint and may violate it.

Less importantly, we've also introduced a potential race condition. If the same user submits two reviews at the same time, it's possible that both pass the check and you end up with two reviews, violating our constraint.

### 

Great Solution: Database Constraint

###### Approach

The right way to handle this is via a database constraint, effectively enforcing the constraint at the database engine level. This can be done via a unique constraint on the user\_id and business\_id fields.

`ALTER TABLE reviews ADD CONSTRAINT unique_user_business UNIQUE (user_id, business_id);`

Now, it's impossible to violate our constraint because the database will throw an error if we try to insert a duplicate. We just need to handle that error gracefully and send it back to the client. In the case of the race condition we mentioned earlier, there will always be one winner (as long as the data is on the same database instance, which it should be). The write attempt that ends up being processed second will throw a unique constraint error.

Generally speaking, whenever we have a data constraint we want to enforce that constraint as close the persistence layer as possible. This way we can ensure our business logic is always consistent and avoid having to do extra work in the application layer.

### 3) How can you improve search to handle complex queries more efficiently?

This is the crux of the interview and where you'll want to be sure to spend the most time. Search is a fairly complex problem and different interviewers may introduce different constraints or nuances that change the design. I'll walk through a couple of them.

The challenge is that searching by latitude and longitude in a traditional database without a proper indexing is highly inefficient for large datasets. When using simple inequality comparisons (> lat and < lat, > long and < long) to find businesses within a bounding box, the database has to perform a full table scan, checking every single record against these conditions. This is also true when searching for terms in the business name or description. This would require a wild card search across the entire database via a LIKE clause.

`// This query sucks. Very very slow. SELECT *  FROM businesses  WHERE latitude > 10 AND latitude < 20  AND longitude > 10 AND longitude < 20 AND name LIKE '%coffee%';`

### 

Bad Solution: Basic Database Indexing

###### Approach

The first thing we could do to improve the performance of this query is to add a simple B-tree index on the latitude and longitude columns. Ideally, this would make it faster to find businesses within a bounding box.

`CREATE INDEX idx_latitude_longitude ON businesses (latitude, longitude);`

###### Challenges

The reality is this approach doesn't work as effectively as we might hope. The simple B-tree index we'd typically use for single-column or composite indexes is not well-optimized for querying 2-dimensional data like latitude and longitude. Here's why:

1. Range queries: When searching for businesses within a geographic area, we're essentially performing a range query on both latitude and longitude. B-tree indexes are efficient for single-dimension range queries but struggle with multi-dimensional ranges.
    
2. Lack of spatial awareness: B-tree indexes don't understand the spatial relationship between points. They treat latitude and longitude as independent values, which doesn't capture the 2D nature of geographic coordinates.
    

To truly optimize geographic searches, we need more specialized indexing structures designed for spatial data, such as R-trees, quadtrees, or geohash-based indexes. These structures are specifically built to handle the complexities of 2D (or even 3D) spatial data and can significantly improve the performance of geographic queries.

### 

Great Solution: Elasticsearch

###### Approach

Instead, there are 3 different types of indexing strategies we'll need in order to search of 3, very different types of filters:

1. **Location**: To efficiently search by location we need to use a geospatial index like [geohashes](https://en.wikipedia.org/wiki/Geohash), [quadtrees](https://en.wikipedia.org/wiki/Quadtree), or [R-trees](https://en.wikipedia.org/wiki/R-tree).
    
2. **Name**: To efficiently search by name we need to use a full text search index which uses a technique called [inverted indexes](https://en.wikipedia.org/wiki/Inverted_index) to quickly search for terms in a document.
    
3. **Category**: To efficiently search by category we can use a simple [B-tree index](https://en.wikipedia.org/wiki/B-tree).
    

There are several technologies which support all three of these indexing strategies (and more). One common example is [Elasticsearch](https://www.hellointerview.com/learn/system-design/deep-dives/elasticsearch).

Elasticsearch is a search optimized database that is purpose built for fast search queries. It's optimized for handling large datasets and can handle complex queries that traditional databases struggle with making it a perfect fit for our use case.

Elasticsearch supports various geospatial indexing strategies including geohashing, quadtrees, and R-trees, allowing for efficient location-based searches. It also excels at full-text search and category filtering, making it ideal for our multi-faceted search requirements.

We can issue a single search query to Elasticsearch that combines all of our filters and returns a ranked list of businesses.

`{   "query": {     "bool": {       "must": [         {           "match": {             "name": "coffee"           }         },         {           "geo_distance": {             "distance": "10km",             "location": {               "lat": 40.7128,               "lon": -74.0060             }           }         },         {           "term": {             "category": "coffee shop"           }         }       ]     }   } }`

Elasticsearch

###### Challenges

The main challenge you want to be aware of when introducing Elasticsearch is that you won’t want to use it as a primary database, this is a no-no. It is not optimized to maintain transactional data integrity with full ACID compliance, nor to handle complex transactions. Additionally, Elasticsearch’s fault tolerance mechanisms, while effective, require careful configuration to avoid potential data loss during node or network failures. It’s best utilized for what it’s designed for: search and analytical operations across large datasets, rather than primary data storage.

As a result, we need a way to ensure the data in Elasticsearch remains in sync (consistent) with our primary database. The best way to do this is to use a [Change Data Capture](https://en.wikipedia.org/wiki/Change_data_capture) (CDC) system to capture changes to the primary database and then apply them to Elasticsearch.

This works by having all DB changes captured as events and written to a queue or stream. We then have a consumer process that reads from the queue or stream and applies the same changes to Elasticsearch.

### 

Great Solution: Postgres with Extensions

###### Approach

One way we can get around the consistency issue all together is to just use Postgres with the appropriate extensions enabled.

Postgres has a geospatial extension called [PostGIS](https://postgis.net/) that can be used to index and query geographic data.

`CREATE EXTENSION IF NOT EXISTS postgis;`

Postgres also has a full text search extension called [pg\_trgm](https://www.postgresql.org/docs/current/pgtrgm.html) that can be used to index and query text data.

`CREATE EXTENSION IF NOT EXISTS pg_trgm;`

Postgres with Extensions

By using these extensions, we can create a geospatial index on the latitude and longitude columns and a full text search index on the business name and description columns without needing to introduce a new service.

Given that this is a small amount of data, 10M businesses x 1kb each = 10GB + 10M businesses x 100 reviews each x 1kb = 1TB, we don't need to worry too much about scaling, something that Elasticsearch excels at, so this is a perfectly reasonable solution.

However, it's worth noting that while PostGIS is excellent for geospatial queries, it may not perform as well as Elasticsearch for full-text search at very large scales.

In some cases, interviewers will ask that you don't use Elasticsearch as it simplifies the design too much. If this is the case, they're often looking for a few things in particular:

1. They want you to determine and be able to talk about the correct geospatial indexing strategy. Essentially, this usually involves weighing the tradeoffs between geohashing and quadtrees, though more complex indexes like R-trees could be mentioned as well if you have familiarity. In my opinion, between geohashing and quadtrees, I'd opt for quadtrees since our updates are incredibly infrequent and businesses are clustered into densely populated regions (like NYC).
    
2. Next, you'll want to talk about second pass filtering. This is the process by which you'll take the results of your geospatial query and further filter them by exact distance. This is done by calculating the distance between the user's lat/long and the business lat/long and filtering out any that are outside of the desired radius. Technically speaking, this is done with something called the Haversine formula, which is like the Pythagorean theorem but optimized for calculating distances on a sphere.
    
3. Lastly, interviewer will often be looking for you to articulate the sequencing of the phases. The goal here is to reduce the size of the search space as quickly as possible. Distance will typically be the most restrictive filter, so we want to apply that first. Once we have our smaller set of businesses, we can apply the other filters (name, category, etc) to that smaller set to finalize the results.
    

### 4) How would you modify your system to allow searching by predefined location names such as cities or neighborhoods?

For staff level candidates or senior candidates that moved quickly and accurately through the interview up until this point, I'll typically ask this follow up question to increase the complexity.

Our design currently supports searching based on a business's latitude and longitude. However, users often search using more natural language terms like city names or neighborhood names. For example, Pizza in NYC.

Notably, these location are not just zipcodes, states, or cities. They can also be more complex, like a neighborhood ie. The Mission in San Francisco.

The first realization should be that a simple radius from a center point is insufficient for this use case. This is because city or neighborhoods are not perfectly circular and can have wildly different shapes. Instead, we need a way to define a polygon for each location and then check if any of the businesses are within that polygon.

These polygons are just a list of points and come from a variety of sources. For example, [GeoJSON](https://geojson.org/) is a popular format for storing geographic data and includes functionality for working with polygons. They can also just be a list of coordinates that you can represent as a series of lat/long points.

We simply need a way to:

1. Go from a location name to a polygon.
    
2. Use that polygon to filter a set of businesses that exist within it.
    

Solving #1 is relatively straightforward. We can create a new table in our database that maps location names to polygons. These polygons can be sourced from various publicly available datasets ([Geoapify](https://www.geoapify.com/download-all-the-cities-towns-villages/) is one example).

Then, to implement this:

1. Create a locations table with columns for name (e.g., "San Francisco"), type (e.g., "city", "neighborhood"), and polygon (to store the geographic data).
    
2. Populate this table with data from the chosen sources.
    
3. Index the name column for efficient lookups.
    

This approach allows us to quickly translate a location name into its corresponding polygon for use in geographic queries.

Now what about #2, once we have a polygon how do we use it to filter businesses?

Conveniently, both Postgres via the PostGIS extension and Elasticsearch have functionality for working with polygons which they call [Geoshapes](https://postgis.net/docs/reference.html#Geography_Shapes) or [Geopoints](https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-point.html) respectively.

In the case of Elasticsearch, we can simply add a new geo\_shape field to our business documents and use the geo\_shape query to find businesses that exist within a polygon.

`{   "query": {     "geo_bounding_box": {        "location": {         "top_left": {           "lat": 42,           "lon": -72         },         "bottom_right": {           "lat": 40,           "lon": -74         }       }     }   } }`

Doing this bounding box search on every request isn't that efficient though. We can do better.

Instead of filtering on bounding boxes for each request, we can pre-compute the areas for each business upon creation and store them as a list of location identifiers in our business table. These identifiers could be strings (like "san\_francisco") or enums representing different areas.

For example, a business document in Elasticsearch might look like this:

`{   "id": "123",   "name": "Pizza Place",   "location_names": ["bay_area","san_francisco", "mission_district"],   "category": "restaurant" }`

Now all we need is an inverted index on the location\_names field via a "keyword" field in ElasticSearch

By pre-computing the encompassing areas we avoid doing them on every request and only need to do them once when the business is created.

## Final Design

After applying all the deep dives, we may end up with a final design that looks like this:

Yelp Final Design

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

So, what am I looking for at each level?

### Mid-level

At mid-level, I'm mostly looking for a candidate's ability to create a working, high-level design while being able to reasonably answer my follow-up questions about average ratings and search optimizations. I don't expect them to know about database constraints necessarily, but I do want to see them problem solve and brainstorm ways to get the constraint closer to the persistence layer. I also don't expect in-depth knowledge of different types of indexing, but they should be able to apply the "correct" technologies to solve the problem.

### Senior

For senior candidates, I expect that you nail the majority of the deep dives with the exception of "search by name string." I'm keeping an eye on your tendency to over-engineer and want to see strong justifications for your choices. You should understand the different types of indexes needed and should be able to weigh tradeoffs to choose the most effective technology.

### Staff+

For staff candidates, I'm really evaluating your ability to recognize key insights and use them to derive simple solutions. Things like using Postgres extensions to avoid introducing a new technology (like Elasticsearch) and avoid the consistency issues, recognizing that the write throughput is tiny and thus we don't need a message queue. Identifying that the amount of data is also really small, so a simple read replica and/or cache is enough, no need to worry about sharding. Staff candidates are able to acknowledge what a complex solution could be and under what conditions it may be necessary, but articulate why, in this situation, the simple option suffices.

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

![Piyush](https://lh3.googleusercontent.com/a/ACg8ocKVHxAshJgusXa5wRToi_0PYLhcI46VOxlcOcnnMoIEUDSIF8E3=s96-c)

Piyush

[• 11 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm12f86fd00aj11yoi1nbsixi)

You locked this post from UI only. The URL still works if you visit https://www.hellointerview.com/learn/system-design/answer-keys/yelp But I would request to keep this write up open for community since guided practice is already paid feature.

Show more

0

Reply

![Jimmy Rao](https://lh3.googleusercontent.com/a/ACg8ocJoqemA31cA0oG6FT1upaTKaxREiCFVv_kPFZ_-HIW-kpsL6OW1=s96-c)

Jimmy Rao

[• 10 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm20zpmou002r666s3h65gzsd)

In addition to Piyush's point, all answer keys are not protected by auth. They are all accessible as long as the user has the url.

Show more

0

Reply

C

CulturalAmethystTrout404

[• 10 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm25kup9e004mo4hzoqajyhlf)

Hi Even, regarding the first deep dive, what is the trade-off between using optimistic locking v.s distributed locking with Redis?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2mc56tr004vvgwsig9fggkn)

This isn't a great use case for a distributed lock because there isn't a TTL requirement, so instead, we just need row-level locking or optimistic locking, which is much simpler and requires no additional infrastructure.

Show more

5

Reply

![Paul England](https://lh3.googleusercontent.com/a/ACg8ocI9EL7D8ZUYgaO3tz6A9B9lvnN4vq3pCnDPV5l0kizwLwta78bH=s96-c)

Paul England

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2r20kwj00m2101jzzhjazn0)

What the user reads is irrelevant. The user should not be sending number\_of\_reviews + 1. They should send a rating and text. SQL can update the number of reviews, plus total points atomically.

UPDATE businesses
SET review\_points = review\_points + n, review\_count = review\_count+1

Then to get the ratings

SELECT review\_points / review\_count AS average\_rating
FROM businesses

A small nitpick but I would always store ints over float in the DB.

Note that this and the unique constraint combined can't guarantee consistency in the face of a service crash though. I suggested a transaction in addition to the unique key constraint and the AI dinged me for complexity.

Consider this happy path, if reading from a queue: Read msg -> update reviews -> update business -> update offset. No locks needed, even if interleaving reviews.

Now consider this unhappy path read msg -> update review -> crash service restarts read msg -> update review -> unique constraint violation.

Data is inconsistent.

Show more

5

Reply

![Sourabh Upadhyay](https://lh3.googleusercontent.com/a/ACg8ocKpj06uaBaRfJhVsAJL98n9F7-IyL3NsYEkFdZuG1m_9wYa4Q=s96-c)

Sourabh Upadhyay

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmdd6mlzl05fcad08j4ysj4pl)

You can easily bypass this in application code. DB might throw an error but application can handle it, get the data from DB, resolve the rating based on timestamp and proceed to update business. Adds a bit of complex code for resolution but should be fine because I see this as an edge case. If my design works for 90 percent use case and I need some special handling for 10% its fine.

Show more

1

Reply

V

VivaciousVioletTortoise990

[• 10 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2bdxuqu026u7f7ipy1dq4kl)

Hey yelp in not in left menu in common problems, not sure if you want to hide it intentionally, though there is a link on whatsapp page for next: yelp

Show more

0

Reply

P

panaali2

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2mc3kmh00frbefpaz27bdkn)

> we can pre-compute the areas for each business upon creation and store them as a list of location identifiers in our business table.

This can be a bit expensive as we might need to go over all locations and see if the business coordinates falls within it's polygon. since the number of citis & towns in the world are around [4 million](https://simplemaps.com/data/world-cities#:~:text=Comprehensive:%20Over%204%20million%20unique,every%20country%20in%20the%20world.) we might need to implement some fancy algorithm (eg: using a quadtree for locations and then for the new business search through the quadtree to find locations that might include that).

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2mc6wrm00gckad7i54hoqto)

We'd store them as strings!

So my business in Marina Del Rey, CA, would have a list of location strings like: \[Marina Del Rey, West LA, LA, Southern California, California\], for example.

Then the lookup is wicked fast. If we add new regions, we have an expensive offline recompute, but this won't happen often.

Show more

0

Reply

P

panaali2

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2mei5xt00h9ee4adnmtwsu6)

I agree with the proposal, my question and proposal was about how we can figure the list of location strings when a new business is added. When users/admins add a new business, they only provide the coordinates and so to figure the coordinates to list of location strings we need to use a geospatial index like quadtree. Am I missing any of the information that the article provided?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2mot7op00swee4a6pmf4utb)

We have a location database that maps location names to a list of Geopoints (lat/long pairs).

Then, there are algorithms that take a set of geopoints and a lat/long and determine whether the point is inside the polygon formed by the geopoints. Elasticsearch supports such searches.

Show more

1

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm480qfrg037o3o1h6zwwjaqz)

Wouldn't this involve checking each of the polygons for every single new business?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm4810ax303bm129i2zz3fcfz)

Yeah, but so what? :) Not that many businesses, and they aren't added that frequently. This is done asynchronously outside of user requests, so we have all day.

To optimize, you'd have a tree structure, so there's no reason to check Manhattan's borders for a business in CA.

Show more

0

Reply

O

omit.outcast.0c

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5hglcfr012g1osrp7t9djht)

Why we are not using Geohash here. For example, if a user provides "San Francisco", we need to convert that string to a Geohash (probably a 6 character to start with). Once we have that, we can run a query to provide sub-strings of this Geohash. Any reason why this option was not preferable?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5hhzyxz014ra8vv9lj1u1dk)

@omit.outcast.0c, because San Francisco (and any other region) has soft edges. using geohashes would either miss businesses near the edges or include too many businesses outside the actual boundaries. Using polygon-based lookups gives us more precise results for predefined, irregularly-shaped regions like this.

Show more

1

Reply

W

WoodenAquamarineSnipe931

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm8lc0p4100m09xqy5pbsjrou)

Maybe edit the article to incorporate the reasoning behind this trade-off?

Show more

0

Reply

O

omit.outcast.0c

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5ht1lqx01enep4lxklke9ph)

I could be wrong here but wouldn't having simplification better. The current approach is over complicated where we have to maintain another DB, keep track of the cities etc. and also apply inverted index. We also have to keep the source up to date as things changes in the map. The only trade-off here is that we might be returning more results but that can be resolved by just applying another filter during runtime based on the requirement? I wonder how Yelp is doing though.

Show more

0

Reply

U

UnitedGoldOwl597

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2qemlxk00scyixonz4so3af)

Small inconsistency, on this page this problem is listed as a "medium" problem but on the practice page it's listed as an "easy" problem.

Show more

2

Reply

R

RoundAquamarineWasp973

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2qfgf4o00amzmroalkgf805)

Would a candidate be expected to write a NoSQL query on the spot (such as the ElasticSearch search query in this example)?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm2qht7y800ugyixoknriwfn5)

No :)

Show more

0

Reply

![Sourab Vadlamani](https://lh3.googleusercontent.com/a/ACg8ocJVN0vB10eQhqxgey-zD15n_ehNvzFagw32FcvBYR3dpJSdnXyD=s96-c)

Sourab Vadlamani

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm4m28t5o00tunbottlvfjs36)

SQL? I'll show myself out.

Show more

6

Reply

A

AppropriateGreenHoverfly664

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm30f15nr00nsipc5o1lqhhk2)

What are the attributes that goes into elastic , all of the business table (businessId,address,location, numsOfReviews, avgRating)attributes in the index? Do we add geohash in the elastic document while indexing or will that be handled by elastic internally and we just add attributes lat,long?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm4811qva037ohco9opccc40c)

The latter! https://www.elastic.co/geospatial

Show more

0

Reply

M

MilitaryBlueNewt787

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm3ohr7kb01dzo0ypqk0wkzkg)

@Evan Using the same DB makes sense but what is the benefit of adding a new separate 'service' for reviews ( with also uses the same path as /businesses ) ? Unless you're thinking of scaling needs individually which might differ, it just seems like another service to maintain, deploy with secrets management etc etc

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm4812zcx00tsql2hyz9ayj4e)

Yah I don't mind that argument. In reality it would depend on request patterns, repo requirements (diff languages?), and organizational structure. In an interview, I'm totally fine with either justification here.

Show more

1

Reply

M

ManyBlackHorse876

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm4apj66300j810dq1vn2hpc6)

Hi, can you add more content regarding geohashing and quadtrees? Like how they work and tradeoffs? Thx

Show more

9

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm4apk2gc00jd10dq1yavgteg)

On our backlog!

Show more

5

Reply

L

LinguisticRedViper140

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm4bqcgec00dhuris68ahmg2e)

Can you please explain this a bit more clearly: "availability > consistency" Like you say it is easy to mention the words but it should be in relation to this design. :) Does it mean available for the user to access the system but not consistent in terms of business/review update.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5ba826u02ssc4lgo7m3emmk)

Yah exactly. Adding both businesses and reviews can be eventually consistent.

Show more

0

Reply

R

RadicalAquaBedbug232

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm52wt8ea031ewbx35iz2m8vw)

small inconsistency: This description marked as medium, while in Practice list difficulty marked as easy

Show more

3

Reply

P

PracticalScarletConstrictor933

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5as7i2v02nnv808jsy8qb61)

I am wondering if using Redis Geo spatial index can be used instead of Elastic Search. In this case we have 10M businesses. Assuming even 1KB per businessid + lat/long , the total memory would still be 10GB which would easily fit in on a Redis Node and it will be faster than Elastic Search. Also wondering why we don't make use of external Geocoding API (like Google Maps or Here ) to get place names to lat/long. We can cache results to reduce costs.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5ba6wf202umke2ryl43xycy)

You could. But then you'd search Redis for location and then go to another datastore to search based on category and full text -- making it 2 hops instead of 1. You could use Redis Search, but at that point, you'd probably rather just be using Elasticsearch or a similarly durable storage.

Show more

1

Reply

![Hamad Khan](https://lh3.googleusercontent.com/a/ACg8ocIAcXpQE5wVrXSXX4iGbf5TmMgV4MoN2SZtun3zbG0FkyGUPA=s96-c)

Hamad Khan

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5en66ey00amyag8mullmdsq)

Heads up, this is tagged as "medium" on this article, but on the practice screen where you can select a problem to practice, it is tagged as "easy".

Show more

1

Reply

P

ParliamentaryBlushTiglon248

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5ghvmf1004f1osrmall2ry9)

Why can’t we get the list of businesses from the address itself? Each business should register with a city info. We could create an index on the city and state. Having said that, cities could be big and the user may still want to search in the vicinity. We would still have to rely on exact lang/lat coordinates. Restricting the search space to a city first could drastically reduce the search space.

Show more

1

Reply

V

VastTomatoPtarmigan305

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5hdl3a700wh10u8ky1z5ovf)

Thanks, Evan! Another great post! I think I might of missed it but how is the final design handling the scale of 100m dau and 10m businesses? I didn't see a deep dive for it. Or is something that can be horizontally scaled, sharding on the database/cache, etc?

Show more

3

Reply

V

VariousBrownNarwhal695

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5jlli4y03bp1osrgozni1k5)

New here, but for maintaining consistency, in the updating of the average rating section, can we first say that Postgres is ACID complaint, so would handle consistency out of the box (assuming we made clear we are using relational DB). And then if asked, can move to optimistic locking ?

Show more

0

Reply

![Ryan Lynch](https://lh3.googleusercontent.com/a/ACg8ocKrmKncNNcAhrMIDU5X6ZlEZNUJJ78J0EH2zkuTDQ-Scor4VA=s96-c)

Ryan Lynch

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm66ptdm601i3a4uu7xrv0cxn)

Just stating ACID probably isn't sufficient. Many people view the "C" in ACID as not really even being a characteristic of the database. ACID databases still need to have some sort of locking strategies in-place that work well with your use case to ensure consistency.

I think pointing to ACID works for atomicity, isolation, and durability, but you need to add some application context when talking about consistency.

Show more

0

Reply

N

NeutralAmethystUnicorn745

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5k7kog8041g1osrwqu2c31s)

For deep dive (2), it seems the Great Solution where we add uniqueness constraint to a table is only relevant to SQL DBs like Postgres. What will the solution look like if we choose NoSQL DBs like Cassandra/DynamoDB?

Show more

1

Reply

N

NeutralAmethystUnicorn745

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5khbisp04i91osrqq6c4hmq)

My solution is we will need two tables with redundant data, one for ensuring uniqueness of <bussiness, user> pair, the other for fetching the reviews with pagination. review\_table1: Both primary key and parition key will be (bussinessID, userID)

review\_table2: primary key will be (bussinessID, timestampUUID\_for\_the\_review). Partition key is bussinessID, sorting key being timestampUUID\_for\_the\_review. The cursor pagination will be based on "timestampUUID\_for\_the\_review".

When writing a new review, the write will directly happen on table1 with Light Weight Transaction. "INSERT INTO review\_table1 ... values ... IF NOT EXISTS". Then we will use CDC to do another insert into review\_table2. The reason why we can't make table2 a global index table of table1 is because table2's primary key does not include table1's, a hard requirement for building global index table.

Show more

1

Reply

O

OlympicBlackLynx843

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm6ztk7f8039ohomje4pbvc9f)

Interestingly there is no mention of the actual DB choice - although authors implied that its SQL in several places because of the decisions they've made (such as adding a constraint). I suspect that would be bad in an actual interview - one would be expected to make DB decision explicit before going into a deep dive and offering a solution that relies on that earlier decision?

Show more

1

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm8joghoy0135phpwh126rwbz)

AWS has a approach for supporting unique key for DynamoDB. Whenever adding an item, insert extra item into the same table, with the pk attribute set to the attribute name

Show more

1

Reply

![Harshitha Gandamalla](https://lh3.googleusercontent.com/a/ACg8ocITRqM0agWKr1NeJFc6MkGyVM2ppFyCuMm644YMOXxwYrKbgwav=s96-c)

Harshitha Gandamalla

[• 7 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5r74igm02jn114yhqt1tt5s)

Hi there, I was working on the yelp system design question and was wondering why storing the avgRating and number of ratings in a db with optimistic control approach is better than using redis (store the average rating and num of ratings per business key - given redis is single threaded. Is it that we would be blocking updates to avg ratings on other business keys?) I'm a little confused on the draw back to using redis. Thank you!!

Show more

2

Reply

![Vineeth Kanaparthi](https://lh3.googleusercontent.com/a/ACg8ocKYbskHNrwvGx9rTeKGgKpG38MvVmBLMWkPfWotKjkCb3dMwRkf=s96-c)

Vineeth Kanaparthi

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5ymqpu800fh7ai1mj924d8o)

Avoids extra query to redis. If volume is an issue with the optimistic concurrency control we can do CDC from the reviews table and use a consumer to asyc update the avg rating.

Show more

0

Reply

P

PhilosophicalHarlequinOrca509

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7h9i43y01vuoa3nmdwit8c8)

I have the same question as you. I can see why having the extra query to Redis is not ideal when we are accessing the businesses table anyway to retrieve search results. But, how sub-optimal is it really? Wouldn't a benefit of having Redis mean we would avoid the problem of locking?

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm8joi6ec0139phpwrn619o4w)

It works but doesn't give much benefit IMO

Show more

0

Reply

P

PresentLimeHare138

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9tabymn00e4ad08r1c2dq0k)

@Evan curious what you think to Harshitha's question. If we use redis (LRU), then we have less concurrency issues to deal with. So I think that might be a pretty good approach.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9taxq4n00mdad08r9qagke5)

Redis isn't better here because we don't need the extra complexity. The write volume is tiny (maybe 1 review/sec), so a simple DB constraint with optimistic locking works fine. Redis would actually make things worse since now you need to worry about consistency between Redis and your DB, handle Redis failures, and maintain another service. Plus, Redis being single-threaded isn't relevant here - the atomic nature of DB transactions with optimistic locking gives us the same guarantees without the overhead.

Show more

1

Reply

![Ryyan Smith](https://lh3.googleusercontent.com/a/ACg8ocKYxrn1HIfsGMl2LA_q1W0CyWegpW1CIJ0Eg-WcYSFTGCrTEW02=s96-c)

Ryyan Smith

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5szkygb000v14528a4b6t88)

I think this query is messed up

// Search for businesses GET /businesses?query&location&category&page -> Business\[\]

should it be?

GET /businesses?name&location&category&cursor -> Business\[\]

Show more

0

Reply

F

FancyTanGoldfish347

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm5tctodx00g9p38vn2lw34sp)

I'm not certain on where we got the 100 reviews per business from in the 4th section of the high level design. It seems like this is something we should define in the entity definition section of the problem (along with the read/write ratio).

Show more

0

Reply

F

FancyTanGoldfish347

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm60i34yz00tl4i6qpu6btqea)

"Given that this is a small amount of data, 10M businesses x 1kb each = 10GB + 10M businesses x 100 reviews each x 1kb = 1TB, we don't need to worry too much about scaling, something that Elasticsearch excels at, so this is a perfectly reasonable solution."

This part should say PostgreSQL

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm60zzpu4012x3linw4bg42ol)

What it's trying to say is that Elasticsearch does a better job for very-large-scale datasets and this is only 1TB, comfortably something which could fit on a single Postgres node.

Show more

0

Reply

![Aidan Lakshman](https://lh3.googleusercontent.com/a/ACg8ocL-gxFmpTBAQtVUiwaAj-vc75MKMC-PHljAXuCWMxRsZAaH3CTP1g=s96-c)

Aidan Lakshman

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm62ebxsk0157hqkj3g9cg9pe)

For calculating the average rating, any reason you didn’t decide to just store NUM\_REVIEWS and TOTAL\_REVIEW\_SCORE in the table, rather than constantly recalculating a running average? You could avoid the concurrent update problem by using commutative operators like atomic increments, and calculating the average rating would be effectively the same time (just doing total\_rating / num\_rating before displaying). A business would need over 400M 5-star reviews to overflow a 32-bit int, so not really a huge concern (or just store it as a long). Seems to me like a much cleaner solution with less concurrency traps.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm62eg5r5015khqkj13ph1izk)

If I understand you're suggesting just moving the division part of the operation to read time instead of write? That works and I agree it avoid some concurrency traps (though they are incredibly minor). Either works.

Show more

1

Reply

![Aidan Lakshman](https://lh3.googleusercontent.com/a/ACg8ocL-gxFmpTBAQtVUiwaAj-vc75MKMC-PHljAXuCWMxRsZAaH3CTP1g=s96-c)

Aidan Lakshman

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm62fkhxd019acf2a48ui0hla)

Yep, that’s what I meant. You’re right though, the concerns are pretty minor either way. Thanks!

Show more

0

Reply

B

BreezyAmberFlamingo112

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm6cswa7f02tdiq3ncrnb8xzs)

I don't know where better to post this, but are y'all aware that the thumbs up functionality for comments is very broken? I can't actually give out thumbs up, but I can remove another users' thumbs up...

Show more

6

Reply

![Andrew Fialho](https://lh3.googleusercontent.com/a/ACg8ocJzHBQXN1-sVrEasgnD4dcVknfEIGw_SLQCgDF-SOKAxOPPCw=s96-c)

Andrew Fialho

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm6mkfz5404ohjgz30nrr836e)

"able to reasonably answer my follow-up questions about average ratings and search optimizations. I don't expect them to know about database constraints necessarily"

As a Mid level, database constraints are way easier to me than geopastial stuffs, at least the unique constraint used in the solution, comes naturally.

Show more

0

Reply

![Abishek Kumar](https://lh3.googleusercontent.com/a/ACg8ocL8Fb-xNT5QWwVFs-6zEUNYC2Be1G7qkd6rtZOBHn71Ypth2A=s96-c)

Abishek Kumar

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm6t81g3903av14j6bd5sn4l7)

Hi Stefan and Evan, What is the reason behind review as a separate service? Submit review requests are very minimal and can be placed within the business service and we can maintain Yelp with a single service.

Show more

0

Reply

![Shashwat Kumar](https://lh3.googleusercontent.com/a/ACg8ocJPvaY3V-bQ9H4_wAZZ9n3pTRB6KOQ_-1xYhupiJM132cEAYQg7=s96-c)

Shashwat Kumar

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm8s4z5a80070ad08owzvqb2h)

You answered your question. Since the submit review requests are minimal, its better to decouple the read service from write service for scaling them independently.

Show more

0

Reply

W

WanderingBeigeCrab571

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7875izm020nl7b1cny3hkqx)

Dont we have to consider following?

1. while searching within 5 mile radius, if we use geohash two point which are across river will be in same geohash but technically they are farther than many other point. How to handle that?

Show more

0

Reply

W

WanderingBeigeCrab571

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7879f4s020tl7b136xiflpe)

I think here;s what it can be done. help me if that is right approach

- Filter candidates using real-world travel distance For each business found in the geohash search, calculate the travel distance using a road network (e.g., Google Maps API, OpenStreetMap, or Mapbox).

Show more

0

Reply

![Alan Mathew](https://lh3.googleusercontent.com/a/ACg8ocIiFPPjryq4t0Kl2mpdZGwsJQVwnzQL8RsiqrjoupD61jaQ0A=s96-c)

Alan Mathew

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm78icdmu02kjl2b7xecftj7r)

Does Elasticsearch have all the data as the primary database? If not, how much data from the primary DB is stored?

Show more

0

Reply

![Aditya Jain](https://lh3.googleusercontent.com/a/ACg8ocJjHh-eky22nfbV7YSOwPct6ROk615alDtDMGafjdNJV1gQcQ=s96-c)

Aditya Jain

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7ciizse00moab600zab6c53)

Why did we prefer optimistic over pessimistic locking for updating avg rating? It is because we have high reads and low writes?

Show more

0

Reply

P

PhilosophicalHarlequinOrca509

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7h97ke001sxxcugrduamw40)

Yes, there are way more reads than writes, and the chances of a write happening at the same time is very low.

Show more

2

Reply

![Will Shang](https://lh3.googleusercontent.com/a/ACg8ocLL-_zI633rdwKuZxrzSInrilUdONavtZFCWy3x-mtPY2VPdA=s96-c)

Will Shang

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7lktqqk000z4y43v6f56wk4)

Hi Evan, for products which serve global users, we always want to make sure our services are distributed globally and DB replicas are co-located with each other. However in the design walk throughs, like this one, it seems like that we only consider if the traffic is high enough requiring scale, otherwise we don't care about adding more servers. I believe this yelp design means the individual service only has one server or maybe for fault tolerance, every service is load balanced? Do we need to explicitly mention this every service is always load balanced?

Show more

2

Reply

F

FormidableRoseMeerkat174

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7qrrkfq02a22ebzu1jz5z5h)

In non-func reqs says < 500ms search but how exactly is this enforced/materialized? Is simply using tuned/optimized Elasticsearch cluster and a decent amount of API servers enough to satisfy this in an interview?

Show more

0

Reply

![Erick Takeshi Mine Rezende](https://lh3.googleusercontent.com/a/ACg8ocK2cjXUIj2mB69_Hc5ZVesE8etdmqFpSR6rfbYMnbwPzc6oy_UH=s96-c)

Erick Takeshi Mine Rezende

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7wt5sr901rsdimnq67msnj1)

In a hypothetical scenario where we have high load of rating writes, what would be the approach to deal with it? Should we move into something like distributed locking? What are the alternatives to implement a consistent average with more load?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9df7896002mad08lmvrvfuu)

Distributed locking would actually make this worse - it'd create a bottleneck and kill throughput. Instead, we'd use a combination of sharding (by business\_id) to distribute the load and optimistic or pesimisstic locking to handle concurrent updates within each shard. For really extreme write loads, we could buffer updates in a message queue and have consumers batch-update the averages, trading some consistency for throughput. But honestly, even massive review sites like Yelp don't see enough review volume to warrant this complexity, you'd need millions of concurrent reviews per second to break the simple approach outlined in the article.

Show more

0

Reply

![Juan Diego Jimenez](https://lh3.googleusercontent.com/a/ACg8ocJfYSyAV6yVa0z0JHVui7dImzFpqndMSA20CyHAI_tsCTuDpPQ=s96-c)

Juan Diego Jimenez

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm7xxyq6x017xtvyjbe6x9861)

How would you know if you need to specify more details like adding a load balancer or cdn, etc?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9df69xw002gad08tgle6d5n)

Images/media was out of scope for us, otherwise CDN would have been good to talk about during scaling. As for LB, its pretty implied nowadays you're going to horizontally scale. But you can mention in a sentence like I do in the attached video.

Show more

0

Reply

![Kushagra Gupta](https://lh3.googleusercontent.com/a/ACg8ocL8DzgJlxVlwejihfvhi7AO9VmkV72f4m_rSbkMCJG3vCGI3i4X=s96-c)

Kushagra Gupta

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm82pda1p00dnhnviik79wlon)

If we are using DynamoDB, we can achieve the same thing using geohashing, right? However, we would still need Elasticsearch for the inverted index.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9df596o002cad08dds8zj3s)

DynamoDB with geohashing would work for location-based queries but it's not ideal on its own. You'd need to maintain multiple geohash indexes at different precisions to handle varying search radii, and range queries on geohashes can return false positives requiring post-filtering. Plus you're right that you'd still need ES or something similar for text search, so might as well just use ES (or Postgres+PostGIS) for everything since it handles both use cases natively and more elegantly.

Show more

0

Reply

F

FederalHarlequinLamprey277

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm8d53rqs01kjehoda48z2ntz)

@Evan, Can you clarify these steps a bit? "we simply need to both add the new review to the Review table while also updating the average rating for the business in the Business table." Steps:

1. Optimistic locking is used to update the rating,
2. Adding the new review in the Review table if step 1 succeeds. Is it possible step 2 is not done due to service restart or other errors etc? Same issue with Online Auction max bid.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9df3jl20026ad08uv8yirdq)

You need a transaction here. The steps should be: 1) Start transaction, 2) Update business rating with optimistic locking, 3) Insert review, 4) Commit transaction. If anything fails (including service restart), the transaction rolls back and neither update happens.

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm8joo5eb013hphpwnhu7c2sz)

""Next, you'll want to talk about second pass filtering. This is the process by which you'll take the results of your geospatial query and further filter them by exact distance.""

How is the geospatial query is done if there is no any filter on that. If there is already filter like an area (where does this area restriction comes from, the 10km constraint?) Then shouldn't the first geospatial query already include the distance filter?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9df319s001yad08ne1hzbc5)

First, we use a spatial index (quadtree/geohash) to quickly find businesses in a bounding box that encompasses our search radius - this is super fast but gives us approximate results. Then we apply the exact distance calculation (Haversine formula) on this smaller subset to get precise results within our 10km radius.

Show more

0

Reply

E

ExoticPeachOpossum454

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm921afzs00miad08hoe8sjrv)

Would you mind making a video for this design?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9df20kq001sad08r8y7uycb)

Done :)

Show more

1

Reply

E

ExoticPeachOpossum454

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9dfd5tr0038ad08g5p0045r)

Thank you :)

Show more

0

Reply

N

NuclearAmethystStork225

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9e0h78v00jead08d2hozdj2)

Please allow access to the written content if not the video, please this will be very much helpful to us.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9fy39pd016zad08gcvgrz6c)

I dont understand? What are you looking for?

Show more

0

Reply

![cy](https://lh3.googleusercontent.com/a/ACg8ocJIVshot7zHFUYT69AFMXTVrD_3aXUT4--Rbh1rrSCOkPzRiZo=s96-c)

cy

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9hmd356003lad0860ttovls)

My team actually uses a queue to do something similar to denormalize the ratings into the parent business here.

A create method should ideally has no synchronous side effect and introduces no locks to other part of the database, e.g. should not introduce contentions with reads and writes to the business table. We should not delay/abort creating a review just because someone is updating the business, and vice versa. More strictly, this method should be a custom method rather than a create method if side effect is introduced, as per https://google.aip.dev/136. Obviously, low qps is a valid justification of this approach.

I would argue the async queue is a better solution because it follows the best practice of separation of concerns. It separates the operation of creating a review with the work that can be done asynchronously. It's future-proof - are there any other async tasks can be done together here? It's a cleaner architecture obviously with some overhead of the message queue and an additional service. When we have a bug in calculating average rating, we don't fail the transaction and we can still allow users to post reviews.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9hpuxon002bad087ht2qvyg)

Agree to disagree! At least in the context of the interview, the decision matters much less than how you justify it. If you were to provide this justification, even if I disagree with the solution, I'd think the interview went great.

Show more

0

Reply

![Shalin Shah](https://lh3.googleusercontent.com/a/ACg8ocKmIhTlZVn6dIJ9-mHsHleTfkGXr52r-Lcznyu_RMBJvJlRHLw=s96-c)

Shalin Shah

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9qn67pk00ckad08a7v2r1xr)

Hey team. Why can't we select DDB in place of Postgres? DDB can have Global Secondary Indexes which can help querying on Lat/Long.

Show more

0

Reply

O

ObviousMagentaAnteater913

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9r8giio00smad08zne0xj3g)

I'm a little skeptical about the usefulness of the polygon bounding box search. In practice, users care about proximity. Only for very niche use case they would want to filter results limited to a precisely delimited area and want to exclude anything outside regardless of proximity.

Even so, if we're talking neighborhood or district for instance, using a radius then filter on the business' address sounds like it would be enough.

Show more

0

Reply

F

FashionableGreenLadybug828

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmd0fsw6905scad08yk9sqcvo)

A user could very well be planing a trip to another place and gathering restaurant ideas. Or a "nearby" restaurant in another neighborhood is actually separated from them by a highway or international border. If they went to the trouble to type out "restaurants in Soho" instead of just a proximity search there was likely a reason for it.

Show more

0

Reply

O

ObviousMagentaAnteater913

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9r8tbpt00syad08uglcongn)

Atomic counters in DynamoDB might be a better alternative to optimistic locking for the average score update, as they would avoid errors altogether. With the currently suggested method, I would be concerned about a high error rate in case the concurrent updates situation is frequent.

Show more

0

Reply

![Rajan Jha](https://lh3.googleusercontent.com/a/ACg8ocJY3LlnnTwEvPjYFbGrj4257FEjvezoWp37LBcyGfE30QqNvA-T=s96-c)

Rajan Jha

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9x7yxoy0033ad08z350y4xy)

I was expecting more granular level detail on Geospatial indexes, their types and pros and cons. I mean its Yelp

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9xdu6g800bmad09pgy4l95l)

More details on that in [db indexing](https://www.hellointerview.com/learn/system-design/deep-dives/db-indexing#geospatial-indexes).

Show more

0

Reply

![Joe](https://lh3.googleusercontent.com/a/ACg8ocJiJ1scmdouk0b6Y6_MFYZG_VjfRTq2vk9g_3HaC6yylMK15A=s96-c)

Joe

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cm9zti5dd02ljad08lczbetor)

I think in the "separate DB per microservice" issue there are 2 different disagreements that get conflated:

1. Seperate "physical" DB instance per microservice (which I think you cover well here, and is primarily operational)
2. Seperate groups of tables/namespaces/"DBs" per microservice (which I don't think is covered here, and is primarily an abstraction concept)

Even if you're happy to eat the operational risks (which in a lot of cases is going to be fine) you still have to decide whether you want your DB schema to be a de facto API by sharing tables between services. It's really easy to fall into that hole accidentally and very hard to get out.

Show more

0

Reply

![Mayur Jain](https://lh3.googleusercontent.com/a/ACg8ocK1wIf98x3clGeeUT1NE-Co8x4YwP9xtsXe-uXUH09gTfZdVFI=s96-c)

Mayur Jain

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cma6e5ex600cmad086lljbnzd)

How can one get access to the Excalidraw used in this design

Show more

1

Reply

K

kstarikov

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cma735ctl00ztad07i1jyssbf)

I've watched almost all videos on this website and nowhere do you explain why your latency constraints hold. It's very irritating.

If I were an interviewer, one of the first thing I'd ask is how you know why your searches with this design will take less than 500ms.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cma740n0d012had08sjjxdju5)

The latency constraint is actually well-supported by the design. With Elasticsearch (or Postgres+PostGIS), geospatial queries on 10M businesses using proper indexing consistently return results in <50ms. The bottleneck isn't the search itself but rather network latency, which for a well-distributed CDN typically adds 100-200ms. Even with additional filtering and ranking, we're comfortably under 500ms.

Every interviewer is going to be a bit different :). The latency constraints hold because of how the data is organized and rough OOM estimates of latencies for each component. Some of these are obvious to more senior engineers and when they aren't, those are places you should expect a probe.

Show more

0

Reply

K

kstarikov

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cma75y1i300ypad073ecr4z0d)

Yeah I expect this explanation to be part of every video where the constraint was defined.

Show more

4

Reply

H

HandsomeIvoryCrow799

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmaanc34i00e9ad084q7ijtb8)

> How would you efficiently calculate and update the average rating for businesses to ensure it's readily available in search results?

Please correct me if I'm wrong but it seems optimistic locking is not necessary as race condition is not possible because UPDATE statements are done sequentially on the same record for the default Read Committed isolation level. See [official document](https://www.postgresql.org/docs/current/transaction-iso.html):

> UPDATE, DELETE, SELECT FOR UPDATE, and SELECT FOR SHARE commands behave the same as SELECT in terms of searching for target rows: they will only find target rows that were committed as of the command start time. However, such a target row might have already been updated (or deleted or locked) by another concurrent transaction by the time it is found. In this case, the would-be updater will wait for the first updating transaction to commit or roll back (if it is still in progress). If the first updater rolls back, then its effects are negated and the second updater can proceed with updating the originally found row. If the first updater commits, the second updater will ignore the row if the first updater deleted it, otherwise it will attempt to apply its operation to the updated version of the row. The search condition of the command (the WHERE clause) is re-evaluated to see if the updated version of the row still matches the search condition. If so, the second updater proceeds with its operation using the updated version of the row. In the case of SELECT FOR UPDATE and SELECT FOR SHARE, this means it is the updated version of the row that is locked and returned to the client.

So, technically we can do something as follows:

UPDATE business\_table
SET avg\_rating = ((avg\_rating\*num\_reviews) + rating) / (num\_reviews + 1),
    num\_reviews = num\_reviews + 1
WHERE business\_id = ...

Moreover, I think we can use [AFTER INSERT trigger](https://neon.tech/postgresql/postgresql-triggers/postgresql-after-insert-trigger) instead of transaction to maintain the data consistency at persistence layer. It's also less costly in terms of retries.

> How would you modify your system to ensure that a user can only leave one review per business?

We can use [INSERT ... ON CONFLICT](https://www.postgresql.org/docs/current/sql-insert.html) statement to upsert the record natively.

Show more

1

Reply

N

NearCopperQuokka731

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmb9udswq0211ad08dptpe32g)

Agreed. Also both repeatable reads and read committed does obtain exclusive row-lock level lock, hence updates can't be done concurrently.

Show more

0

Reply

L

LivelyCrimsonSnail889

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmaewaxle00a8ad07egcsigft)

The video doesn't load!

Show more

0

Reply

D

DizzyJadePerch635

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmaexuou600d6ad070me0d1e0)

Why this GET /businesses/:businessId/reviews?page= -> Review\[\] instead of GET /reviews/:businessId ?

Show more

0

Reply

![GGu T](https://lh3.googleusercontent.com/a/ACg8ocIdslsqsPj4kru3lxTs-5qRxv4nSsxr4X5YU93-rd7bfkNkdg=s96-c)

GGu T

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmahc59vw02htad070k812apr)

For updating rating, can we simply do "update business set total\_score = total\_score + 9 and review\_count=review\_count+1 where business\_id=123"? When FE fetches avg rating, backend returns total\_score and review\_count, and FE easily calculates avg rating. Since we are using mysql/postgres which guarantees ACID, we don't need any optimistic lock at all. Just this simple sql is enough.

Show more

0

Reply

![Jagrit](https://lh3.googleusercontent.com/a/ACg8ocLVQF2_4KCJMgtZ0FF2xVqiW2qacI3u57liReHgnXzKXfc-iOwX=s96-c)

Jagrit

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmakibmsq038had075dw763qv)

Awesome video as usual. Minor requests to add CCs to videos and link to the excalidraw just like youtube :D. Thanks a lot

Show more

1

Reply

![Luiz Felipe Mendes](https://lh3.googleusercontent.com/a/ACg8ocLiaxrQj_iBKa0tyykfUuU5Jx14UBC9Sf_7rf2bEHaxFQBwLX9dNg=s96-c)

Luiz Felipe Mendes

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmamoc13i00ptad08xauahqi2)

Can I use Elastic Search as my only database? It does everything we need, it does not have transactions but i do not think it is completely necessary on this problem.

Show more

0

Reply

![Vivek Nagarajan](https://lh3.googleusercontent.com/a/ACg8ocLsWmrgcEKa9r-IaNuSi3LGAiUcQ9NxIvdzMQjhvdPDpBM8WRaE=s96-c)

Vivek Nagarajan

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmaq2z26l003bad08tfqpu0ce)

I'm a bit confused about how/why to lock a row when handling concurrent reviews of the same business using transactions.

My understanding is that transactions are expected to handle:

Atomicity – ensuring that updates to both the business and review tables occur as a single atomic unit.

Concurrency control – by using the READ COMMITTED isolation level to prevent dirty read/write. This is implemented via row level locking for the same record.

However, I'm not sure if I still need to explicitly lock the row (e.g., using SELECT ... FOR UPDATE) to avoid issues when multiple users submit reviews for the same business simultaneously.

Could you clarify specifying the isolation level alone is sufficient since it implements row level locking under the hood to prevent dirty writes as in this case

Show more

0

Reply

![abhinay natti](https://lh3.googleusercontent.com/a/ACg8ocJsCbVr1a1xmAmO4lGu17xxmA5Q-6JXuFdEKGSo1dBpjO03sg=s96-c)

abhinay natti

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmar5d4yt002lad08z58ub44u)

How can we provide read-your-writes consistency so that the user can see the reviews he has posted? We have given eventual consistency as the SLA, but shoudn't we also priortize read-your-write consistency here? Will this apporach be able to hande this: We use an in-memory cache like redis with some ttl (time it takes for the review to be eventually consistent + some buffer) and store the writes of the reviews in the cache. Key could be something like userId\_businessId and we merge the reviews from the cache and the DB

Show more

1

Reply

![Chinmay Bijwe](https://lh3.googleusercontent.com/a/ACg8ocKm1zES8pSU5lg0eScwtAfBlfVruwB1W5nTQuhnRRbmTqbWgba0fw=s96-c)

Chinmay Bijwe

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmay2aesh019tad08l1tyqifn)

This is a great solution. I did the practice interview and learned a lot. I like the push for 1 min accurate rating after someone posts a review, but I'd argue it doesn't reflect the reality. I think it's good to play both scenarios in an interview though. For example on Yelp, they take a while to post the review, and I believe it goes through some async checks for content (safety etc), so there could be a chance your review doesn't get posted. Something interesting to play with in the interview, which is where I had brought in the message queue and async worker as well, but your push to keep the simplicity is very valid. My other thought was around updating/rebuilding the Elastic Search index. It usually isn't instant, so I was curious if that could meet the 1-minute requirement to make the updated rating available (unless we say it's available on the business details page, but the search can lag behind). Would love to hear your thoughts

Show more

0

Reply

![Meng Li](https://lh3.googleusercontent.com/a/ACg8ocLgYcQNCM7IYF8X1JypdzotMp3ZQrcJEohQtNeUS8Oej4bWfBk=s96-c)

Meng Li

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmb1r11w3003aad0947fztoaz)

Great write-up on the Yelp design—I completely agree with your assessment that directly writing to the database is the ideal choice for a scenario like averaging ratings at ~1 QPS.

However, I wanted to discuss another scenario with you. Suppose we introduce a simpler action, like comment upvotes/downvotes. With 100M DAU, assuming 20% of users perform this action an average of 3 times per day, we're looking at around 60 million events daily (~700 QPS). Technically, even 700 QPS can comfortably be handled directly by Redis without a message queue.

But considering the broader architectural context, if our infrastructure team already provides a centralized counting service (e.g., Kafka + Flink-based) used across multiple use-cases (not just this one), would it be reasonable, from a business extensibility and consistency perspective, to leverage this existing counting pipeline for our scenario as well—even if it's slightly more complex than strictly necessary for just 700 QPS?

Additionally, I fully support your point on having the database as the single source of truth for deduplication. But practically speaking, client-side UI disabling of repeated clicks and API Gateway request-level deduplication—although not strictly trustworthy—can significantly reduce redundant requests hitting the backend. Do you think highlighting these practical layers of defense during a system design discussion would be beneficial?

Thanks again for your thoughtful content!

Show more

0

Reply

B

BareCoffeeBasilisk390

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmb2en7so000dad08z637ap13)

I would still consider a queue because the optimistic locking will preventing a race condition via failure usually (hibernate or JPA usually throw an exception). someone. Whereas the queue prevents failure from even happening. SQS cost for up to a billion message based on 64 K payload is $0.40 per month. So not seeing the over - engineering.

Show more

0

Reply

O

OutsideRedXerinae444

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmb32lk4y00ugad07rbv8lucg)

For the search for restaurants in Santa Monica, why not adding zip codes to the business table and matching some zip codes

Show more

0

Reply

S

seantech1999

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmbjmclbk00qtad08qi5ml0xu)

Hi Evan, I cannot hear the audio, can you pls help check? thanks!

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmbjmeghh00sjad08wxcyepd4)

Audio is fine! Check your system :)

Show more

1

Reply

S

StableLavenderLocust829

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmbtozwps04no07adk2fksu3v)

Hi Mr Evan/ Mr Stevan

I am a premium member.

1. This particular video of Yelp and all others you discussed. Your explanation is from Product Architecuture perspective or System Design perspective ?
    
2. I have watched most of your videos here and all makes sense to me. Should I go for Product Architecture or System Design based on your videos ?
    

Please help.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmbtpoexl04on07addat0eusq)

1. Both
2. Check this out! https://www.hellointerview.com/blog/meta-system-vs-product-design

Show more

0

Reply

S

StableLavenderLocust829

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmbv16o7o00dh08adbzmnpxid)

So for Product Architecture interview, should we discuss Non-Functional requirements ?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmbv1sfyf00il08ad3g1ml1i7)

yes

Show more

0

Reply

S

StableLavenderLocust829

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmbwt6rz505jf08ad74q8td6o)

Hi Mr Evan/ Mr Stefan

Thanks to your videos. I am able to build API, Entities & HLD. In functional requirements, I am not able to cover edges cases, bottlenecks & failover points.

For example, in URL shortener, when I am doing by myself, I missed uniqueness of the URL.

How to take care of these ? Please help/ guide.

Show more

0

Reply

![Xiaozhen Zhu](https://lh3.googleusercontent.com/a/ACg8ocKbAK5FeTjSbe2DKDDdb7in5Rzaaq-oObREGHwmoT-AlQrg2t0=s96-c)

Xiaozhen Zhu

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmc54mf9906j4ad08e6z49yep)

Thank you so much for this video, it's probably the best I've seen so far among the ones you made in the platform. You addressed many unclear points that have been bugging me for quite some time, e.g. how to justify using a single db in a microservice architecture, or the design choices made in the rest apis. Just wow.

Show more

0

Reply

F

FitMaroonBoa535

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmc9rk4hx01oead077hcr5gmc)

Is not anti REST that we have a business service responsible for endpoints prefixed with /business and a review service also having endpoints starting with /business?

Show more

0

Reply

J

jaswanthsai917

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmcofmyuz03yrad085kibfl6l)

Why cant we have the review to be updated in a singletranscation having both review table and business table since the scale is not very large for submitting reviews?

Show more

0

Reply

![Priyankar Raj gupta](https://lh3.googleusercontent.com/a/ACg8ocLj4znexnJYoaFwdkTmM26gju9vXeJeZHeGkBO0YPITob8d3Rsl=s96-c)

Priyankar Raj gupta

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmcte7lph06c9ad08z3dzkwcy)

For review table user\_id and business\_id is composite primary key right ? So unique constraint is automatically applied there, isnt that not right ? Reviews

- userId
- businessId
- rating
- text

why do we need locking ? I dont understand.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmctg2fab01t2ad08lnd39cuy)

The unique constraint in the deep dive is about preventing duplicate reviews (one user can only review a business once). The locking discussion is about preventing race conditions when updating the average rating. So these are different problems on different data :)

Show more

0

Reply

![Priyankar Raj gupta](https://lh3.googleusercontent.com/a/ACg8ocLj4znexnJYoaFwdkTmM26gju9vXeJeZHeGkBO0YPITob8d3Rsl=s96-c)

Priyankar Raj gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmcti2com02rlad08ppy49s4r)

Sorry for mixing locking and unique constraint.

Coming to first part of my comment, do you still think we need to explicitly add the unique constraint given composite primary key PRIMARY KEY (user\_id, business\_id):

> For review table user\_id and business\_id is composite primary key right ? So unique constraint is automatically applied there, isnt that not right ? Reviews
> 
> userId businessId rating text

Show more

0

Reply

![sjhello](https://lh3.googleusercontent.com/a/ACg8ocL56BDQABmAx8VZ7PKMIL1KlKovYlvewaGLfEZWkd8mWbPY9w=s96-c)

sjhello

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmd6hrpc200f9ad09dlndmh28)

Shouldn't the business review table have keys /schema in the order {businessId,reviewId,userId}

ReviewId is a monotonically increasing UUID that increases with time.

We can define a unique Index key on {businessId,userId} to enforce only 1 review on the business by the user.

We can get all the reviews of business sorted by time without having to do a join.

Show more

0

Reply

A

AdorableApricotGoat478

[• 25 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmdmmyevq002mad09q69cxz03)

can you share excalidraw link for this design?

Show more

0

Reply

![Aditya Sridhar](https://lh3.googleusercontent.com/a/ACg8ocJFdKFULXsN2myMcn2qHoMu0BgChqI9zVYT-qbaLPm6kfb66U-ocA=s96-c)

Aditya Sridhar

[• 25 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmdmnn0q300abad08c6qanti2)

Hello, very useful content. Thank you! When do you use Elastic search vs Redis for location-based search. My mental model is - if the storage requirement is small enough (say 10 - 20 GB), we use Redis as it provides faster read (< 1ms) by virtue of it being in-memory. Are there any advantages to using Elastic search over Redis?

Show more

0

Reply

![Guy Alster](https://lh3.googleusercontent.com/a/ACg8ocLS3RRNQfbNhhf51MAl0azntTjU2y_yoqpj8U7DLMAqTVxnpSSu=s96-c)

Guy Alster

[• 25 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmdmp04ql00cdad08v3uz71to)

I believe this design drill down is lacking and is of lower quality than the others. For example, when I tried solving it myself I immediately thought of having elastic search as the main search engine for searching as opposed to using the DB for it. In the last diagram you are using ES, but I didn't see you mention it before. Also, CDC is not always the best way to move the data between the DB and the Elastic Search, it is often better to use an OutBox pattern in the DB and have a job update ES from the DB. I think that the ranking section should be elaborated on in the requirements, and not left as a section to talk about in the high level design.

Show more

0

Reply

P

PrivateAzureAlbatross620

[• 23 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmdouvi5e02edad07056q2hih)

That Unique constraint is exactly why I'm on team DBs should be written to by a single service. That service is the right place for business logic, and that business logic might evolve in a way which isn't supported by the database.

Show more

0

Reply

![Vrajesh Patel](https://lh3.googleusercontent.com/a/ACg8ocIVdNLCet9U1l7iLF0kH_yzFRl1WOjyrt7QZZYTYdhHyrL-MJ9aHg=s96-c)

Vrajesh Patel

[• 8 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmeahy0n0006pad08lgt8s0wh)

For the 1 user review per business constraint, we could also have a composite primary key of userId and businessId as well, that will inherently apply the unique constraint on the userId and BusinessId pair.

Show more

0

Reply

![Arpit Shah](https://lh3.googleusercontent.com/a/ACg8ocIYgUYnzPcFRViVA97eKjOfUKokGqd6pz85EFJ-STtlNb7nVed5=s96-c)

Arpit Shah

[• 8 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmebc1oz606nzad07ujozjbyn)

How about the use of geo-hash?

- We can calculate full 8 char geo-hash, from Lat,long and store it in DB( business table. \[ full 8 char geo-hash gives precision of ~40 meters\]. Using PostGIS, we can search on geohash in postgre to get nearby businesses. \[ convert lat/long from Search API to geohash and search for first 6-7 character from DB \]. @Evan King

Show more

0

Reply

![Arpit Shah](https://lh3.googleusercontent.com/a/ACg8ocIYgUYnzPcFRViVA97eKjOfUKokGqd6pz85EFJ-STtlNb7nVed5=s96-c)

Arpit Shah

[• 8 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp#comment-cmebc83h906oyad073km7mqbn)

Can w euse hybrid approach. Step-1: Search Elastic Search DB based on the search query Step2: Pass the business Ids from Step-1 to Postgis to search on geo hashing( more granular search), to improve accuracy. What do you think on that? @Evan King

Show more

0

Reply
