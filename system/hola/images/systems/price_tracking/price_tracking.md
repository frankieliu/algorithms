# Design a Price Tracking Service

[![Evan King](/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75&dpl=2c00fec7d898fc5204eb9b0f9b4dae7f4e34d61b)

Evan King

Ex-Meta Staff Engineer

](https://www.linkedin.com/in/evan-king-40072280/)

medium

Published Jun 3, 2025

---

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

Everyone. Let's talk about how we would design the price tracking site

Deep Dives

1:00:49

Play

Mute

0%

0:00

/

1:02:28

Premium Content

Closed-Captions On

Chapters

Settings

AirPlay

Google Cast

Enter PiP

Enter Fullscreen

## Understanding the Problem

**📈 What is [CamelCamelCamel](https://camelcamelcamel.com/)?** CamelCamelCamel is a price tracking service that monitors Amazon product prices over time and alerts users when prices drop below their specified thresholds. It also has a popular Chrome extension with 1 million active users that displays price history directly on Amazon product pages, allowing for one-click subscription to price drop notifications without needing to leave the Amazon product page.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements)

**Core Requirements**

1. Users should be able to view price history for Amazon products (via website or Chrome extension)
    
2. Users should be able to subscribe to price drop notifications with thresholds (via website or Chrome extension)
    

CamelCamelCamel has a popular Chrome extension with 1 million active users that displays price history directly on Amazon product pages, allowing for one-click subscription to price drop notifications without needing to leave the Amazon product page.

**Below the line (out of scope):**

- Search and discover products on the platform
    
- Price comparison across multiple retailers
    
- Product reviews and ratings integration
    

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements)

The scale and performance requirements for CamelCamelCamel are driven by Amazon's massive product catalog and the need for timely price notifications.

**Core Requirements**

1. The system should prioritize availability over consistency (eventual consistency acceptable)
    
2. The system should handle 500 million Amazon products at scale
    
3. The system should provide price history queries with < 500ms latency
    
4. The system should deliver price drop notifications within 1 hour of price change
    

**Below the line (out of scope):**

- Strong consistency for price data
    
- Real-time price updates (sub-minute)
    

We're building a system that must be "polite" to Amazon while providing valuable price tracking to millions of users. This creates interesting technical challenges around data collection, storage efficiency, and notification delivery that we'll address in our deep dives.

Here is how your requirements might look on the whiteboard:

CamelCamelCamel Requirements

## The Set Up

### Planning the Approach

We'll follow the Hello Interview framework by building our design sequentially through each functional requirement, then use our non-functional requirements to guide our deep dive discussions on scaling challenges.

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

I like to begin with a broad overview of the primary entities we'll need, without getting bogged down in all the details yet. We can always add more complexity as we iterate on the design.

Just make sure that you let your interviewer know your plan so you're on the same page. I'll often explain that I'm going to start with just a simple list, but as we get to the high-level design, I'll document the data model more thoroughly.

To satisfy our key functional requirements, we'll need the following entities:

1. **Product**: Represents an Amazon product we're tracking, including its current price and metadata needed for price history display and notifications
    
2. **User**: Represents a person using our service, containing their contact information and preferences for receiving price drop notifications
    
3. **Subscription**: Links users to products they want to monitor, specifying the price threshold that triggers notifications
    
4. **Price**: Time-series data capturing price changes over time for each product, enabling us to generate historical price charts
    

In the actual interview, this can be as simple as a short list like this. Just make sure you talk through the entities with your interviewer to ensure you are on the same page.

CamelCamelCamel Entities

### [The API](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#4-api-or-system-interface)

The API is the main way users will interact with our price tracking service, both through our website and the Chrome extension. Let's design endpoints that directly support our two core functional requirements.

`// Retrieve historical price data for charts and displays GET /products/{product_id}/price?period=30d&granularity=daily -> PriceHistory[]`

We include product\_id in the URL path rather than the request body for better caching and REST compliance. The granularity parameter allows us to return daily averages for long periods or hourly data for recent timeframes.

`// Subscribe to price drop notifications with threshold POST /subscriptions {   product_id,   price_threshold,   notification_type, } -> 200`

### [Data Flow](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#optional-data-flow-5-minutes)

Before diving into the technical design, let's understand how data flows through our system. The data flow shows how information moves from initial collection to final user output.

Understanding this flow early in our design process serves multiple purposes. First, it helps ensure we're aligned with our interviewer on the core functionality before getting into implementation details. Second, it provides a clear roadmap that will guide our high-level design decisions. Finally, it allows us to identify potential bottlenecks or issues before we commit to specific architectural choices.

In this case, the "hidden" requirement is that we need to be able to get the data in the first place!

1. Web crawlers and Chrome extension collect current prices from Amazon product pages
    
2. Price data is validated and stored in our price database
    
3. Price changes trigger events for notification processing
    
4. User receives email notification when price drops below threshold
    

Note that this is simple, we will improve upon as we go, but it's important to start simple and build up from there.

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes)

We'll build our design incrementally, starting with the most fundamental requirement and adding complexity as we address each successive need. This ensures we deliver a working system first, then layer on additional capabilities.

Amazon is not friendly to price tracking services. They don't provide an API, actively discourage scraping, and implement rate limiting (typically 1 request per second per IP address). We must design our system to work within these constraints while remaining respectful of Amazon's terms of service.

### 1) Users should be able to view price history for Amazon products (via website or Chrome extension)

Users want to see how prices have changed over time for specific Amazon products. This serves as the foundation of our entire system and breaks down into two distinct challenges: acquiring price data from Amazon and serving historical price queries efficiently.

For the first iteration, we'll establish the basic data collection and storage infrastructure. Our architecture needs a web scraping system to gather price data, a storage layer for historical prices, and an API service to serve price history to users.

CamelCamelCamel High Level Design - Price History

Here's what we're building:

1. **Client**: Our client is either the website or the Chrome extension. In either case, they're how the user views price history graphs.
    
2. **API Gateway**: Handles authentication, rate limiting, and routes requests to appropriate services
    
3. **Price History Service**: Manages price data retrieval and coordinates with the database
    
4. **Web Crawler Service**: Scrapes Amazon product pages to collect current prices
    
5. **Price Database**: Stores both current product information and historical price data
    
6. **Primary Database**: Stores all other tables aside from price like Users, Products, and Subscriptions.
    

We choose to separate the Price History Service from the crawler because they have different scaling characteristics. The history service needs to handle many concurrent read requests from users viewing charts, while the crawler operates on a scheduled basis with consistent write patterns. This separation also allows us to optimize each service independently.

As for the databases, we'll make a strategic separation based on scale and access patterns. Our Primary Database will store Users, Products, and Subscriptions together since these have similar operational characteristics - they're relatively small and have traditional CRUD access patterns. However, we'll put our Price Database completely separate because price history data has fundamentally different requirements: it will grow to billions of rows, is append-only, requires time-series optimizations, and can tolerate eventual consistency. This separation allows us to optimize each database independently and scale them based on their specific needs.

We'll choose to "black box" the web crawler service for now. We can assume it periodically scrapes Amazon product pages and updates the Price table. Of course, we'll go into more detail about just how this might be accomplished in our deep dives.

When a user requests price history through our website or Chrome extension:

1. Client sends GET request to /products/{product\_id}/price?period=30d
    
2. API Gateway authenticates the request and routes to Price History Service
    
3. Price History Service queries the Price table filtered by product\_id and time range
    
4. Historical prices are returned as JSON, ready for chart rendering
    
5. Chrome extension or website displays the price chart to the user
    

### 2) Users should be able to subscribe to price drop notifications with thresholds (via website or Chrome extension)

Building on our price history foundation, users now want to receive notifications when product prices drop below their specified thresholds. This introduces new requirements for user management, subscription storage, and notification delivery.

We'll add new components while leveraging our existing infrastructure. For our initial implementation, we can start with a dead simple polling approach that gets the job done, knowing we can optimize this later in our deep dives.

CamelCamelCamel High Level Design - Subscriptions

Our enhanced architecture includes:

1. **Subscription Service**: Manages user subscriptions and price threshold settings
    
2. **Notification Cron Job**: Periodically checks for price changes and sends email notifications
    

We separate the Subscription Service from the Price History Service because they serve different user actions and have different data access patterns. The Subscription Service handles user account operations (create, update, delete subscriptions), while the Price History Service focuses on read-heavy price queries.

For our initial notification system, we have a straightforward batch processing approach:

1. Cron job runs every 2 hours to check for notifications
    
2. Job queries Price table for any price changes in the last 2 hours
    
3. For each price change, query Subscriptions table from the Primary Database to find users with thresholds >= new\_price
    
4. Send email notifications for any triggered subscriptions
    
5. Mark notifications as sent to avoid duplicates
    

This polling approach works but has obvious limitations. Users might wait up to 2 hours for price drop notifications, and we're doing expensive database scans every few hours. We'll explore much better real-time approaches in our deep dives that provide sub-minute notification delivery.

When a user subscribes to price alerts:

1. User submits subscription via website or Chrome extension
    
2. API Gateway routes POST request to Subscription Service
    
3. Subscription Service creates record in Subscriptions table (user\_id, product\_id, price\_threshold)
    
4. Service returns confirmation to user
    

This simple approach makes sure we have a working subscription system that satisfies our functional requirements. Users get price drop alerts (just not immediately), and our system remains easy to understand and debug. The periodic nature also prevents notification processing from interfering with our core price tracking functionality.

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes)

Time for the fun part. We'll take our existing, simple design and layer on complexity via our deep dives. When we're done, we'll have a system that satisfies all of our functional and non-functional requirements.

As one of my favorite sports journalists Fabrizio Romano would say, "Here we go!"

### 1) How do we efficiently discover and track 500 million Amazon products?

Up until now, we've been "black boxing" our web crawler, simply assuming it somehow gets price data from Amazon and updates our database. But when we confront the reality of tracking 500 million products while respecting Amazon's rate limiting constraints, we realize this is actually one of our biggest technical challenges.

We're solving two distinct but related problems here.

The first challenge is product discovery: we need to find all 500 million existing products when we launch, plus discover the approximately 3,000 new products Amazon adds daily.

The second challenge is price monitoring: we need to efficiently update prices for millions of known products while prioritizing which ones to check most frequently.

Both challenges face the same fundamental constraint of Amazon's rate limiting, but understanding them separately helps us design more effective solutions.

### 

Bad Solution: Naive Web Crawling Approach

##### Approach

The most straightforward approach involves deploying traditional web crawlers that systematically browse Amazon's website. We start from seed URLs like Amazon's homepage and category pages, extract product links from each page, and recursively follow these links to discover the entire product catalog. Each crawler runs through categories methodically, extracting product IDs, prices, and basic metadata before moving to the next page.

From an implementation perspective, this is largely what we designed in [Design a Web Crawler](https://www.hellointerview.com/learn/system-design/problem-breakdowns/web-crawler).

This treats Amazon like any other website to be indexed. We implement a frontier queue containing URLs to visit, a visited URL tracker to avoid cycles, and crawlers that process URLs sequentially. When a crawler encounters product recommendation links or "customers also viewed" sections, it adds these new URLs to the frontier queue for future processing. The system maintains a simple database of discovered products and updates prices whenever a product page is revisited.

Naive Web Crawler

##### Challenges

The fundamental problem is scale arithmetic. With 500 million products to track and Amazon's rate limit of approximately 1 request per second per IP address, a single crawler would require over 15 years just to visit each product page once. Even if we deploy 1,000 servers with different IP addresses, we're still looking at 5+ days for a complete catalog refresh, during which millions of price changes will go undetected.

This also creates a discovery lag problem. New products added to Amazon might not be found for weeks or months, depending on where they appear in our crawling sequence. Popular products that change prices multiple times per day will have stale data for extended periods, making our service unreliable for time-sensitive deals. The massive infrastructure cost of running thousands of crawling servers makes this economically unfeasible for most companies.

### 

Good Solution: Prioritized Crawling Based on User Interest

##### Approach

The key realization here is that we have fundamentally limited crawling resources, so we better use them wisely. Product popularity follows a Pareto distribution - a small percentage of Amazon's products get the vast majority of user attention, while there's a long tail of products that people rarely care about. Instead of treating all 500 million products equally, we can tier our crawling based on actual user interest and dramatically improve our service quality where it matters most.

Pareto

We can implement a priority scoring system where products get higher scores based on active user engagement. Products with many price drop subscriptions receive the highest priority since users are actively waiting for updates on these items. Products that users search for frequently on our website also get elevated priority, indicating current interest even if subscriptions haven't been created yet.

The crawling system maintains priority queues where high-interest products might be checked every few hours, medium-interest products get daily updates, and low-interest products are refreshed weekly or even less frequently. We also implement feedback loops where successful price drop notifications (ones that lead to user clicks or purchases) boost a product's priority score, creating a self-reinforcing system that focuses on the most valuable updates.

This allows us to achieve excellent coverage for the products that matter most to our users while using a fraction of the infrastructure required for comprehensive crawling. Popular products stay current while niche products still receive periodic updates.

Priority Queue

##### Challenges

The fundamental limitation is coverage gaps for new or trending products that haven't yet built up user interest signals. A hot new product release might not get discovered quickly if no users are searching for it yet, potentially missing early adoption opportunities when prices are most volatile.

### 

Great Solution: Chrome Extension + Selective Crawling

##### Approach

Remember that Chrome extension with 1 million users we mentioned earlier? It turns out to be more than just a convenience feature, it's actually our secret weapon for data collection.

The most elegant solution leverages our Chrome extension's 1 million users as a distributed data collection network. When users browse Amazon with our extension installed, it automatically captures product IDs, current prices, and page metadata, then reports this information to our backend services. This crowdsourced approach transforms user browsing behavior into our primary data collection mechanism.

The extension operates transparently, extracting structured data from Amazon's product pages using DOM parsing and sending updates to our price reporting API. We receive real-time price data for products that users are actively viewing, which naturally prioritizes popular and trending items. This user-generated data covers the products people actually care about without requiring extensive crawler infrastructure.

Extension Crowdsource

Our traditional crawlers now just need to handle products that haven't been viewed by extension users recently. The system also uses extension data to discover new products; when users visit previously unknown product pages, we add them to our Product table. Brilliant!

##### Challenges

This introduces dependency on user adoption and browsing patterns. Products in niche categories with low user interest might receive infrequent updates, creating coverage gaps. But most importantly, we must carefully handle the data validation challenge since user-generated price reports could be manipulated or incorrect, requiring verification systems we'll explore in our next deep dive. We'll talk about this next.

This hybrid approach solves our scale challenge by turning our biggest constraint (the need to monitor millions of products) into our competitive advantage through crowdsourced data collection. Instead of fighting Amazon's rate limits with massive crawler infrastructure, we leverage user behavior to naturally prioritize and collect data for the products that matter most.

If you're thinking to yourself, "How the hell would I know this? I know nothing about Chrome Extensions!" don't sweat it. When I ask this in an interview, I usually lead with, "Your extension is installed on millions of browsers actively viewing product pages. Can you make use of that somehow?"

### 2) How do we handle potentially malicious price updates from Chrome extension users?

The Chrome extension approach from our previous deep dive creates a powerful data collection advantage, but it introduces a critical reliability challenge. With 1 million users potentially submitting price data, we must assume that some percentage will be malicious actors, mistaken users, or systems experiencing technical issues.

Consider this scenario: A malicious user reports that the latest iPhone costs $0.01, triggering price drop notifications to thousands of subscribers. Users rush to Amazon only to find the normal price, damaging our credibility and user trust. Our notification system's effectiveness depends entirely on data accuracy, making this challenge essential to solve properly.

So what should we do?

### 

Good Solution: Consensus Based Validation

##### Approach

One thing we could do is validate by committee. While we may have a few bad actors, they're not likely to be pervasive. Instead of immediately accepting any price report, we wait until we have N independent users report the same price change before marking it as validated and triggering notifications.

The system works by holding price updates in a "pending validation" state when they first arrive from extension users. We maintain a consensus table that tracks how many unique users have reported each specific price for each product within a time window (say, 1 hour). Only when we reach our consensus threshold - perhaps 3 independent users for popular products, or 2 for less popular ones - do we accept the price change as legitimate and process notifications.

For products with high subscription counts or significant price drops, we can require higher consensus numbers. A $500 price drop on the latest iPhone might need 5 independent confirmations, while a $2 change on a book might only need 2. The system tracks user reputation over time, so reports from consistently accurate users might count more heavily toward the consensus requirement.

This creates a natural defense against both malicious actors and technical errors. A single bad actor can't trigger false notifications because they need multiple accomplices. Technical issues like browser cache problems or network glitches are unlikely to affect multiple users simultaneously in the same way, so these get filtered out automatically.

##### Challenges

The fundamental limitation is delay for legitimate price changes, especially for niche products. If only a few users monitor a specialized product category, it might take hours or even days to reach consensus threshold, meaning subscribers miss time-sensitive deals. Flash sales or limited-time offers could expire before enough users browse the product to validate the price drop.

Popular products create their own problems during major sales events. When Amazon drops prices on thousands of items simultaneously, our system might be overwhelmed with pending validations, creating backlogs that delay notifications even for products that quickly reach consensus.

Lastly, it's still possible for bad actors to coordinate and "beat" your consensus system. While maybe unlikely, it's still a possibility.

### 

Great Solution: Trust-But-Verify with Priority Verification

##### Approach

The best approach is simple: trust the extension data immediately, but verify it quickly with our crawlers when something looks suspicious.

Here's how it could work. When our extension reports a price change, we accept it right away and can send notifications immediately. But if the change seems fishy - like a huge price drop, or it's from a user who's been wrong before, or it's a popular product with lots of subscribers - we automatically queue up a verification crawl job to check Amazon directly within a few minutes.

We use our existing crawler infrastructure but give these verification jobs high priority. Instead of waiting hours or days for our regular crawling schedule, suspicious price updates get checked within 1-5 minutes. If our crawler finds out the extension data was wrong, we can send correction notifications and mark that user as less trustworthy.

For really important products, we can wait for multiple users to report the same price change before we fully trust it. If we see conflicting reports from different users, we immediately send a crawler to figure out who's right.

The nice thing is that most extension data gets processed immediately (fast notifications), but we catch the bad stuff quickly enough that it doesn't cause real damage. Users get fast alerts for legitimate deals, and we maintain trust by correcting the occasional mistake.

##### Challenges

This does add significant complexity to our crawling infrastructure, requiring priority queue management and rapid response capabilities that consume additional server resources. The verification crawling creates more load on Amazon's servers, potentially increasing our risk of rate limiting or IP blocking.

This trust-but-verify approach strikes a really nice balance, providing immediate user value while maintaining the data integrity essential for long-term trust.

### 3) How do we efficiently process price changes and notify subscribed users?

Right now, we're using a straightforward cron job that runs every 2 hours to scan for price changes and send notifications. While this might work for basic functionality, it fails to meet our non-functional requirement of delivering notifications within 1 hour of price changes. More critically, the polling approach creates massive database load as we scale to millions of price updates and subscriptions.

The notification system's effectiveness directly impacts user satisfaction and retention. Users expect timely alerts for price drops, especially for competitive deals that might sell out quickly. A 2-hour delay often means missing time-sensitive opportunities, making our service less valuable than competitors with faster notification systems.

The fundamental problem with our current polling approach is that we're asking the wrong question at the wrong time. Instead of asking "what changed in the last 2 hours?" every 2 hours, we should be asking "who cares about this change?" immediately when a change happens.

Moving to event-driven notifications is straightforward. When a price changes, we immediately know which users need alerts without expensive database scans. We have two solid ways to implement this, each with different trade-offs.

The first option is database change data capture (CDC), where we configure our database to automatically publish events whenever price data changes. When our crawlers or extension data processors insert new price records, database triggers fire and send events to Kafka (or similar) containing the product ID, old price, and new price. Our notification service subscribes to these events and immediately queries the subscriptions table to find affected users.

Database Change Data Capture

This is clean because price changes automatically trigger notifications without any application-level coordination. We don't need to remember to publish events in our application code since the database handles it for us. The data flow becomes completely automatic.

The second option is dual writes, where our price collection services write to both the database and publish events to Kafka simultaneously. When crawler data or extension updates come in, the same service that writes to the price database also publishes structured events to our notification stream.

The dual-write approach lets us be smarter about which changes trigger notifications. We can filter out tiny price fluctuations that wouldn't interest users or batch multiple rapid changes before publishing events. We also avoid the overhead of database triggers on every single price update.

Regardless of which option we choose, our notification consumers work the same way. They read price change events from Kafka and immediately query the subscriptions table to find users whose thresholds have been triggered. For each matching subscription, we send an email notification through our existing notification service.

The beauty is that we only process actual price changes rather than expensive table scans of millions of records. Most products don't change price very frequently, so we're talking about a manageable number of events even at scale. When a popular product's price drops, we run one query to find all subscribers with thresholds above the new price and send those notifications right away.

### 4) How do we serve fast price history queries for chart generation?

Looking at our current data storage approach, we're keeping price history in a straightforward PostgreSQL table with basic indexing on (product\_id, timestamp). This works for initial functionality, but fails to meet our < 500ms latency requirement when serving price charts for popular products with extensive history data.

Consider the scale challenge that comes from aggregating price data for popular products. Popular products might have price data points every few hours for several years, resulting in many thousands of records per product. When users request a 2-year price chart, we must aggregate this data into appropriate time buckets (daily or weekly averages) and return it quickly enough for smooth user experience. Raw database queries struggle with this aggregation workload.

### 

Good Solution: Scheduled Pre-Aggregation with Cron Jobs

##### Approach

The straightforward solution is pre-computing price aggregations at different time granularities using scheduled batch jobs. Every night, we run a job that calculates daily, weekly, and monthly price summaries for all products, storing these aggregations in a simple table optimized for fast chart queries.

Our batch job processes the previous day's price data and computes relevant statistics: average price, minimum price, maximum price, opening price, and closing price for each time period. We store all this in a single price\_aggregations table with a granularity column that indicates whether each row represents daily, weekly, or monthly data.

When users request price charts, our API queries this aggregation table instead of the raw price data. A 30-day chart queries for daily aggregations, while a 2-year chart uses monthly aggregations. With proper indexing on (product\_id, granularity, date), these queries return dozens of pre-computed records in milliseconds instead of aggregating thousands of raw price points.

Scheduled Pre-Aggregation

##### Challenges

Pre-aggregation creates data freshness problems since charts show data that's up to 24 hours old. For a price tracking service, this staleness is generally acceptable since users are looking at historical trends rather than real-time prices. The batch processing approach also requires additional storage space for all the pre-computed summaries, though this scales linearly with the number of products and time periods we support.

### 

Great Solution: TimescaleDB for Real-Time Price Analytics

##### Approach

Rather than running these cron jobs, we can use [TimescaleDB](https://www.timescale.com/), a time-series extension for PostgreSQL that's purpose-built for this type of workload. Since we're already using PostgreSQL for our operational data (users, products, subscriptions), TimescaleDB lets us handle price history analytics within the same database ecosystem while getting specialized time-series performance.

With TimescaleDB, we can perform real-time aggregations directly without any pre-computation. When a user requests a 6-month price chart, we run a query like SELECT time\_bucket('1 day', timestamp), avg(price) FROM prices WHERE product\_id = ? GROUP BY 1 and get results in milliseconds even with billions of price records. TimescaleDB's automatic partitioning and compression make these aggregations incredibly fast while maintaining PostgreSQL's familiar interface.

This keeps our architecture simple. We don't need complex pre-aggregation jobs or multiple database systems. Users can request any time range or granularity, and TimescaleDB computes the results on demand. The system handles our scale requirements naturally since TimescaleDB is designed for exactly this type of time-series workload, while we maintain all the operational benefits of staying within the PostgreSQL ecosystem.

TimescaleDB

Alternatives like ClickHouse could provide even higher analytical performance, but TimescaleDB offers the best balance of performance and operational simplicity by keeping our entire data stack unified.

The TimescaleDB approach provides the performance and flexibility needed for production-scale price chart serving while maintaining the real-time responsiveness users expect. By choosing the right tool for analytical workloads, we achieve both simplicity and performance without complex caching or pre-aggregation strategies.

## Final Design

Taking a step back, we've designed a scales system that can intelligently collect price data from Amazon, validate it, and notify users when prices drop all while serving price history charts to users in real-time. Not bad!

As for the final diagram, I'll admit the drawing got a little out of hand with all the crossing arrows. But we end up with a final design that looks something like this!

Final Design

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

So, what am I looking for at each level when I ask candidates to design a price tracking system like CamelCamelCamel?

### Mid-level

At mid-level, I'm looking for candidates who can create a working high-level design that addresses the core functional requirements - price history display and notification subscriptions. They should understand the basic challenge of collecting price data from Amazon and propose a simple web scraping approach, even if it's not scalable. I expect them to design a straightforward database schema with products, users, subscriptions, and price history. While they might initially suggest polling every product frequently, with some guidance they should recognize the scale limitations and understand why prioritization might be necessary.

### Senior

For senior candidates, I expect you to quickly identify that data collection is the core technical challenge and drive the conversation toward solutions like the Chrome extension approach. You should understand the trade-offs between different crawling strategies and understand that crawling 500M products will require either immense resources or endless time. I'm looking for solid understanding of notification system architectures, you should be able to explain why polling doesn't scale and propose event-driven alternatives using message queues. Ideally you go deep on at least of couple of deep dives, even if not the ones I have above.

### Staff+

For staff candidates, I'm evaluating your ability to see the bigger picture and balance technical elegance with operational reality. You should proactively recognize that the Chrome extension isn't just a user feature, but that it can be used to solve the fundamental data collection scaling problem. I expect you to discuss system evolution thoughtfully: how do we start simple but design for the scale we'll eventually need? You should understand the business implications of technical choices, for example, why data validation affects user trust and retention, not just technical correctness. Strong candidates often surface concerns I haven't asked about, like handling Amazon's anti-scraping measures, managing extension user privacy, or planning for what happens when Amazon changes their page structure. You demonstrate systems thinking by considering how different components affect each other and proposing solutions that prioritize long-term maintainability alongside immediate functionality.

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

D

DarkSalmonGoose801

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbjwpvly002gad08c9njbwxd)

Great content, you guys rock!

Show more

10

Reply

![Pankaj Bhambani](https://lh3.googleusercontent.com/a/ACg8ocJOHUj8AKKD6NYXryJK9cTIMleH4ZarFm_cf_oIcwgqyF3TubhZiw=s96-c)

Pankaj Bhambani

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbku4jj500x8ad077vptucs8)

Hey, just a quick suggestion, during one of my interviews, I received feedback that I didn’t discuss the trade-offs between different technologies. For example, I mentioned using Postgres without explaining why it would be better or comparing it to alternatives like NoSQL. It might help to include the trade-offs between options when discussing tech choices.

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbkz4qb0012iad08t71ba92e)

Good shout. I did that less in this one than normal (aside from the DB choice). Oftentimes, when this is the case, it's because the decision matters less. But let me go through and call out a couple of places for next release.

Show more

8

Reply

FH

Frank H.

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmblk9e7m000v08adc0l7e2u2)

Love it! I’ve been looking for something like this for a while. Finally, I get the chance to compare my own system design approach with this high-quality reference and spot any gaps.

Big thanks to the team for putting this together!

Show more

1

Reply

![Aashima A](https://lh3.googleusercontent.com/a/ACg8ocJDjLQArHZ5jqceZ1gn1WZJMaZui4uqNr4KfS5GX1z9vYdLwA=s96-c)

Aashima A

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmblqloh3006308adqohi75m9)

Is crawler updating the price db or is the chrome extension updating it? Its hard to say because of the dual arrows.

- why do we need a price change worker from primary db? what is it doing?
- crawler is updating prices in primary db and adding products to price db? or that incorrect?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbmegwup00ig08advm1nycse)

Price change worker needs to read the primary db to get the subscriptions! So it can compare them against price changes and send notifications.

For your second bullet, yes, correct!

Show more

0

Reply

W

WillowyWhiteOx710

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmd79obs3029sad08q7ois4tx)

If we just auto parse details using the chrome extension, how would a user report malicious price updates for a product?

Show more

0

Reply

![RITU RAJ](https://lh3.googleusercontent.com/a/ACg8ocKCtDQcR6quJz6ry7KOfyTO-iegYVxIT6m5q9s24zaafnT9XsY=s96-c)

RITU RAJ

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbm1uvwn00aj08ads8zuwm7z)

The arrow for updarte price and add products seem to be wrong,it should be the other way around

Show more

1

Reply

FH

Frank H.

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbmsu5t700er08adxqtgj849)

I am thinking the same. In addition, what does the arrow mean from Price DB to Scheduler? same question for the arrow from Primay DB to Scheduler. If the Scheduler is for schedule the Product to the queue, then the arrow should come from the Subscription Service or Price Update Service.

Show more

2

Reply

B

BareCoffeeBasilisk390

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbmoalab006v08adgnti8jos)

Thanks great content as alwats.

When you introduce the CDC to the Price Change Stream, yuo suggested Kafka or similar. By similar did you mean something like Kinesis or do you think SQS could be an option here?

And for the queue used in the web crawler, I'm thinking SQS is a beter call there because you can get more fialure without HTTP requests, thoughts on that?

Show more

0

Reply

R

RelevantWhiteAntelope452

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbmye43j00ko08adxq3d3e0v)

Shouldn't the texts for the "Add products" and "Update prices" arrows from the Crawler be opposite?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbpemqvr006u08ad7gopls09)

The crawler is finding new products and new prices and writing the udpates back to the DBs respectively

Show more

0

Reply

R

RelevantWhiteAntelope452

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbphtj5h00a108adjeijx7lb)

Right, but price updates should go to the Price DB, and product additions should go to the Primary DB, isn't it? Your current diagram shows the opposite, which is what needs to be fixed imo.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbpizl4u00ci08adyige7226)

Oh i see! My b, will fix :)

Show more

0

Reply

U

UnchangedBlackGorilla909

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbo7w8nu01q308adgww66hff)

The idea of having customers sending the data is very clever, I'm cusrious if there is a way we could leverage and authN/Z mechanism to be able to get our servers to get to "yeah this is definetly coming from the chrome extension" since the client that sends the data baked into the extention. I was thinking some sort of userAgent check or header check but then realized those can be manipulated by any client, then I thought maybe we can do create a SIGv4 auth header and very server side? not sure if this would actually work but am curious if anyone has any dieas on that

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbo83wck01cu08ad2xtp08am)

You definitely can't trust the data coming from the client, no matter the encryption scheme, but you can make it harder to manipulate.

For adversarial problems, you generally want to ask "what benefit would an attacker have?" and try to blunt that. In this case, maybe a merchant wants more traffic to their product so they try to get the price to appear dropped ... "how would you minimize the benefit they get here?" is half the battle.

Show more

2

Reply

J

johnyric7

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbsl8c1j003n08ad446vgoxu)

Even when there are a million active users, it will still make a call to Amazon right? How would we make sure we are handling it politely? It will still be a lot of calls right? May be have a queue and keep track of the previous call and then call/not call accordingly?

Show more

0

Reply

N

NursingScarletFly558

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmcdb7n7l05kfad0885yklt7d)

Browser extensions have access tro the browser DOM, that is the extension can "see" the HTML in the page. If a user is browsing amazon.com already they have already made the API call to amazon to get the HTML. Now our extension simply has to use javascript to identify the price and product ID from the webpage and make a request to our server.

Show more

2

Reply

F

FriendlyWhiteParrot680

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbo8x36k01kf08adhnbp1w5s)

1. On containing the product ID, old price, and new price since our price history DB is append only, how would you generated this CDC event? I don't think in this design we are doing an UPDATE of a row. We could implement this like so: add another column called price\_changed. When we append the row, we read the previous row and see if this new price being inserted is different.
2. Alternatively, we can only insert rows when price changes - that way CDC is only configured for Appends
3. For notification service, if the price of the product is $10 and user's threshold is below $5 and product price drops down to $2, we should be mindful to not send notification if the price increases to $3. Since, we already sent one at $2 or we should inform them that the price is back on the rice. TL; DR we need some semantics of "notifiction history".

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbpelp0d006s08adn9q1tc0w)

For #1 and #2: CDC events don't require UPDATEs, they work fine with INSERTs. The CDC system can compare the latest price with the previous one for the same product\_id. No need for a price\_changed column or to only insert on changes. For #3: Spot on about notification history. We need to track the last notification sent per subscription and only trigger new ones when price drops below threshold AND we haven't already notified for a lower price. Could store this as last\_notified\_price in the subscription table.

Show more

3

Reply

N

NobleGreenOrca860

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmc080ggd056v08adi2pgvovq)

It would be awesome if you could write a technology deep dive on CDC mechanism, and its various usecases. It seems a bit tricky

Show more

0

Reply

![KK](https://lh3.googleusercontent.com/a/ACg8ocKLYy4a8XaThL-Nnj1QyGTaV4iZ2XTEw6SyGGAOimZyD5-wC6hd=s96-c)

KK

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmde70udf00c4ad08yt7kqg61)

"The CDC system can compare the latest price with the previous one for the same product\_id." Would you please elaborate on that? Based on my understanding, CDC is just reflecting the data that you changed in the database or tables that you subscribed, just a streaming of events. How could we implement the compare logic with the old price? doesn't that require a database lookup?

Show more

1

Reply

B

Bobbiejim

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbp7xom902dd08adno24gfe2)

Deep dive 3) should have a bad/good/great breakdown so its more readable/follows the same structure as the other deep dives, since there are two solutions you can pick from.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbpekwno006q08add3jx68ar)

I opted not to just because I didn't feel it was worth as deep a tradeoff discussion. Either works!

Show more

0

Reply

![ShinyArceus493](https://lh3.googleusercontent.com/a/ACg8ocJI5i6TIxrEev--SrgMIk-Csqrf72M5BeneEA0ZHWKjmx1zoA=s96-c)

ShinyArceus493

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbpu172u005v08adrhqaa4fq)

You guys are doing God's work at **Hello Interview**! Thanks a ton on behalf of everyone here.

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbsf27sw000a08ads7hy07gc)

🫶🏻

Show more

1

Reply

R

RemoteBlueGuan642

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbsr4btb008y08ad2ow3qmr7)

Would have loved to see back of the envelope calculations after the final design. I am not able to visualize how this solution fits the requirements.

1m chrome extension, assuming 10% daily amazon users. 100K visits. If we track item search results, each user generates ~50 prices.So 5M products/day. Assuming duplicates let's take 40% of this: 2M products/day Now adding 100K from the crawler. It's still ~2M /day

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbsr9z1c009e08adrhkjdzeq)

Yah this is important. In the interview, the discussion usually goes, "I can't meet the SLA for ALL products, but I can meet it for the products that matter. Here's how."

Show more

0

Reply

![Ahmad Awad](https://lh3.googleusercontent.com/a/ACg8ocIyavhH2pmR4YTSKjR5Cqor7NBvHjJ2N7Mgm-8_umh6DukkKq2D=s96-c)

Ahmad Awad

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbv7u8ib000f08ad6w2ls402)

Thanks for the great and detailed content!

But my concern is, without a prior knowledge or seeing this problem before, I doubt if I could solve this problem in a real interview.

Show more

3

Reply

L

LinearSapphireJaguar130

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbvw70sy00a508adjs2yqaxu)

We are talking about 500 million products in amazon. What if there are changes to about 2million product at the same time? How does Kafka handle this much of a load? As I understand it can support upto 800k ~ 1M events per sec per cluster. How do we scale that? Filter out the events to only products where we have any active subscription or there is better way?

Show more

0

Reply

![Ge Xu](https://lh3.googleusercontent.com/a/ACg8ocIUwlvTlvr8XDYlJr0mawAAAJrlIz-ROily9Zg_FOKgpjV_7mqN=s96-c)

Ge Xu

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbwlaq3203md08adotrfpvx6)

I am curious if dynamoDB is another choice. The tables are simple and no complex query with join. It handles heavy write, and we can set priceUpdateTime as sort key.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbwlcz2503ni08adexvisnzx)

works great!

Show more

1

Reply

S

SpontaneousSalmonHalibut342

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbxs92vt0b1f08aduwzqa0fz)

We need to query all subscriptions for a product for alerting purpose.

Select \* from Subscriptions where ProductId=123;

Show more

1

Reply

S

SpontaneousSalmonHalibut342

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmbxs3ah70b0308adc41ylaex)

I was asked this in the interview. It's so much different that other questions. I messed it up. I skipped this as this was a new entry and wasnt in top asked question.

Interviewer asked for the tradeoff for database. I used DynamoDB. In hurry, I added all subscribed products in the UserID entity which now seems like a bad approach.

Show more

0

Reply

K

kstarikov

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmc3tg68900qb08admrdb4x7v)

> When our crawlers or extension data processors insert new price records, database triggers fire and send events to Kafka

Which database has triggers that send events to Kafka?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmc3tlm5800q408addogfyq4x)

Like which database technology or which database in our diagram?

Show more

0

Reply

K

kstarikov

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmc6kvtqp07ftad082q6hwn99)

Which database technology. This design shows that the CDC component is connected to the Postgres database for prices, but Postgres doesn't have the triggers with the functionality that you mention.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmc6l2idn07cqad08gj54j6f0)

PostgreSQL absolutely supports triggers that can fire on insert/update/delete and these triggers can be used to implement CDC. You can write triggers in PL/pgSQL that publish changes to external systems, and there are mature CDC tools like Debezium that work directly with Postgres' write-ahead log (WAL) to capture changes.

Show more

1

Reply

K

kstarikov

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmc80ivym0hdhad08x9tnbglw)

> PostgreSQL absolutely supports triggers that can fire on insert/update/delete and these triggers can be used to implement CDC. You can write triggers in PL/pgSQL that publish changes to external systems

Everyone reading this should do their homework on why this is a bad idea and why existing Postgres's CDC utilities don't work like this.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmc83119j0indad09lcrj3tkn)

Teach us :) Fwiw, strong opinions in an interview based on experince is great!

Show more

0

Reply

C

ChristianPlumTakin500

[• 26 days ago• edited 26 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmdmcguj70521ad086t7urqge)

TLDR; huge hit on performance and potential data-loss using triggers, because it impacts primary operations — "Triggers execute as part of the same transaction that modifies the data"

From Gemini,

> While Evan King is technically correct that triggers can be used for CDC in PostgreSQL, kstarikov's point is more aligned with modern best practices. Using triggers for CDC is generally considered a legacy or less-than-ideal approach due to the significant performance and maintenance drawbacks.

> The industry has largely moved towards log-based CDC solutions like Debezium precisely because they avoid the pitfalls of the trigger-based method. Therefore, in a system design interview, advocating for a WAL-based approach for CDC with PostgreSQL would be the stronger and more informed answer.

Show more

0

Reply

F

FancyTanGoldfish347

[• 20 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmdup2ftt0b3jad07j0dg981d)

The PostgreSQL WAL CDC approach is still a valid path it presents minimal performance impact compared to queries/triggers.

Show more

0

Reply

![Vankshu Bansal](https://lh3.googleusercontent.com/a/ACg8ocJs_boSzlEWXXmgwq72Mj8aYHqsOX0Vwi6X5pQP2JG1YWf6Rw=s96-c)

Vankshu Bansal

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmd43nt8o05tfad08zlvyli9t)

Aurora Postgres supports CDC (or it will, i was a part of the team working on this).

Show more

0

Reply

![Sudhanshu](https://lh3.googleusercontent.com/a/ACg8ocLNhPeK2pYoUwd2DcMaipO6CWYcGvCOODBZnu3Nh2h6RlFS7qfwSA=s96-c)

Sudhanshu

[• 2 months ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmc9fk6x901q5ad08g8yjsg5d)

Real time notification solution will also depend upon number of subscription being made on products. let's say we have 500M product but only 100K are subscribed for price change. Then, CDC will be cause more trouble than harm since we are reading off all the price changes and discarding most of it. A better approach in that case will still be sharding our db and having dedicated notification cron for each shard which reads off from the db and notifies the user about any price change.

What do others think?

Show more

0

Reply

F

FancyTanGoldfish347

[• 20 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmduppy0s0b9gad07o2yfd93x)

Based on some discussions above, if we plan to discard price update events for products with 0 subscribers, we wouldn't trigger any notifications to begin with. We wouldn't even store the historical prices for these products until we get a single subscriber, which means there would be no notifications sent.

Show more

0

Reply

T

tinder

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmch0dwyi032dad08yugjbgas)

Assume Pareto Distribution of 80% users subscribed to the top 20% of products. When one of these products has a price change, would price change worker be overwhelmed with all these emails to send?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmcl5sdle03arad0858nzmxqd)

Not really a concern - we can handle this with basic message queue patterns. When a price change happens, we publish one event to Kafka. Our notification service can then use multiple consumers in a consumer group to parallelize sending emails, with each consumer handling a subset of subscribers. We also don't have that many users (relatively).

Show more

0

Reply

![Nitin Sharma](https://lh3.googleusercontent.com/a/ACg8ocJOCGPyried71gZAPdk11e4pu2dx6FQ-YG2r-AFm6G7MhA_a2GG=s96-c)

Nitin Sharma

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmckumqi3005nad08scg1bhzn)

How about having composite primary key for combination of columns userId & productId in subscriptions table instead of having a column named id?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmcl5rjeq039vad08ea4vcbnm)

Maybe, just means that it would prevent users from having multiple subscriptions to the same product at different price points, which could actually a useful feature to have. Would just ask my interviewer. Good tradeoff discussion.

Show more

0

Reply

![John Van Schultz](https://lh3.googleusercontent.com/a/ACg8ocLbRXRLT82E76qsGGh-hjJqg-EYLRKPBf5jWkI34xke7wd4skWQ=s96-c)

John Van Schultz

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmco1h8cr00zead082oj8q95r)

I'm a little confused about the deep dive on the price aggregations. What exactly is being aggregated. Say for example if a product had a price change every day for the last year and then you the year aggregate. Is the price being returned the minimum value of the product during that time?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmco1vuoo019lad08rd0i0z2t)

You'd typically aggregate multiple data points: min price, max price, average price for each time period (day/week/month). When showing a year's worth of data, we'd return daily or weekly aggregates to keep the chart readable - not just a single minimum value for the year. The aggregation level (daily vs weekly) depends on the time range being viewed, similar to how stock charts work.

Show more

0

Reply

![Yang Pan](https://lh3.googleusercontent.com/a/ACg8ocJOrLzpVSfyONBl1-kijENPu2cLDDM8hWDFcdGieuZq_RaF9g=s96-c)

Yang Pan

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmcofv4e903z9ad08za9ckpn5)

Can I use time series database like InfluxDB (updated through CDC) to serve history price queries?

Show more

0

Reply

Q

QuixoticYellowHoverfly945

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmcqcft7r09cead08mwgkg2cz)

Are we storing the product urls whenervr a product is crawled for the first time. Because, thats the only way we coudl proritize re-crawling over products that we need.

Show more

0

Reply

![Priyankar Raj gupta](https://lh3.googleusercontent.com/a/ACg8ocLj4znexnJYoaFwdkTmM26gju9vXeJeZHeGkBO0YPITob8d3Rsl=s96-c)

Priyankar Raj gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmd1ssmyq0b85ad08nm678vyf)

does something like casandra intead of timescaleDB also works?

Show more

0

Reply

F

FederalOliveReptile537

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmd28vz8z003zad08fkawubu9)

How do we decide if we want to use time series database or OLAP database?

e.g. In AdClick aggregator we used OLAP db whereas here we used time series db.

Show more

2

Reply

M

ModerateMoccasinHawk364

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmd9tihdg039had08dbcau19o)

I was asked a similar question in my interview. One of the deep-dives / follow-ups was the following:

"Consider there is a popular product with several millions of people subscribed for price change notifications. How will you efficiently handle sending notifications to all these users, given that there is a strict SLA that price change notification must be sent to all subscribed users with X seconds (say, X is a small value like 5 seconds)."

Could you critique this approach (or provide what you think might be a better approach):

I believe the main bottleneck is the following:

We have a hot partition-key in the subscriptions table: assumes a GSI with partition-key == productId & sort-key == userId. This GSI is used to query by partition-key to return all subscribers for a product, to whom we need to send notifications.

To solve this problem we use a key-fanout technique: Add random numbers to the partition key. For e.g. product\_id\_A is split into product\_id\_A:1, product\_id\_A:2, product\_id\_A:3, product\_id\_A:4.

We can now use 4 different workers to query 4 different partitions to return all subscribers more efficiently, parallelized across 4 workers, and resulting in 4 times less load/query-time than in the case of a single partition.

Each of the 4 workers can parallely send messages to the notification queue/service.

Based on actual SLA and scale, we can use larger/smaller number than 4 mentioned above.

How does this sound? any better approaches?

Show more

1

Reply

![Anna Penka](https://lh3.googleusercontent.com/a/ACg8ocKzMp5mrjzgsWXclloH8fLFsE_XYTVUHeU0x78A9QUTBNGse9IISA=s96-c)

Anna Penka

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmdazfb9l01xzad09gjvwkkje)

such a good example thar covers a lot! question, why not to use a flink job to aggregate price changes and write the aggregated values to the db?

Show more

0

Reply

A

AdvisoryMaroonPanda905

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmdbupw7e08pxad09abmohboc)

hi Evan, Stefan,

I got probed on handing contentions woth accessing multiple vendor APIs and potential limitations with vendor APIs Oauth tokens using refresh tokens. I mentioned using a caching mechanism to store the tokens with set TTL where the server can reuse valid tokens and handle refreshing tokens as needed. The interviewer was pushing for storing the tokens in the DB which I challenged since the caching layer is quicker and can prevent a trip to the DB. what trade offs have I not considered that would make the DB solution more preferred? Are there resources thst you recommend that has explores challenged with handling integrations with external services? much thanks

Show more

0

Reply

D

DistinctAmberHarrier825

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmdfomhjf0248ad07ax9u66jp)

My favourite video so far. Amazing content. would it be ok to suggest some kind of expiry/ ttl of may be 1 hour on the notification kafka topic to ensure not sending stale notifications. Tia.

Show more

0

Reply

C

ChristianPlumTakin500

[• 26 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmdmey62201q2ad07xldl7jfb)

Honestly, I found the design a bit off. Here's what I came up with.

### Data Model

**Item**

\- item\_id
- amazon\_ref\_id
- amazon\_page\_url
- price
- version
- last\_checked\_at
- last\_price\_updated\_at

**PriceUpdates** (can be time series datastore like timescale db)

\- update\_id
- item\_id
- new\_price
- previous\_price
- created\_at
- index: (item\_id)

**Subscriptions**

\- subscription\_id
- item\_id
- user\_id
- type: START | STOP
- index: (item\_id), (item\_id, type), (user\_id)

**Notifications**

\- notification\_id
- item\_id
- last\_sent\_at
- last\_price

**ItemCache**

- tracks item — item\_id, current\_price, last\_updated\_at
- helps make meaningful updates, protects Items table

**ItemPriceQueue**

- all price updates from client added to this
- partitioned by item\_id so that updates are tracked in order

**ItemTrackerService**

- receives updates about items from clients, and adds to ItemPriceQueue
- the separate workers will process the queue

**PriceUpdatesQueue**

- acutal price updates added to this via CDC on PriceUpdatesTable

**NotificationsQueue**

- batch payloads of notifications with 1000 users per payload

### Flow

#### **Ingestion & Price Updates**

1. **Items**
    1. tracks details about items
    2. source of truth for current price
    3. last\_price\_updated\_at helps to ignore updates added to **PriceUpdates** table
2. **ItemCache**
    1. tracks item — item\_id, current\_price, last\_updated\_at
    2. helps make meaningful updates, protects Items table. ex: can ignore update if price is same
    3. on cache miss, populates cache — but to avoid db load, spend more on this to ensure we cover most of the items to avoid db load
3. **ItemPriceQueue**
    1. all price updates from client added to this
    2. partitioned by item\_id so that updates are tracked in order
4. **ItemTrackerService**
    1. receives updates about items from clients, and adds to **ItemPriceQueue**
    2. the separate workers will process the queue
    3. reads from ItemCache to see if update makes sense, ignores otherwise
5. **PriceUpdates** table
    1. When the service detects that price is indeed updated, writes a new row to PriceUpdates table and updates Item table automically; Automic updates are OKAY here since number of actual updates are fairly low, and is guarded by caches to ignore stale updates.
    2. The CDC will capture this update and goes into **PriceUpdatedQueue**

#### **Reading Price History**

1. For a given item **PriceUpdates** db will provide data aggregation capabilities for time range, and also pagination to view history
2. Can be partitioned by item\_id if needed, to handle heavy writes
3. Use replica for scaling reads, and also cache reads as needed

#### **Notifications**

1. A price update goes into **PriceUpdatedQueue**
2. A subscription is tracked in **Subscriptions** table
3. **priceUpdatedListenerWorkers** reads the queue, loads the users subscribed to this product
4. Since this could be large numbers, batches the payload in 1000s and adds to **NotificationQueue**
5. **Subscriptions** table have read replica, and shared by item\_id if needed
6. **notificationWorkers** reads from this queue and sends notifications to these users via email

Show more

0

Reply

![Alex Ilyenko](https://lh3.googleusercontent.com/a/ACg8ocItEb6gSi3Y9bGtUXZZJw0xFkFJkFq1AIVmq95nRJ4wQcw3-W5l9g=s96-c)

Alex Ilyenko

[• 23 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmdr4nqtv04zwad08k5ysj82p)

Hey Evan! Thanks for this content! Can you please share excalidraw diagrams (links) for private content as well?

Show more

0

Reply

S

SlightMoccasinGopher187

[• 16 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cme1a6ojt07g6ad08nt3vzo9z)

How does the design change if we want the user to be notified for a particular price drop exactly once and no duplicate emails?

Show more

1

Reply

J

junos.ny

[• 9 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmeb0642c04c1ad084szj2yic)

Thanks for the great analysis, as usual. I am a bit confused by the source of data collection. You mentioned that it is kind of like a crowdsource system, but uses a browser extension. The former sounds like a manual way, while the latter is an automatic way. Could you clarify/elaborate a little bit here? Thanks in advance.

Show more

0

Reply

E

EvolutionaryMaroonPinniped894

[• 5 days ago• edited 5 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmeh9mims01mxad08oid0vwwm)

To create a system design for the tracking stock prices on broker/exchange platform, can this above given solution be used? Stock brokers/exchanges have limit orders (where users specify the threshold and sell happens when that value is triggered). There also price history of the stock is displayed to customers. How much to modify this above solution to make it work? First concern, choosing strong consistency in the Non Functional requirements and much more.

Show more

0

Reply

![shashank reddy](https://lh3.googleusercontent.com/a/ACg8ocKcTZoqzG50xSB3Xx92aKjdcRQceLX4q68B4gWS_wlT21Azejoy=s96-c)

shashank reddy

[• 4 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmehvl2fk013uad087ngrqobf)

How do we handle query performance for subscriptions? Let's say we have 1M DAU and a majority of them subscribed to a new iPhone. And the price changes every 3-4 hours in a day. This maybe a case during cyber monday where the price might change based on the inventory size.

In this case, do we query the Subscription table for all the 1M users subscribed to this product? (If this is a m2m relationship)?

Or, even if we have a dict (product->userID), do we loop all the 1M subscribed users every time the price changes?

Just wondering if there is any optimal way to handle this. Thanks!

Show more

0

Reply

I

InvisiblePinkChipmunk238

[• 2 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/camelcamelcamel#comment-cmelpotxx00qmad081qlot755)

I was recently asked a similar question where

1. User submits a web url of a product from any Ecommerce platform (Not limited to Amazon)
2. The tracking platform would have website and mobile app as user interface

All the other requirements are same. I suggested the web crawler approach but interviewer was not very satisfied with it. Could someone suggest an optimal approach in that case?

Show more

0

Reply
