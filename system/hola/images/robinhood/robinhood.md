# Design Robinhood

Real-time Updates

[![Joseph Antonakakis](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fjoe-headshot.acf0af6f.jpeg&w=96&q=75&dpl=e097d75362416d314ca97da7e72db8953ccb9c4d)

Joseph Antonakakis

SWE at Notion, ex-Meta Staff SWE

](https://www.linkedin.com/in/joseph-antonakakis-323b92a8/)

hard

Published Jul 14, 2024

* * *

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

## Understanding the Problem

**📈 What is [Robinhood](https://robinhood.com/)?** Robinhood is a commission-free trading platform for stocks, ETFs, options, and cryptocurrencies. It features real-time market data and basic order management. Robinhood isn't an exchange in its own right, but rather a stock broker; it routes trades through market makers ("exchanges") and is compensated by those exchanges via [payment for order flow](https://en.wikipedia.org/wiki/Payment_for_order_flow).

### Background: Financial Markets

There's some basic financial terms to understand before jumping into this design:

-   **Symbol**: An abbreviation used to uniquely identify a stock (e.g. META, AAPL). Also known as a "ticker".
    
-   **Order**: An order to buy or sell a stock. Can be a _market order_ or a _limit order_.
    
-   **Market Order**: An order to trigger immediate purchase or sale of a stock at the current market price. Has no price target and just specifies a number of shares.
    
-   **Limit Order**: An order to purchase or sell a stock at a specified price. Specifies a number of shares and a target price, and can sit on an exchange waiting to be filled or cancelled by the original creator of the order.
    

Outside of the above financial details, it's worth understanding the responsibilities of Robinhood as a business / system. **Robinhood is a brokerage and interfaces with external entities that actually manage order filling / cancellation.** This means that we're building a brokerage system that facilitates customer orders and provides a customer stock data. _We are not building an exchange._

For the purposes of this problem, we can assume Robinhood is interfacing with an "exchange" that has the following capabilities:

-   **Order Processing**: Synchronously places orders and cancels orders via request/response API.
    
-   **Trade Feed**: Offers subscribing to a trade feed for symbols. "Pushes" data to the client every time a trade occurs, including the symbol, price per share, number of shares, and the orderId.
    

For this interview, the interviewer is offering up an external API (the exchange) to aid in building the system. As a candidate, it's in your best interest to briefly clarify the exchange interface (APIs, both synchronous and asynchronous) so you have an idea of the tools at your disposal. Typically, the assumptions you make about the interface have broad consequences in your design, so it's a good idea to align with the interviewer on the details.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements)

**Core Requirements**

1.  Users can see live prices of stocks.
    
2.  Users can manage orders for stocks (market / limit orders, create / cancel orders).
    

**Below the line (out of scope)**

-   Users can trade outside of market hours.
    
-   Users can trade ETFs, options, crypto.
    
-   Users can see the [order book](https://www.investopedia.com/terms/o/order-book.asp) in real time.
    

This question focuses on stock viewing and ordering. It excludes advanced trading behaviors and doesn't primarily involve viewing historical stock or portfolio data. If you're unsure what features to focus on for a feature-rich app like Robinhood or similar, have some brief back and forth with the interviewer to figure out what part of the system they care the most about.

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements)

**Core Requirements**

1.  The system prefers high consistency for order management; it's _essential_ for users to see up-to-date order information when making trades.
    
2.  The system should scale to a high number of trades per day (20M daily active users, 5 trades per day on average, 1000s of symbols).
    
3.  The system should have low latency when reflecting symbol price updates and when placing orders (under 200ms).
    
4.  The system should minimize the number of active clients connecting to an external exchange API. Exchange data feeds / client connections are typically expensive.
    

**Below the line (out of scope)**

-   The system connects to multiple exchanges for stock trading.
    
-   The system manages trading fees / calculations (we can assume fees are not in scope).
    
-   The system enforces daily limits on trading behavior.
    
-   The system protects against bot usage.
    

Here's how it might look on a whiteboard:

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/784a0cb26bf28be6b87b3371314a4355)

For this question, given the small number of functional requirements, the non-functional requirements are even more important to pin down. They characterize the complexity of these deceptively simple live price / order placement capabilities. Enumerating these challenges is important, as it will deeply affect your design.

## The Set Up

### Planning the Approach

Before you move on to designing the system, it's important to start by taking a moment to plan your strategy. Generally, we recommend building your design up sequentially, going one by one through your functional requirements. This will help you stay focused and ensure you don't get lost in the weeds as you go. Once you've satisfied the functional requirements, you'll rely on your non-functional requirements to guide you through the deep dives.

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

Let's go through each high level entity. I like to do this upfront before diving into other aspects of the system so we have a list of concepts to refer back to when talking about the details of the system. At this stage, it isn't necessary to enumerate every column or detail. It's all about laying the foundation.

For Robinhood, the primary entities are pretty straightforward:

1.  **User**: A user of the system.
    
2.  **Symbol**: A stock being traded.
    
3.  **Order**: An order for a buy or sell, created by a user.
    

In the actual interview, this can be as simple as a short list like this. Just make sure you talk through the entities with your interviewer to ensure you are on the same page.

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/093e1f2ff2ba56e32bdd4909c37df8a8)

### [The API](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#api-design-5-minutes)

The API is the primary interface that users will interact with. It's important to define the API early on, as it will guide your high-level design. We just need to define an endpoint for each of our functional requirements.

Let's start with an endpoint to get a symbol, which will include details and price data. We might have an endpoint like this:

`GET /symbol/:name Response: Symbol`

To create an order, an endpoint might look like this:

`POST /order Request: {   position: "buy",   symbol: "META",   priceInCents: 52210,   numShares: 10 } Response: Order`

Note we're using priceInCents instead of price to avoid floating point precision issues. Especially for financial application it's better to use integers to avoid errors and [financial scams](https://screenrant.com/justice-league-incarnate-superman-iii-scheme-easter-egg).

To cancel an order, the endpoint could be as simple as:

`DELETE /order/:id Response: {   ok: true }`

Finally, to list orders for a user, the request could be:

`GET /orders Response: Order[] (paginated)`

With each of these requests, the user information will be passed in the headers (either via session token or JWT). This is a common pattern for APIs and is a good way to ensure that the user is authenticated and authorized to perform the action while preserving security. You should avoid passing user information in the request body, as this can be easily manipulated by the client.

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes)

### 1) Users can see live prices of stocks

The first requirement of Robinhood is allowing users to see the live price of stocks. This might be one stock or many stocks, depending on what the user is viewing in the UI. To keep our design extensible, let's assume a user can see many live stock prices at once. To support this design, let's analyze a few options.

### 

Bad Solution: Polling Exchange Directly

##### Approach

This solution involves polling the exchange directly for price data per symbol. The client would poll every few seconds (per symbol), and update the price shown to the user based on the response.

##### Challenges

This is a simple approach that will not scale and will not minimize exchange client connections / calls. There's a few fundamental problems with this design:

-   **Redundant Exchange Calls**: This approach is a very inefficient way to get price information in terms of exchange call volume. It involves polling, which happens indiscriminately, even if the price has not changed. Additionally, it involves many clients requesting the same information from the exchange, which is wasteful. If 5000 clients are requesting a price at the same time, the price isn't different per client, yet we're expending 5000 calls (repeatedly) to disperse this information.
    
-   **Slow Updates**: If we're pursuing a polling solution, we'll see slower updates than we'd like to pricing information of symbols. In the non-functional requirements, we indicated we wanted a reasonably short SLA to update clients about symbol prices (under 200ms), and the only way we'd guarantee that with this solution is if we poll data every 200ms, which is unreasonable.
    

For a design like this, we can rule out polling the exchange directly as a viable option.

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/1cedb71de8cbc28f119aae1b90df0e6c)

### 

Good Solution: Polling Internal Cache

##### Approach

This solution still involves polling for price information, but we're polling a symbol service that performs a key-value look-up on an internal cache that is kept up-to-date by a symbol price processor that is listening to prices on the exchange.

This approach prevents excess connections to the exchange by "proxying" it with a service that listens and records the most essential detail: the symbol price. This price is then made available to clients of Robinhood via polling.

##### Challenges

This approach is certainly an improvement, but is still subject to some issues. Clients still indiscriminately poll, even if the price has not changed, leading to some wasted HTTP traffic. Additionally, this approach is a slow way to get price updates; the polling interval dictates the worst-case SLA for a price update propagating to the client. Can we do better?

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/d018fdc8c54abd7e37e2481bb94486b6)

### 

Great Solution: Server Sent Event (SSE) Price Updates

##### Approach

A great approach here involves Server Sent Events (SSE). SSE is a persistent connection (similar to websockets), but it is unidirectional and goes over HTTP instead of a separate protocol. For this example, the client isn't sending us data, so SSE is a superior choice to websockets. For more details on reasoning through the websocket vs. SSE trade-off analysis, feel free to reference our [FB Live Comments write-up](https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-live-comments#2-viewers-can-see-all-comments-in-near-real-time-as-they-are-posted).

###### Pattern: Real-time Updates

Real-time stock price updates are a perfect example of the real-time updates pattern. Here we use Server Sent Events (SSE) to establish persistent connections that allow our servers to push live price changes to clients instantly. This approach handles the networking fundamentals of real-time communication while ensuring users see current market prices without the inefficiency of constant polling.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates)

This approach involves re-working our API to instead have a /subscribe POST request, with a body containing a list of symbols we want to subscribe to. Our backend can then setup a SSE connection between the client and a symbol service, and send the client an initial list of symbol prices. This initial list of prices is serviced by a cache that is kept up-to-date by a processor that is listening to the exchange. Additionally, that processor is sending our symbol service those prices so that the symbol service can send that data to clients that have subscribed to price updates for different symbols.

##### Challenges

Adding SSE introduces challenges, mainly involving maintaining a connection between client and server. The load balancer will need to be configured to support "sticky sessions" so that a user and server can maintain a connection to promote data transfer. Additionally, a persistent connection means we have to consider how disconnects / reconnects work. Finally, we need to consider how we route user connections and symbol data. Several of these details are covered later in our deep dive sections, so stay tuned.

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/5f5ae9a71a982d2c286518381a09723c)

### 2) Users can manage orders for stocks

The second requirement of Robinhood is allowing users to manage orders for stocks. Users should be able to create orders (limit or market), cancel outstanding orders, and list user orders.

Let's consider our options for creating and cancelling orders via the exchange.

### 

Bad Solution: Send Orders Directly to Exchange

##### Approach

This solution involves directly interacting with the exchange to submit orders. Any orders issued by the client are directly submitted to the exchange.

##### Challenges

While this is a "mainline" way to submit orders that cuts out any incurred latency from a backend proxying the exchange, it can lead to large number of exchange clients and concurrent requests, which will be very expensive. Additionally, there's no clear path for the client to check status of an order, outside of polling the exchange or directly listening to trade feeds, both of which aren't viable solutions. Finally, the client is exclusively responsible for tracking orders. This isn't great, as we can't consider the client's storage to be reliable; the user might uninstall the app or the phone hardware might fail.

Let's consider other solutions.

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/b6edc4f13d7b1448cc0fe9830437363b)

### 

Good Solution: Send Orders to Dispatch Service via Queue

##### Approach

This solution involves sending orders to an order service, which enqueues them for an order dispatch service. This soluton avoids excess exchange clients by proxying the exchange order execution with the order dispatch service. This service can bear the responsibility of efficient exchange communication. This service sends orders to this dispatch service via a queue. The queue prevents the dispatch service from being overloaded, and the queue volume can serve as a metric that the dispatch service could elastically scale off of.

##### Challenges

This approach is on the right track as it proxies the exchange and allows a path for elastic scalability in the face of increased order load (e.g. bursts in trading traffic). However, this approach breaks down when we consider our tight order SLA (under 200ms as a goal).

If we consider moments of high queue load, perhaps during high trading traffic and before the order dispatch service has scaled up, orders might take a while to be dispatched to the exchange, which could violate our goal SLA. In particular sensitive moments of trading, this can be really bad for users. Imagine a user who wants to quickly order stocks or quickly cancel an outstanding order. It would be unacceptable for them to be left waiting for our dispatcher to eventually handle their order, or for our service to start more machines up to scale up given increased queue load.

What other options do we have?

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/f0d0bd82550331e7720b5ad7d3c7e90d)

### 

Great Solution: Order Gateway

##### Approach

This approach involves sending our orders directly from the order service to an order dispatch gateway. The gateway would enable external internet communication with the exchange via the order service. The gateway would make the requests to the exchange appear as if they were originating from a small set of IPs.

For this approach, our gateway would be an [AWS NAT gateway](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html) which would allow for our order services to make requests to the exchange but then appear under a single or small number of IPs ([elastic IPs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html), in AWS terms).

Given that the gateway is managing outbound requests, this approach relies on the order service that is accepting requests from the client to play a role in actually managing orders. This service will run business logic to manage orders and will scale up / down as necessary given order volume. Given this search is being routed to from clients, we might make the auto-scaling criterion for this service quite sensitive (e.g. auto-scale when average 50% CPU usage is hit across the fleet) or we might over-provision this service to absorb trading spikes.

##### Challenges

This approach requires our order service to do more to manage orders, meaning it will need to be written in a way that is both efficient for client interaction and efficient for exchange interaction (e.g. potentially batching orders together).

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/2f078a2cf5e4ffe7b8e12d02654d2fa2)

Now that we know how we'll dispatch orders to the exchange, we also must consider how we'll store orders on our side for the purposes of exposing them to the user. The user should be able to GET /orders to see all their outstanding orders, so how might we keep data on our side to service this request?

In order to track orders, we can stand up an order database that is updated when orders are created or cancelled. The order database will be a relational database to promote consistency via ACID guarantees, and will be partitioned on userId (the ID of the user submitting order changes). This will make querying orders of a user fast, as the query will go to a single node.

The order itself will contain information about the order submitted / cancelled. It will also contain data about the state of the order (pending prior to being submitted to the exchange, submitted when it's submitted to the exchange, etc.). Finally, the order will contain an externalOrderId field, which will be populated by the ID that the exchange responds with when the order service submits the order synchronously.

Additionally, to keep these orders up-to-date, we'll need some sort of trade processor tailing the exchange's trades to see if orders maintained on our side get updated.

Of note, this trade processor would be a fleet of machines that are connected to the exchange in some way to receive price updates. The communication interface here doesn't matter too much. For most systems like this, a client of the exchange like Robinhood would setup a webhook endpoint and register that with the exchange, and the exchange would call the webhook whenever it had updates. In the case of webhooks, the trade processor would have a load balancer and a fleet of machines serving that webhook endpoint. For the sake of simplicity, we visualize the trade processor as a single square on our flowchart.

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/007f3c6006c64264d5d49df65ba2dc75)

You might be wondering how we concretely reflect updates based on the exchange's trade feed. Stay tuned, as we'll dive into this in one of our deep dive sections.

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes)

### 1) How can the system scale up live price updates?

It's worth considering how the system will scale live price updates. If many users are subscribing to price updates from several stocks, the system will need to ensure that price updates are successfully propagated to the user via SSE.

The main problem we need to solve is: **how do we route symbol price updates to the symbol service servers connected to users who care about those symbol updates?**

To enable this functionality in a scalable way, we can leverage [Redis pub/sub](https://redis.io/docs/latest/develop/interact/pubsub/) to publish / subscribe to price updates. Users can subscribe to price updates via the symbol service and the symbol service can ensure it is subscribed to symbol price updates via Redis for all the symbols the users care about. The new system diagram might look like this now:

![](https://d248djf5mc6iku.cloudfront.net/excalidraw/251b2fe402024e62942d14d398f6323f)

Want to learn more about Redis? Check out our [Redis deep dive](https://www.hellointerview.com/learn/system-design/deep-dives/redis) for in-depth discussion of the different ways Redis can be used practically in system design interviews.

Let's walk through the full workflow for price updates:

1.  A user subscribes via a symbol service server. The server tracks Symbol -> Set<userId> mapping, so it adds an entry for each symbol the user is subscribed to.
    
2.  The symbol service server is managing subscriptions to Redis pub/sub for channels corresponding to each symbol. It ensures that it has an active subscription for each symbol the user is subscribed to. If it lacks a subscription, it subscribes to the channel for the symbol.
    
3.  When a symbol changes price, that price update is processed by the symbol price processor, which publishes that price update to Redis. Each symbol service server that is subscribed to that symbol's price updates get the price update via subscription. They then fan-out and send that price update to all users who care about that symbol's price updates.
    
4.  If a user unsubscribes from symbols or disconnects (detected via some heartbeat mechanism), the symbol service server will go through each symbol they were subscribed to and removes them from the Set<userId>. If the symbol service server no longer has any users subscribed to a symbol, it can unsubscribe from the Redis channel for that symbol.
    

The above would scale as it would enable our users to evenly distribute load across the symbol service. Additionally, it would be self-regulating in managing what price updates are being propagated to symbol service servers.

### 2) How does the system track order updates?

When digging into the order dispatch flow, it's worth clarifying how we'll ensure our order DB is updated as orders are updated on the exchange.

Firstly, it's not clear from our current design how the trade processor might reflect updates in the orders DB, based off just a trade. There's no efficient way for the trade processor to look-up a trade via the externalOrderId to update a row in the Order table, given that the table is partitioned by userId . This necessitates a separate key-value data store mapping externalOrderId to the (orderId, userId) that is corresponds to. For this key-value store, we could use something like [RocksDB](https://rocksdb.org/). This key-value store would be populated by the order service after an order is submitted to the exchange synchronously.

This new key-value store enables the trade processor to quickly determine whether the trade involved an order from our system, and subsequently look up the order (go to shard via userId -> look up Order via orderId) to update the Order's details. The new system diagram might look like this:

Order update deep dive

### 3) How does the system manage order consistency?

Order consistency is extremely important and worth deep-diving into. Order consistency is defined as orders being stored with consistency on our side and also consistently managed on the exchange side. As we'll get into, fault-tolerance is important for maintaining order consistency.

Before we dig in, we firstly can revisit our order storage mechansim. Our order database is going to be a horizontally partitioned relational database (e.g. Postgres). All order updates will happen on a single node (the partition that the order exists on). All order reads will also occur on a single node, as we'll be partitioning by userId.

When an order is created, the system goes through the following workflow:

1.  Store an order in the order database with pending as the status. _It's important that this is stored first because then we have a record of the client could. If we didn't store this first, then the client could create an order on the exchange, the system could fail, and then our system has no way of knowing there's an outstanding order._
    
2.  Submit the order to the exchange. Get back externalOrderId immediately (the order submission is synchronous).
    
3.  Write externalOrderId to our key-value database and update the order in the order database with status as submitted and externalOrderId as the ID received from the DB.
    
4.  Respond to the client that the order was successful.
    

The above workflow seems very reasonable, but it might break down if failures occur at different parts of the process. Let's consider several failures, and ways we can mitigate these failures if they pose a risk to our consistency:

-   **Failure storing order**: If there's a failure storing the order, we can respond with a failure to the client and stop the workflow.
    
-   **Failure submitting order to exchange**: If there's a failure submitting the order to the exchange, we can mark the order as failed and respond to the client.
    
-   **Failure processing order after exchange submission**: If there's an error updating the database after an exchange submission, we might consider having a "clean-up" job that deals with outstanding, pending orders in the database. Most exchange APIs offer a clientOrderId metadata field when submitting an order (see [E\*TRADE example](https://apisb.etrade.com/docs/api/order/api-order-v1.html#/definitions/PreviewOrderRequest)) so the "clean-up" job can asynchronously query the exchange to see if the order went through via this clientOrderId identifier, and do one of two things: 1) record the externalOrderId if the order did go through, or 2) mark the order as failed if the order didn't go through.
    

Now that we've considered the order flow, let's consider the cancel flow. When an order is cancelled, the system goes through the following workflow:

1.  Update order status to pending\_cancel. _We do this first to enable resolving failed cancels later._
    
2.  Submit the order cancellation to the exchange.
    
3.  Record the order cancellation in the database.
    
4.  Respond to the client that the cancellation was successful.
    

Let's walk through different failures to ensure we are safe from inconsistency:

-   **Failure updating status to pending\_cancel**: If there's a failure updating the order status upfront, we respond with a failure to the client and stop the workflow.
    
-   **Failure cancelling order**: If there's a failure cancelling the order via the exchange, we can respond with a failure to the client and rely on a "clean-up" process to scan pending\_cancel orders (ensure they are cancelled).
    
-   **Failure storing cancelled status in DB**: If there's a failure updating the order status in the DB, we can rely on a "clean-up" process to pending\_cancel orders (ensure they are cancelled, or no-op and just update status to cancelled if they have already been cancelled).
    

Based on the above analysis, we have 1) a clear understanding of the order create and cancel workflows, and 2) identified the need for a "clean-up" background process to ensure our order state becomes consistent in the face of failures at different points in our order / cancel workflows. Below is the updated system diagram reflecting our changes:

Order consistency deep dive

### Some additional deep dives you might consider

Robinhood, like most fintech systems, is a complex and interesting application, and it's hard to cover every possible consideration in this guide. Here are a few additional deep dives you might consider:

1.  **Excess price updates**: If a set of stocks has a lot of trades or price updates, how might the system handle the load and avoid overwhelming the client? This might be interesting to cover.
    
2.  **Limiting exchange correspondence**: While we certainly covered ways we'd "proxy" the exchange and avoid excess concurrent connections / clients, it might be worthwhile to dive into other ways the system might limit exchange correspondence, while still scaling and serving the userbase (e.g. perhaps considering batching orders into single requests).
    
3.  **Live order updates**: It might be worthwile to dive into how the system would propagate order updates to the user in real time (e.g. if the user is looking at orders in the app, they see their orders get filled in real time if they're waiting on the exchange).
    
4.  **Historical price / portfolio value data**: In this design, we didn't focus on historical price / portfolio data at all, but some interviewers might consider this a requirement. It's worthwhile to ponder how a system would enable showing historical price data (over different time windows) and historical user portfolio value.
    

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

Ok, that was a lot. You may be thinking, "how much of that is actually required from me in an interview?" Let’s break it down.

### Mid-level

**Breadth vs. Depth:** A mid-level candidate will be mostly focused on breadth (80% vs 20%). You should be able to craft a high-level design that meets the functional requirements you've defined, but many of the components will be abstractions with which you only have surface-level familiarity.

**Probing the Basics:** Your interviewer will spend some time probing the basics to confirm that you know what each component in your system does. For example, if you add an API Gateway, expect that they may ask you what it does and how it works (at a high level). In short, the interviewer is not taking anything for granted with respect to your knowledge.

**Mixture of Driving and Taking the Backseat:** You should drive the early stages of the interview in particular, but the interviewer doesn’t expect that you are able to proactively recognize problems in your design with high precision. Because of this, it’s reasonable that they will take over and drive the later stages of the interview while probing your design.

**The Bar for Robinhood:** For this question, an E4 candidate will have clearly defined the API endpoints and data model, landed on a high-level design that is functional for price updates and ordering. I don't expect candidates to know in-depth information about specific technologies, but the candidate should converge on ideas involving efficient price update propagation and consistent order management. I also expect the candidate to know effective ways of proxying the exchange to avoid excess connections / clients.

### Senior

**Depth of Expertise**: As a senior candidate, expectations shift towards more in-depth knowledge — about 60% breadth and 40% depth. This means you should be able to go into technical details in areas where you have hands-on experience. It's crucial that you demonstrate a deep understanding of key concepts and technologies relevant to the task at hand.

**Advanced System Design**: You should be familiar with advanced system design principles (different technologies, their use-cases, how they fit together). Your ability to navigate these advanced topics with confidence and clarity is key.

**Articulating Architectural Decisions**: You should be able to clearly articulate the pros and cons of different architectural choices, especially how they impact scalability, performance, and maintainability. You justify your decisions and explain the trade-offs involved in your design choices.

**Problem-Solving and Proactivity**: You should demonstrate strong problem-solving skills and a proactive approach. This includes anticipating potential challenges in your designs and suggesting improvements. You need to be adept at identifying and addressing bottlenecks, optimizing performance, and ensuring system reliability.

**The Bar for Robinhood:** For this question, E5 candidates are expected to quickly go through the initial high-level design so that they can spend time discussing, in detail, how to handle real-time price propagation and consistent orders. I expect the candidate to design a reasonable, scalable solution for live prices, and I expect the candidate to design a good order workflow with some mindfulness of consistency / fault tolerance.

### Staff+

**Emphasis on Depth**: As a staff+ candidate, the expectation is a deep dive into the nuances of system design — I'm looking for about 40% breadth and 60% depth in your understanding. This level is all about demonstrating that, while you may not have solved this particular problem before, you have solved enough problems in the real world to be able to confidently design a solution backed by your experience.

You should know which technologies to use, not just in theory but in practice, and be able to draw from your past experiences to explain how they’d be applied to solve specific problems effectively. The interviewer knows you know the small stuff (REST API, data normalization, etc.) so you can breeze through that at a high level so you have time to get into what is interesting.

**High Degree of Proactivity**: At this level, an exceptional degree of proactivity is expected. You should be able to identify and solve issues independently, demonstrating a strong ability to recognize and address the core challenges in system design. This involves not just responding to problems as they arise but anticipating them and implementing preemptive solutions. Your interviewer should intervene only to focus, not to steer.

**Practical Application of Technology**: You should be well-versed in the practical application of various technologies. Your experience should guide the conversation, showing a clear understanding of how different tools and systems can be configured in real-world scenarios to meet specific requirements.

**Complex Problem-Solving and Decision-Making**: Your problem-solving skills should be top-notch. This means not only being able to tackle complex technical challenges but also making informed decisions that consider various factors such as scalability, performance, reliability, and maintenance.

**Advanced System Design and Scalability**: Your approach to system design should be advanced, focusing on scalability and reliability, especially under high load conditions. This includes a thorough understanding of distributed systems, load balancing, caching strategies, and other advanced concepts necessary for building robust, scalable systems.

**The Bar for Robinhood:** For a staff-level candidate, expectations are high regarding the depth and quality of solutions, especially for the complex scenarios discussed earlier. Exceptional candidates delve deeply into each of the topics mentioned above and may even steer the conversation in a different direction, focusing extensively on a topic they find particularly interesting or relevant. They are also expected to possess a solid understanding of the trade-offs between various solutions and to be able to articulate them clearly, treating the interviewer as a peer.

Mark as read

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

G

GiganticApricotSwordfish902

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm44aaxg4009i5zn7qdn3mul7)

regarding the Good solution vs. Great solution with the Order Dispatch service and Order Gateway, I don't understand what the difference is? To me it seems the only difference is having a queue in between them.

Show more

28

Reply

F

FastJadeGrouse442

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm44vtofl00yd3bhc7l6qern9)

I think the NAT gateway proxies the Order Service’s network requests to ensure all outgoing requests share the same IP address.

Show more

0

Reply

![Magic Potato](https://lh3.googleusercontent.com/a/ACg8ocJeHx-rpaU4T1D7H8hToMCixBmGc40S0KnCZEzJm9h5SjaysClY=s96-c)

Magic Potato

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm48uzo7u000whh2e6iyotrmp)

Does sending requests from small set of IPs ensure we can handle high scale ?

Show more

2

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm48v26ap000zhh2eoifc26uh)

The intent is less to ensure scale and more to ensure security. Financial systems frequently manage access via allowlists/whitelists (e.g. our bank requires a set of IPs to submit ACH transfers for our mock interview coaches).

Show more

11

Reply

E

ExcessBlackAardvark647

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4gpdfio00rw8jlrth1fihnj)

so the order gateway doesn't actually resolve order queue's problems and it also loses the benefits brought by the queue. the only benefit the "great solution" is hiding clients' real ip behind a "fake" one, right? if so, I don;t quite get why this is a great solution. if the requests have to be sent from an allowlisted IPs, then the "good solution" should not be called "good" as it is not working. but if there is no such requirement from the bank, then the so-called "great solution" doesn't look like "great" to me. But PLEASE let me know if I understand the flow incorrectly and explain it more. Thanks!

Show more

22

Reply

![Anirudh Kaki](https://lh3.googleusercontent.com/a/ACg8ocJnh2a8FYhXHXBBH9NOrnr_OXXIPpM3ux4a_ZuCuuIQo8dQcnPF=s96-c)

Anirudh Kaki

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4hiuhsq01ex5gbotivg0vn4)

The intent is to avoid queues between OrderService and OrderDispatcher to prevent delays. Direct communication ensures immediate order processing, allowing quick responses to market changes and instant order modifications.

Show more

5

Reply

E

ExcessBlackAardvark647

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4hlh2qp01hd5gbonoifj704)

Thanks for the reply. Then how does the OrderDispatcher Gateway prevents the dispatch service from being overloaded, and what should be the metric to indicate the dispatch service to elastically scale off(these are the benefits brought by the queue)?

Show more

4

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4u9bn2d00ube4jkenkuo5hy)

true that, I believe queues are great when you want to avoid any load bringing down your infrastructure and also not wanting to lose any incoming requests. in this system design, as long as u scale ur own structure as a conduit of trade placement, the actual load is someone else's problem (i.e. exchange)

Show more

1

Reply

![nikhil singh](https://lh3.googleusercontent.com/a/ACg8ocIJGDLcL0BK5EVRQ44eYjcVW1xk3cEaubuneolVd3gPGNJXYw=s96-c)

nikhil singh

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5c9nymj0052kz0hpv23wp95)

"Then how does the OrderDispatcher Gateway prevents the dispatch service from being overloaded" -> the post mentions that "we might make the auto-scaling criterion for this service quite sensitive (e.g. auto-scale when average 50% CPU usage is hit across the fleet) or we might over-provision this service to absorb trading spikes"

Show more

6

Reply

![nikhil singh](https://lh3.googleusercontent.com/a/ACg8ocIJGDLcL0BK5EVRQ44eYjcVW1xk3cEaubuneolVd3gPGNJXYw=s96-c)

nikhil singh

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5c9ry8y003zm9c3eo9fjhsi)

We use queue instead of over provisioning of services because of cost factor as over provisioning of services is expensive but if the latency requirement is extremely low, then, over provisioning is a good idea rather than using queue

Show more

2

Reply

J

jokerchendi

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5i16fiq01ka5sp8itrw2rha)

Exactly!

A key architecture decision here is: if low latency is top priority, there's no point using a queue.

Think about it. If we add a queue in between, to ensure low latency, we must always keep the number of unprocessed messages in the queue low (close to zero). So, why use a queue in the first place?

Show more

4

Reply

![Katie McCorkell](https://lh3.googleusercontent.com/a/ACg8ocJhhgdomrgC_-4WZgadOwN2X8drRXio1OaPECBGUAB1SRYpPV4UHw=s96-c)

Katie McCorkell

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8epnv7g01hi40wj215s47hu)

thank you guys for this discussion, I had all the same questions. very useful. I think the writing in that part of this blog is unclear.

Show more

12

Reply

M

MathematicalLimePuffin340

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm90xvclq0053ad08s3sta7ab)

Not using a queue is trading off Reliability for lower latency. What if Exchange is down and orders have to be processed? What if Order Dispatch Gateway goes down? From a usability standpoint, the enqueue + dequeue adds 100ms which seems a reasonable wait time if it adds reliability for stock transaction.

I would argue that we choose Reliabilty over Latency for Order Execution and prioritize latency for receiving stock price update information

Show more

6

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma4v7mf700m6ad0781a5f1av)

I think this a good discussion point in the interviews. However - if the exchange itself goes down, I think there are much larger problems at hand :) This will cause potentially millions in losses for the exchange. It also might not make sense to queue up these while the exchange is down since that will cause a huge build-up of orders (potentially causing an even longer latency).

The argument here for the queue may be to allow the user to fire and order and forget, which could be a valid point (since a lot of these brokerages allow you to schedule orders).

Show more

1

![WH L](https://lh3.googleusercontent.com/a/ACg8ocKcEkRZ406Sjv2WAg8Qb5SxAtpZLsd-xt5-gGvitF0qc0xBqQ=s96-c)

WH L

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmbd2jhuf00xwad08yfims6oi)

you didn't mention any security issue for your elastic ip solution.

Show more

0

Reply

![Sergey Zavitaev](https://lh3.googleusercontent.com/a/ACg8ocK5G0GxPnJ5_3Bdqvh2t1XAusqfAqejvNZ-qDuGFRJUvnBvPHv_=s96-c)

Sergey Zavitaev

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm45mye2l00jm4dcr44e8ufjm)

Greate article ! The only question i have: "The symbol service server is managing subscriptions to Redis pub/sub for channels corresponding to each symbol"

why we need to have separate pub/sub channels per symbol ? Why not create single pub/sub channel and send updates through this single channel with symbolId, price and other things that we need ?

I think it would be more flexible and extendable and we don't need to manage new channels in case new symbols addition and so on

Show more

2

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4f73we104ma1i3r2nfhczxt)

that is because with a single channel, there will be too much noise that most servers don’t care about. say there are 3000 symbols and each server cares about 500 of them, then we avoid triggering the server for 2500 symbol updates

Show more

4

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma4v9bsh00mcad07bco2lqb6)

You want to keep the traffic that each server receives to a minimum - if you are sending all ticker data over a single channel, to every single server, a lot of servers are going to receive data they dont actually need.

Show more

0

Reply

Y

YammeringTealBovid101

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4azwbds011gpgt8bg5v9xsj)

Wondering why to add a RocksDB instead of add index on externalOrderId in the Order DB? With 2 DBs, extra complexity is added to handle data consistency.

Or maybe this is a typo? The article mentions "externalTradeId", wondering whether it's the same thing with External Order ID or not? If we think about order partially filled case. it makes more sense to me that, 1 order can have 1 external order ID, but multiple external trade IDs. So we need a Trade table to store the trade IDs to order Id maping. But still, 1 DB with multiple tables seems enough here. Especially if we want to maintain the orderID and tradeID relation.

Show more

16

Reply

P

pssharma1699

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4e54qmc02g1amnh4z7rr7k9)

I think externalTradeId and externalOrderId are the same. The main reason why we are using rocksDB is we cannot have index on externalTradeId as the data would be partitioned based on userId and we might have to go through all shards before we can update db. Not sure if index work across shards.

Show more

1

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4f759ly04md1i3rru52lf29)

why rocksdb and why not redis?

Show more

4

Reply

P

pssharma1699

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4fauqd104pe1i3rypoy44zh)

Dont think there is a hard rule to use rocks DB, some properties should be

1.  persistent
2.  fast Redis with persistent config should work imo.

Show more

2

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4fsahzu03y1qfete6hsz7hx)

I don't know if we need persistent. it only needs to be saved until the trade is settled

Show more

1

Reply

![Hello Interview](https://lh3.googleusercontent.com/a/ACg8ocL37V560KNRasR0i4GkleYZIgsyB7aBt4jhXrr_x3cYay5VXQ=s96-c)

Hello Interview

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm9u8cqx200qrad075p2xlx8m)

yes i also tthink so , since its embedded no extra network call , but how is the initial mapping getting created for an orderId -> externalOrderId , we wont have the external orderId unless we send it to exchange or rather exchange responds back with it

Show more

0

Reply

F

FastIvoryMink623

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm65h7bdv00ndwnto1l8n13qt)

We have similar situation at work and but the external system will accept our custom key and response to us with that key, our key is "ourteamID\_orderid\_userid" so we just need to parse the key to get all these information.

Show more

3

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8gxpkyg018it1if95yddpzo)

Even the database is partitioned based on the userId, we can have a GSI on the externalOrderId

Show more

3

Reply

B

BiologicalMoccasinTahr305

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8vyezqy030zad074y8wy7ye)

GSIs are in DynamoDB not postgres. You'd either create a new table within postgres and shard that by externalOrderId OR you'd use a separate db like the blog post does

Show more

4

Reply

C

ControlledTealParakeet247

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmbv7qlrs00jd07ad6hwaj8gk)

THis is the real problem that not many databases support global secondary indexes. the index that you create from external order id to order-id when partitioned by user-id will not have any global presence. which means you would have to query all the shards. Hence use an external store to simulate the GSI.

Show more

0

Reply

![hardcorg](https://lh3.googleusercontent.com/a/ACg8ocLFA0W78X2jXYmrz1Rd8ooNrRWJKBcMsf2iRGfx1PySOVlnq47L=s96-c)

hardcorg

[• 13 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cme6ecupp0sirad08jwdfyn64)

My take: Pick a distributed durable storage layer that supports cross-table/cross-partition propagation.

I would caution against maintaining fault tolerance in updating two distinct databases.

Consider this: the order is submitted to the exchange, the order db is updated, but the host process before rocksdb is updated.

You would need some persisted WAL for the process to come back up and restore state.

Or the more extreme case: The host dies and you need to provision a new one. RocksDB is local and embedded. You'd need to use something like TiKV to have fault tolerance. Otherwise, the rocksdb data is gone for good.

Show more

0

Reply

![RunningVioletAphid451](https://lh3.googleusercontent.com/a/ACg8ocKQGdTpblvPrifBACmSg8865PeqoIiJIPbmPQwV9VZpL1z4Q_E=s96-c)

RunningVioletAphid451

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmaja0c1i01m3ad085r9f769o)

Yes, i think in postgres. partition by user id might still work with index on externalOrderId. But sharding by user id will make index on externalOrderId not that efficient.

Show more

0

Reply

P

ProspectiveCoffeeRhinoceros457

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmdgte0oz015gad07k8xwj6o6)

But if you are building an index on externalOrderId, that doesn't necessarily help you that much, as you have to still query potentially all Postgres write shards (because the corresponding order could be in any of them) if I understand correctly

Show more

0

Reply

C

ConfidentialCrimsonFlamingo719

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4dtb51q039q1i3rtj16a2yu)

Great write up!

I don't really think you need the External Order Metadata DB. Most good external apis offer you a metadata field where you can pass in whatever you want, and they will return it to you in their webhooks. In this case, we can pass in a userId and orderId when we create an order from the exchange. When those webhooks come in, we receive back those ids and can look up our order directly in our Orders DB.

I would think most senior and all staff level folks have experience with this metadata field in apis that offer webhooks.

Show more

31

Reply

G

GlobalPinkLadybug365

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm7it9jbt048ixcuguy0ljwsd)

+1

Show more

0

Reply

E

ElegantMoccasinSwan892

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8dhq0ho0068tuvwr686wils)

+1 I was about to write the same comment :)

Just something to consider — passing "internal" userId or orderId to third-party systems (like the Exchange) might not align with a company's security or privacy policy. I think a safer approach could be using indexable columns such as an externalUserId or externalOrderId that can be shared with third-party systems.

Show more

2

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmcagykov034zad08qfmqozi0)

can we not use redis instead of rocksdb, as redis is also a KV store and we need this mapping as long as order is not filled(success/failed) or till the time market closes, whichever happens first.

Show more

0

Reply

V

VerticalBlackFlea403

[• 28 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmdl2lx9w0287ad073iybbsnu)

Redis can crash before the order is filled. besides, if it's limit order, it may not be filled for days

Show more

0

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 20 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmdvp1fhg0k9bad08qyp5eg5u)

That can be handled with high availability redis sentinal used in cluster setup.

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4f715r203mnqfetgl64lwqz)

what about a validation service that checks if the user has enough funds to buy, enough stock units to sell?

Show more

1

Reply

G

GrandPlumSilkworm276

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm9qgrilk0044ad08i6mbh4or)

that's beyond the scope of this design

Show more

0

Reply

S

SpecialTanNewt956

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4jo5fxc002cfmmi4woh36pq)

I don't understand the need of PostgreSQL + separate KV store here. We don't require ACID. We require consistency which Dynamo can also provide via Quorum consistency. I think Dynamo will be a good choice since it provides GIS also. So, Two Heads, One Bullet. What do you think?

Show more

2

Reply

I

InclinedSapphireScallop143

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5cx9ikw00kjm9c3xgran0zb)

dynamo ensures eventual consistency but here we have strong requirements on ACID for orders

Show more

2

Reply

H

HeavyLimeWolverine413

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5okba26007skzmh30zz1sc5)

DynamoDB can provide ACID properties these days. It can support transactions. GSI is a different story though, that would be eventually consistent.

Show more

1

Reply

N

NobleAquaRoundworm111

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5z7yarj004b12uxpbj83ngk)

I'm also thinking if we can use DDB for this case.

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8gxlxwi018et1ifdv6a60js)

Yeah, from the story, i don't see a transaction with multiple query. So we don't actually use the ACID properties. Can we use Cassandra for the high write qps? The replication strategy should use synchronous replication regardless

Show more

0

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma4vze9300muad0714hqg70r)

You need ACID when you need to ensure the stock # in the account is synced, the cash in the account is not going to fall negative if two orders happen at the same time, etc.

Without a consistent write, you risk partial updates.

There may be multiple tables being written to at once, and you want all of them to happen or not.

Show more

2

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4p9w4ji026gjs6syd80k3ju)

1.  The gateway avoids the delay of having a queue, but it doesn't alleviate the excessive communication problem. I don't know if the IP thing is important, because the services would run in a VPC with a limited number of IPs. But I'm no AWS networking expert, so, not sure about that. In order to truly reduce the number of calls, the orders will have to be batched; this is mentioned in additional deep dives. But then we're back to the delay similar to the queue, so, this writeup never really addresses the NFR #4.
    
2.  Why not just create a unique index on external order id instead of introducing a KV store?
    

Show more

7

Reply

I

ian.ornstein

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm7o1ht8d007sl12q7wacatsl)

Regarding the unique index on external order id:

Adding a secondary index was my original thought. I think the important thing is to make sure you are able to discuss tradeoffs of each approach.

Some drawbacks of adding a secondary index is that every new index slows down writes. We already have a delay of responding to the user via purchases because we have write one to our order DB, write 2 to the exchange itself, then write 3 to the order DB. Adding a local index slows down the final write. The other approach allows a 4th write to the key-value database to happen in parallel with this final write, so it speeds up this process a bit.

In addition, now when we get an update, we receive external order Id and we get to look it up in the DB. Your approach is fine if we have a single node, since the local secondary index secondary on external order id lets us look it up lightening fast.

But if our database is horizontally scaled this becomes a problem. Because now we have to either a) check every shard for the presence of the external order id or b) have to use a global secondary index (which now means all writes have to go to at least two nodes, using two phase commit to ensure they both update, which will definitely slow down the writes)

Then again, will the Orders DB ever need to scale horizontally?

20M users \* 5 trades/day \*365trades/year ~ 36B orders a year. in 3 years we get 100B orders. what is the size of each order? orderId - 8bytes userId - 4bytes externalOrderId - 8bytes symbol - 1byte shares - 4 bytes price - 4 bytes state - 1 byte ... more data timestamps, etc? - 20bytes so a total of around 50bytes per order 5000B = 1 TB every 3 years.

This is definitely manageable for a single large db node, so we wouldn't have to scale.

The other drawback of the unique secondary index on external order id is that your DB has to support it being null for the first write. Which postgres does allow, so that is fine.

Show more

2

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma4wo85000i8ad08adpicwsi)

Just to clarify, RDMBS doesnt use a GSI or LSI - thats for NoSQL DBs.

The addition of a nonclustered index on the ordersDB for the externalOrderID will likely be fine in practice (typically issues with indexing comes from a huge number of indexes and the amount of data).

Depending on how many rows in the table, partitioning may be a better option since all of the data will live on the same node, and honestly, sharding a RDBMS comes with its own set of challenges (managing cross-node transactions, 2 phase commits, etc).

As some other people mentioned, its entirely possible to pass in metadata to a webhook such that when the Exchange makes a request through the webhook, it can return the metadata key back so we dont necessarily need the second database.

Show more

3

Reply

D

djmo0000

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmar2th3e00jbad08484pv64b)

Since the data records for orders are pretty narrow, in an RDBMS, your index can often already contain all the fields you need for your query. You can always add additional small fields to your index to never have to go to the table. The downsides are (1) less records get pulled into memory when a block of the index is retrieved and (2) updates to those added fields will require changes to the index. An example is updating a status field we don't need in the index, but we want the index to span our query pattern. But if the status field is the last field in the index, changing the value likely won't actually change the sort order.

If it actually changes sort order, it would likely be to a location in the same block of the index already in memory, making the update really fast. You might just be changing the value in a single location of that block while impacting nothing around it.

Show more

0

Reply

![Shreeharsha Voonna](https://lh3.googleusercontent.com/a/ACg8ocJFOfqMA6ZDt2EamTjU3FJJThz35r5s7qZyqePh9qAYYlJq_K8p=s96-c)

Shreeharsha Voonna

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4u8sler00ssqchthnd4maax)

Thanks for great content Joseph, Evan and Stefan. Questions:

1.  In the "How can the system scale up live price updates?" deep dive section, you mentioned that Symbol service host subscribes to the price updates for all the symbols which its connected users are interested in. When we assign a user to Symbol service host for sticky session, I believe its via consistent hashing on userId. That means each Symbol service host could be subscribing to potentially all the channels (all the symbol updates) given that user that are connected to it have a wide variety of portfolio. This further means, each symbol service host is constantly getting updates for each of the symbol anyways. So how is Redis pub-sub helping here? Why can't Symbol service host just read latest price from Redis cache directly?
    
2.  In the "Users can manage orders for stocks" section, you mentioned having Orders Gateway is a great solution. But how does that solution take care of insane amount of order requests that are coming in during peak hours when there is no message queue in between? All I could understand is Gateway going to ensure requests going to Exchange come from certified/allowed set of IPs.
    

Thanks

Show more

3

Reply

![Joe](https://lh3.googleusercontent.com/a/AAcHTtc-eLHVxd8Q5Ppk5tLveDuet1s0BtYKRPzMX6iVpQ=s96-c)

Joe

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5x8hxxm00jyjtq11vc62tjc)

1.  We need to tail price feeds vs. read from the cache so that we can push updates to user clients proactively vs. have them poll the cache. When it comes to distributing load, the load balancer might balance load across machines by choosing machines that are already subscribed to a symbol so as to avoid machines subscribing to a ton of symbols and being overwhelmed with price inflow / fan out. Generally, an application load balancer would be favorable here so it could use application heuristics to decide where to route a user.
2.  The order gateway can be internally scaled horizontally to handle load. This would involve some sort of internal load balancer. This detail is a bit implied, but is also explicit in the sense that we mention potentially over-provision the service to handle high load of orders.

Show more

1

Reply

S

SubstantialHarlequinHerring381

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm4yqoyrs003g13vumiq0jtux)

Hi, Is it okay to directly send SSE Events from the service to the client or should this go through API Gateway? Wouldn't exposing our backend service to client directly have security implications?

Show more

0

Reply

M

MinorAquamarineHare569

[• 6 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm789lmkg0265l2b7mprhhj9r)

AWS\* API Gateway does not support SSE. If AWS, this is a mistake by the author.

Show more

0

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma4xct5p00izad08v5n1sfds)

Are you sure? A lot of tech (NGINX, Kong, Envoy, etc) seem to say they support this

Show more

0

Reply

![Akhil Razdan](https://lh3.googleusercontent.com/a/ACg8ocJjkDdOJSlZqYI67eSJoU3gDT54OHhWrgkk6tGVoT6phVwsBLWV=s96-c)

Akhil Razdan

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5abbq6o0234c4lg2j5hsxk0)

Why do the sessions need to be sticky? The symbol service doesn't seem to be stateful for the connections to be sticky.

Show more

2

Reply

I

InclinedSapphireScallop143

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5cxal1500kmgzymi3s9ig6a)

SSE connections are stateful and needs a persistent TCP connection

Show more

2

Reply

J

jasonliu0499

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm79i6b3z00bkngojp8aizjxw)

This is not correct, SSE is stateless and rely on longlived HTTP connection, on the event of disconnect, the client/browser will init re-connect

Show more

3

Reply

F

FinancialGreenTick391

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8d0lxit01hu11ory2oxohs5)

I guess that if we don't have sticky sessions, the user could be peered with many different servers, leading to many different redis un/subscribe events to symbols. You'd prefer the user to stay on one service.

Show more

1

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma4xea2w00j5ad08u6v5msbw)

wont this cause hot servers where potentially a disproportional amount of users are connected to a single server?

Show more

1

Reply

I

IndividualGrayToucan573

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5g5w4hn00azepsu9wupzjnh)

Is it better to merge Order Dispatch Gateway into Order Service as a single component? what is the advantage of having two services?

Show more

1

Reply

H

HeavyLimeWolverine413

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5okcxcr007mqsjkbmw2o7y1)

+1, we can use NAT gateways on the Order Service itself to limit IP addresses right? This just introduces an extra layer imo

Show more

0

Reply

![Joe](https://lh3.googleusercontent.com/a/AAcHTtc-eLHVxd8Q5Ppk5tLveDuet1s0BtYKRPzMX6iVpQ=s96-c)

Joe

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5x8juj600gozd19oxiks6ct)

This is a fine idea! One service with baked in support for external requests from a limited set of IPs is perfectly fine, so long as you explain the responsibilities of the service.

Show more

0

Reply

K

kkp151993

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5j5uy8l02ywa8vvsnxgq7kz)

-   Can someone explain how could we handle system if instance of symbol service goes does?
-   Additionally do we need to have CDC on orders table to send notification to client regarding order status updates?

Show more

0

Reply

![Joe](https://lh3.googleusercontent.com/a/AAcHTtc-eLHVxd8Q5Ppk5tLveDuet1s0BtYKRPzMX6iVpQ=s96-c)

Joe

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5x8msqi00h4zd195vbl8awg)

If the symbol service goes down, clients will disconnect and retry to re-subscribe for symbol updates. This will be routed back to the symbol service and they will start getting updates from another set of symbol service machines. The machine that died can be replaced and start bearing the load of new clients quickly.

Order status updates aren't necessarily a requirement here, so that isn't a concern for this system (based on the functional requirements). I encourage you to think about how you might support order updates as an extension of this question, as it's a meaningful extension to the system!

Show more

0

Reply

H

HeavyLimeWolverine413

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5okqr8v008mkzmhs8o4zi4s)

For the user to subscribe and get real time price updates, is the solution eluding that there are multiple Symbol Service servers maintaining connections with 1 client? If for example a client wants updates for AAPL and TSLA, then 2 separate servers would be sending the price updates to the client?

Show more

1

Reply

![Joe](https://lh3.googleusercontent.com/a/AAcHTtc-eLHVxd8Q5Ppk5tLveDuet1s0BtYKRPzMX6iVpQ=s96-c)

Joe

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5x8nw0y00khjtq1yk0m1vbp)

That's entirely possible. It might be more efficient for 2 separate servers to support the live feeds for 2 separate symbols. Having 2 socket connections on the client would be cheap.

Show more

0

Reply

![Wang lei](https://lh3.googleusercontent.com/a/ACg8ocIBwcYDZiesH-WGea9evEz-VtPcpOiYrYCCdqZM0uHbfMdpWw=s96-c)

Wang lei

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm86sxtzk00ztkzp58vvlm128)

how can system fan out single request from user about watching multiple symbols to multiple symbol service hosts? also single client has multiple SSE per symbol?

Show more

0

Reply

![prakhar](https://lh3.googleusercontent.com/a/ACg8ocKDXXoGteJDNOueUSTQRS4w_FcP_2LM2rmDgu5vmw2SK28XSQ=s96-c)

prakhar

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5qv41qu02gawu5hl6inrgoa)

Hey, I have a suggestion. The FR and NFR which are out-of-scope are the most interesting once in almost all of the design problem that I am had read so far ( Maybe I am saying because they are actually more unique wrt the problem). It may make sense to create the levels of specific system design problem. For example robinhood:::standard and robinhood:::no\_scope\_included. <standard> can be enough for clearing interview. <no\_scope\_included> can actually be more advanced / research based.

Show more

0

Reply

![Rajat Mishra](https://lh3.googleusercontent.com/a/ACg8ocK1VeQLni_QF_BrSquYJkb2LFFdlEUZHKhT9t6o5JuSTmt_OwU=s96-c)

Rajat Mishra

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5wvsqsb0014x457mro6joef)

Redis streams are better than redis pubsub. Pubsub is not scalable.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm5ww0raf0015123d0660iq2c)

Why is pubsub not scalable?

Show more

1

Reply

N

NuclearAmethystStork225

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm9e38v0300o8ad08574o5ocr)

Hey Can you please unlock just the written article for free? this is very helpful to us

Show more

0

Reply

I

InterestingPlumPython739

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm60cvje400oy4i6q2kp7ckpg)

Thanks for the comprehensive article. I have question regarding consistency requirment.

In the funcional requirement we added: "The system prefers high consistency for order management; it's essential for users to see up-to-date order information when making trades."

Wondering how is it achieved?

Even with postgres db as order DB and we can lock a row when read and write. User can still see an order "pending" in db, but the order may already be finished in the exchange, because the order submission is asynchronize.

Do we need 2pc actually to make internal and external status strongly consistent? But it will be too slow. In the article , it kind of addresses it by returning successful response to user only when receiving order success from exchange. But if user calls get(orderId) before exhange calls back, there is still inconsistency status.

Show more

0

Reply

![Joe](https://lh3.googleusercontent.com/a/AAcHTtc-eLHVxd8Q5Ppk5tLveDuet1s0BtYKRPzMX6iVpQ=s96-c)

Joe

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm62dntph016xgudu6g0nvrjq)

No 2PC needed here and there may be some degree of asynchrony RE order's true status given that we're relying on the exchange and participants of it to fill the order. However, as soon as our system is made known of updates, we should enforce consistency. We should also have strategies to ensure that an order goes through "states" consistently when it is submitted to our system, submitted to the exchange, and eventually filled (or failed). This characterizes the consistency requirement of this question.

If we preferred a highly available system (sacrificing consistency), we might experience inconsistent reads of order state, even when our system has been made known of order status via corresponding with the exchange. This would be considered undesirable in a system like this.

Show more

0

Reply

I

InterestingPlumPython739

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm649a6dh0196ufnhy3jv6p7o)

Thanks for replying, Joe. Given order status consistency is crucial when trading, we don't want to show order as "pending: when it's actually successful/failed, especally in high freqenct trading, can we NOT provide the get(orderid) api as we cannot gurantee it's correctness. Can I bring this proposal up to the interviewer? Thanks

Show more

0

Reply

D

DemocraticSalmonLemming370

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm63ynufm00o3y43nico4azi2)

Trade updates are sent using webhooks. How does the symbol price processor get price updates from the exchange ? Also, how does the order service notify the client when there is an update to the order ?

Show more

0

Reply

T

ThoughtfulPeachGull276

[• 7 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm65gxtck00mhwntowm7or3ps)

Great stuff; thank you!

I'm curious if an alternative pub/sub solution like Kafka would work the same or even better here. Could you create Kafka partitions based on a singular stock or cryptocurrency token like BTC and that way clients could subscribe (via hooks or websockets) to the particular partition(s) they need?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6pnvafs036v41j3krtqq1ap)

Kafka works well here, yeah.

Show more

0

Reply

F

FlexibleSapphireEchidna890

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm7ctqk3p02pf5v9j5gj1k3fn)

If we were to use Kafka, will that eliminate the need for symbol service to listen to price updates? Can the Robinhood app/client then directly listen to Kafka topics using Kafka client library?

Symbol service's role will be then limited when client pulls the price for the first time upon opening up the app. Does this sound right?

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8gxzs200192t1ifxmqejqja)

With Redis pub/sub, we are creating one stock per channel, that's 1000s channel. Can kafka support that?

Show more

0

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd9n1a5p020gad087wnzp9w8)

A Kafka queue will add a time lag to the updates, which violates the real-time NFR.

Show more

0

Reply

H

hossain00

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm67780or002q10oxpmgqii5w)

Hello Stefan, I just have one query to clarify. I thought RocksDb is used as embedded db. According to my understanding you are using as non-embedded mode. could you please clarify why?  
I am sure it will shard with externalOrderId as sell and scale horizontally. Please clarify.

Show more

2

Reply

V

VisualVioletReindeer787

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm67khdva001k86ajjoz6gm20)

Small pedantic suggestion - in the create order API, the price is listed as 522.10. To avoid rounding errors, prices should always be in integers in minor units, i.e. 52210. Even if internationalization is beyond the scope of this design, storing all currencies as integers prevents rounding error.

Won't make any difference to the overall design, but if you're interviewing for an e-commerce or fintech role, it will win you some points!

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm67ki3aj0027l4pes82zdyr4)

Yeah, this is a solid point. Will make some adjustments, thank you.

Show more

1

Reply

V

VisualVioletReindeer787

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm67kvwcf002686ajccfp9u97)

Another small tip that could win you points as a candidate! The design mentions the use of a "clientOrderId" -> this is exactly the right answer, but calling it an "idempotency token" or "idempotency key" will definitely help with some interviewers :)

Show more

1

Reply

![Di Ye](https://lh3.googleusercontent.com/a/ACg8ocJOUtDjlvIB25alFL0hZyexChUImYjCM06uaO-gUSmZP93atw=s96-c)

Di Ye

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm69pb6d702ape7bzyizwpt34)

Redis pub/sub has at-most-once delivery semantics. Given the critical nature of real-time data in a trading platform, should we use a messaging system that provides strong delivery guarantees?

Show more

1

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8gy2a8y019bt1if7z7q606c)

IMO, that's fine. The prices changes every seconds, if it missed 1s price, the next second comes in immediately.

Show more

1

Reply

S

SelectiveMaroonTapir734

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6conzax02hu6hpqry0m4edu)

Nice write up! Loved this one.

I am wondering for more fault-tolerance and durability if a Kafka event bus ought to be placed between the Redis pub/sub and Symbol Price Processor. In the case a subscriber goes down or SSE connection disconnects, it will miss some of the messages from the Redis pub/sub (since they're ephemeral) - Kafka being durable will persist the updates for the subscriber or SSE connection to "catch up" on. Perhaps missing some updates (given market volatility) is also fine here.

It's impossible to enumerate every situation. This problem is great because I can see so many places you could go deeper. Thanks again!

Show more

0

Reply

D

DemocraticSalmonLemming370

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6k0d8jy01qojgz3z1tx38vr)

thanks for the write up. How does the symbol price processor get price updates from the exchange in real time ? Does the exchange manage SSE connections to symbol price processor service for pushing price updates?

Show more

0

Reply

![Anton Ushakov](https://lh3.googleusercontent.com/a/ACg8ocI81pWSMOXkQ-lhHDlxCnW3dqkFSlwiIYr3yj-NTdgVNNPfXQ=s96-c)

Anton Ushakov

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6mosemt0506oi5wpbnzovrk)

why do we need a separate store for mapping ExternalOrderID -> orderId provided that orders table holds ExternalOrderID? Can't we easily UPDATE orders .. WHERE ExternalOrderID = $1 from Trade Processor?

Show more

1

Reply

P

PleasedPlumHalibut980

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6oj2wfy01tao9xsixdfqft6)

The 'Order cleanup' job queries the exchange and updates the Order DB. How will it get ExternalOrderId to query the information from an exchange?

Show more

0

Reply

![Tommy Loalbo](https://lh3.googleusercontent.com/a/ACg8ocKJIn8OPXYOxiFFjMUkH5UDjWWCbOFuGt2Srsu9sGECWCgexFCq=s96-c)

Tommy Loalbo

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6pn0ljf034rv55h607yx6po)

Would we be wrong to argue that websockets are necessary here? This is financial business. Can we really gamble on the damage to the reputation that could occur due to following a fragile protocol like SSE in this business case? I can understand accepting the loss in a social media scenario, but i am imagining some serious complaints and lawsuits coming out of using SSE here if people catch on by chatting with fellow traders over the phone or something. I understand we could try to use some complex logic to detect unsucessful and the need for websockets, but that just seems like it would significantly reduce the likelihood of issues. Help me wrap my brain around this. (Going for staff level)

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6pn2mxi035441j3k9colly2)

What's the fragility you're referring to?

Show more

1

Reply

![Tommy Loalbo](https://lh3.googleusercontent.com/a/ACg8ocKJIn8OPXYOxiFFjMUkH5UDjWWCbOFuGt2Srsu9sGECWCgexFCq=s96-c)

Tommy Loalbo

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6pnn02r035lo9xs7z9pprzz)

I always appreciate your prompt replies Stefan! I am saying that like people can be behind proxies and firewalls that block SSE.

For example:

1.  corporate networks, ISPs, or public Wi-Fi networks may have proxies or firewalls that interfere with long-lived HTTP connections.
    
2.  Mobile carriers often deploy carrier-grade NATs and load balancers to manage the massive amount of traffic from mobile devices. This means that the client’s connection might be routed through these intermediary devices, which can affect how long-lived connections (like SSE) are handled.
    
3.  Public networks (e.g., in cafes, airports, or hotels) sometimes use proxies or load balancers to manage bandwidth and ensure security, which may cause similar issues.
    
4.  Some Internet Service Providers use load balancing or caching proxies at a network level to optimize traffic flow and improve performance for their users. This means that, even at home, a client’s connection might be routed through such infrastructure.
    

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6pnt9si036ev55hy8alpmlg)

Note that NAT's are not going to keep SSE from functioning, but there are certainly some network configurations which will cause issues (e.g. proxies which buffer chunked encoding).

I think it's a fair callout. Trivial to detect with a simple heartbeat. I don't think the lawsuit angle holds much weight here, especially if it's a misconfigured proxy and not Robinhood themselves which caused the issue. Generally speaking you're unlikely to get or be docked points for this in an interview setting, but if you want to say "WS because I need realtime updates AND I want more mature support" — sounds good!

Show more

0

Reply

![Tommy Loalbo](https://lh3.googleusercontent.com/a/ACg8ocKJIn8OPXYOxiFFjMUkH5UDjWWCbOFuGt2Srsu9sGECWCgexFCq=s96-c)

Tommy Loalbo

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6poe68g0374v55hi89iqpd2)

Thanks. I was thinking in a stock based app where seconds matter we wouldn't want users to have to go through the timeout to be able to access the information (like the first time they sign on, they are sitting in a hotel and news hits and the seconds cost them thousands). Is that a valid argument? Or is that a product question?

Also, I have been burning to get answers on some of the questions Evan lead for days now that i commented on. Would you be able to answer some of them?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6pon1ce038341j369jyma8s)

Typically, you have to pay more for real time feeds. I don’t know about Robinhood in specific but most retail feeds are 3-5 seconds delayed anyways. And Robinhood lets Citadel front run your trades anyways. All to say: the casino vibes of Robinhood aren’t accidental.

We don’t offer any SLAs on comments. We try our best to get to as many as we can. Sorry!

Show more

1

Reply

M

MarvellousCrimsonEchidna107

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6q4a89v042j124quvnqufof)

I believe we need multiple service to listen to price change and push to client through SSE. If the service need to track the map of symbol<-> List<userId>, does that mean a user that tracks the list of symbol would connect to multiple services? That seems more resource consuming to maintain multiple connections with a client and the service group.

Show more

0

Reply

![DifficultSalmonBear341](https://lh3.googleusercontent.com/a/ACg8ocIUjkx8EZ2pe68ytUu9_RgBphRrYRLNwZwpQw1jhXPpk9ef6sed6g=s96-c)

DifficultSalmonBear341

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm6wnfblw01kpzdjvzhicq0fh)

I think the introduction of a key-value store just to look up externalOrderId -> orderId is an overkill. It's a whole another system to maintain just for this purpose and practically keeping a hash index on externalOrderId is sufficient to make the lookups faster.

Show more

0

Reply

![Misha Borodin](https://lh3.googleusercontent.com/a/ACg8ocITt8_C-XimHao0Gj-BqF28IKe3WXyA8ppWstGFMnewgZtPMQ=s96-c)

Misha Borodin

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm7jfzl0c04rwoa3n3z2hut20)

Why do we need a stand-alone cache for Symbol prices if there're just 10^4 or so symbols? Why not to put an entire copy directly into RAM of each symbol service?

Show more

0

Reply

![Dmitry Grigorenko](https://lh3.googleusercontent.com/a/ACg8ocKWISD4A7ZGdhKGbgPoOHdMPuzpoX6CD_cK-cO1JJDXH3mKrZOeuA=s96-c)

Dmitry Grigorenko

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm7zr5ovs03mm5693ry3rqz36)

RocksDB is an engine, not a standalone DB, so it definitely just does not fit here.

Show more

0

Reply

![Katie McCorkell](https://lh3.googleusercontent.com/a/ACg8ocJhhgdomrgC_-4WZgadOwN2X8drRXio1OaPECBGUAB1SRYpPV4UHw=s96-c)

Katie McCorkell

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8epp4t801ho40wj24ohc30r)

I think there is a typo in the middle sentence here. "Store an order in the order database with pending as the status. It's important that this is stored first because then we have a record of the client could. If we didn't store this first, then the client could create an order on the exchange, the system could fail, and then our system has no way of knowing there's an outstanding order."

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8gy5z4j019lt1ifctkcg6s0)

This section "How does the system manage order consistency" is actually discussing how to make system fault tolerant and consistent with state machine. Would it be cleaner to expand the discussion around the state machine with reconciler?

Show more

0

Reply

H

HeavyCopperPython311

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8kdfgq10008139l0et5ira8)

From Vote for New content , Stock Exchange question, It is directing to this question, isn't Robinhood a stock broker rather a stock exchange. or it doesn't matter regarding system design perspective?

Show more

0

Reply

S

suparv2204

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8uow6bn01jfad071fmmjsk4)

Do we not need a status as "Unfulfilled"? What if the exchange could not match the order?

Show more

0

Reply

![Garrett Mac](https://lh3.googleusercontent.com/a/ACg8ocKvc-WsjvNfRoWnWXLh2G95DQ31dmFv83g1P2q-nlplgPZOnQ=s96-c)

Garrett Mac

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm8vz5z0402yhad092uwqtm3x)

Would love to have a link to the excalidraw in all these breakdowns

Show more

0

Reply

G

GradualSalmonBooby605

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm98xp9yg00evad08sb4vu44p)

I am curious to know why we used RockDB here! it's kind of out of the blue in the entire HelloInterview tech stack, but not only that, we also added new software to our design's tech stack, which sounds hard to justify given our use case and can be argued by adding more maintenance and operational complexity. Can't we stick with Redis (AOF), as it's already in our tech stack, and keep a key value like "(externalOrderId: {userId, orderId})"

Show more

1

Reply

G

GradualSalmonBooby605

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm98ym37k00g5ad08m5xks7r4)

It sounds like the article has not covered limit order while it has been mentioned in the core requirements.

Show more

3

Reply

U

UnlikelyAquaCougar275

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm9c9pxwc003dad08ussq39cw)

Is partitioning Postgres on basis of userId a good idea? Postgres doesn't scale well for millions of Partitions? Would mongoDB be a better alternative if we have to partition on basis of userId?

Show more

1

Reply

G

GradualSalmonBooby605

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm9d901p000b3ad080tmxkbs9)

There is not a 1:1 relationship between the number of users and the number of partitions. Many users can fall into one partition. For example, you can have more than 20M users in one partition, which in a five-partition setup can handle 100M users.

Show more

0

Reply

![Silver Blaze](https://lh3.googleusercontent.com/a/ACg8ocKmug9aG0CmQxixVcD8H9bOlYnpVU2VZhoWSRVGt1cuY_wDywY=s96-c)

Silver Blaze

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmcqt35p800kead087kzvznyd)

Then what is the benefit of partioning by user\_id if there can be millions of user in a single one ? Should we not partition by date and create an index on user\_id column for every partition ?

Show more

0

Reply

I

IncreasedAquamarineMole954

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm9geieh702dnad08jtq4drw5)

For "Users can see live prices of stocks", wouldn't we want to skip out on caching in this case since stocks get such frequent price updates?

Show more

0

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmcahv7pl0392ad086xfjwfa8)

instead of cache i feel better to put in TSDB to serve historical data without going back to exchange.

Show more

0

Reply

![Hoang Le](https://lh3.googleusercontent.com/a/ACg8ocIkQDbUaH17x63SWVqDKGYMWPW0kqD8sugFMc7BMNYFPa0prhXA=s96-c)

Hoang Le

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cm9jflbuq0033ad07t1hv1cb2)

I'm curious to know why we need a separate trade processor to tail the trade but can't double up the trade dispatcher to listen to the confirmation from the exchange?

Show more

0

Reply

E

ExoticPeachOpossum454

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma2oy41700inad08djaq4kdd)

Maybe I missed some parts while reading the solution, but I'm curious about some of the choices in the design: 1- Why are we adding the key-value DB to map: {external order ID: orderId, user ID}? Why not add a new table to PostgreSQL? 2- Why are we mapping externordertid to:orderId, userId? OrderIds are unique, no? and we can map: externordertid to:orderId

3-Where did we use ACID properties in this design? Is it to serialize multiple orders or write them from the Order Service?

Show more

0

Reply

V

VitalGoldHerring686

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma4mve8i0064ad07ib5rtcw6)

How does order validation work? What if the user places multiple orders and their available balance can only support one of them? How do we make sure the balance updates are done in a way that the concurrent requests are not reading a stale value?

Show more

0

Reply

C

ConcreteBrownMinnow590

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma8d89hq00nzad08cw7ilsr7)

Give 20M DAU and 1000s tickers, I don't think simply mentioning about Redis pubsub would be helpful here. It's important to know about the nature of this: We're fanning out ticker price from the source to all users that (presumably) mark it as favorite. For popular ticker (i.e. SPY), we will have 20M subscribers - which requires Redis pubsub to be scaled up with custom sharing strategy - not an easy task. An alternative is to use Kafka with topics set to be tickers - in the context of this problem, it would seems to be a better tool for the scale.

Show more

0

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd9mz2080200ad08jx9t8mhs)

A Kafka queue will add a time lag to the updates, which violates the real-time NFR.

Show more

0

Reply

![Kuldeep Arora](https://lh3.googleusercontent.com/a/ACg8ocLRhvB7Ax8bDEu9fmbzE_Cj9jlkZZzh7LzuVjnvMNLd79sLPVe8=s96-c)

Kuldeep Arora

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cma9w6yg202cqad08e36j4ahs)

Great write up.

Small question: We have our symbol service, which manages a map of symbols to user\_ids.

If we our working on a large scale, we will have to shard this symbol service into multiple instances based on the symbol id, (using consistent hashing).

Now when a user is calling the subscribe() function, and passes 100 symbols it wants to subscribe to, let's assume these 100 symbols are managed by 10 different symbol service instances, so this would mean 10 different client<-> server connections, which is very inefficient.

Ideally we should have another layer between the client and server, which will manage the websocket/SSE connection, and fetch the prices from all different servers based on the symbols.

Show more

0

Reply

M

matariyasavanh

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmafetvl600m2ad0842gtt7sw)

I didn't understand the use of Order Dispatch Gateway fully, shouldn't we use the queue which is mentioned in "Good Solution" because that will provide decoupling, and we can retry the order to exchange.

Show more

0

Reply

A

AddedLimeCarp593

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmaogncd200c9ad09a9xrxgpw)

Why do we need the Orders database to be ACID?

Show more

0

Reply

![Oussama H](https://lh3.googleusercontent.com/a/ACg8ocJa3ZWTqotfg35U1Ga4aimQBVn7hoX0x4QD0vXTOgTbPv9L5Q=s96-c)

Oussama H

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmb5nwuee00rvad07ty6jtmsn)

This seems wasn't written by neither Evan nor Stefan. Might be worth having a video about, seems a few points are just skimmed through. (Or may be I learned too much and this is enough depth lol)

Show more

1

Reply

![Ata Marzban](https://lh3.googleusercontent.com/a/ACg8ocIgSgndjO_pqn4iQrf_4kw4BDLwQdbzookWWW9_YcumAol5J2g=s96-c)

Ata Marzban

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmb9f56yr01quad08kt1ne3mj)

The clean-up job approach means that the user will not know for some time (depending on clean-up frequency) what happened with their order, wouldn't e.g. a Kafka reconciliation queue + SSE order status updates be better?

Show more

0

Reply

![Harshit rastogi](https://lh3.googleusercontent.com/a/ACg8ocL00iaqE5Zyne-_crxYXfhS_Ih-VoS7Wv4wXwJmXCsDcYO4b0HL=s96-c)

Harshit rastogi

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmbepfsk3032kad08a345qok0)

when the system receives order confirmation from Exchange, the Trade processor needs to update External DB and Order DB. What happens if update is failed for one of the DB ?

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmbf3hyih00bgad07ejmha15k)

"internal cache that is kept up-to-date by a symbol price processor that is listening to prices on the exchange." - Symbol price processor listening to exchange - how does this happen can you please explain

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmbf46de300boad07qpfw7e9d)

our gateway would be an AWS NAT gateway- why gatway is shown as a different service.

Show more

0

Reply

Q

QuixoticCyanReindeer555

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmbqcn61900jq07adf6gdnac2)

Hey, thanks for the article. I just did not understand why we are using dispatch gateway here ? It mentions that it will make requests look like under small number of IPs. But why we need this feature ? Is this service doing similar thing to limit the request number or some other functionalities here ?

Show more

1

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd9mwyym01zlad08d9zjgn1q)

It is a non-functional requirement in the question. Typically, a 3p API (the exchange here) will give you a limited number of API keys and QPS quota. It's easier to maange when you restrict all API requests to a small number of (gateway) servers

Show more

0

Reply

A

AG

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmbykjna203o409adn0ws4e7q)

How are payments for the orders processed?

Show more

1

Reply

D

DoubleCrimsonGecko613

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmc57mde300anad08qvd1hxsw)

Thanks for the article. I wanted to understand why we need the External Order Metadata DB. Can't we create an index on externalOrderId in Order DB so the Trade Processor can efficiently look it up directly in the Order DB?

Show more

0

Reply

![Helen Gu](https://lh3.googleusercontent.com/a/ACg8ocIBme-fdzbG5XpPTZecqNQGLckKixAnX-S629BqTtKGBVbxGdY=s96-c)

Helen Gu

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmc6yt4jx0artad08k5dv13xf)

Hi! Really appreciated with your coverage. It would be awesome if you can cover LIMIT order part.

Show more

0

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd9mtok201yfad08j6jgo6gu)

I think the type of order (limit or market) is passed on to the third-party exchange. There's no infrastructure complexity to it

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmcwuj1xv03srad0856cevocn)

Redis fan-out pressure: if 10 servers subscribe to a hot symbol, Redis must push that update to 10 channels. Consider sharding by symbol or using Redis Streams if fan-out becomes a bottleneck.- is using redis stream a correct solution??

Show more

0

Reply

A

anup.mayank

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd2xhddg083xad08ro2demwn)

Why is there a need for orderid to externalOrderId in a separate DB? Why can't it be maintained in the same Order DB

Show more

1

Reply

C

cloud

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd4o7fb409szad08gej9wqkh)

Regarding the last deep dive to maintain order consistency - why an additional service 'clean up job' is needed where 'track order' service can do the same thing? It is already periodically updating the both external DB and order DB based on the order status in the external exchange so having another service seems redundant.

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd7jhhox03xqad08tul2xncu)

Hi! Thanks for writing this article. Is it possible to get the tips text (not sure what else to call them) in a different color other than black? In dark mode they are hard to see. For example this text is black: "With each of these requests..."). If we can get that text in a shade of blue like the green in "Note we're using priceInCents...", that would be awesome!

Also, there's a minor typo with the word "effectivew". Thank you!

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd7kukxx04ryad08y5eubdz9)

Absolutely yes, thank you. Fixing both!

Show more

1

Reply

N

NationalTurquoisePorpoise312

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmd9lu0b001q2ad08z7p5r64p)

Can you explain how to create a RESTful method for subscribe? It's not one of the REST verbs. Do we define it over the collection?

This approach involves re-working our API to instead have a /subscribe POST request, with a body containing a list of symbols we want to subscribe to.

Show more

0

Reply

M

MonthlyBeigeBeetle805

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmda5y1qc05cjad08lz4jra4r)

NFR: The system connects to multiple exchanges for stock trading. Can you also share a deep dive for the NFR above?

Show more

0

Reply

D

DistinctIndigoTurtle210

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmdensbce03bfad08lap30e4i)

"It's important that this is stored first because then we have a record of the client could. " -> I think this is a mistake

Show more

0

Reply

D

DistinctIndigoTurtle210

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmdeod6ds03g5ad08yvrccha7)

In the "Some additional deep dives you might consider" section in this deep dive, no indication/hint/summary is given about the right direction to go in. I noticed the same section in the Online Auction deep dive did have this information which I found very useful

Show more

0

Reply

S

SystematicBlueCoral547

[• 28 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmdkmyonp01v4ad08eaph5cf0)

Etrade is not an exchange, its just another broker like Robinhood. Nasdaq, NYSE, CBOE are exchanges.

Show more

0

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 20 days ago• edited 20 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmdvq3c6v0ki0ad08bjhsnwc7)

The order submission process and order cancellation process both have interaction with multiple components and services and have multiple failures scenarios to handle, to me these are good candidate for durable execution via engines like temporal to handle failures more gracefully.

Show more

0

Reply

E

ExactAmethystLark466

[• 18 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmdz5n1bd0429ad08txk8727h)

I was asked this question with an extension that we have 100 exchanges and we need to find the best match by the specified price and order type and handle the cases when exchanges can go down. Consider exchange latency into consideration as well when picking an exchange to execute order.

Show more

0

Reply

H

hoyt\_lin

[• 11 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cme9ox7cc0b9mad08v3ctqn78)

I am a new on System Design, just wonder if we can leverage SSE between Symbol Pricer Processor and Exchange as well?

Show more

0

Reply

P

PleasantAquamarineHoverfly685

[• 10 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmeaav3xg01yfad08xp2kupbs)

Evan, can you do a video on this as well. Thanks :)

Show more

0

Reply

A

ActualMagentaBuzzard104

[• 5 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmehkrlvf00udad07lcoijb1t)

Trade Processor and Dispatch Gateway can be merged into one. Usually with these systems there is an “Exchange Gateway” that is responsible for both dispatching and handling acknowledgements. It’s one TCP connection after all. You send them data, they reply.

Show more

0

Reply

U

UniformMoccasinLion191

[• 3 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/robinhood#comment-cmekimfvp012sad085jubicch)

I might be missing something here. But in the requirement we also need to support limit order, where we need to watch for the specific price change for placing an order to the exchange. I believe we need a real time watcher to kick off the buy/sell, which I think is the most challenging part of this design. But doesn't seem the solution is addressing this part.

Show more

0

Reply
