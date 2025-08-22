# Design Strava

[![Evan King](/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75&dpl=1a01e35ef00ef01b910d317b09313e145b78f47f)

Evan King

Ex-Meta Staff Engineer

](https://www.linkedin.com/in/evan-king-40072280/)

medium

Published Jul 12, 2024

---

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

## Understanding the Problem

**🏃‍♂️🚴‍♀️ What is [Strava](https://www.strava.com/)?** Strava is a fitness tracking application that allows users to record and share their physical activities, primarily focusing on running and cycling, with their network. It provides detailed analytics on performance, routes, and allows social interactions among users.

While Strava supports a wide variety of activities, we'll focus on running and cycling for this question.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements)

**Core Requirements**

1. Users should be able to start, pause, stop, and save their runs and rides.
    
2. While running or cycling, users should be able to view activity data, including route, distance, and time.
    
3. Users should be able to view details about their own completed activities as well as the activities of their friends.
    

**Below the Line (Out of Scope)**

- Adding or deleting friends (friend management).
    
- Authentication and authorization.
    
- Commenting or liking runs.
    

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements)

**Core Requirements**

1. The system should be highly available (availability >> consistency).
    
2. The app should function in remote areas without network connectivity.
    
3. The app should provide the athlete with accurate and up-to-date local statistics during the run/ride.
    
4. The system should scale to support 10 million concurrent activities.
    

**Below the Line (Out of Scope)**

- Compliance with data privacy regulations like GDPR.
    
- Advanced security measures
    

Here is how it might look on the whiteboard:

Strava Non-Functional Requirements

## The Set Up

### Planning the Approach

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

Let's begin by outlining the main entities of our system. At this point, we don't need to delve into every specific detail or column. We'll address the finer points, such as individual fields, once we have a better overall picture. For now, identifying these key entities will help shape our thinking and provide a solid foundation as we move towards defining the API.

To satisfy our key functional requirements, we'll identify the following core entities:

1. **User**: Represents a person using the app. Contains profile information and settings.
    
2. **Activity**: Represents an individual running or cycling activity. Includes activity type (run/ride), start time, end time, route data (GPS coordinates), distance, and duration.
    
3. **Route**: A collection of GPS coordinates recorded during the activity (this could also just be a field on the Activity entity).
    
4. **Friend**: Represents a connection between users for sharing activities (note: friend management is out of scope, but the concept is necessary for sharing).
    

In the actual interview, this can be as simple as a short list like this. Just make sure you talk through the entities with your interviewer to ensure you are on the same page.

Strava Entities

### [The API](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#4-api-or-system-interface)

We'll define the APIs that allow the mobile app to interact with the backend services.

Our goal is to simply go one-by-one through the core requirements and define the APIs that are necessary to satisfy them. Usually, these map 1:1 to the functional requirements, but there are times when multiple endpoints are needed to satisfy an individual functional requirement or where a single endpoint meets the needs of multiple functional requirements (as is the case with our activity state update endpoint below).

First, we'll define an endpoint that can be used to create a new activity.

`// Create a new activity POST /activities -> Activity {     type: "RUN" | "RIDE" }`

Next, we'll add an endpoint that can be used to update the state of that activity.

`// Update the state of an activity PATCH /activities/:activityId -> Activity {     state: "STARTED" | "PAUSED"  }`

We use PATCH here because we only want to update a subset of the activity's fields. Whereas PUT would require sending the entire resource, with PATCH we only send the fields that are being updated -- a, albeit minor, improvement in efficiency.

In your interview, PUT and PATCH are functionally equivalent. However, you will come across some interviewers who are RESTful API hawks and will argue this point with you, so it's best to be prepared.

When it comes to tracking activity data, we'll add a POST endpoint that is used to update the activity's route and distance.

`// Add a new location to the activity's route POST /activities/:activityId/routes -> Activity {     location: GPSCoordinate, }`

When it comes to sharing activities with friends, we'll say that this happens automatically when an activity is saved, so instead of a endpoint for sharing, we can just update our existing endpoint that updates the activity state to include a "COMPLETE" state.

`// Mark an activity as complete using the same endpoint we use to update the activity state PATCH /activities/:activityId -> Activity {     state: "COMPLETE" }`

To view activities, we'll need a list view that shows either our activities or the activities of our friends. This list will just show basic information about the activity, such as the distance, duration, and date.

`// Get a list of activities for a user GET /activities?mode={USER|FRIENDS}&page={page}&pageSize={pageSize} -> Partial<Activity>[]`

The Partial<Activity> type simply means that we are returning a subset of the activity fields. This is done to limit the amount of data we send over the network. This notation is what I use and is derived from TypeScript, but you can use whatever notation you'd like. This is just a personal preference.

When we click on an activity, we'll want to see more details about like the maps with the route and any additional metrics or details (that are out of scope for this question).

`// Get an activity by id GET /activities/:activityId -> Activity`

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes)

Let's address each functional requirement and design components to satisfy them. The best strategy here, especially if it's a question you've never seen before, is to start with the simplest possible system that satisfies the functional requirements and then layer on complexity to satisfy the non-functional requirements during your deep dives.

I often get asked, "If I already know the more complex design, should I still start simple?" The answer is nuanced. If you have a well-thought-out, more sophisticated solution in mind, it's perfectly acceptable to present it during the high-level design. That said, it's crucial to strike a balance. Going deep too early can result in you running out of time before you've met all the functional requirements and, thus, not answered the question.

### 1) Users should be able to start, pause, stop, and save their runs and rides.

We need to design a system that correctly handles both our create activity endpoint and our update activity state endpoint.

We can start by sketching out a very basic client-server architecture:

Users should be able to start, pause, stop, and save their runs and rides.

1. **Client App**: Users interact with the system through a mobile app.
    
2. **Activity Service**: Handles incoming requests for creating and updating activities.
    
3. **Database**: Stores information about activities, including route, distance, and time.
    

A user is going to start by opening the app and clicking the "Start Activity" button:

1. The client app will make a POST request to /activities to create a new activity, specifying whether it's a run or a ride.
    
2. The Activity Service will create the activity in the database and return the activity object to the client.
    
3. If the user opts to pause or resume their activity, they'll make a PATCH request to /activities/:activityId with the updated state and the Activity Service will update the activity in the database accordingly.
    
4. When the activity is over, the user will click the "Save Activity" button. This will trigger a PATCH request to /activities/:activityId with the state set to "COMPLETE".
    

One interesting part of the design is how to handle the activity time when a user pauses their activity. A naive approach to handling the amount of time elapsed in an activity would be to manage time based on a startTimeStamp and then, when an activity is ended, calculate the elapsed time by subtracting the startTimeStamp from the current time.

However, this approach would be problematic if a user pauses their activity. The startTimeStamp would remain unchanged and the elapsed time would continue to increment, which would be inaccurate.

One common way we could handle this, we can maintain time via a log of status update and timestamp pairs. For example, we could have the following log:

`[     { status: "STARTED", timestamp: "2021-01-01T00:00:00Z" },     { status: "PAUSED", timestamp: "2021-01-01T00:10:00Z" },     { status: "RESUMED", timestamp: "2021-01-01T00:15:00Z" },     { status: "STOPPED", timestamp: "2021-01-01T00:20:00Z" } ]`

When the user clicks "Stop Activity", we can calculate the elapsed time by summing the durations between each pair of timestamps, excluding pauses. In the example above, the elapsed time would be 15 minutes (10 minutes + 5 minutes).

This may seem like overkill for our simple requirement, but it would allow for natural expansions into a feature that shows athletes when they were paused and for how long, as well as a breakdown of "total time" vs "active time."

### 2) While running or cycling, users should be able to view activity data, including route, distance, and time.

When the activity is in progress we need to both update the state and display the activity data to the user.

The first, and most important, part of this is to accurately track the distance and route of the activity. We can do this by recording the user's GPS coordinates at a constant interval. We make the assumption that they are using a modern smartphone with GPS capabilities, so we can use the GPS coordinates to calculate the distance using the [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula).

While running or cycling, users should be able to view activity data, including route, distance, and time.

Here is how this would work:

1. The client app will record the user's GPS coordinates at a constant interval, let's say 2 seconds for a bike ride and 5 seconds for a run. To do this, we'll utilize the built-in location services provided by both iOS and Android:
    
    - For iOS: We'll use the Core Location framework, specifically the CLLocationManager class. We can set up location updates using startUpdatingLocation() method and implement the locationManager(\_:didUpdateLocations:) delegate method to receive location updates.
        
    - For Android: We'll use the Google Location Services API, part of Google Play Services. We can use the FusedLocationProviderClient class and call requestLocationUpdates() method to receive periodic location updates.
        
    
2. The client app will then send these new coordinates to our /activities/:activityId endpoint.
    
3. The Activity Service will update the Route table in the database for this activity with the new coordinates and the time the coordinates were recorded.
    
4. We will also update the distance field by calculating the distance between the new coordinate and the previous one using the Haversine formula.
    

You absolutely do not need to know the libraries and frameworks to use here to get GPS coordinates. The point of this question is to test your knowledge of system design, not mobile development. However, it is important to understand that most modern smartphones have built-in GPS capabilities and that most mobile apps use the relevant libraries and frameworks to access this data. I've just included them here to give you an idea of how it works in practice.

Astute readers are likely already yelling at their screen as they've realized that we don't need to send these updates to the server, instead handling the logic locally on the client. You're right. We'll get into this in the deep dive. The reality is most of the time I ask this question, candidates overlook this optimization and we get to it later in the interview.

### 3) Users should be able to view details about their own completed activities as well as the activities of their friends.

Once activities are completed, we should have them already saved in the database and all we needed to do was update their state to "COMPLETE" using the same endpoint we used to update the activity state.

When it comes to viewing activities, we only need to query the DB for the activities with this "COMPLETE" state, while also filtering on the mode query param to either show the user's activities or their friends' activities which can be done with a simple WHERE clause.

Example query:

`SELECT * FROM activities  WHERE state = "COMPLETE" AND userId === :userId -- if mode is USER AND userId IN (SELECT friendId FROM friends WHERE userId = :userId) -- if mode is FRIENDS LIMIT :pageSize OFFSET (:page - 1) * :pageSize`

Our friends table will have two columns, userId and friendId. We'll consider relationships to be bi-directional, so for each new friendship we will add two entries to the table, one with the userId and friendId and the other with the userId and friendId reversed. By making the first column the primary key, we can ensure querying for a user's friends is efficient.

This initial GET request returns a list of activities with just the basic information and the activityId for each activity. When the user clicks on an activity, we can make a second, more detailed GET request using the activityId to get the full activity details.

Users should be able to view details about their own completed activities as well as the activities of their friends.

The full flow is thus,

1. User navigates to the activities list page on the app.
    
2. Client makes a GET request to /activities?mode={USER|FRIENDS}&page={page}&pageSize={pageSize} to get the list of activities.
    
3. The list of activities is rendered in the UI and the user clicks on an activity to view details.
    
4. Client makes a GET request to /activities/:activityId to get the full activity details to render on a new details page.
    

For showing the map, we can use a combination of the route data and the Google Maps API. We can pass the array of coordinates to the Google Maps API and it will draw a line connecting all the points.

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes)

At this point, we have a basic, functioning system that satisfies the functional requirements. However, there are a number of areas we could dive deeper into to improve efficiency, offline functionality, scalability, and realtime-sharing.

This list of deep dives is, of course, not exhaustive. It is instead an accumulation of some of the most popular directions I've seen this interview take.

### 1) How can we support tracking activities while offline?

Many athletes will opt to workout in remote areas with no network connectivity, but they still want to be able to track their activities. This brings us to an impactful realization that re-shapes the entire system.

The key insight is that, so long as we don't support realtime-sharing of activities, we can record activity data locally, directly on the clients device and only sync activity data back to the server when the activity completes and/or the user is back online.

Importantly, this actually solves several problems for us:

1. We can track activities without worrying about network reliability on the client side.
    
2. Even when the client has a fine network connection, we save network bandwidth by avoiding pinging location updates to the server every few seconds (instead, only sending on run completion or pause).
    
3. Showing accurate and up-to-date activity data to the user is easy as it all happens synchronously on their device.
    

This question highlights an important, often overlooked aspect of system design: the client as an active participant. Modern smartphones and browsers have access to a wide variety of sensors, substantial memory and processing power, and significant local storage. Many candidates overlook the client's capabilities, focusing solely on server-side solutions. However, for systems like Strava, Dropbox, or Spotify, the client often plays a critical role in the overall design, enabling features like offline functionality, reducing server load, and improving user experience.

So how does this work in practice?

1. Just like in our high-level design, we use on-device location services to record the user's GPS coordinates at a constant interval.
    
2. We record this event data locally on the device in an in-memory buffer (e.g., an array of GPS coordinate and timestamp pairs).
    
3. To prevent data loss in case of device shutdown or battery depletion, we'll periodically persist this buffer to the device's local storage every ~10 seconds:
    
    - For iOS: We can use Core Data for larger, structured datasets, or UserDefaults for smaller amounts of simple key-value data.
        
    - For Android: We can use Room database for larger, structured datasets, or SharedPreferences for smaller amounts of simple key-value data.
        
    
4. When the app is reopened or the activity is resumed, we first check local storage for any saved data and load it into our in-memory buffer before continuing to record new data.
    
5. Once the activity is complete and the device is online, we send all the accumulated data to our server in a single request. For very long activities, we might need to implement a chunking strategy to handle large amounts of data efficiently.
    
6. We can also implement a background sync mechanism that attempts to upload data periodically when a connection becomes available, even if the activity isn't complete yet. This balances efficiency with data durability.
    
7. Upon confirmation that the data was saved to our remote database, we can delete the local buffer and are ready to go again.
    

This approach ensures that even if the device unexpectedly shuts down, we'll only lose a maximum of 10 seconds of activity data (or whatever interval we choose for our periodic saves). However, it's important to balance the frequency of GPS tracking and data persistence with battery life considerations.

How can we support tracking activities while offline?

The exact implementation details may vary depending on the mobile platform and the size of the data being stored. For larger datasets, using a local SQLite database might be more appropriate than options like UserDefaults or SharedPreferences. But the details of this are reserved more for Mobile System Design interviews, not a general SWE interview.

We're done with our first deep dive, and we still have a simple client-server architecture. Why? Especially when most of our other breakdowns use microservices? The truth is, there is no one right answer here. But in this case, we offloaded so much of the write load to the client that we don't have a ton for our backend to do. What we do have still needs to scale, and we will get to that in a bit, but we don't need separate microservices to scale independently given no significant read/write skew across different paths. It could be argued that this will make it harder for the engineering team to grow, with everyone touching the same codebase, but if Meta can famously manage it, so can we.

### 2) How can we scale to support 10 million concurrent activities?

Now that we track activities locally, scaling our backend became pretty easy! We cut down on the number of requests our backend services receive by a factor of 100x since we only send data once a run completes now. Our production engineering team is going to be thrilled, so too the finance team.

We do still have some potential bottlenecks that are worth addressing.

Let's start by looking at our database. With ~100M DAU doing an activity each day we add up to ~100M new activities each day. Over a year, that's ~36500M activities or ~36.5B activities.

To estimate the amount of storage we need for each activity, we can break it down as follows:

- Basic metadata like status, userID, start/end time, etc should be pretty cheap. Let's say ~100 bytes per activity.
    
- The route is the most expensive part. If the average activity is 30 minutes and we take GPS coordinates, on average, every ~3 seconds, then we'll have about 600 points per activity. Each route row needs a latitude and longitude field as well as a timestamp. This is an additional (8bytes + 8bytes + 8bytes) \* 600 = ~15KB.
    

15KB \* 36.5B = 547.5TB of data each year

This is a good amount of data, but it's far from unmanageable. Here are some things we could do to handle this:

1. We can shard our database to split the data across multiple database instances. We can shard by the time that the activity was completed since the majority of our queries will want to see recent activities.
    
2. We could introduce data tiering. The chances that someone wants to view a run from several years ago are pretty low. To reduce storage costs, we could move older data to cheaper storage tiers:
    
    - Hot data (recent activities) stays in fast, expensive storage
        
    - Warm data (3-12 months old) moves to slower, cheaper storage
        
    - Cold data (>1 year old) moves to archival storage like S3
        
    
3. We can introduce caching if needed. If we find that we are frequently querying the same activities, we can introduce caching to reduce the read load on our database. This would not be a priority for me, as read throughput should be pretty low. but its the first thing we would do if load times become unacceptable.
    

What database should we choose?

Hot take. It doesn't really matter. This is a large but totally manageable amount of data; there is no crazy read or write throughput, and data is relational but not heavily so. Realistically, all popular major database technologies would work great here.

For the Activity Service, given both read and write throughput should be pretty low, I see no reason to break this into microservices that scale independently. Instead, when we run into issues with memory, cpu, or network limitation we can just scale the service horizontally.

How can we scale to support 10 million concurrent activities?

### 3) How can we support realtime sharing of activities with friends?

One common follow up question in this interview is, "what happens when you want to allow friends to follow along with activities in real-times". In this case, friends don't just see activity statistics once completed, but they can watch you mid run/bike ride -- seeing your stats and routes update in near real-time.

This moves the design closer to other popular realtime systems like [FB Live Comments](https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-live-comments) or [Whatsapp](https://www.hellointerview.com/learn/system-design/problem-breakdowns/whatsapp).

To enable real-time sharing with friends, we'll reintroduce periodic server updates during activities. While maintaining local device logic for user-facing data, we'll send location updates to the server every 2-5 seconds. As the server gets these updates, they'll be persisted in the database and broadcast to all of the user's friends.

Now, I know what you're likely thinking, "Websockets!" This isn't wrong per se, you could absolutely implement this with websockets or SSE, but I'd strongly argue it's over-engineering the problem.

While you could introduce a real-time tracking service which connects to friends clients via Websocket or SSE and use pub-sub to broadcast updates to friends, this introduces a lot of unecessary complexity.

Instead, there are two key insights that suggest a simpler, polling mechanism will be more effective:

1. Updates are predictable: Unlike with Messenger or Live Comments, we know that the next update should come in the next 2-5 seconds. This predictability allows for efficient polling intervals.
    
2. Real-time precision isn't critical: Friends don't need to see up-to-the-second information. A slight delay of a few seconds is acceptable and won't significantly impact the user experience.
    

Informed by these insights, we can implement a simple polling mechanism where friends' clients periodically request updates at the same interval that the athlete's device is sending updates to the server (offset by a few seconds to account for latency).

We can further enhance the user experience by implementing a smart buffering system. This approach involves intentionally lagging the displayed location data by one or two update intervals (e.g., 5-10 seconds). By buffering the data, we can create a smoother, more continuous animation of the athlete's movement. This eliminates the jarring effect of sudden position changes that can occur with real-time updates. To friends viewing the activity, the athlete will appear to be in constant motion, creating a more engaging, "live-stream-like" experience. While this approach sacrifices absolute real-time accuracy, it provides a significantly improved visual experience that better matches users' expectations of how a live tracking feature should look and feel. The intentional lag also helps compensate for any network latency, ensuring a more consistent experience across different network conditions.

### 4) How can we expose a leaderboard of top athletes?

Another natural extension to this problem could be to expose a leader board of the top athletes by activity type and distance. We could filter by country, region, city, etc.

Here are some approaches we could take:

### 

Bad Solution: Naive Approach

##### Approach

The simplest thing we can do is to query the DB for all activities and then group them by user and sum the distance. We can assume we have some user table with basic information like name, email, country, etc.

This would look something like this:

`SELECT     u.name,     SUM(a.distance) as total_distance FROM activities a JOIN users u ON a.userId = u.userId WHERE a.state = "COMPLETE" AND a.type = "RUN" GROUP BY u.userId ORDER BY total_distance DESC`

Naturally, this query will be pretty slow given it is running an aggregation over millions of activities. There are several ways we can optimize this.

##### Challenges

This approach faces significant scalability issues. As the number of activities increases, query performance will degrade rapidly. The system would need to scan and aggregate millions of records for each leaderboard request, leading to high latency and increased database load. Additionally, this method doesn't account for the dynamic nature of leaderboards, where rankings can change frequently as new activities are logged.

### 

Good Solution: Periodic Aggregation

##### Approach

To improve on the naive approach, we can implement a periodic aggregation system. This involves creating a separate leaderboard table that stores pre-calculated totals for each user. A background job would run at regular intervals (e.g., daily) to update these aggregates based on new activities. The leaderboard query then becomes a simple SELECT from this pre-aggregated table, sorted by total distance.

Periodic Aggregation

##### Challenges

While this method significantly reduces query time for leaderboard requests, it introduces eventual consistency. The leaderboard won't reflect real-time changes, potentially leading to user confusion if their latest activity isn't immediately reflected in their ranking. In addition, determining the optimal frequency for updates can be tricky – too frequent updates could strain system resources, while infrequent updates lead to stale data.

### 

Great Solution: Real-time Leaderboard with Redis

##### Approach

For a more scalable and real-time solution, we can leverage [Redis](https://www.hellointerview.com/learn/system-design/deep-dives/redis), an in-memory data structure store. We'll use Redis Sorted Sets to maintain leaderboards, with user IDs as members and their total distances as scores. When a new activity is logged, we increment the user's score in the sorted set. To query the leaderboard, we simply retrieve the top N members from the sorted set, which Redis can do extremely efficiently.

What happens when we want to filter by country or time range?

To handle filtering by country or time range, we can extend our Redis-based approach:

1. For country filtering:
    
    - We can create separate sorted sets for each country, e.g., "leaderboard:run:USA", "leaderboard:run:Canada", etc.
        
    - When an activity is logged, we update both the global leaderboard and the country-specific leaderboard.
        
    - To query a country-specific leaderboard, we simply use the appropriate country-specific sorted set.
        
    

`# Add an activity to the leaderboard redis.zadd(f"leaderboard:run:global", {user_id: timestamp}) redis.zadd(f"leaderboard:run:USA", {user_id: timestamp}) # Query the leaderboard redis.zrange(f"leaderboard:run:global", 0, -1, desc=True)`

1. For time range filtering, we have a more sophisticated approach that combines both sorted sets and hashes:
    
    - We'll use a combination of Redis sorted sets and hashes to store both timestamp and distance information.
        
    - For each activity, we'll store:
        
        1. A sorted set entry with the activity ID as the member and timestamp as the score.
            
        2. A hash entry with the activity ID as the key, containing fields for user ID and distance.
            
        
    - When querying the leaderboard for a specific time range:
        
        1. Use ZRANGEBYSCORE to get activity IDs within the desired time range.
            
        2. For these activity IDs, retrieve the corresponding user IDs and distances from the hash.
            
        3. Aggregate the distances by user ID in-memory.
            
        4. Sort the aggregated results to produce the final leaderboard.
            
        
    

Here's a pseudo-code example of how this might work:

`# When logging a new activity activity_id = generate_unique_id() redis.zadd("activities:timestamps", {activity_id: timestamp}) redis.hset(f"activity:{activity_id}", mapping={"user_id": user_id, "distance": distance}) # Querying the leaderboard for last 24 hours now = current_timestamp() day_ago = now - 24*60*60 activity_ids = redis.zrangebyscore("activities:timestamps", day_ago, now) user_distances = {} for activity_id in activity_ids:     activity_data = redis.hgetall(f"activity:{activity_id}")     user_id = activity_data["user_id"]     distance = float(activity_data["distance"])     user_distances[user_id] = user_distances.get(user_id, 0) + distance # Sort users by total distance leaderboard = sorted(user_distances.items(), key=lambda x: x[1], reverse=True)`

This approach allows us to efficiently filter by time range and aggregate distances, providing a real-time leaderboard that can be customized for different time periods. We can cache the results with a short TTL to limit the in-memory aggregation while still ensuring freshness.

Strava System Design

##### Challenges

The main challenge with this approach is ensuring data consistency between Redis and our primary database. We need to implement a robust system to handle failures and retries when updating Redis. Additionally, we must consider Redis's memory limitations – storing complete leaderboard data for all possible combinations of activity types, time ranges, and geographical filters could consume significant memory. To mitigate this, we might need to implement a caching strategy where we only keep the most frequently accessed leaderboards in Redis and calculate less common ones on-demand.

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

I

ImplicitEmeraldVicuna183

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm2u0kjk4019kory2962cu7p3)

Thanks for this. Are you planning on adding auction system?

Show more

2

Reply

A

AppropriateGreenHoverfly664

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm2vsk47800m811onp9ddvdp7)

Auction system please.. I have my interview soon :)

Show more

1

Reply

![Aingkaran Jega](https://lh3.googleusercontent.com/a/ACg8ocLTK7lvICDZYKWWxIES4nJvNAD7NlMyTBa8NLyIxYCcfJqddA=s96-c)

Aingkaran Jega

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm2ytf0y800s51388rnd8lwcj)

for the last deep dive you have the leader board for the countries in the redis sorted set as : redis.zadd(f"leaderboard:run:USA", {user\_id: timestamp})

but shoudn't it be user\_id: score, I assume for this we're just making separate sets for each country so timestamp would be irrelevant here right?

Show more

13

Reply

Y

YouthfulBlueVole468

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm31cytyg00b6lbtu2py2nysk)

For top athletes leaderboard - shouldn’t we use Kafka solution from top k videos? - as we don’t need to sort everything

Show more

1

Reply

Y

YouthfulBlueVole468

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm31e22kx00av4pbcodnviuyp)

Also ‘ For time range filtering‘ - it’s not going to work well if range is large

Show more

0

Reply

![Eric Tsend](https://lh3.googleusercontent.com/a/ACg8ocKUIBTY_0Ek2qtHMDTb8l9PUrMADnlfCbfs7sFwzdSCTZ0L3b0=s96-c)

Eric Tsend

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm5pu90kq01eziuxn6e1qeyna)

Like specified in the problem, I don't think we need to look past 24 hours before the current time.

Show more

0

Reply

A

AppropriateGreenHoverfly664

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm33nbvf000ptlajc0ilbgi4p)

what database are we using dynamoDb?

Show more

0

Reply

![Aleksandr Kulikov](https://lh3.googleusercontent.com/a/ACg8ocLXuGeaSGLLNhNuxMekUk0VbrJk6nw5ACBHgJrmaPObMKoqcgCrgw=s96-c)

Aleksandr Kulikov

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm6dl2gsm03ob8twy2vl6yl6m)

The topic says "It doesn't really matter" but why? 10M concurrent activities is a huge amount of RPS. For example, if every active activity send updates to server every 5 sec it will be: 10M / 5 = 2M RPS straight to our DB.

Show more

0

Reply

![Chris Cichielo](https://lh3.googleusercontent.com/a/ACg8ocLbrf2s1ZKvqQIovKRk4xolrY7sEDZRnffdGH2qZ0AbJcPOe9jF=s96-c)

Chris Cichielo

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm6xxoqxs00q2homjqgh1ebyt)

If the updates are being calculated on the client ( phone / wearable ) and only sent when an activity is COMPLETED - then there is not updates every 5 seconds going to the server.

I see your point still though Aleksandr - worst case is 10M concurrent activities finishing at the same time but with 100M DAU with 1 activity per day.

A DynamoDB table can scale to 40K WCU - and with requests being 15KB that's 15WCU per update - that's ~2,500 writes per second - and ~1 hour to process all of that data. Depending on density of users in a given area this may or may not be an issue. In US just having a global table in 4 regions would bring this down to 15 minutes.

Show more

1

Reply

![Kamry Bowman](https://lh3.googleusercontent.com/a/ACg8ocJypclOxGpLvUXw3UNWZ8tOUqE7erFtecnq3iJB-jgRH1oANw=s96-c)

Kamry Bowman

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm7zep2t1035k5693lmt4qz1o)

Presumably only a fraction of the activities are being live streamed to friends.

Show more

0

Reply

A

AppropriateGreenHoverfly664

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm33nxmtf00pjium6cbr4hyhb)

If user1 is primaryKey in the friends table how will the user have more friends? Will it not be always just 1. I'm bit confused here

Show more

8

Reply

![Eric Tsend](https://lh3.googleusercontent.com/a/ACg8ocKUIBTY_0Ek2qtHMDTb8l9PUrMADnlfCbfs7sFwzdSCTZ0L3b0=s96-c)

Eric Tsend

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm5pub13u01f4iuxnf0ck69o7)

I thought it was duplicated for each relationship which seems redundant and costly, but shouldn't that take care of your concern here?

Show more

1

Reply

P

PremierPlumAmphibian934

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm3v27sbj014x9a0t6s98nvr9)

Hi Evan, In the Deep Dive "How can we scale to support 10 million concurrent activities?" you were discussing about database bottlenecks. I am bit confused. Below are couple of questions

1. How databased bottlenecks are related to supporting 10 million concurrent activities? What about Scaling of Activity Server to support these concurrent activities?
2. Does 10 Million Concurrent activities is Per Sec/Min/Hour. Can you please explain a bit?

Show more

1

Reply

![Eric Tsend](https://lh3.googleusercontent.com/a/ACg8ocKUIBTY_0Ek2qtHMDTb8l9PUrMADnlfCbfs7sFwzdSCTZ0L3b0=s96-c)

Eric Tsend

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm5pudswh01ct114yhx59449p)

1. I would guess the database is the primary bottleneck here since application service can always be replicated and scaled horizontally, while DB replication and sharding raises more concerns for data consistency and sync, also we need to remember that most of the data tracking is done on the client side, so all we need is to persist the data
2. By definition, "concurrent" must mean simultaneously which means down to the second in this case

Show more

0

Reply

S

StableCoffeeCheetah166

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm4553rox006f4dcrrvh0j2k0)

Hi, since we are using mobile device here, should the API be in graphQL instead of just normal REST API

Show more

0

Reply

P

pssharma1699

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm4fas33d03qfqfet7fwrrkzu)

Thanks for the explanation, I am little confused about the database choice here, lets say there are 100M DAU and 600 route related updates are being sent by per user per day that transforms to 60k updates/sec (also this is considering there is no peak load and every 3 seconds we will send route update to server). Handling this doesn't seem to be very straight forward. We would need some kind of write optimised database. This problem might be solved by using the batched approach where we are batching updates in client and sending at once to server (even if we sent it per 10 mins the write load is 1000 updates/sec) which is edge limit of postgres. Batching approach will not work in case we are planning to send these updates in realtime to friends as then we would need updates to be sent at regular interval.

One solution I can think of is:

1. Create a different route service (Rationale being that the traffic pattern is different between Activity and Route service) and use cassandra to store the routes which can handle the heavy load.

Show more

5

Reply

F

FancyTanGoldfish347

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm58wcv6r00zpe1nm8zdy9wjs)

I think this is quite a good point to dive into, especially when it come to real machine count + write ops benchmarks from different sources. Of courses there are many factors that play into this (all DBs can eventually scale to this traffic demand if cost and hardware was infinite), but DBs like Cassandra would be able to scale its # of write ops with fewer machines making it more efficient. https://www.linkedin.com/pulse/nosql-mongodb-vs-cassandra-shrutika-poyrekar

But, we don't really need to worry about such high QPS since it sounds like the design mentioned doing direct writes to Redis for live activities since it is considered the "hot data". We would need to consider a cadence for flushing this data to the slower DB we decide to choose. Using Cassandra doesn't seem necessary based on this design decision alone.

Show more

1

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmc4sc2m105kqad08shpv67gf)

Can cassandra serve for large read throughput or what can be a good approach? I feel we can even use Redis here to put the coordinate data as redis can also support such high throughput, and we can push the coordinate data to other persistent DB(such as postgres) in batches for the case of later view of route.

Show more

0

Reply

S

SecureTomatoLynx933

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm4i53r16024m8jlr7syr460y)

Why not partition Activity on the user\_id, and use a timestamp as the sort key? Wouldn't that allow for rapid retrieval of a user's activities, especially with a database such as DynamoDB?

Show more

0

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm4p4krxy01wop041kd9u4ait)

1. The Friend table can't have user1 as the PK, since user1 may have multiple friends. It needs to be a composite key (user1, user2).
2. In HLD 2.2, the coordinates should be sent to /activities/:activityId/routes as previously outlines during API Design.
3. In Deep Dive 3, filtering by time, what happens if the user does more than one activity in the last 24 hours? If the user id is the member, the sorted set wouldn't be able to handle that.

Show more

7

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm4p4r2e001zrjs6siqlvha48)

The last point is moot, the sorted set member is the activity id, which is unique. I got confused since the same section shows sorted set by user id.

Show more

0

Reply

H

HistoricalAzureCuckoo704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm546nr5l03q8r5q69vl5kd0t)

re 1. I think this also goes for the route table. Can't have the activity\_id as the PK, needs a route\_id as PK and possibly set the activity\_id as the Foregn key

Show more

1

Reply

K

kstarikov

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmaip21te00z1ad08ultswjzg)

> It needs to be a composite key (user1, user2)

This will make it difficult to run queries like 'get all friends of user1'.

Show more

0

Reply

![Thomas Bao](https://lh3.googleusercontent.com/a/ACg8ocIO8YpJG5G-78wQvloOEL_SE6DiycxQL-jz3084vXtrKG7BE-oj=s96-c)

Thomas Bao

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm4qqfzka00y3z3r0w0ojk6to)

The LLM gave feedback to use the Haversine formula but that is overkill to improve estimates by <10^-6 percentage points. The delta between straight line distance and haversine distance is tiny when the points are close relative to radius of earth which is ~6k km. The distance between gps points during a run/bike etc, would be miniscule. If two gps points are 1 km apart, the delta between haversine and straight line distance is < 1 micrometer

Show more

2

Reply

E

ExactSalmonMongoose135

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm7dw8zhk00yodbe082ij3ve1)

Kindly agree here, I don't think a backend engineer will know this formula. I was a little surprised when AI feedback told me this is the way to go.

Show more

0

Reply

F

FreshCrimsonTick387

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm4xmyhjv01oh116251c1xzyd)

For the time based leadership board,

Isn't this very slow? there might be millions of activities between specific timestamp. Getting all activities -> getting the distance for these activities -> aggregating to prepare the leaderboard

I am wondering this is going to be very slow?

Show more

0

Reply

F

FancyTanGoldfish347

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm4ywcvfh009gr5q6l8zpaj91)

For the second deep dive, how did we get a request decrease factor by 100x? I don't think I saw any mention of an average activity duration with requests being sent to the server at some interval. The only interval I see is every 10 seconds for flushing buffer data to local storage on the client for the first deep dive.

Show more

1

Reply

S

space59

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm5edi9s800m36zzausq5httu)

Can someone shed light on the idea of sharding by completion time? To look up an Activity by activityId, how would I determine which shard it's on? Perhaps the idea is that archived records would be sharded by completion time, which makes more sense to me.

Show more

1

Reply

![Asmita Negi](https://lh3.googleusercontent.com/a/ACg8ocKTeGtgePeqOwOr6Ldv39nBufTG7QSodEpv-DbCrtth1TYzk1uC=s96-c)

Asmita Negi

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm5std4vj0193qf8e5uzn66se)

For 10 Million concurrent users I was thinking solving the problem using redis pubsub approach like we did in Uber Because the load is too high. If we are sending location data every 10 seconds it still adds to 1Million location updates every seconds. What about the approach where clients send the location to their redis channel and then we have workers that are reading off that client and write all the locations in the channel to the db? Here we have 1 worker for multiple channels because we dont need db to have the most recent data so even if db write happens once in every minute it should be acceptable

Show more

1

Reply

E

EnthusiasticAzureBasilisk385

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm6v66my7002jzdjvzptai89a)

Why we can't use a high throughput DB like Cassandra for millions of concurrent writes?

Show more

0

Reply

I

InterimBlueDove307

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm630w3os021zhqkjse5vtu98)

The guided practice links to anchor that doesn't exist.

https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#4-how-will-you-efficiently-update-the-total-distance-that-is-displayed-to-the-user

Also, I've never heard of the formula for calculating distances. Could hand-waving in an interview and just saying "Euclidean distance for now, but can revisit more accurate formulas later if needed" work?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm6325xcr023shqkjdshv4ne7)

Yes. At these low of distances the error implied by the curvature of the earth is minimal.

Show more

1

Reply

![Spandan Pathak](https://lh3.googleusercontent.com/a/ACg8ocJ4mZkoGI0NnDWTsqKXBR7-x8i5U27JydTrixG75peiTw2Ma_7yuA=s96-c)

Spandan Pathak

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm8mmii53003wg02tdwnadczw)

Spotted a flat earther :)

Show more

1

Reply

F

FunctionalCopperWalrus129

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm66c62s3021jwntoxbv4fd67)

Thank you for the detailed explanation. Could you please add a breakdown of what's expected/preferable for each level?

I have noticed that you have a preference for Redis when it comes to key-value stores. Redis is certainly amazing, but when it comes to the interview, especially for senior and staff+ engineers, would it be more preferable to mention so many specific details of Redis or to keep it high level and just talk about the key value store (mentioning Redis as an example)?

Show more

0

Reply

I

IndividualGrayGecko945

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm67ei84000fjpuj07bbw9ra1)

why don't we use a graph database for friend relationships?

Show more

1

Reply

![Tommy Loalbo](https://lh3.googleusercontent.com/a/ACg8ocKJIn8OPXYOxiFFjMUkH5UDjWWCbOFuGt2Srsu9sGECWCgexFCq=s96-c)

Tommy Loalbo

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm6pc4rto02opv55hz72p37gx)

I don't see storing all points in the activity table being realistic. First of all we would have to update that entire record every time we want to write instead of just slapping on new coordinate (Which we could just compress post workout). But the main reason is that SQLs 8kb record limit gives us about ~511 points we could store. Storing a point every second, that gives about 1.4km! which is feel like is an unrealistically small number for fitness apps where people will be biking multiple miles. I am aware you could use postgre's TOAST feature, but i don't think that cuts it! Am i missing something?

Show more

0

Reply

![Spandan Pathak](https://lh3.googleusercontent.com/a/ACg8ocJ4mZkoGI0NnDWTsqKXBR7-x8i5U27JydTrixG75peiTw2Ma_7yuA=s96-c)

Spandan Pathak

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm8mmhfj2003sg02tqmp9howj)

Time to solve Run Length Encoding on leetcode :)

Show more

0

Reply

V

ValuablePeachTrout847

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm729w4cv00m0z4df3zre30mn)

There is type with "w"hat" in "One common follow up question in this interview is, w"hat happens when you want to allow friends to follow"

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm72a90pu00ns100m122y9kzz)

Thanks! Fixing next release

Show more

0

Reply

![Walid Dib](https://lh3.googleusercontent.com/a/ACg8ocKyS6PyCrvAt6jagAlRPc2kw6EK5eAJ42vSIqN-f0oxzgZusQ=s96-c)

Walid Dib

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm7cb2bu3009uab60vbymlj0r)

random Q: can I just casually pull one of these bad boys in an interview if my estimations are weird?

https://m.media-amazon.com/images/I/71rYNyUn8LL._AC\_SL1500_.jpg

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm7cb9xq001ph19rh2ugcxpc8)

I'd actually recommend you reduce the precision of your estimates. It's kludgy to carry around 5045.4k TPS or worse when you have a lot of zeros ("500000000. Now how many zeroes was that?")

But if you show a literal casio calculator I'm sure your interviewer isn't going to think you're typing into ChatGPT on the side :)

Show more

1

Reply

![Aditya Jain](https://lh3.googleusercontent.com/a/ACg8ocJjHh-eky22nfbV7YSOwPct6ROk615alDtDMGafjdNJV1gQcQ=s96-c)

Aditya Jain

[• 5 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm7h0m73401gwimqn6poacu2m)

Wouldnt it be wise to shard the database by activity\_id (instead of time) , because it may become hot shard if many users are online at the same time? I think we should shard by activity id and then partition it by time.

Show more

5

Reply

F

FinancialGreenTick391

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm8yd1vfq00kiad081la7mwtu)

On one hand, It feels like "the majority of our queries will want to see recent activities" is a good reason not to shard by timestamp, since many queries will hit a specific shard. On the other hand, when fetching the (chronological) feed, we don't want to hit multiple shards.

Show more

0

Reply

L

LiquidCrimsonGoat313

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm7rc4s7y000j2d763oby2846)

In deep dive 3 "How can we support realtime sharing of activities with friends?"

The activities from clients are sent every 2-3secs and with 10M concurrent activities these would results in 3-5M RPS. And similarly all friends are polling the data every 5 secs and this get traffic would also result in 2M RPS.

Expecting some buffer for spikes would single DB be able to handle that much traffic of around 10M RPS in both read and write or am I missing something?

Show more

0

Reply

![Julian Pineda](https://lh3.googleusercontent.com/a/ACg8ocKrOubKiSqCIaxZpgaanDD8PcKvk_QZfniwO8o02_d6Di72Hw=s96-c)

Julian Pineda

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm7xin4wy00k75693an6vpqe5)

Maybe out-of-scope observation: while using screenreaders or apps like Speechify to listen to this, some text is not read (Below the Line (Out of Scope), Core Requirements). When you are listening and reading it at the same time, it feels like you are missing info and/or losing context that may be really important. A quick fix on those using proper HTML is a huge win for users using screen readers or read-out-loud apps like Speechify.

Show more

0

Reply

![Yash Shukla](https://lh3.googleusercontent.com/a/ACg8ocKlbomzbIWDTFkGY-SJEEdy4jwqoSpK0iVpB1ARZZl4qLzVnQ=s96-c)

Yash Shukla

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm8ay116g004311zxabahq9ic)

I

Show more

0

Reply

V

VocationalMoccasinHippopotamus822

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm8q6aqj2015bfg39qaq6e2s6)

Could the device's in memory buffer be used for the whole activity that's in progress, adding route points as they are collected, and then push the activity to the db when it is completed? This would preserve the whole route if the phone shuts down unexpectedly or there's no service, support pausing, and decrease writes to the db (you'd only need to save an activity once). I realize this would not allow for tracking friends' activities in progress, but could work for the base functional requirements.

Also, for viewing a user's friends' completed activities in a feed, would it be overkill to cache friend's activities (with a short TTL for availability > consistency) in a redis hash (user\_id: activity\_ids).

Show more

0

Reply

E

ExoticPeachOpossum454

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm9217g6f00h7ad08uu6m311r)

Would you mind making a video for this design?

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm9218fyi00qiad07tm07pgim)

Soon! Slowly adding vids for all premium content

Show more

3

Reply

N

NuclearAmethystStork225

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm9e0jccm00jlad0897geowh4)

Hey Evan, please kindly make the written article for free for premium content, it will be helpful to use

Show more

0

Reply

R

RapidAquaTuna163

[• 4 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm9pr7qza02i2ad0852fbir81)

In the other HelloInterview articles, we have learnt not to trust the client. In this case, we are trusting the activity data sent by the client. For eg, the timestamps of the activity can be fudged. How do we evaluate which approach to take (trusting the client or sending real-time updates to the server)?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm9t9oxx600hvad08puxej7ft)

In this case the client is all we have. They are the source of truth. Now you're right that we would not want a client to say they ran a mile in 2 minutes. We could have some basic server side validation to "sniff out" funky looking data from clients.

Show more

2

Reply

P

PreliminaryScarletToucan490

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cm9qlyuqm001uad079k5nnyhl)

Why do we have a status column and a startTime column if we have a statusUpdateEvents column in the Activity table that holds both of these pieces of information?

Show more

0

Reply

A

ArchitecturalPlumReptile288

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmabkkciz01eyad08q1t08g27)

A simple question: Do we need to write REST API in SWE system design? I am asking because i never learn REST from my experience.

Show more

0

Reply

H

HandsomeIvoryCrow799

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmaesi277006rad08atumkr5g)

I remember that I posted this comment the other day but not sure why it's lost now.

Why don't we store route as a field for activity record? This way, we don't need to handle transaction.

Show more

0

Reply

M

MildOliveAngelfish918

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmawse2rn01iyad086aake7ej)

How would we handle potentially 10 million concurrent writes if all users ping the server with an update at the same time? Especially in the case of live activities (we don't just send the final route).

That seems like a major design bifurcation (supporting live events vs not). Or would we argue that live feature is opt-in, and so the number of users would be far less than 10 million?

Show more

0

Reply

K

kstarikov

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmb2qapi500mtad08llafuyf8)

The realtime sharing of an activity isn't in the requirements or the API or the design itself. I think it should either be in all three, or not mentioned at all.

Show more

0

Reply

![Hemant Khurana](https://lh3.googleusercontent.com/a/ACg8ocKBFGoJgC0uZoxuIgimqRxYaW2Se8F_CA2LneGQH0Z8OuiasA=s96-c)

Hemant Khurana

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmbi334ho00lnad09lgtug9y2)

There is a routeId in the activity table but the primary key in Route table is activityId? Where are getting the routeId from? Is the id in Activity table and activityId in Route table the same key?

Show more

0

Reply

![Vankshu Bansal](https://lh3.googleusercontent.com/a/ACg8ocJs_boSzlEWXXmgwq72Mj8aYHqsOX0Vwi6X5pQP2JG1YWf6Rw=s96-c)

Vankshu Bansal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmbp07nf001z308ada9qagtgw)

'We can shard by the time that the activity was completed since the majority of our queries will want to see recent activities'

Shouldn't we be sharding by something which we know won't cause hot shards? I think we can shard on userId or activityId, rather than the time.

Show more

0

Reply

![Vankshu Bansal](https://lh3.googleusercontent.com/a/ACg8ocJs_boSzlEWXXmgwq72Mj8aYHqsOX0Vwi6X5pQP2JG1YWf6Rw=s96-c)

Vankshu Bansal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmbp1boye01zl08adv1i2yzxl)

'We can cache the results with a short TTL to limit the in-memory aggregation while still ensuring freshness'

Why do we need TTL? Why can't we keep say 2k entries and throw out the rest?

Show more

0

Reply

M

ModestBlackJackal289

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmbzx91gj034208advsn3k3jt)

As an avid Strava user, I think the requirements completely miss how strava is actually used. 95% of users are uploading data via a watch apple/garmin/coros. Hardly anyone uses the strava app to monitor their run/cycle in progress. Most of Strava is analyzing past efforts and and calculating personal bests and as called out in the final deep dive - route leaderboard.

Another interesting deep dive missed is finding new routes in new cities / locations. Users search for popular routes and even view Strava generated heat maps.

Show more

0

Reply

G

GothicCopperPanther728

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmcia15pm0chxad086a3ltwe9)

Wouldn’t sharding by activity completion time be a bad idea since the newest activities would have the most reads, leading to very uneven partition load?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmcia3iuv000mad082kypvvw0)

The high read load on recent activities is exactly why we want time-based sharding, it naturally separates hot data (recent activities that get frequent reads) from cold data (old activities that rarely get accessed). Plus, most queries are time-based (e.g., "show me activities from last week"), so having time-based shards makes these queries more efficient by limiting the number of shards we need to hit. Solving for hot reads is easy, we can add a cache.

Show more

1

Reply

![Uday Naik](https://lh3.googleusercontent.com/a/ACg8ocICzn0414Dtx8UumksG29Uf1wGCi1RpuKRpDVNjtL7zSe0cBw=s96-c)

Uday Naik

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmcjgwu6v00v7ad0827xyr5h1)

I would add a Q for asynchronous event write to DB. This write can be every 10 - 15 seconds. The user is getting realtime info on his device already so no need to rush. This ensures a smooth user experience and provides scaling.

Then server combines event info with other info on the database and pushes a dashboard to the user over pub/sub every 15 - 30 seconds

example:

1. user gets step count, location, distance travelled already. without the server.
2. the system takes the activity info (step count / pace etc) and calculates e.g. calories burnt, ideal heart rate etc. from the server
3. this info is put on pub/sub. Reader reads it off the topic and pushes it via SSE or long polling
4. user dashboard has real time step count info and a consolidated view refresh every 15-30 seconds

Show more

0

Reply

B

BlushingRoseBoar963

[• 3 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/strava#comment-cmehi078b052aad08q56o8e9e)

I was a little frustrated by the practice for this problem. I immediately thought that we should track all the activity data on the client and only post an activity on completion. However, the feedback for the API section said I had it all wrong and needed more APIs to handle start/stop etc.  
I ended up quitting the practice halfway through the high-level design and just reading the solution, which was what I had in the first place :).

Show more

0

Reply
