design view count

track video views
display video views

300m active users
don't double the count
user can refesh the page count once

view count must be fetched within 200ms
while video streaming

eventual consisteny 100% withing 10 mins
availability with 99.99% SLA

First crack

[counter](counter.drawio.png)

API

POST /videos/videoId/views
{
    userId, timestamp
}

GET /videos/videoId/views
{
    videoId
}
{
    videoId:
    viewCOunt:
    lastUpdated:
}

# Single DB

200e6 * 5 videos/day = 1e9 updates/day

1e9/1e5 = 1e4 requests/s

peak 50k to 100k RPS

- DB cannot handle this load (~10k typical for mysql)
- DB is single point of failure
- DB 1. 

# Sharding
1. View count sharded across instances
1. database_id = hash(video_id)%10

# Hot partition problem
1. also shard viral videos
1. but read increases in latency

# Handling idempotency
Create idempotence_id = video_id+user_id+timestamp

[

Write



](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------)

[

](https://medium.com/search?source=post_page---top_nav_layout_nav-----------------------------------------)

[Sign up](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fitnext.io%2Fscaling-distributed-counters-designing-a-view-count-system-for-100k-rps-0567f6804900&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

[Sign in](https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fitnext.io%2Fscaling-distributed-counters-designing-a-view-count-system-for-100k-rps-0567f6804900&source=post_page---top_nav_layout_nav-----------------------global_nav------------------)

![](https://miro.medium.com/v2/resize:fill:64:64/1*dmbNkD5D-u45r44go_cf0g.png)

[

## 

ITNEXT



](https://itnext.io/?source=post_page---publication_nav-5b301f10ddcd-0567f6804900---------------------------------------)

·

[Follow publication](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fsubscribe%2Fcollection%2Fitnext&operation=register&redirect=https%3A%2F%2Fitnext.io%2Fscaling-distributed-counters-designing-a-view-count-system-for-100k-rps-0567f6804900&collection=ITNEXT&collectionId=5b301f10ddcd&source=post_page---publication_nav-5b301f10ddcd-0567f6804900---------------------publication_nav------------------)

[

![ITNEXT](https://miro.medium.com/v2/resize:fill:76:76/1*yAqDFIFA5F_NXalOJKz4TA.png)



](https://itnext.io/?source=post_page---post_publication_sidebar-5b301f10ddcd-0567f6804900---------------------------------------)

ITNEXT is a platform for IT developers & software engineers to share knowledge, connect, collaborate, learn and experience next-gen technologies.

[Follow publication](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fsubscribe%2Fcollection%2Fitnext&operation=register&redirect=https%3A%2F%2Fitnext.io%2Fscaling-distributed-counters-designing-a-view-count-system-for-100k-rps-0567f6804900&collection=ITNEXT&collectionId=5b301f10ddcd&source=post_page---post_publication_sidebar-5b301f10ddcd-0567f6804900---------------------post_publication_sidebar------------------)

You're reading for free via [Animesh Gaitonde's](https://animeshgaitonde.medium.com/?source=post_page-----0567f6804900---------------------------------------) Friend Link. [Become a member](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fitnext.io%2Fscaling-distributed-counters-designing-a-view-count-system-for-100k-rps-0567f6804900&source=-----0567f6804900---------------------post_friend_link_meter------------------) to access the best of Medium.

Member-only story

# Scaling Distributed Counters: Designing a View Count System for 100K+ RPS

## Architecture, challenges and bottlenecks in counting views at scale

[

![Animesh Gaitonde](https://miro.medium.com/v2/resize:fill:42:42/0*jDWGFWcAZRiyW3TG.)





](https://animeshgaitonde.medium.com/?source=post_page---byline--0567f6804900---------------------------------------)

[Animesh Gaitonde](https://animeshgaitonde.medium.com/?source=post_page---byline--0567f6804900---------------------------------------)

Follow

11 min read

·

2 days ago

[

](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fvote%2Fitnext%2F0567f6804900&operation=register&redirect=https%3A%2F%2Fitnext.io%2Fscaling-distributed-counters-designing-a-view-count-system-for-100k-rps-0567f6804900&user=Animesh+Gaitonde&userId=307ef0382b4b&source=---header_actions--0567f6804900---------------------clap_footer------------------)

46

[](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fbookmark%2Fp%2F0567f6804900&operation=register&redirect=https%3A%2F%2Fitnext.io%2Fscaling-distributed-counters-designing-a-view-count-system-for-100k-rps-0567f6804900&source=---header_actions--0567f6804900---------------------bookmark_footer------------------)

[

Listen









](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3D0567f6804900&operation=register&redirect=https%3A%2F%2Fitnext.io%2Fscaling-distributed-counters-designing-a-view-count-system-for-100k-rps-0567f6804900&source=---header_actions--0567f6804900---------------------post_audio_button------------------)

Share

# Introduction

The Youtube video Baby Shark Dance has got **16,086,737,725** views world wide. It’s one of the most watched videos in the world. 🚀

You can read the free version of the article [here](https://animeshgaitonde.medium.com/0567f6804900?sk=8b054f6b9dbcf36086ce2951b0085140).

It’s followed by others such as Despacito, Wheels on the bus, etc, that have got billion views. Have you wondered how Youtube stores and displays these accurate view counts? 🤔

At the surface-level, it seems like a simple addition problem. Keep a counter, increment it for every view and then show it.

However, the problem is far more complex than you think. The system must handle billion views daily with traffic spikes without compromising the accuracy.

In this article, we will build a view counting system from the ground up. We will start with the most basic solution and refine it to meet our goals. The article will go over the pros/cons of each approach and explain the trade-offs.

By the end, you will learn the core principles used while designing a view counting system. Also, you will be able to level up your system design skills, architect scalable systems and also crack system design interviews.

With that, let’s get started and understand the problem statement.

# Problem Statement

Here’s a one-liner for the problem:

_Design a functionality to display the view counts in a video streaming service like Youtube._

Assume that the video streaming service already exists, and we’ll focus only on the video count feature.

Next, let’s break the problem down into functional and non-functional requirements.

# Functional Requirements

Let’s formalize what the system must do. For the scope of this problem, the system must:

- **Track video views** (**FR-1**)— The system must track how many times a video is viewed by the users.
- **Display video views** (**FR-2**)— While the video is streaming, the system should display its view count.

# Non-Functional Requirements

We will now define the non-functional requirements in the context of each functional requirement.

## Track video views (**FR-1)**

- **Scalability** — The system must handle the scale of **300 Mn** active users. It must also handle viral videos that show a sudden spike in the views.
- **Idempotency** — The system shouldn’t double count the views. A user may refresh the page multiple times but system must only count the view once.
- **Performance** — The performance of the existing system shouldn’t be impacted due to the new functionality.

## Display video views (FR-2)

- **Performance** — The view count must be fetched within **200 ms** while video streaming.
- **Eventual Accuracy** — The view count must be 100% accurate but may lag by a few minutes (**< 10 mins**). In other words, the counts may be eventually accurate.
- **Availability** — The system must show the view count with **99.99%** SLA.

> Food for thought: Why do you think view the count accuracy is so important? Can’t websites like Youtube just show approximate counts? _🤔 (Leave your thoughts in the comments)_

Now that we are clear with both the functional and non-functional requirements, let’s move on and design the system from scratch.

# Entities and Actors

The system comprises of the following entities:-

1. **Users** — The users would watch the video and fetch the view count.
2. **Videos** — Every video would have an associated view count attribute.

So far, the system looks simple with only two entities. Let’s now understand how these entities would interact with each other.

# APIs and interfaces

We will now define how each functional requirement would be met through APIs.

## Track video views (FR-1)

**Track Video API**

**Track video request**

**Track Video request**

**Track video response**

**Track Video response**

## Display video views (FR-2)

**Display video view API**

**Request parameters:**

- `videoId` (Path param): ID of the video.

**Video count response**

**Display video view response**

Now that the APIs are defined, let’s now sketch the high-level flow showing the interaction between the different components.

# High-Level Design

At a high-level, we would need the following two data flows:-

- **Write path** — It would capture the user view and increment the view count.
- **Read path** — It would fetch the view count and display it to the users.

Every time a video is viewed, the system captures the view and displays the current view count. Hence, the read-write ratio would be 1:1.

We can achieve the above through:-

1. **Video Count service —** A service that would expose the APIs for incrementing and fetching the view count.
2. **Video Count data store** — A data store that manages the view counts.

The diagram below illustrates the high-level design along with the different flows.

![](https://miro.medium.com/v2/resize:fit:924/1*Y7C7_0HaEUnySwG8RWm8Og.png)

**Video Counting System**

Now that we have a good high-level understanding, let’s build a solution that meets our requirements.

We will start with a basic approach, understand its drawbacks, iterate and then improve the solution. So, let’s get started.

# Single database storing view count per row

We’ll start with a single service backed by a database as described in the previous section. Let’s choose a relational database for now.

The below diagram shows the data model and the architecture.

![](https://miro.medium.com/v2/resize:fit:924/1*wBtdVYpFp0j_YjaZzc6hPA.png)

**Video View data model**

![](https://miro.medium.com/v2/resize:fit:1320/1*NXdxl1JABjuu1Dk08eHUaA.png)

**Single database architecture**

Would the above solution work for 300 Mn users? Let’s find out through calculations.

## Calculations

Assume that 200 Mn users are active out of 300 Mn and each user watches 5 videos daily. So, total number of read/write requests sent to the system daily becomes 200 Mn \* 5 = 1000 **Mn**.

_Average requests per second=(Total number of requests)/(number of seconds in a day)_

Average requests per second = 1000 Mn / 86,400 ~ **10,000**.

**10K** RPS is average, and the peak can be anywhere between **50K** and **100K** RPS. 😱

What would happen at **100K RPS**? We will horizontally scale our service to handle the increasing load. But, would our single instance database scale? 🤔

The answer is **NO**.

Here are some drawbacks of this approach:

- **Single point of failure** — The system’s availability would be compromised if the database crashes.
- **Database throughput** — Relational databases such as PostgreSQL or MySQL aren’t optimized to handle **100K** RPS throughput. Requests would fail impacting the availablity and accuracy.
- **Performance** — Due to database throttling, the performance would be impacted and requests would take a long time (more than **1 sec**)to complete.
- **Duplicate views** — The system only stores and increments the counts. It’s not designed to handle duplicate views. Hence, it might add duplicate views and show a higher view count.

> **Note**_: We will face similar challenges with non-relational database if it substitutes the relational database. Non-relational databases also have an upper bound on the number of requests. Further, the costs increase with additional load._

In this approach, the database is the primary bottleneck. So, let’s now think of ways to overcome the bottleneck.

# View count sharded across database instances

Instead of a single database instance, we would use several database instances and shard the data across the different instances.

The below diagram illustrates the working:

![](https://miro.medium.com/v2/resize:fit:1320/1*5YdnTypzgYhomcejNPkPyQ.png)

**Video Counter with sharded database**

If a single database can handle **10K** RPS and we receive **100K** RPS, then we would 10 such instances.

The solution addresses the drawbacks of the previous approach. But do you think the solution would work always?

> The best way to identify bottlenecks is to evaluate the solution against all non-functional requirements and edge cases.

What would happen if a video suddenly becomes viral?

We would have a surge in the read/write requests. Since a video maps to a single database shard, one of the shards would see a spike. This is known as **hot partition problem**.

The approach fails for viral videos — if many go viral across different shards, all shards get impacted.

We can still overcome the hot partition problem if we store the viral video on multiple shards. The service can then uniformly distribute the count for the video across the shards.

This is explained in the following diagram:

![](https://miro.medium.com/v2/resize:fit:924/1*q0py0aquzpEcJt_79kAeSQ.png)

**Video counter hot partition sharding**

However, now the read path would have to aggregate the data from multiple database shards. This would impact the latency to fetch the view counts.

We tried scaling the reads/writes through sharding and it resulted in great improvements. However, we haven’t still solved to handle duplicate views. Let’s now refine the solution further.

# In-memory aggregation followed by persistence

So far, the previous two approaches didn’t address a critical requirement of **Idempotency** or handling duplicate views. Let’s now address it

## Idempotency

We need to uniquely identify a view event within a specific deduplication window (e.g. 1 hour or 1 day). Any repeated events from the same source shouldn’t be double counted.

The server can create an idempotency key to uniquely identify such events. The following code shows the logic:

**Idempotence key generation logic**

We can introduce a cache to store the idempotency keys for the videos. The keys can have a 1 hr or 1 day TTL, so that they get evicted.

Now that we have solved for unique views, let’s see how we can tackle the view count aggregation.

## View count aggregation

In the previous approaches, we were directly updating the count in the database. This led to challenges such as exceeding database throughput and hot partitions.

Do we always need to perform database writes? What if we aggregate the view counts in memory and then perform a batch write?

Aggregating the counts in the memory would reduce the database load. We can then periodically (every **10 mins**) update the database.

Further, a cache can be introduced to store and fetch the view counts. We can set **10 mins** as the TTL on each **_video\_id_** since the data needs to be eventually accurate (refer the Non-functional requirements).

The diagram below shows how idempotency and aggregation can be tackled.

![](https://miro.medium.com/v2/resize:fit:1320/1*0jBmwcYDRKTqNlUNUGqcMA.png)

In-memory aggregation followed by persistence

This approach addresses most of the pain-points of the previous approaches. Here’s how it achieves the goals:

- Reduces the overall database load by buffering the writes.
- Reads perform a simple cache lookup. In case of a cache miss, the data is fetched from the database.
- Guarantees no duplicate views through idempotency.
- The view counts would be eventually accurate after **10 mins**.

Where do you think would the system fail?

> Plan for failure — it’s a hallmark of great system design.

Since we are aggregating the writes in-memory, what if the service instance goes down?

We would lose all the temporary aggregated counts in case a service crashes. Since this is a data loss, we won’t get the writes back and the view count would be inaccurate.

While we have improved the solution, there are still scenarios where the solution would fail.

Let’s again iterate and solve for the failure scenarios.

# Event-driven architecture with Kafka and Stream processing

As observed in the previous approach, durability of the data is important. It helps to avoid duplicate views and also show accurate view count.

We need a system that does the following:-

1. Supports high write throughput at low-latency.
2. Guarantees 100% data durability.
3. Performs aggregation for different time windows.

Kafka is best suited to solve for 1, and 2. For 3, we can use a stream processing platform like Apache Flink.

The below diagram shows how the system would work:

![](https://miro.medium.com/v2/resize:fit:1320/1*rquB7HNonvcUFO1IH5aiBQ.png)

**Event-driven architecture with Kafka and Stream processing**

The data would be partitioned on the **_video\_id_** and would be distributed across the different Kafka partitions.

Flink would aggregate the data at **10** **mins** interval and then update the database. It would use Redis to dedup the events and perform a lookup based on the idempotence key.

Moreover, we can add consumers to read all events and store them in a time-series database or another data store. This enables event auditability.

However, the solution still has some drawbacks:-

- **Potential delays** — Viral videos would lead to a surge in the Kafka events and consumers might slow down. The video count shown to the users may be stale.

So, how do we solve this problem? We can use either of the following two approaches:

1. **Show the stale view count** — Engineers can discuss with the product manager whether it’s acceptable to show stale count for the viral videos. However, the surge in the events may also impact view counts of non-viral videos.
2. **Dedicated Kafka cluster** — Create a dedicated Kafka cluster for viral events. Add logic to detect viral events and then direct them to this cluster. However, this increases both the cost and the complexity.

Out of the above two, you would select the one that aligns with the product’s vision and reduces the engineering complexity.

Additionally, it’s a challenge to provision and manage the Kafka cluster due to its distributed nature and configuration. With the increase in the throughput, the system would require rebalancing of the topic partitions.

The additional complexity is an acceptable trade-off given that the design meets all the requirements.

# Conclusion

Designing a view count feature at Youtube scale isn’t a simple addition problem. The system must solve for accuracy, scalability, durability and duplicate view counts.

In this article, we started with a basic solution, iterated and ended up solving it through an event-driven system based on Kafka and Flink.

Here’s how the solution addresses the different challenges:-

1. **Scalability** — Kafka handled high throughput at low latency. Aggregation in the Flink layer reduced the overall database load.
2. **Durability** — Kafka avoided data loss through persistence.
3. **Eventual accuracy** — The aggregation through Flink may lead to delays but given the **10 min** window aggregation, the view counts would be updated within **10 mins**.
4. **Idempotency** — Event persistence ensures Kafka consumer (Flink) could perform deduplication and the system could avoid duplicate views.
5. **Performance** — The read path was kept simple with a lookup on either Cache/Database ensuring < **200 ms** latency.

We traded off Kafka’s operational complexity to meet our requirements. An additional cluster dedicated for viral videos would also help in scaling.

The diagram below maps the different approaches to the initial stated non-functional requirements and shows how we logically progressed to the final solution.

![](https://miro.medium.com/v2/resize:fit:1320/1*T5fFOsO2hnKda2ygCNfpXg.png)

**Approach vs. Non-Functional Requirement Mapping**

The approach described in this article can be applied to any other software design problem. Here are some key learnings:-

- Always start with a basic solution.
- Identify the bottlenecks by checking if the solution meets all the non-functional requirements.
- Resolve each bottleneck and understand the drawbacks of any new solution.
- Finalize a solution that meets all your requirements and completes the design.

Do you think the same system can we leveraged to handle other counting use cases such as Web page views, ad clicks, social media likes, etc. If not, what would be the best solution? Leave your thoughts in the comments below.

Before you go:

- 👏 for the story and follow me for more such articles
- Subscribe to my free engineering newsletter [here](https://engineeringatscale.substack.com/)
- 🔔 Follow me: [LinkedIn](https://www.linkedin.com/in/animesh-gaitonde/), [Twitter](https://twitter.com/animesh3436), [Medium](https://medium.com/@animeshgaitonde)

[

Software Development

](https://medium.com/tag/software-development?source=post_page-----0567f6804900---------------------------------------)

[

Software Engineering

](https://medium.com/tag/software-engineering?source=post_page-----0567f6804900---------------------------------------)

[

System Design Interview

](https://medium.com/tag/system-design-interview?source=post_page-----0567f6804900---------------------------------------)