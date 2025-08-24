# Numbers to Know

Learn about the numbers you need to know for system design interviews.

* * *

Our industry moves fast. The hardware we build systems on evolves constantly, which means even recent textbooks can become outdated quickly. A book published just a few years ago might be teaching patterns that still make sense, but quoting numbers that are off by orders of magnitude.

One of the biggest giveaways that a candidate has book knowledge but no hands-on experience during a system design interview is when they rely on outdated hardware constraints. They do scale calculations using numbers from 2015 (or even 2020!) that dramatically underestimate what modern systems can handle. You'll hear concerns about database sizes, memory limits, and storage costs that made sense then, but would lead to significantly over-engineered systems today.

This isn't the candidate's fault – they're doing the right thing by studying. But understanding modern hardware capabilities is crucial for making good system design decisions. When to shard a database, whether to cache aggressively, how to handle large objects – these choices all depend on having an accurate sense of what today's hardware can handle.

Let's look at the numbers that actually matter in 2025.

## Modern Hardware Limits

Modern servers pack serious computing power. An AWS [M6i.32xlarge](https://aws.amazon.com/ec2/instance-types/m6i/) comes with 512 GiB of memory and 128 vCPUs for general workloads. Memory-optimized instances go further: the [X1e.32xlarge](https://aws.amazon.com/ec2/instance-types/x1e/) provides 4 TB of RAM, while the [U-24tb1.metal](https://aws.amazon.com/blogs/aws/ec2-high-memory-update-new-18-tb-and-24-tb-instances/) reaches 24 TB of RAM. This shift matters because many applications that once required distributed systems can now run on a single machine.

Storage capacity has seen similar growth. Modern instances like AWS's [i3en.24xlarge](https://aws.amazon.com/ec2/instance-types/i3en/) provide 60 TB of local SSD storage. If you need more, the [D3en.12xlarge](https://aws.amazon.com/ec2/instance-types/d3/) offers 336 TB of HDD storage for data-heavy workloads. Object storage like [S3](https://aws.amazon.com/s3/) is effectively unlimited, handling petabyte-scale deployments as a standard practice. The days of storage being a primary constraint are largely behind us.

Network capabilities haven't stagnated either. Within a datacenter, 10 Gbps is standard, with high-performance instances supporting up to 20 Gbps. Cross-region bandwidth typically ranges from 100 Mbps to 1 Gbps. Latency remains predictable: 1-2ms within a region, and 50-150ms cross-region. This consistent performance allows for reliable distributed system design.

These aren't just incremental improvements – they represent a step change in what's possible. When textbooks talk about splitting databases at 100GB or avoiding large objects in memory, they're working from outdated constraints. The hardware running our systems today would have been unimaginable a decade ago, and these capabilities fundamentally change how we approach system design.

## Applying These Numbers in System Design Interviews

Let's look at how these numbers impact specific components and the decisions we make when designing systems in an interview.

### Caching

In-memory caches have grown exponentially in both size and capability. Gone are the days of 32-64GB Redis instances that required careful memory management and partial dataset caching. Today's caches routinely handle terabyte-scale datasets with single-digit millisecond latency, and a single instance can process hundreds of thousands of operations per second. This shift in scale changes the entire approach to caching strategy.

Numbers to know:

-   Memory: Up to 1TB on memory-optimized instances, with some configurations exceeding this for specialized use cases
    
-   Latency
    
    -   Reads: < 1ms within the same region
        
    -   Writes: 1-2ms average cross-region for optimized systems
        
    
-   Throughput
    
    -   Reads: Over 100k requests/second per instance for in-memory caches like ElastiCache Redis on modern Graviton-based nodes
        
    -   Writes: Sustained throughput of hundreds of thousands of requests per second
        
    

When to consider sharding:

-   Dataset Size: Approaching 1TB in size
    
-   Throughput: Sustained throughput of 100k+ ops/second
    
-   Read Latency: Requirements below 0.5ms consistently (if being exceeded, consider sharding)
    

These capabilities fundamentally change caching strategy. The ability to cache entire databases in memory, even at hundreds of gigabytes, means you can often avoid complex partial caching schemes altogether. This "cache everything" approach, while seemingly brute force, typically costs less than engineering time spent on selective caching logic. When you do need to scale, the bottleneck is usually operations per second or network bandwidth, not memory size – a counterintuitive shift from just a few years ago.

### Databases

The raw power of modern databases surprises even experienced engineers. Single PostgreSQL or MySQL instances now routinely handle dozens of terabytes of data while maintaining millisecond-level response times. This isn't just about storage either. Modern databases efficiently handle tens of thousands of transactions per second on a single primary, with the bottleneck often being operational concerns rather than performance limits.

Numbers to know:

-   Storage: Single instances handle up to 64 TiB (terabytes) for most database engines, with Aurora supporting up to 128 TiB in some configurations
    
-   Latency
    
    -   Reads: 1-5ms for cached data, 5-30ms for disk (optimized configurations for RDS and Aurora)
        
    -   Writes: 5-15ms for commit latency (for single-node, high-performance setups)
        
    
-   Throughput
    
    -   Reads: Up to 50k TPS in single-node configurations on Aurora and RDS
        
    -   Writes: 10-20k TPS in single-node configurations on Aurora and RDS
        
    
-   Connections: 5-20k concurrent connections, depending on database and instance type
    

When to consider sharding:

-   Dataset Size: Approaching or exceeding 50 TiB may require sharding or distributed solutions
    
-   Write Throughput: Consistently exceeding 10k TPS indicates scaling considerations
    
-   Read Latency: Requirements below 5ms for uncached data may necessitate optimization
    
-   Geographic Distribution: Cross-region replication or distribution needs
    
-   Backup/Recovery: Backup windows that stretch into hours or become operationally impractical
    

While the largest systems in the world (social networks, e-commerce giants, etc.) absolutely need sharding to handle their scale, many candidates jump to distributed solutions too early. For systems handling millions or even tens of millions of users, a well-tuned single database can often handle the load. When you do need to scale, carefully consider what's driving the decision: is it pure data volume, operational concerns like backup windows, or the need for geographic distribution? Understanding these tradeoffs leads to better scaling decisions.

More often then not I see candidates reaching for scaling too quickly. They have 500GB or a couple of terabytes of data and they're start explaining how they'd shard the database. Slow down, do the math, and make sure sharding is actually needed before you start explaining how you'd do it.

### Application Servers

Modern application servers have evolved beyond the resource constraints that shaped many traditional design patterns. Today's servers routinely handle thousands of concurrent connections with modest resource usage, while cloud platforms enable near-instant scaling in response to load. CPU processing power, rather than memory or connection limits, typically determines your server's capabilities.

Numbers to know:

-   Connections: 100k+ concurrent connections per instance for optimized configurations
    
-   CPU: 8-64 cores
    
-   Memory: 64-512GB standard, up to 2TB available for high-memory instances
    
-   Network: Up to 25 Gbps bandwidth in modern server configurations
    
-   Startup Time: 30-60 seconds for containerized apps
    

When to consider sharding:

-   CPU Utilization: Consistently above 70-80%
    
-   Response Latency: Exceeding SLA or critical thresholds
    
-   Memory Usage: Trending above 70-80%
    
-   Network Bandwidth: Approaching 20 Gbps
    

The implications for system design are significant. While the trend toward stateless services is valuable for scaling, don't forget that each server has substantial memory available. Local caching, in-memory computations, and session handling can all leverage this memory to improve performance dramatically. CPU is almost always your first bottleneck, not memory, so don't shy away from memory-intensive optimizations when they make sense. When you do need to scale, cloud platforms can spin up new instances in seconds, making aggressive auto-scaling a viable alternative to over-provisioning. This combination of powerful individual instances and rapid scaling means you can often achieve high performance through simple architectures.

### Message Queues

Message queues have transformed from simple task delegation systems into high-performance data highways. Modern systems like Kafka process millions of messages per second with single-digit millisecond latency, while maintaining weeks or months of data. This combination of speed and durability has expanded their role far beyond traditional async processing.

Numbers to know:

-   Throughput: Up to 1 million messages/second per broker in modern configurations
    
-   Latency: 1-5ms end-to-end within a region for optimized setups
    
-   Message Size: 1KB-10MB efficiently handled
    
-   Storage: Up to 50TB per broker in advanced configurations
    
-   Retention: Weeks to months of data, depending on disk capacity and configuration
    

When to consider sharding:

-   Throughput: Nearing 800k messages/second per broker
    
-   Partition Count: Approaching 200k per cluster
    
-   Consumer Lag: Consistently growing, impacting real-time processing
    
-   Cross-Region Replication: If geographic redundancy is required
    

The performance characteristics of modern queues challenge traditional system design assumptions. With consistent sub-5ms latencies, you can now use queues within synchronous request flows—getting the benefits of reliable delivery and decoupling without forcing APIs to be async. This speed, combined with practically unlimited storage, means queues can serve as the backbone for event sourcing, real-time analytics, and data integration patterns that previously required specialized systems.

## Cheat Sheet

Here is a one-stop-shop for the numbers you need to know in 2025. These numbers represent typical values for well-tuned systems with specific workloads - your requirements may vary based on workload, hardware, and configuration. Use them as a starting point for capacity planning and system design discussions, not as hard limits. Remember that cloud providers regularly update their offerings, so while I'll try to keep this up to date, it should be treated more as a starting point than a hard limit.

Component

Key Metrics

Scale Triggers

**Caching**

\- ~1 millisecond latency  
\- 100k+ operations/second  
\- Memory-bound (up to 1TB)

\- Hit rate < 80%  
\- Latency > 1ms  
\- Memory usage > 80%  
\- Cache churn/thrashing

**Databases**

\- Up to 50k transactions/second  
\- Sub-5ms read latency (cached)  
\- 64 TiB+ storage capacity

\- Write throughput > 10k TPS  
\- Read latency > 5ms uncached  
\- Geographic distribution needs

**App Servers**

\- 100k+ concurrent connections  
\- 8-64 cores @ 2-4 GHz  
\- 64-512GB RAM standard, up to 2TB

\- CPU > 70% utilization  
\- Response latency > SLA  
\- Connections near 100k/instance  
\- Memory > 80%

**Message Queues**

\- Up to 1 million msgs/sec per broker  
\- Sub-5ms end-to-end latency  
\- Up to 50TB storage

\- Throughput near 800k msgs/sec  
\- Partition count ~200k per cluster  
\- Growing consumer lag

## Common Mistakes In Interviews

### Premature sharding

The single biggest mistake I see candidates make is assuming sharding is always necessary. They introduce a data model and immediately explain which column they'd shard on. It comes up almost every time with [Design Yelp](https://hellointerview.com/learn/system-design/problem-breakdowns/yelp) in particular. Here we have 10M businesses, each of which is roughly 1kb of data. This is 10M \* 1kb = 10GB of data! 10x it to account for reviews which we can store in the same database and you're only at 100GB, why would you shard?

The same thing comes up a lot with caches. Take a [LeetCode](https://hellointerview.com/learn/system-design/problem-breakdowns/leetcode) leaderboard where we have 100k competitions and up to 100k users per competition. We're looking at 100k \* 100k \* 36b ID + 4b float rating = 400GB. While even more than what we store on disk with Yelp, this can still fit on a single large cache -- no need to shard!

### Overestimating latency

I see this most with SSDs. Candidates tend to vastly overestimate the latency additional to query an SSD (Database) for a simple key or row lookup. We're talking 10ms or so. It's fast! Candidates will oftentimes justify adding a caching layer to reduce latency when the simple row lookup is already fast enough -- no need to add additional infrastructure.

Note, this is only for simple row lookups with an index. It is still wise to cache expensive queries.

### Over-engineering given a high write throughput

Similar to the above, incorrect estimates routinely lead to over-engineering. Imagine we have a system with 5k writes per second. Candidates will often jump to adding a message queue to buffer this "high" write throughput. But they don't need to!

Let's put this in perspective. A well-tuned Postgres instance with simple writes can handle 20k+ writes per second. What actually limits write capacity are things like complex transactions spanning multiple tables, write amplification from excessive indexes, writes that trigger expensive cascading updates, or heavy concurrent reads competing with writes. If you're just inserting rows or doing simple updates with proper indexes, there's no need for complex queueing systems at 5k WPS.

Message queues become valuable when you need guaranteed delivery in case of downstream failures, event sourcing patterns, handling write spikes above 50k+ WPS, or decoupling producers from consumers. But they add complexity and should be justified by actual requirements. Before reaching for a message queue, consider simpler optimizations like batch writes, optimizing your schema and indexes, using connection pooling effectively, or using async commits for non-critical writes.

The core point is to understand your actual write patterns and requirements before adding infrastructure complexity. Modern databases are incredibly capable, and simple solutions often perform better than you might expect.

## Conclusion

Modern hardware capabilities have fundamentally changed the calculus of system design. While distributed systems and horizontal scaling remain necessary for the world's largest applications, many systems can be significantly simpler than what traditional wisdom suggests.

Understanding these numbers helps you make better scaling decisions:

-   Single databases can handle terabytes of data
    
-   Caches can hold entire datasets in memory
    
-   Message queues are fast enough for synchronous flows (as long as there is no backlog!)
    
-   Application servers have enough memory for significant local optimization
    

The key insight isn't that vertical scaling is always the answer – it's knowing where the real limits are. This knowledge helps you avoid premature optimization and build simpler systems that can grow with your needs. In system design interviews, demonstrating this understanding shows that you can balance theoretical knowledge with practical experience – a crucial skill, especially for the more senior levels.

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

B

BiologicalMoccasinTahr305

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm60jcu4100uw3lincp94qj7u)

Can you give an example of where you would use message queues for synchronous flows?

Show more

5

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm613fsu5016h3lin2go2bbm2)

Its still not very popular, but showing up more and more. Check this out: https://www.rabbitmq.com/tutorials/tutorial-six-python

Show more

6

Reply

![Harika Reddy](https://lh3.googleusercontent.com/a/ACg8ocJKCewnSnBVS4eu7us6BUYCA1Cb_RreTnqilDSQzyjizsH6fidjTg=s96-c)

Harika Reddy

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm65ya57d00t2a4uuelb3qzlp)

I currently use these RPC blocking calls in my product for interservice communication

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm65yb27601nmtbmru8rlbvp4)

Nice!

Show more

0

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 10 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmeateq75022rad08fk91vogs)

https://medium.com/@sinha.k/system-design-using-asynchronous-pipelines-to-handle-synchronous-flows-19c498de9cc6

Show more

0

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 10 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmeaqzeqr019yad08sjuosdpi)

I'd say it is not very uncommon when write throughput is way higher than server can handle but the processing may require an external service. Writing to db won't make sense in this scenario if external service needs to poll. The two can be decoupled using a RPC queue where external service can poll the queue and write response to a reply-queue.

Show more

0

Reply

VB

vishal bajoria

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm610t6e80120wppf05dpeaz9)

Thank you for this excellent article! I truly appreciate the effort in putting this together. I have a question regarding modern hardware capabilities and caching best practices. With modern RAM capacities reaching up to 24 TB, why is it recommended to limit Redis cache to 1 TB and shard when memory usage approaches 80% of this limit? Should we consider leveraging higher RAM capacities for Redis, or are there specific constraints that justify the 1 TB recommendation?

Show more

8

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm610x5uw011l1ds5xohefebu)

No strict rule here, but large instances become harder to back up, replicate, or recover from failures

Show more

15

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm610yfgl014d3linp43kc91o)

Also remember for Redis specifically that it's single-threaded! You'll probably be CPU-bound first. Other implementations can help with this (e.g. Dragonfly) but otherwise you'll need to shard anyways to increase CPU utilization.

Show more

20

Reply

VB

vishal bajoria

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm6124538007umy5kfqsrul3a)

thank you. it makes sense.

Show more

1

Reply

B

BigGoldStoat775

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm61krfp600ktcf2adtqbn84s)

Thank you so much, a much-needed article. Provides a lot of explicit context for the decisions that more experienced people intuitively make.

Show more

5

Reply

C

CuddlyCopperWildebeest704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm61nkbas00o6cf2a9hfv7yr3)

If an application server nowadays can handle 100k+ connections, why should we consider sharding at only Connection Count: Nearing 15k per instance?

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm61nm13j00mghqkjhqgxzzbu)

Because I made an edit to the number of connections and forgot to update the scaling suggestion :) fixing

Show more

7

Reply

N

NeutralCrimsonCapybara540

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm6y3qqpx00z0cv3wysx65qmx)

It still says 15k connection as scaling trigger?

Show more

1

Reply

C

CapableEmeraldPiranha843

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7jpbmvw002huknb7a93nnxn)

Then what would the correct number be?

Show more

0

Reply

P

PositiveEmeraldParakeet596

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7kndbd9008o4scglk972vzf)

15K is still there

Show more

0

Reply

C

CuddlyCopperWildebeest704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm61o3j6y00k0u73xz6bgitsw)

"Modern systems like Kafka process millions of messages per second with single-digit millisecond latency, while maintaining weeks or months of data. "

This is very interesting. I was actually thinking about it in the context of top k videos where throughput is 700k/s

Assuming 1Mb/message \*1M messages/s = 1TB per second 1TB/s \* 30days/month \* 100,000s/day = 3 000 000TB/month That would mean Kafka for this scenario would require 3000PB \* 2(replication x2 => follower and leader) = 6000 petabytes of storage? This would require a lot of brokers, wouldn't it? Or am I missing something

Show more

1

Reply

![Yaroslav Shevchuk](https://lh3.googleusercontent.com/a/ACg8ocKbcdEeJJXs5j0egEstxH3q40OiMhooT5wF3gCcxc58xvENMehTkw=s96-c)

Yaroslav Shevchuk

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7kv6yrx00jx4scgbxwv43fd)

Why would you want to post 1Mb messages?

Show more

4

Reply

C

CuddlyCopperWildebeest704

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7l76jgp010g8gbta88n5aw2)

this is upper bound since best practice for Kafka best practice recommends max size of msg to be 1MB.

Show more

0

Reply

Z

ZealousFuchsiaLeopon345

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9d2563i006mad08dwvoknnr)

But your message size won't actually be 1MB (10^6 bytes), would it?

Show more

0

Reply

T

toby

[• 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmddjypor01jbad09ajcvv6nn)

1TB per second over the network on a single machine? Good luck. Kafka may handle 100M messages/sec at maybe 1KB per message or less, but anything beyond is into some pretty serious networking speeds. AWS currently has a c6in.32xl which can handle 200GB networking (I think by bonding cards), but that seems pretty specialised.

Show more

1

Reply

C

CuddlyCopperWildebeest704

[• 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmddk41ms01e2ad08q3o1bqjs)

good point! did not think about networking limit. thank you!

Show more

0

Reply

G

GleamingChocolateChinchilla112

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm622u61f00ychqkjvh1rfm3e)

How do you stay up-to-date on these numbers or ideas? Where do you learn from?

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm62ehkx4012su73xe6jx0a2l)

An easy way is to see what the big cloud providers are offering in terms of hardware and when they make updates.

Show more

10

Reply

![Nikhil Desai](https://lh3.googleusercontent.com/a/ACg8ocL7PtE3QPnse_xzmT_iPOk3OAN4lZ7G9m2_nnSyWsfWFehtdbtD=s96-c)

Nikhil Desai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm62ehcmg017mcf2au0kyey3r)

Just to get your thoughts: Do you think some of the design decisions might change if done in a start up Vs big companies where there is some luxury of money. Meaning since some of the high end machines contributes to more opex, will horizontal scaling (with added operational and application complexities) make sense there?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm62eiwdt012wu73xfu7oe6kc)

For sure. Cost is a meaningful factor not discussed in this deep dive. But the reality is, while it's nice to acknowledge cost in an interview, it plays far less of a practical role in your decision-making in an interview setting than it would in real life.

Show more

5

Reply

A

aditya.bhardwaja

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm62wekdi01rgu73x116vvvky)

Excellent article! I can relate to this from my current work experience. About five years ago, we developed certain features to optimize caching under the assumption that our instances couldn't handle data > 200GB , spent lot of effort in eviction strategies and hot/cold reload. However, those features have become irrelevant due to modern hardware limits

Show more

8

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm675scqo000a6dn8awcmx4fx)

It's wild how quickly things change, right!

Show more

3

Reply

T

ThoughtfulPeachGull276

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm633pefr024yhqkjwglkxzh3)

Regarding read throughput for in-memory caching, the 100k RPS for ElastiCache Redis on Graviton seems low. Doesn't ElastiCache for Redis enable something like a million RPS with 7.1?

Genuinely just trying to understand. AWS' post from 2023 states that "for example on r7g.4xlarge, you can achieve over 1 million requests per second (RPS) per node, and 500M RPS per cluster."

Is the 100k RPS posted here on a base Graviton node?

Links or references to sources for many of the numbers posted would really help in diving deeper into them/building a stronger mental model.

Thanks, guys! Love all the content!

Show more

7

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm6706wi7001al9zcpjek0sdo)

They can get up to 1M with the right conditions and small value sizes. I would not necessarily optimize for 1M in an interview, I'd stick in the 100k order of magnitude, but you're not wrong. Agree on references. Can try to bolster

Show more

3

Reply

B

BreezyAmberFlamingo112

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm641bkta00sv110mr55rrz10)

Do those databases numbers apply to NoSQL DBs like Dynamo / Cassandra? Or are they primarily for relational DBs like PostGres and MySQL?

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm65yck9j01nttbmra6chj518)

More relational. For managed services like DDB the number like this are abstracted away to RCU and WCU (read and write capacity numbers). For Cassandra, you could handle higher write throughput on the same hardware

Show more

3

Reply

N

NetChocolateBovid916

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7efpw9t00wikwlw9if0qtyq)

I really like how you expressed the "Scale Triggers", where we need to start doing sharding. Similarly, it would be good to have guidance on when increased write throughput should lead us to use Cassandra. (aside from the cross-region aspects). When write throughput increases, we can either add shards, or start using Cassandra for data where query patterns are predictable and transactions are not needed? I suppose it would be more cost-effective to use Cassandra in some cases ? (e.g, much less reads than writes?)

Show more

3

Reply

![Kevin Do](https://lh3.googleusercontent.com/a/AGNmyxbmeu4bkpdBIf6Bv4bu4LF5IIGaehhapyYf3_N0I8o=s96-c)

Kevin Do

[• 7 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm649hksm0199c9i58mdemyqs)

Can you expand on application servers handling 100k+ concurrent connections? Does this mostly apply to quick stateless requests like HTTP request to GET/POST data?

What about persistent long-lived connections like WebSocket? Do we have a benchmark for how many unique user websocket connections a M6i can hold?

Show more

2

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm83h1p19016soh5f8yz3v58v)

Nothing offhand. If someone wants to run this test I'd happily cover the AWS costs.

Show more

2

Reply

C

CuddlyCopperWildebeest704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm64jwwyn01s3c9i5v9jldkan)

Could you also add section for numbers to know for pub/sub (Redis pub/sub), AWS SQS, Flink?

Show more

10

Reply

E

ExtraAmaranthDuck389

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmat5p6b000niad09h85a4jo5)

@evan, could you please? Thank you!

Show more

0

Reply

O

OfficialCoralRoundworm778

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm66z6n3700b9cfoc796ishb2)

"Write throughout - 10-20k TPS in single-node configurations on Aurora and RDS". So would adding more nodes linearly scale the throughout? Can we use such a system in scenarios like Ad-click aggregator instead of a stream processor? The scale required there was 10k clicks per second.

Show more

0

Reply

![E Z](https://lh3.googleusercontent.com/a/ACg8ocIvzX0SuEb-25SaNWiD-Ye0PAgT4B_Bjg2gbwo6kHyq995G5U8=s96-c)

E Z

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm69n494w023db3ex0skpyl70)

great article!

For the caching section, the statement below Throughput Reads: Over 100k requests/second per instance for in-memory caches like ElastiCache Redis on modern Graviton-based nodes Writes: Sustained throughput of hundreds of thousands of requests per second

hundreds of thousands of requests per second is like 100k/second. So read and writes have similar numbers, does it make sense?

Show more

1

Reply

B

BackHarlequinMarten245

[• 7 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm69qjq5x027gl4pe9xegqras)

Thanks for the excellent content! Just curious, what is the maximum QPS a modern application server can handle if the query is simple and doesn't require heavy operations?

Show more

0

Reply

U

UniversalBronzeTakin563

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm6la5v6r0375lhflxuhf1f8b)

can you please provide numbers for total users and DAU and MAU for most of the systems you spoke about too at one place

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm6la8ypa038qjgz329ndq2pz)

You can typically just ask your interviewer. But for estimation of MAU

1.  Biggest companies (meta, google): 1-3B
2.  Big companies (Uber, Dropbox): 100-900M
3.  Pretty Big (Yelp): 50-100M

Show more

4

Reply

R

ryanvu87

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm6pnl4lp035go9xsl5bqxf4n)

Would it be possible to provide estimates on latency due to network hops between microservices.

E.g.

Service A <--> Service B Service A <--> Service B <--> Service C

How much additional time does Service A have to wait when Service C is added to the network chain?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm6pnmrq1036841j3nurzd2mh)

-   Single hop within same availability zone (AZ): 0.5-1ms
-   Single hop across AZs in same region: 1-2ms
-   Single hop across regions: 50-150ms (varies by geographic distance)

Show more

9

Reply

I

ImmenseMaroonIguana707

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm6q99l2o0465ob0khlu5do5m)

Very useful article. Thanks a lot for putting these numbers together. Could you please also add the references to where you have obtained each number from?

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm70p08gr049f8xn7gtwd75tw)

Check the hyperlinks in the "Modern Hardware Limits" section

Show more

0

Reply

K

karzhouandrew

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7273j4p00gbti7oxvoj7j8f)

Thanks a lot for such an amazing article. It's really helpful. I would appreciate if you could elaborate a bit more on "operational concerns" from "Modern databases efficiently handle tens of thousands of transactions per second on a single primary, with the bottleneck often being operational concerns rather than performance limits." What do you include to "operational concerns" and what limitations do we have here?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm727xc8m00ieihpg1b7mfg78)

Operational concerns include backup windows, maintenance downtime, data migration complexity, and disaster recovery planning. As databases grow, these tasks become more time-consuming and risky. For example, a 50TB database might take hours to back up, impacting performance. Upgrades and schema changes become nerve-wracking operations. Recovery time objectives (RTO) stretch beyond acceptable limits. These issues often force sharding or distribution before raw performance does. It's about manageability, not just throughput.

Show more

6

Reply

H

HappyAquamarineVole837

[• 6 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm758tspv024galw2hfwbtme7)

Can you elaborate on this:

"Reads: 1-5ms for cached data, 5-30ms for disk (optimized configurations for RDS and Aurora)"

vs. in the cheat sheet, one of the db scale triggers is: "Read latency > 5ms uncached".

So even though the lower bound is usually 5ms for uncached reads (disk), it's a scale trigger when we reach let's say 6 or 7ms? Is that premature? Especially given that this says optimized configurations for RDS/Aurora?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7jbhuef04n8tygeivggyqk4)

You're right to question this. The 5ms threshold in the cheat sheet is overly aggressive. For most systems, 5-30ms for uncached reads is perfectly acceptable. Scaling should be considered when latency consistently exceeds 30ms or impacts application performance. The cheat sheet oversimplifies; real-world decisions depend on specific workload patterns, SLAs, and cost considerations. Good catch on the inconsistency.

Show more

2

Reply

H

HappyAquamarineVole837

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7jpv0l700298rk2cvc2sanf)

Ok great. Thanks for the clarification, this really helped lock my understanding.

Show more

0

Reply

![Robert](https://lh3.googleusercontent.com/a/ACg8ocLzJOwmklmw_sU9vcF1R-hRXkUuvy4eEAaU4mc12lsZ0oK7Idw=s96-c)

Robert

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7az4omq007g6mfll50hbngh)

I really like this type of article, learned a lot about storage limits. Thank you!

Show more

1

Reply

W

WeeOrangeLeopon734

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7dp6tsg00p9gd7z6ibzr43j)

How would you approach an interviewer that is "stuck" on antiquated upper limits?

I'm thinking that you could say something like:

"Ok, after 3 years of writes we'll be approaching 30TB of data. I believe that while that used to enough where we should consider a NoSQL database for ease of horizontal sharding, modern database servers are offering 64TB of storage. We are interested in doing some joins on the data so a PostgreSQL database will work nicely."

Phrasing it like this as a development I believe shows that you are evolving your understanding of System Design as the landscape evolves and brings your interviewer along with you.

Maybe I'm over thinking things.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7dp9ril00okvkj1h4m4cg8y)

Seems like a good idea! Always good to watch the interviewer as you're talking. If they grimace or cringe it's a good signal that you may need to show them more evidence.

Show more

3

Reply

S

SecureTomatoLynx933

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7i9sud003c0xcugy6b7hasg)

Can you please include references/links for these numbers where possible?

Show more

0

Reply

M

MentalAquaScallop449

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7ib7o4v03fpxcug7p8rmcf8)

This is excellent starting point, but I could not associate it in terms of cost considerations. Maybe higher configuration hardware often comes with higher costs and may not be the best choice. Replication as well may pose challenge with vertical scaling instance, log file writes may pose IO challenges. I am not sure about how to answer these questions by reading these numbers independently.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7ihpwym03t3tygeqdta6eis)

Few system design interviews are going to be directly considering cost. This is different than the discipline of system design in general.

Your interviewer might scoff if, for instance, you inefficiently use a resource which is known to be expensive — but there is rarely an expectation that you're tabulating dollars and using those as a basis for comparison.

In short: you should aim to make efficient use of resources, but probably don't need to be overly concerned with the cost dimension _for interviews_.

Show more

3

Reply

B

BeneficialBrownBarnacle904

[• 6 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7icki8r03hhtygeblavy81h)

Thanks for the article! Could you please elaborate on sharding message queues based on partition count: "Approaching 200K per cluster" ? Is this suggesting a single broker can host upto 200K partitions?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7jbgjwz04m8xcugrujeaiyn)

Nope, that's not quite right. The 200K partition limit is typically for the entire Kafka cluster, not a single broker. It's a practical limit based on factors like ZooKeeper's ability to handle metadata, broker resource consumption, and overall cluster stability. A single broker can handle thousands of partitions, but nowhere near 200K. The exact number depends on hardware, but 4000-10000 partitions per broker is more typical. When you approach cluster-wide partition limits, it's time to consider scaling strategies like adding brokers or even creating separate Kafka clusters.

Show more

2

Reply

Z

ZealousFuchsiaLeopon345

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9d2bspv006uad08u0c45ugg)

How will adding more brokers help, if 200k is the cluster wide limit?

Show more

0

Reply

R

RightAzureSnail808

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm7zyxne903rge77fk0rvw8o4)

To clarify my understanding, those numbers relate mostly to performance and throughput concerns. So while I might not need more servers because of sharding, I would still want extra servers for fault tolerancy and high availability. Is that right?

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm83grvce017810qjaomdt7y8)

Definitely. You'll have some aspect of horizontal scaling regardless.

Show more

1

Reply

![Amulya Sharma](https://lh3.googleusercontent.com/a/ACg8ocKFu0znrrTgadxeTs6nwsRCemjMGROiMJDiJxgRtUEOFTasVLjfgw=s96-c)

Amulya Sharma

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm83gfshu015roh5fbyzbsnm7)

Sharding is also important for reducing the blast radius. One large database failing has broader implication as compared to one smaller database failing (out of fleet of 10 databases). If availability is important larger unit size may not be deliver better results.

Show more

3

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm83gukcf01a8hnviaj8a6k8j)

Yes, kind of. In expectation they're identical from an availability standpoint since you've also multiplied the potential points of failure. But there are definitely places (e.g. multi-tenant applications) where you want to isolate clients/users so they don't impact others and a [cell-based architecture](https://docs.aws.amazon.com/wellarchitected/latest/reducing-scope-of-impact-with-cell-based-architecture/what-is-a-cell-based-architecture.html) can be useful for this!

Show more

0

Reply

S

StraightAmaranthRook358

[• 5 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm8f4qxyc01sm104do9jcb5k6)

Is there a typo in Caching/When to consider sharding -> "Throughout" should be "Throughput"

Show more

0

Reply

C

ConservationChocolateBasilisk953

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm8z2f2e801wyad086vzzs5fc)

Why Redis(cache) supports less 100K TPS compared to message broker 1M TPS? Again is it because Redis is single threaded and CPU bound? I was thinking writing to memory should be faster than writing to log.

Show more

1

Reply

W

walnatara2

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm92ksz8x00ewad08ug896gep)

"Writes: 10-20k TPS in single-node configurations on Aurora and RDS"

is this using SSD or HDD?

Show more

1

Reply

W

walnatara2

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm92kwi5100f0ad088j0cv0nd)

ohh I think it use SSD https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP\_Storage.html

Show more

2

Reply

Z

ZealousFuchsiaLeopon345

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9d11oap005ead08f148wy4d)

> "Reads: Over 100k requests/second per instance for in-memory caches like ElastiCache Redis on modern Graviton-based nodes Writes: Sustained throughput of hundreds of thousands of requests per second"

Am I reading it correctly? Does Redis have higher write throughput than read throughput?

Show more

0

Reply

H

HolyTanKite597

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmd0m01os00m1ad084gkkr19k)

#following

Show more

0

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 10 days ago• edited 10 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmeapuffc00wcad08gfz4y6sk)

A local get/set on redis takes near 5 micro seconds (due to RAM access). That makes it nearly 200k/sec on the higher side with the batching. As far as RAM access is concerned, reads and writes won't have much difference.

Show more

0

Reply

E

EntitledIvorySwordtail645

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9n3lj9500qxad0861k8ryi6)

I see vertical scaling coming up a lot, but to me it sounds like a difficult solution to recommend, specially considering if it goes down during some deployment or due to any other reason. It seems that despite the instance itself being more than capable of handling traffic, there are significant upsides (in terms of resiliency) to horizontal scale by default (with less powerful instances if needed).

What are the caveats that come with this single-node recommendation? For example, do you include replication as mandatory here?

Show more

1

Reply

E

EntitledIvorySwordtail645

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9n3oepm00r8ad08natdn0sl)

I totally see the point of partitions not being needed, but I think the wording leaves out horizontal scaling as a method of ensuring high availability and resiliency

Show more

1

Reply

![Cathy Liu](https://lh3.googleusercontent.com/a/ACg8ocJmCvdH_IsRWWU27J-ArQYMkSqqhNX8V_Mv8kIlMZvqDcNPBofk=s96-c)

Cathy Liu

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmb312ph700y8ad07a4irem88)

I don't think this article is suggesting running a single instance without failover strategies, it's just telling you what a single instance is capable of. You should at least proactively tell your interviewer that you opt for an active-passive configuration.

Show more

1

Reply

S

SurroundingCyanPlanarian290

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9qnfjb803ugad08fvrscabw)

The blurb at the top seems to be mistakenly copy/pasted from the Kafka article:

> Learn about how you can use Kafka to solve a large number of problems in System Design.

Show more

0

Reply

![Ashith E N](https://lh3.googleusercontent.com/a/ACg8ocLjCKGmtvNl4E_NwnRyZX-t-cGwodr2YJNvV02l4dHMHvDGVw=s96-c)

Ashith E N

[• 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9sy6pr3004rad08xwuvxdyt)

Hi,in caching section its mentioned: Writes: 1-2ms average cross-region for optimized systems

does this mean redis average cross-region write latency is 2ms? As I read in other sources, cross region network latency itself is >10ms.

Can you please clarify this?

Show more

3

Reply

![Ivan Atroshchenko](https://lh3.googleusercontent.com/a/ACg8ocKs2RdLYX_h3iuQWXlqtX5XXhV9XOGR24Ytb7Tq4fmMDCBRj68b=s96-c)

Ivan Atroshchenko

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9tj96l5003rad08ektk6du1)

So, on the majority of projects we work on, we event don't need to scale horizontally)? it's enough for us to just use single instance of yet powerful machine, single optimized db instance single broker and one instance of cache)?

Show more

1

Reply

![Cathy Liu](https://lh3.googleusercontent.com/a/ACg8ocJmCvdH_IsRWWU27J-ArQYMkSqqhNX8V_Mv8kIlMZvqDcNPBofk=s96-c)

Cathy Liu

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmb30y5yo00xzad07lx3fxf5s)

IBM's mainframes are indeed still running on powerful hardware for many banks today, but your interviewers are not interested in knowing that... Even if you are able to to fit everything on a single machine, you should at least have an active-passive configuration and multiple replicas for backup.

Show more

1

Reply

A

AddedLimeCarp593

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9yq14pg00rcad08e50wsep6)

What about disk space on application servers?

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9yq3ham00t4ad0803axvq9f)

Disk space on app servers isn't typically a constraint worth discussing in system design interviews. Modern cloud instances can have 100GB+ by default, and app servers shouldn't be storing significant data anyway - that's what object stores (S3) and databases are for. The only disk space you might care about is for temporary files or local caching, but even then we're talking negligible amounts.

The only exceptions I can think of are candidates using local NVMe.

Show more

3

Reply

A

AddedLimeCarp593

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9yqbl4s00s8ad08yph04x08)

Great article, thanks. However I'm struggling to reconcile this article with the recommendation in the Delivery Framework to avoid upfront capacity estimations. It seems like upfront capacity estimation is exactly what is needed (along with knowing the numbers in the article) to realize that, for example, you don't need to shard your DB to hold the entirety of Yelp places.

The Delivery Framework says "perform calculations only if they will directly influence your design". Well, how do I know if they will or won't without performing them and comparing them against the numbers in the cheatsheet?

Show more

2

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cm9yrw9sa014wad082jh7vrm5)

The short answer is intuition. If you don't have it yet, go ahead and practice and perform every calculation you can think of. What you'll quickly realize is that 80% of them won't help you, and you'll develop a sense for when that is.

Interviewers aren't tasked with assessing whether you can do mental arithmetic. They _are_ tasked with making sure you're building a sensible design that is grounded in realistic assumptions. The quicker you can get to those answers, the more time you'll have for other things.

Show more

4

Reply

![Lovedeep Singh](https://lh3.googleusercontent.com/a/ACg8ocLz6zxfhEAbASWcvmGuiyB3iNT3lBO3JcMzEdBITqbNoFddEt_V=s96-c)

Lovedeep Singh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmb2dghuc00ghad0724czm3qb)

Can we get a video on this as well?

Show more

2

Reply

B

bronze.gazer.6w

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmb59ilfr03b2ad07rzmyymvw)

Thanks! This is a great article. I've always wondered if I'm making the right decision by talking about sharding, etc when I practice. I feel like I'm just repeating concepts I've learnt without actually applying them. These numbers will help me question myself before I suggest adding more resources in the design.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmb59m1p10458ad08nls619gp)

Exactly!

Show more

0

Reply

A

abrar.a.hussain

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmb5lg70q00ktad08hu8yqbtw)

> Message queues become valuable when you need guaranteed delivery in case of downstream failures, event sourcing patterns, handling write spikes above 50k+ WPS, or decoupling producers from consumers.

This wouldn't apply if your writes are going to Cassandra right? I've seen interviewers suggest a couple of times adding a queue in front of Cassandra for something like say publishing messages for a Chat system design. But from my POV you should just be able to scale up your cluster so that the queue isn't required right? No need for event sourcing. They're useful for handling bursts but if you're able to justify why traffic should be roughly consistent I imagine that's a better argument.

Show more

0

Reply

![Robin Zheng](https://lh3.googleusercontent.com/a/ACg8ocImhjYaZUS-GKUpX0FGyzgm5-j4QzdVtkGojE0G2gLzVCKsCw=s96-c)

Robin Zheng

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmb62aucz007xad08c6a6t48d)

This is the best numbers I have ever seen, and exactly I can not agree more with what in here ". They do scale calculations using numbers from 2015 (or even 2020!) that dramatically underestimate what modern systems can handle. You'll hear concerns about database sizes, memory limits, and storage costs that made sense then, but would lead to significantly over-engineered systems today." Thanks Evan and Stefan!

Show more

1

Reply

C

CharacteristicCoralWhitefish627

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmbo1ta06019u08ad4gr9ppkb)

Thank you for this post and the cheat sheet with scale triggers. What would be some "numbers to know" for Redis Pub/Sub? Specifically what would be scale triggers in addition to clients or throughput? Is the number of channels a factor with Redis Pub/Sub at all?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmbo68whv01hm08adtyu4m5dh)

Pub/Sub should support > 1M TPS per core. It's _very_ dumb and thus _very_ efficient. Sharded pub/sub allows you to scale this out. This guy has some nice benchmarks on AWS: https://www.youtube.com/watch?v=F27loUSoIno

In terms of memory, you're going to incur a small (O(100 bytes)) memory hit on each subscription, when publishing messages you'll have the inverse of that. The worst case scenario would be a small number of channels with a very large number of subscribers.

Show more

1

Reply

I

InitialCopperLizard654

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmbw97xxi01db08ados03mia9)

Might be asked here before but I'll ask anyway: Say the interview question ended up needing a simple solution with single servers etc. Wouldn't it look like I don't really know more advanced topics? I mean, the interviewer might want to see that I know how to use message queues, sharding, and more, but given the requirements, I don't need to introduce it, so how would he know that I'm familiar with these topics?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmbwdpui9022a08adykby52d7)

Are you asking "if the interviewer doesn't ask about a specific topic how will they know I'm familiar with it?"

The direct answer is: they won't. Interviewers are asking questions that demonstrate the competencies they're trying to assess, it's not your responsibility to inject distributed systems knowledge into a conversation that doesn't require it!

Show more

2

Reply

F

FiscalIndigoSalmon526

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmc0ztmup09si08ad3lwhpe0h)

Over the past few years, I went through many youtube videos and courses on system design. Every single person goes around saying you need to estimates to determine how to scale the system. But not many actually tell you when you should consider. Instead, they all just default to "trust me bro" when they shard. It is understandable because machines can be cheap or expensive. This article provided some really good insight on how far we have came and what we can use these days.

Show more

0

Reply

C

CheerfulTurquoiseGuppy693

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmc54gssu07wgad083d8qv2qy)

Would be great if there is a dedicated article on statefullnes of a system. When and where we endup introducing the state in the system. When to avoid and when not to avoid the state.

Show more

0

Reply

R

ReliableSapphireSalamander717

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmc6tc2e109hiad08tm0uz8gh)

Copy paste error in the description below the title "Numbers to Know Learn about how you can use Kafka to solve a large number of problems in System Design."

Show more

0

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 2 months ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmc8f6t3p01n8ad0850dv6kpi)

How does an application server scale to 100k+ concurrnet connections, while a Database maxes out at 5-20k connections?

Show more

0

Reply

E

ExpensiveMagentaCrocodile897

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmcb2l22k01rlad08wx9ps8v4)

Hopefully this helps if no one else replies:

-   Your DB would typically return data in < 1s (say, something like 1ms to 100ms)
-   Appservers sometimes have connection pools - I think one per process (and maybe a process has a large number of threads, but a smaller number of connections ready to be picked from in their connection pool) (but I'm not sure if its typical for a connection to release its use of a db connection mid-request, or if it usually hold onto it for the whole duration of a request)

So if an appserver connection isn't spending all of its time with a DB request, then the DB connection limit might not be as constraining

Show more

0

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmcb32vno01ryad08cp0qej60)

Thanks! I think that both the numbers are for number of concurrent connections, not the number of queries/second. So the time taken to serve a request is irrelevant.

If the numbers are correct, it can be partly be explained that webservers spend a lot of time waiting to fetch data from databases or other servers. They do not actively compute for 90% of requests's execution time. As a result, they can sustain 10x more concurrent connections.

On the other hand, databases get requests and execute them (there may be some waiting time) from local data. So they are actively working on the 5-20k concurrent connections.

Cassandra's Writes Average p99 is under 10 ms @ 40k QPS: https://www.datastax.com/blog/apache-cassandra-benchmarking-40-brings-heat-new-garbage-collectors-zgc-and-shenandoah

Show more

0

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 10 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmeapbmr600q1ad08gsannngr)

I'd want to add that the connections from the pool will count towards concurrent connections until those are active. If you create multiple pools from different services for a single database, there would be atleast those many concurrent connections. Though the real question is whether or not all of those connections are in use by application - that's a different story. Pool's cleanup policy may influence number of actively used vs num of connections that are waiting to be used.

I have used AWS RDS PG extensively for a recent work and i noticed that running beyond 800 concurrent connections for expensive writes is strictly a no-no.

Show more

0

Reply

A

ArmedAmberLimpet331

[• 1 month ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmcq2ti2z083pad07gl3a34z8)

In the section on caches,

> Writes: 1-2ms average cross-region for optimized systems

should be corrected to same-region?

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 28 days ago• edited 28 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmdla5v1a04bhad07obdm7sso)

if my database choice is RDBMs and I need write throughput of 10k /sec - so we dont need to shard ?

Show more

0

Reply

![Gabriel Martins](https://lh3.googleusercontent.com/a/ACg8ocImF1OV3k2pi72cAj3EW6Nra3gEwrNdQQf_V_BkAf0wY6oXLg=s96-c)

Gabriel Martins

[• 27 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmdmhksp20703ad07sbueq71m)

Even though current hardware allows for that type of accomplishments, running a system with 50k concurrent users, and 20TB of db data on a single db instance and single application server goes against best practices for high availability, fault tolerance, and redundancy.

How to balance between not adding scaling complexity versus being bound to a single instance that may fail and bring everything down?

Show more

0

Reply

![Kody Thach](https://lh3.googleusercontent.com/a/ACg8ocLzAqwA-xzynNbCFhGvdOD9_st7hnwvDx_5x9YsNxrT-2fxEsE=s96-c)

Kody Thach

[• 18 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cmdyvfy9500npad08jhpqborl)

Was wondering, should this page also include details from this article?

https://www.hellointerview.com/blog/mastering-estimation

Show more

1

Reply

![Tina Wu](https://lh3.googleusercontent.com/a/ACg8ocKtRX_CrSp8NxlW0_0GJMFuyES7HhcULU3aNmVmA8Ij5diWsg=s96-c)

Tina Wu

[• 14 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cme4eo5yy07jrad072ipinl5n)

Why we start to scale DB if the write request is larger than 10k TPS while the throughput can support 5 times more at 50k tps?

Show more

0

Reply

![Sanjay Gandhi](https://lh3.googleusercontent.com/a/ACg8ocKRPWQaHVDnQ3vgyP9LRpTIBMCnd4lzG4RzQjH8ImdIC0W6qg=s96-c)

Sanjay Gandhi

[• 14 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cme5k0llb0j3yad08upqd14gx)

Why you have separate notation for Terabyte? TB vs TiB. was it intentional?

Show more

0

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 11 days ago](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know#comment-cme980nth059yad08u6gpt7j9)

The part where it mentions message queue and sharding needs more clarity. A single partition is a shard and there is always an upper limit per shard. For example, Kinesis imposes a 1k/s and 1MB/s per shard so you don't have a choice but to partition or shard the message queue if you want to scale.

Also, the numbers are misleading, a PG instance won't really scale to 20k writes/sec in practical terms. Probably no index and simple inserts (as you pointed out) may lead to that but i don't think that's even practical to propose it. Additionally using a 400GB redis instance requires you to have atleast 2 replicas sized similarly. This will blow the cost compared to a simple cluster that does partition. My point is - some of these are impractical and may reflect badly on the interviewee.

Show more

0

Reply
