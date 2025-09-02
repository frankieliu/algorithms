##### Design Yelp

by LivingCoralGopher356• Senior• 1 month ago

Design Yelp. Kept it simple and tried not to over complicate anything. Wen't with Postgres for full-text search with PostGIS extension for geospatial search so that I don't need to introduce another technology. The read throughput should be low enough. LRU cache for popular businesses with a relatively tight TTL so they don't go too stale. Would cache their reviews too.

Collapse Description

3

10

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Tomer Amir](https://lh3.googleusercontent.com/a/ACg8ocJ_u6mXy5P0HLb4MsPN8aW0EvXHCi3jB2Kas1nLJEK8OOhdvcj_QQ=s96-c)

Tomer Amir

[• 1 month ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmdik90kw043oad087f0gd5ze)

This is cool design! Did you have any deep-dives you did here?

Show more

0

Reply

![Tomer Amir](https://lh3.googleusercontent.com/a/ACg8ocJ_u6mXy5P0HLb4MsPN8aW0EvXHCi3jB2Kas1nLJEK8OOhdvcj_QQ=s96-c)

Tomer Amir

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmdikbipb0446ad08gnb0fa8s)

Were you asked to do this in an interview?

Show more

0

Reply

L

LivingCoralGopher356

[• 1 month ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmdj6dm8q00fjad0888ky70iw)

I was asked this in a mock actually. The main depth was about the geospatial aspect. i think they expected elasticsearch but were impressed when i justified keeping things in postgres and using extensions given the scale. I also briefly explained r-tree and geohashing

Show more

0

Reply

P

ParentalRedSole744

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmdnypumq032aad0848aqonjl)

Looks good!

1.  If you would cache reviews, it might be worth talking about how happens to the cache when a user posts a review. It might be jarring to the user if they don't see their review after posting it. You could also cache their review on the frontend, just for the reviewer. More elegant, imo.
2.  If you're using OCC, it might be worth talking about what database version that you want to be "optimistic" about. HelloInterview has a very good breakdown about why using avg rating in OCC can be problematic. The article can be found [here](https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention).

Show more

0

Reply

F

FutureHarlequinMeerkat461

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmdo41jfq04v7ad085kuysxpi)

IMO, this design lacks depth.

1.  If you have a Postgres handing search request from 100M DAU handling both searches (reads) and reviews (writes), I'd expect you to explain your sharding strategy for Postgres in great detail (with calculations - how many nodes would be required) - either using Citus or application level partitioning. A single Postgres instance could handle approximately 10K TPS. Citus adds a level of indirection for sharding and uses a distributed transaction (2PC) for handling transactions across shards. Discussions around performance again.
    
2.  Your Postgres instance is a single point of failure - if that goes down, you business goes down. How are you handling fault tolerance here? If using replicas - is it synchronous / async replication? Async will have consistency issues, sync - performance penalty.
    
3.  You mention avg\_rating updates using OCC - You have 2 separate tables to update when a new review comes in - add an entry in the review table and then update the Business table with the new average rating. What if the former succeeds but the later fails due to conflict. How would your SQL query look like?
    
4.  Will your cache consist of average rating for a business? If so, what TTL would you keep - if not, how will you show updated average ratings to users?
    

Show more

0

Reply

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 26 days ago• edited 7 days ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmdq5iqiu007tad3iv5z2fat7)

Great work on this Yelp system design! You've made some solid architectural choices that will help the system scale to handle the projected load. I like how you've kept the design relatively simple while still addressing the key requirements. Let's dive into the specifics:

1.  **Positives**
    
    -   Using Postgres with full-text search and PostGIS is a good choice to efficiently handle the search functionality without needing a separate search technology
    -   Caching popular businesses and reviews in an LRU cache will help reduce read latency for frequently accessed data
    -   Separating the Review Service allows reviews to be stored and scaled independently from business data
2.  **Potential Issues** **Single database bottleneck**
    
    -   Using a single Postgres database will limit write throughput to around 10k writes/second which may not support 100M DAU
    -   High write traffic from reviews could start to bottleneck and increase write latency
    -   Sharding the database by business\_id or geographic region may be needed to distribute the write load
    
    **Missing rate limiting and authentication**
    
    -   There is no mention of rate limiting for posting reviews or viewing businesses
    -   Malicious users could spam businesses with reviews or scrape business data at a high rate
    -   Authentication is mentioned for posting reviews but no details on how it's implemented or how auth tokens are passed from the client
    
    **Unclear caching strategy**
    
    -   Caching only "popular" businesses with a tight TTL could lead to frequent cache misses
    -   The criteria for a "popular" business and the cache TTL are not defined
    -   Stale caches could cause inconsistency between reviews and avg rating if the TTL is too long
3.  **Follow-up Questions**
    
    -   How do you plan to handle the high write throughput for reviews as the user base scales?
    -   What is the strategy for keeping the LRU cache in sync with the primary database as reviews are added and ratings change?
    -   How will you protect against abuse like review spam or scraping the business API at a high rate?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

L

LivingCoralGopher356

[• 7 days ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmehtlvyc00dnad08jnuswu50)

For read throughput I have the cache. For writes, I'm less worried as, realistically, businesses will get less than 5 reviews on avg a day (some will get a lot, most will get none). So that's just like 500 writes. What do you think?

Show more

0

Reply

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 7 days ago• edited 7 days ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmehtms63000sad3b0p6z7pwe)

Caching for reads makes sense. For writes, I’d sanity‑check the numbers.

-   If you truly mean ~5 reviews per business per day across 10M businesses, that’s ~50M reviews/day ≈ 580 writes/sec sustained (plus spikes). Still doable with Postgres, but not “just 500 writes.”
-   If you meant site‑wide only ~500 reviews/day, that’s trivial—but with 100M DAU that seems unrealistically low. A more realistic back‑of‑the‑envelope: even 0.1% of DAU writing one review/day is ~100k/day ≈ 1.2/sec; 1% is ~1M/day ≈ 11.6/sec.

The real risk isn’t the average, it’s skew and bursts: a few businesses can get hammered, which creates hot-row contention when you update avgRating/numRatings.

Practical takeaway:

-   If you’re under a few hundred writes/sec sustained, a single well‑tuned Postgres primary is fine.
-   Mitigate hot spots by updating aggregates atomically (e.g., maintain sumRatings and numRatings with UPDATE ... SET num=num+1, sum=sum+rating) instead of OCC on avg, or compute aggregates asynchronously for very hot businesses.
-   Expect peak bursts; plan for retries/backoff and simple partitioning by business\_id later if needed.

So yes, writes may be manageable—but model them explicitly and plan for hot spots rather than relying on the “average per business.”

Show more

1

Reply

L

LivingCoralGopher356

[• 7 days ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmehtnx0k00ctad08pos19nwx)

Makes sense! Thanks

Show more

0

Reply

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 7 days ago](https://www.hellointerview.com/community/submissions/cmdgqbiww00d4ad07eso041d6#comment-cmehto1mn000iad3aexl9vvc3)

You’re welcome! Glad it helped. If you want, I can share a quick example of the atomic UPDATE pattern for rating aggregates or review a revised flow—just say the word.

Show more

0

Reply