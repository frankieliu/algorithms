##### Design Top K Songs Widget for Spotify

by Noam• Staff• 18 days ago

_Design a system that allows each user to retrieve their Top 10 most-listened-to songs from the past 7 days. I looked at meta reports to decide on this variation._

### Aggregation Service

We receive play events via a Kafka stream, where each event includes a song\_id, user\_id, event\_id, and timestamp.

The aggregation service consumes events from Kafka and maintains in-memory counters per (user\_id, song\_id) grouped by minute: (user\_id, song\_id, minute\_bucket). Every minute, the service flushes these counts to the OLAP database. This significantly reduces write pressure, especially under high traffic (~10,000 events/sec).

To avoid concurrency issues, users are partitioned across workers so that each worker processes a disjoint set of users. Aggregation is done based on processing time, not event time, and we retain both the current and previous minute in memory to handle out-of-order arrivals or flush overlaps.

### OLAP Storage

Flushed data is written to an OLAP database (e.g., ClickHouse or Druid), with rows structured as:

(user\_id, song\_id, minute\_bucket, play\_count)

The table is partitioned by day and ordered by (user\_id, song\_id) for efficient filtering and grouping. Only the latest 7 days are stored, and a scheduled job prunes older partitions. All raw events are also written to S3 for backup.

### Daily Rollups

To optimize query performance, we roll up per-minute data at the end of each day:

-   Aggregate into daily totals: (user\_id, song\_id, day, total\_plays)
    
-   Keep only the top 50 songs per user per day to cap row count for heavy users
    
-   Result: at most 350 rows per user for a 7-day window
    

The current day remains in minute-level granularity, while previous six days are accessed through daily aggregates.

### Top K Service

A stateless API receives client requests and queries OLAP to retrieve the top 10 songs for a given user over the past 7 days. It aggregates daily totals and current-day minute-level data, sorts by total play count, and returns the top results.

This service is autoscaled and deployed behind an API Gateway. Being stateless, it’s easy to horizontally scale.

### Reliability and Fault Tolerance

-   Kafka offsets are committed only after successful OLAP flush
    
-   If a worker crashes, a replacement reprocesses from the last committed offset
    
-   Processing is idempotent, ensuring no data loss or duplication
    
-   Time-based buffering (current + previous minute) mitigates minor delays or overlaps
    

### Caching

We do not cache Top K results:

-   Queries are personalized, limiting cache reusability
    
-   Frequency is low per user (a few times per week)
    
-   Maintaining cache coherence adds complexity without clear benefit
    

### Performance and SLA

-   Most users generate <100 rows/week
    
-   Heavy users are capped at ~350 rows via daily Top 50 filtering
    
-   OLAP queries only scan 7 partitions and a few hundred rows max
    
-   With pruning, ordering, and columnar scans, sub-200ms latency is achievable for most queries
    

If needed, we can add a materialized Top K store refreshed every few minutes, but current performance is expected to meet SLA.

Collapse Description

5

15

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 18 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme1ga434002jad397lip1k48)

Nice design! You've created a well-thought-out architecture for a Top-K songs system that demonstrates good understanding of stream processing and OLAP patterns. Your use of Kafka for event ingestion, in-memory aggregation with periodic flushes, and ClickHouse for analytical queries shows solid architectural thinking. The daily rollup strategy to maintain only top 50 songs per user is a clever optimization that keeps data volume manageable while preserving the ability to compute accurate top-10 results.

1.  **Positives**
    
    -   Smart use of in-memory aggregation with per-minute flushes reduces write amplification to the OLAP database while maintaining near real-time updates
    -   Excellent data reduction strategy with daily rollups keeping only top 50 songs per user, which bounds storage to ~350 rows per user per week
    -   Good choice of ClickHouse with day-based partitioning and (user, song) sorting for efficient analytical queries and automatic data expiration
2.  **Potential Issues** **Worker Assignment Mechanism Undefined**
    
    -   You mention each worker handles "a distinct set of users" but don't specify how users are assigned to workers
    -   Without consistent hashing or a coordination service, users could be processed by multiple workers simultaneously after scaling events
    -   This would cause race conditions and duplicate counts, breaking the accuracy of play counts
    
    **Memory Overflow Risk**
    
    -   In-memory aggregation has no defined memory limits or spill-to-disk mechanism
    -   A single user playing thousands of unique songs in a minute could exhaust worker memory
    -   This would cause worker crashes and data loss since Kafka offsets aren't committed until successful flush
    
    **Query Performance at Scale**
    
    -   Querying 7 days of data across 100M users without caching may not meet 200ms SLA consistently
    -   Even with efficient partitioning, scanning 350 rows per user and sorting for top-10 requires significant computation
    -   Peak query loads could overwhelm the OLAP cluster without result caching or pre-computation
3.  **Follow-up Questions**
    
    -   How do you ensure consistent user-to-worker mapping during auto-scaling or worker failures?
    -   What happens when a worker's in-memory state exceeds available memory before the minute flush?
    -   How do you handle late-arriving events that belong to a previous minute bucket?
    -   What's your strategy for handling OLAP query timeouts or failures - do you fall back to stale data?
    -   How do you prevent hot partitions when many users query simultaneously at peak times?
    -   What happens to the minute-level data after daily rollup - is it immediately deleted or kept for debugging?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 18 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme1ibcq8096sad08mdqbx16k)

Noam doing all the heavy hitters! You rock.

Do you mind if I change the name of this one to [Design Top K Songs Widget for Spotify](https://www.hellointerview.com/premium/questions/cm865cqor00fxetkc7odo11qe) so it's easy for people to find?

Show more

0

Reply

N

Noam

[• 18 days ago• edited 18 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme1jzmb5004jad098l44sxw2)

No problem! I had to prep for my Meta interview anyway, so I’m glad I could give something back to the site :) Learned a ton from you.

One feature idea: a space to add writeup (markdown preferably). I’ve got some short write-ups that don’t really fit cleanly when pasted into the high-level section.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 18 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme1k69lx0082ad088dzsqrlb)

Great feedback. Was going to ask you about any feedback now that you've had some good use with it. Would me making the description area markdown and larger solve the issue?

Show more

0

Reply

N

Noam

[• 18 days ago• edited 18 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme1kjn8f00ayad09kfwzx4vj)

Sounds like a quick win :) Another a bit more complicated option, make it a dedicated section that can be expanded to show just the write-up. That way you can see the full write-up along with the design.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 18 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme1kl02v00blad08bn1x88kr)

Should be easy to make it expand to take the whole left panel. Can figure something out there. Appreciate the feedback! Keep it coming as you think of more :)

Show more

0

Reply

N

Noam

[• 18 days ago• edited 18 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme1kqa7s00ctad09xltbfscz)

I'll beta test when ready. I have 3 deep dives for the leaderboard and distributed crawlers :)

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 18 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme1zle23000kad08a5sxvbh2)

Ok added! See what you think :)

Show more

0

Reply

N

Noam

[• 18 days ago• edited 17 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme2enrg704z6ad08im79suy8)

That was quick! I added deep dives to the distributed crawler (and here). It all fits in nicely.

https://www.hellointerview.com/system-design/submissions/cmdy7pqug056dad07octdds1a

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 17 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme2yt16h0854ad08c4g8b3zk)

Wonderful!

Show more

0

Reply

![Vardhan](https://lh3.googleusercontent.com/a/ACg8ocJKdxaF-EUELK8EMLot2BCWPVv0EhCvHeDvOJcNaT9T5HkdiA=s96-c)

Vardhan

[• 17 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme2zkvae088iad08s9yscnat)

Would it be alright to just say "Top K Service" as a black box for this question like in the picture ? Would we be expected to also draw out the top K infra here?

Show more

0

Reply

N

Noam

[• 16 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cme5ctgja0hrwad075lhjto7j)

I think so. It's only a lean service that queries the OLAP.

Show more

0

Reply

V

VerticalBlackFlea403

[• 4 days ago](https://www.hellointerview.com/community/submissions/cme1g9p1p08mqad08yggbgpod#comment-cmem1jvuu04lmad08zklow0d5)

Is there a reason why Flink is not used here for aggregation?

Show more

0

Reply