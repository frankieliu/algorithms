##### Design a Job Scheduler

by UnderlyingApricotGoldfish326• Senior• 19 days ago

We want to build a distributed task scheduler service. The service will allow clients to schedule tasks via an API.

Each task is an HTTP call which is a REST request with a URL, HTTP method, and optional request body that will be executed at a future time. The service will support one-time and recurring tasks defined by using a cron syntax. There should be high reliability and scalability, with no single points of failure. It should handle massive scale, capable of handling hundreds of millions of scheduled tasks per day at a < 5s execution delay.

\===

NB: I got this question and the interviewer asked why there's only one Watcher, which I explained that we could have a really powerful Watcher with another one on standby to take its place. We could not agree on having a single Watcher and the interviewer thought that one Watcher was not scalable, and talked more about how we could scale the Watcher in other ways. Just an FYI if this becomes an issue during your interview. There was a comment from Evan on the Job Scheduler article where he answered that DynamoDB can do millions of reads per second. If we have 100 million DAU, this would be about 1200 reads per second. Unfortunately I didn't remember to mention this and wasn't able to provide a convincing argument for a single watcher instance.

Collapse Description

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

[• 19 days ago](https://www.hellointerview.com/community/submissions/cme1zab5p0572ad08e12hhlzl#comment-cme1zaqng0003ad39k54ts2z5)

Nice design! You've created a solid foundation for a distributed job scheduler with good separation of concerns between job creation, execution, and querying. Your use of DynamoDB with hourly bucketing for time-based queries and SQS for reliable job delivery shows thoughtful consideration of the scale requirements. The most critical issue is the single Watcher component which creates a bottleneck and single point of failure for a system handling 100M+ executions daily.

1.  **Positives**
    
    -   Using DynamoDB with hourly bucket partitioning is clever - this enables efficient time-range queries for upcoming jobs while avoiding hot partitions
    -   Leveraging SQS's built-in delay feature (up to 15 minutes) reduces complexity and provides reliable message delivery with automatic retries
    -   Separating Jobs Service from Query Service allows independent scaling of write vs read workloads
2.  **Potential Issues** **Single Watcher Bottleneck**
    
    -   Having only one Watcher instance polling DynamoDB creates a critical bottleneck for 100M executions/day (1,157 jobs/second average, likely 3-5x at peak)
    -   If the Watcher fails or can't keep up, jobs won't get scheduled to SQS, causing widespread execution delays
    -   This violates the "no single points of failure" requirement and makes the <5s execution delay impossible to guarantee at scale
    
    **Missing Execution Deduplication**
    
    -   No mechanism prevents duplicate executions if a job is accidentally scheduled multiple times or if the Watcher crashes and restarts
    -   Workers could execute the same job multiple times, potentially causing duplicate charges, emails, or other side effects
    -   This breaks idempotency requirements for a reliable scheduler
    
    **Unclear Recurring Job Handling**
    
    -   The design shows workers updating "next execution if required" but doesn't explain how recurring jobs generate future executions
    -   Without a clear strategy for pre-computing or dynamically generating future executions, recurring jobs may be missed
    -   This could lead to skipped executions or inconsistent scheduling behavior
3.  **Follow-up Questions**
    
    -   How do you handle jobs that need to run more frequently than SQS's 15-minute delay limit (e.g., every minute)?
    -   What happens when a worker fails mid-execution - how do you ensure the job is retried without creating duplicates?
    -   How do you scale the Watcher horizontally while preventing multiple instances from scheduling the same job?
    -   What's your strategy for handling time zone changes and daylight saving time transitions for cron-based schedules?
    -   How do you monitor and alert on jobs that consistently fail or take too long to execute?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply