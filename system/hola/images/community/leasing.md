##### Design a leasing system

by prasad parshallu• Mid-level• 29 days ago

There are five dedicated environments where engineers can reserve for running their tests. The tests range from 20 minutes long to 3 hours long, such as stress tests, chaos tests, and load tests. There should be only one test running per environment at the same time, and engineers should be able to reserve an environment at a future time. One key thing to consider is to properly handle messy failures, such as test jobs hanging or environments being bogged down. You can assume there's a blackbox test framework that you can use that interacts with each environment.

Collapse Description

1

3

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

P

ParentalRedSole744

[• 28 days ago• edited 28 days ago](https://www.hellointerview.com/community/submissions/cmdnsfzii00zvad09jtb67mn8#comment-cmdnxrejb02p3ad083bkmbthf)

Great design! I'm finding it hard for me to piece together some details:

1.  I might've left Rate limiting out for a system like this: it's implied that this is internal tooling and thus won't be getting too many requests.
2.  It might've helped to distinguish between Job and Reservation. For example, your Watcher polls the database for jobs that will end in 5 minutes but your database has no table definition for Job. It would make things easier like making sure an environment only has one job running at a time.
3.  What triggers jobs being interrupted? It appears to be another job needing to run. If so, what if there's no job in queue but the job is hanging - past 3 hours?
4.  I'm not sure what Redis is doing here. The Env:StartTime:EndTime isn't in proper key:value format. More importantly, I don't think a cache is necessary at this scale.
5.  The interaction between the Booking service and Test Env is unclear. Isn't it the Status Updater that's starting the job?
6.  To be _above mid-level_ hire, it might be nice to detail what a Test Env could be. E.g. a routable VM pre-provisioned with some service.

Show more

1

Reply

![prasad parshallu](https://lh3.googleusercontent.com/a/ACg8ocLrYymPKBe6LJgW9XqgXBLxsdQARhC1IwY7AsDLqGlcLHpwUZsj=s96-c)

prasad parshallu

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdnsfzii00zvad09jtb67mn8#comment-cmdnzf9qj02xxad08zmexf5ur)

I have updated the design a bit. 1.I feel its good to have rate limiting with higher thresholds if requirement aren't strict. 2. My bad I wanted to say poll for environment in running status. 3. We will fetch the environments if they are with "RUNNING" status and endTime is in next 5mins, even though if we don't have another jobs to run we will interrupt the test env. as every job is costing us. 4. I used redis to check if the requested time is already locked. I could have directly queried the DB but at scale redis locks will help.  
5\. Added the description. 6. I just went with blackbox as mentioned in the description.

Show more

1

Reply

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdnsfzii00zvad09jtb67mn8#comment-cmdryqdeg003sad3d9n76jv0c)

Nice design! You've created a thoughtful architecture for the leasing system with good attention to consistency requirements and failure handling. Your use of Redis for fast reservation checking and SQS for scheduled job management shows solid understanding of distributed systems patterns. However, there are some critical issues around concurrent booking prevention and the polling-based architecture that need addressing to make this production-ready.

1.  **Positives**
    
    -   Using Redis with environment:startTime:endTime keys for fast reservation checking is a smart choice that enables sub-100ms booking latency
    -   Separating booking and reservation fetching services provides good separation of concerns and allows independent scaling of read vs write paths
    -   Including a watcher component to handle job lifecycle management shows good foresight for handling the messy failure scenarios mentioned in requirements
2.  **Potential Issues** **Race Condition in Booking Logic**
    
    -   Checking Redis and then writing to Postgres without distributed locking creates a race window
    -   Two users could check Redis simultaneously, both see the slot is free, and both proceed to book
    -   This violates the core requirement that only one test should run per environment at a time
    
    **Inefficient Polling Architecture**
    
    -   The watcher polls for reservations ending/starting in next 5 minutes continuously
    -   This creates unnecessary database load and introduces up to 5-minute delays in job scheduling
    -   For 10k environments, this could mean thousands of wasted queries per second
    
    **Missing Cleanup for Failed Tests**
    
    -   No mechanism to detect when tests hang or exceed their allocated time
    -   A test that hangs indefinitely would block the environment forever
    -   This breaks the requirement to "properly handle messy failures" like hanging test jobs
    
    **Unclear SQS Delay Mechanism**
    
    -   SQS maximum delay is 15 minutes, but tests can be scheduled up to 7 days in advance
    -   The design doesn't explain how to handle delays beyond 15 minutes
    -   This makes the scheduled job execution unreliable for future reservations
3.  **Follow-up Questions**
    
    -   How do you ensure atomic reservation creation when both Redis and Postgres need to be updated?
    -   What happens when a test runs beyond its scheduled end time - how is this detected and handled?
    -   How does the system handle the 15-minute SQS delay limit for reservations scheduled days in advance?
    -   What's the strategy for Redis key expiration vs actual reservation cleanup coordination?
    -   How do you prevent environment starvation if users consistently book all slots?
    -   What monitoring exists to detect when environments are in a failed state?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply