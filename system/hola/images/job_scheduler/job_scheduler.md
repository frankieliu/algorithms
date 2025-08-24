# Design a Job Scheduler

[![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.66fdc8bf.png&w=96&q=75&dpl=e097d75362416d314ca97da7e72db8953ccb9c4d)

Evan King

Ex-Meta Staff Engineer

](https://www.linkedin.com/in/evan-king-40072280/)

hard

Published Jul 8, 2024

* * *

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

0:00

Play

Mute

0%

0:00

/

1:05:32

Premium Content

Closed-Captions On

Chapters

Settings

AirPlay

Google Cast

Enter PiP

Enter Fullscreen

## Understanding the Problem

**⏰ What is a Job Scheduler** A job scheduler is a program that automatically schedules and executes jobs at specified times or intervals. It is used to automate repetitive tasks, run scheduled maintenance, or execute batch processes.

There are two key terms worth defining before we jump into solving the problem:

-   **Task**: A task is the abstract concept of work to be done. For example, "send an email". Tasks are reusable and can be executed multiple times by different jobs.
    
-   **Job**: A job is an instance of a task. It is made up of the task to be executed, the schedule for when the task should be executed, and parameters needed to execute the task. For example, if the task is "send an email" then a job could be "send an email to [john@example.com](mailto:john@example.com) at 10:00 AM Friday".
    

The main responsibility of a job scheduler is to take a set of jobs and execute them according to the schedule.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements-1)

**Core Requirements**

1.  Users should be able to schedule jobs to be executed immediately, at a future date, or on a recurring schedule (ie. "every day at 10:00 AM").
    
2.  Users should be able monitor the status of their jobs.
    

**Below the line (out of scope)**

-   Users should be able to cancel or reschedule jobs.
    

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements-1)

Now is a good time to ask about the scale of the system in your interview. If I were your interviewer, I would explain that the system should be able to execute 10k jobs per second.

**Core Requirements**

1.  The system should be highly available (availability > consistency).
    
2.  The system should execute jobs within 2s of their scheduled time.
    
3.  The system should be scalable to support up to 10k jobs per second.
    
4.  The system should ensure [at-least-once](https://blog.bytebytego.com/i/51197752/At-Least-Once) execution of jobs.
    

**Below the line (out of scope)**

-   The system should enforce security policies.
    
-   The system should have a CI/CD pipeline.
    

On the whiteboard, this might look like:

Requirements

## The Set Up

### Planning the Approach

For this question, which is less of a user-facing product and more focused on data processing, we're going to follow the delivery framework outlined [here](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery), focusing on the core entities, then the API, and then the data flow before diving into the high-level design and ending with deep dives.

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

We recommend that you start with a broad overview of the primary entities, even for questions more focused on infrastructure, like this one. It is not necessary to know every specific column or detail yet. We will focus on the intricacies, such as columns and fields, later when we have a clearer grasp of the system.

Just make sure that you let your interviewer know your plan so you're on the same page. I'll often explain that I'm going to start with just a simple list, but as we get to the high-level design, I'll document the data model more thoroughly.

To satisfy our key functional requirements, we'll need the following entities:

1.  **Task**: Represents a task to be executed.
    
2.  **Job**: Represents an instance of a task to be executed at a given time with a given set of parameters.
    
3.  **Schedule**: Represents a schedule for when a job should be executed, either a CRON expression or a specific DateTime.
    
4.  **User**: Represents a user who can schedule jobs and view the status of their jobs.
    

In the actual interview, this can be as simple as a short list like this. Just make sure you talk through the entities with your interviewer to ensure you are on the same page.

Entities

### [The API](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#4-api-or-system-interface)

Your goal is to simply go one-by-one through the core requirements and define the APIs that are necessary to satisfy them. Usually, these map 1:1 to the functional requirements, but there are times when multiple endpoints are needed to satisfy an individual functional requirement.

First, let's create a job:

`POST /jobs {   "task_id": "send_email",   "schedule": "0 10 * * *",   "parameters": {     "to": "john@example.com",     "subject": "Daily Report"   } }`

Next, let's query the status of our jobs:

`GET /jobs?user_id={user_id}&status={status}&start_time={start_time}&end_time={end_time} -> Job[]`

### [Data Flow](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#optional-data-flow-5-minutes)

Before diving into the technical design, let's understand how data flows through our system. The data flow represents the journey from when a request enters our system to when it produces the final output.

Understanding this flow early in our design process serves multiple purposes. First, it helps ensure we're aligned with our interviewer on the core functionality before getting into implementation details. Second, it provides a clear roadmap that will guide our high-level design decisions. Finally, it allows us to identify potential bottlenecks or issues before we commit to specific architectural choices.

1.  A user schedules a job by providing the task to be executed, the schedule for when the task should be executed, and the parameters needed to execute the task.
    
2.  The job is persisted in the system.
    
3.  The job is picked up by a worker and executed at the scheduled time.
    
    -   If the job fails, it is retried with exponential backoff.
        
    
4.  Update the job status in the system.
    

Note that this is simple, we will improve upon as we go, but it's important to start simple and build up from there.

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes-1)

We start by building an MVP that works to satisfy the core functional requirements. This does not need to scale or be perfect. It's just a foundation for us to build upon later. We will walk through each functional requirement, making sure each is satisfied by the high-level design.

### 1) Users should be able to schedule jobs to be executed immediately, at a future date, or on a recurring schedule

When a user schedules a job, they'll provide the task to be executed, the schedule for when the job should be executed, and the parameters needed to execute the task. Let's walk through how this works:

1.  The user makes a request to a /jobs endpoint with:
    
    -   Task ID (which task to run)
        
    -   Schedule (when to run it)
        
    -   Parameters (what inputs the task needs)
        
    
2.  We store the job in our database with a status of PENDING. This ensures that:
    
    -   We have a persistent record of all jobs
        
    -   We can recover jobs if our system crashes
        
    -   We can track the job's status throughout its lifecycle
        
    

Schedule Jobs

When it comes to choosing a database, the hard truth is that any modern database will work. Given no need for strong consistency and our data has few relationships, I'm going to opt for a flexible key value store like [DynamoDB](https://www.hellointerview.com/learn/system-design/deep-dives/dynamodb) to make scaling later on easier. If you're interviewing at a company that prefers open-source solutions, you might go with [Cassandra](https://www.hellointerview.com/learn/system-design/deep-dives/cassandra) instead. But, like I said, Postgres or MySQL would work just as well, you'd just need to pay closer attention to scaling later on.

Let's start with a simple schema for our Jobs table to see why this approach needs refinement:

`{   "job_id": "123e4567-e89b-12d3-a456-426614174000",   "user_id": "user_123",   "task_id": "send_email",   "scheduled_at": 1715548800,   "parameters": {     "to": "john@example.com",     "subject": "Daily Report"   },   "status": "PENDING" }`

This works fine for one-time jobs, but it breaks down when we consider recurring schedules. Consider a daily email report that needs to run at 10 AM every day. We could store the CRON expression (0 10 \* \* \*) in our table, but then how do we efficiently find which jobs need to run in the next few minutes? We'd need to evaluate every single CRON expression in our database—clearly not scalable.

This brings us to a key insight: we need to separate the _definition_ of a job from its _execution instances_. Think of it like a calendar: you might have an event that repeats every Monday, but in your calendar app, you see individual instances of that event on each Monday. This is exactly what we need.

Let's split our data into two tables. First, our Jobs table stores the job definitions:

`{   "job_id": "123e4567-e89b-12d3-a456-426614174000",  // Partition key for easy lookup by job_id   "user_id": "user_123",    "task_id": "send_email",   "schedule": {     "type": "CRON" | "DATE"      "expression": "0 10 * * *"  // Every day at 10:00 AM for CRON, specific date for DATE   },   "parameters": {     "to": "john@example.com",     "subject": "Daily Report"   } }`

Then, our Executions table tracks each individual time a job should run:

`{   "time_bucket": 1715547600,  // Partition key (Unix timestamp rounded down to hour)   "execution_time": "1715548800-123e4567-e89b-12d3-a456-426614174000",  // Sort key (exact execution time and the jobId since partition key and sort key must be unique)   "job_id": "123e4567-e89b-12d3-a456-426614174000",   "user_id": "user_123",    "status": "PENDING",   "attempt": 0 }`

By using a time bucket (Unix timestamp rounded down to the nearest hour) as our partition key, we achieve efficient querying while avoiding hot partition issues. For example, to find jobs that need to run in the next few minutes, we only need to query the current hour's bucket and possibly the next hour's bucket. The time bucket can be easily calculated:

`time_bucket = (execution_time // 3600) * 3600  # Round down to nearest hour`

First, by querying only 1-2 partitions to find upcoming jobs, we maintain efficient read operations. The hourly buckets ensure jobs are well-distributed across partitions, preventing hotspots. Second, when a recurring job completes, we can easily schedule its next occurrence by calculating the next execution time and creating a new entry in the Executions table. The job definition stays the same, but we keep creating new execution instances.

This pattern of separating the definition of something from its instances is common in system design. You'll see it in calendar systems (event definition vs. occurrences), notification systems (template vs. individual notifications), and many other places. It's a powerful way to handle recurring or templated behaviors. See the [GoPuff Breakdown](https://www.hellointerview.com/learn/system-design/problem-breakdowns/gopuff) for another example of this.

When a worker node is ready to execute jobs, it simply queries the Executions table for entries where:

-   execution\_time is within the next few minutes
    
-   status is "PENDING"
    

The worker can then look up the full job details in the Jobs table and execute the task.

Execution

This simple approach is a great start, but don't stop reading here! We expand on this significantly later in the deep-dives in order to ensure the system is scalable, fault tolerant, and handles job failures gracefully.

### 2) Users should be able monitor the status of their jobs.

First, the obvious bit, when a job is executed we need to update the status on the Executions table with any of COMPLETED, FAILED, IN\_PROGRESS, RETRYING, etc.

Monitoring

But how can we query for the status of all jobs for a given user?

With the current design, querying jobs by userId would require an inefficient two-step process:

1.  Query the Jobs table to find all job\_ids for a user (an inefficient full table scan)
    
2.  Query the Executions table to find the status of each job
    

To solve this, we'll add a [Global Secondary Index (GSI)](https://www.hellointerview.com/learn/system-design/deep-dives/dynamodb#secondary-indexes) on the Executions table:

-   Partition Key: user\_id
    
-   Sort Key: execution\_time + job\_id
    

This GSI allows us to efficiently find all executions for a user, sort them by execution time, support pagination, and filter by status if needed. The index provides a fast and scalable way to query execution data from the user's perspective.

The GSI adds some write overhead and cost, but it's a worthwhile trade-off to support efficient user queries without compromising our primary access pattern of finding jobs that need to run soon.

This is a common pattern in DynamoDB (and similar NoSQL databases) where you maintain multiple access patterns through GSIs rather than denormalizing the data. The alternative would be duplicating the user\_id in the base table, which would make the data model more complex and harder to maintain.

Now, users simply need to query the GSI by user\_id and get a list of executions sorted by execution\_time.

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes-1)

### 1) How can we ensure the system executes jobs within 2s of their scheduled time?

Our current design has some key limitations that prevent us from executing jobs at the precision we require.

The single, most obvious, limitation is that we are querying the database every few minutes to find jobs that are due for execution. The frequency with which we decide to run our cron that queries for upcoming jobs is naturally the upper bound of how often we can be executing jobs. If we ran the cron every 2 minutes, then a job could be executed as much as 2 minutes early or late.

The next consideration would be to run the cron more frequently. However, to meet our precision requirement, this would mean running it every 2 seconds or less. This won't work for a whole host of reasons.

1.  With 10k jobs per second, each query would need to fetch and process around 20k jobs (jobs due in the next 2 seconds). This is a large payload to transfer and process frequently.
    
2.  Even with proper indexing, querying for 20k jobs could take several hundred milliseconds. Add in network latency and serialization time, and we might spend 500ms just getting the job data.
    
3.  After retrieving the jobs, we still need time to initialize them, distribute them to workers, and begin execution. This processing overhead further reduces our precision window.
    
4.  Running such large queries every 2 seconds puts significant load on the database, which could impact other operations and overall system stability.
    

As you can see, this isn't the best idea. Instead, we can get more clever and introduce a two-layered scheduler architecture which marries durability with precision.

1.  **Phase 1: Query the database**: Just like in our current design, we will query the Executions table for jobs that are due for execution in the next ~5 minutes (leaving some buffer for network latency, etc.).
    
2.  **Phase 2: Message queue**: We'll take the list of jobs returned by our query and push them to a message queue, ordered by execution\_time. Workers will then pull jobs from the queue in order and execute the job.
    

Low Latency

This two-layered approach provides significant advantages by decoupling the database querying from job execution. By running database queries just once every 5 minutes, we reduce database load while maintaining precision through the message queue. The message queue's high throughput and ordering guarantees mean we can execute jobs with sub-second precision, removing the upper bound on execution frequency that existed in our previous design.

Okay, we're making progress, but what about new jobs that are created and expected to run in less than 5 minutes? Currently, we'd write this job to the database, but the cron that runs every 5 minutes would never see it. We either missed it entirely, or we'd catch it the next time the cron runs, which could be as much as 5 minutes later.

We could try to put the job directly into the message queue, but this introduces a new problem. Message queues like Kafka process messages in order, so any new job would go to the end of the queue. This means if we have a 5 minute window of jobs in the queue, the new job would wait behind all of them before being processed - even if it's scheduled to run sooner. This would make it impossible to execute the job within our 2 second precision requirement.

Instead, we could replace our log-based message queue with a queue system that can handle job prioritization based on execution time. Let's examine our main options:

### 

Good Solution: Redis Sorted Sets

###### Approach

Redis Sorted Sets (ZSET) offer a straightforward way to implement a priority queue, using the execution timestamp as the score for ordering. When a job needs to be scheduled, we can add it to the sorted set with its execution time as the score. Workers can then efficiently query for the next jobs to execute by fetching entries with scores less than the current timestamp.

###### Challenges

While Redis provides excellent performance characteristics with sub-millisecond latency and atomic operations, it comes with significant operational complexity in a distributed environment. You'll need to implement your own retry logic, handle failure scenarios, and carefully manage Redis replication to avoid single points of failure. The solution can work well at moderate scale, but requires substantial operational expertise and custom code to run reliably in production.

### 

Good Solution: RabbitMQ

###### Approach

RabbitMQ provides a robust message queuing system with a delayed message exchange plugin for scheduling. The system handles message persistence out of the box and provides publisher confirms to ensure reliable delivery. Jobs can be scheduled by publishing messages with a delay parameter, and RabbitMQ's clustering capabilities provide high availability.

###### Challenges

While RabbitMQ is a mature message broker, its delayed message functionality is plugin-based rather than native, which can be less reliable at scale. The system requires manual implementation of retry logic and has higher operational overhead than managed solutions. Scaling horizontally requires careful cluster configuration and monitoring, and the delayed message implementation may not be as precise as purpose-built scheduling systems.

### 

Great Solution: Amazon SQS

###### Approach

Amazon SQS provides a fully managed queue service with native support for delayed message delivery. When scheduling a job, we simply send a message to SQS with a delay value (called a "SQS Delivery Delay"), and SQS handles the rest.

For example, if we want to schedule a job to run in 10 seconds, we can send a message to SQS with a delay of 10 seconds. SQS will then deliver the message to our worker after the delay. Easy as that!

This eliminates the need for managing infrastructure while providing all the features we need out of the box. The delivery delay feature ensures messages remain invisible until their scheduled execution time, while visibility timeouts handle worker failures after messages are consumed. Dead-letter queues capture failed jobs for investigation. High availability is guaranteed across multiple availability zones, and the service scales automatically to handle our load. The native delayed message delivery feature effectively turns SQS into a distributed priority queue, but without the operational complexity of managing it ourselves.

For our use case, SQS would be the best choice due to its native support for delayed message delivery (effectively making it a priority queue based on time), automatic handling of worker failures through visibility timeouts, and excellent scaling characteristics for our load of 10k jobs per second.

Timely Execution

To recap, our new two-layered scheduler architecture looks like this:

1.  A user creates a new job which is written to the database.
    
2.  A cron job runs every 5 minutes to query the database for jobs that are due for execution in the next ~5 minutes.
    
3.  The cron job sends these jobs to SQS with appropriate delay values.
    
4.  Workers receive messages from SQS when they're ready for execution.
    
5.  If a new job is created with a scheduled time < 5 minutes from the current time, it's sent directly to SQS with the appropriate delay.
    

Keep in mind that many interviewers or companies will prefer that you avoid managed services like SQS. If this is the case, you can implement your own priority queue using Redis or a similar data store. Just check in with them to get a gauge on what is and isn't allowed.

### 2) How can we ensure the system is scalable to support up to 10k jobs per second?

In any interview, when you get to talking about scale my suggestion is to work left to right looking for any bottlenecks and addressing them one-by-one.

**Job Creation**

For us, this means starting with job creation. If job creation was evenly distributed with job execution, this would mean we have 10k jobs being created per second. This is unlikely, since we support recurring jobs, but it's an upper bound. I would likely ask my interviewer at this point if we have any expectations about what percent of jobs are recurring vs. one-time. If the proportion is small, meaning most jobs are one-time, then our Job Creation service might end up being a bottleneck.

To handle high job creation rates, we can introduce a message queue like Kafka or RabbitMQ between our API and Job Creation service. This queue acts as a buffer during traffic spikes, allowing the Job Creation service to process requests at a sustainable rate rather than being overwhelmed. The message queue also enables horizontal scaling by letting us add more Job Creation service instances as consumers, while providing durability to ensure no job creation requests are lost if the service experiences issues.

Adding a message queue between the API and Job Creation service is likely overcomplicating the design. The database should be able to handle the write throughput directly, and we'll always need some service in front of it that can scale horizontally. Unless there's expensive business logic in the Job Creation service that we want to isolate, it's better to keep the architecture simpler and just scale the service itself.

In an interview setting, while this pattern might impress some interviewers, it's worth noting that simpler solutions are often better. Focus on properly scaling your database and service layer before adding additional components. I'll leave it in the diagram but consider it optional.

Scalability

**Jobs DB**

As discussed earlier, we chose DynamoDB or Cassandra for our Jobs and Executions tables. This was a good choice for scale since DynamoDB can handle thousands of writes per second per partition. We've already set up our partition keys optimally:

-   Jobs table: Partitioned by job\_id to distribute load across jobs
    
-   Executions table: Partitioned by execution\_time to enable efficient querying of upcoming jobs
    

With proper provisioning, these tables should easily handle our load of 10k operations per second and scale nearly infinitely to support an increasing number of jobs.

Notably, once a job has been executed, we need to keep it around for users to query. But once a reasonable amount of time has passed (say 1 year) we can move it off to a cheaper storage solution like S3.

**Message Queue Capacity**

Let's do some quick math to understand our SQS requirements. For each 5-minute window, we need to process around 3 million jobs (10k jobs/second \* 300 seconds). While SQS messages can be up to 256KB in size, our messages will be quite small, around 200 bytes each containing just the job ID, execution time, and metadata. That's just 600MB of data per 5-minute window.

SQS automatically handles scaling and message distribution across consumers, which is one of its biggest advantages. We don't need to worry about manual sharding or partitioning - AWS handles all of that for us under the hood. The service can handle our throughput requirements of 10,000 messages per second without any special configuration, though we'll need to request a quota increase from AWS as the default limit is 3,000 messages per second with batching.

We might still want multiple queues, but only for functional separation (like different queues for different job priorities or types), not for scaling purposes.

Scale

Even if you went with the Redis priority queue, 3 million jobs would easily fit in memory, so there's nothing to worry about there. You would just end up being more concerned with fault tolerance in case Redis goes down.

**Workers**

For the worker layer, we need to carefully consider our compute options. The two main choices are containers (using ECS or Kubernetes) and Lambda functions.

1.  **Containers**: tend to be more cost-effective for steady workloads and are better suited for long-running jobs since they maintain state between executions. However, they do require more operational overhead and don't scale as elastically as serverless options.
    
2.  **Lambda functions**: are truly serverless with minimal operational overhead. They're perfect for short-lived jobs under 15 minutes and can auto-scale instantly to match workload. The main drawbacks are that cold starts could impact our 2-second precision requirement, and they tend to be more expensive for steady, high-volume workloads like ours.
    

Given our requirements - processing 10k jobs per second with 2-second precision in a steady, predictable workload - I'd use containers with ECS and auto-scaling groups, but there is no wrong answer here. Containers give us the best balance of cost efficiency and operational simplicity while meeting our performance needs. We can optimize this setup by using spot instances to reduce costs, configuring auto-scaling based on queue depth, pre-warming the container pool to handle our baseline load, and setting up scaling policies to handle any unexpected spikes in demand.

Workers

### 3) How can we ensure at-least-once execution of jobs?

Our main concern is how we process failures. If a worker fails to process a job for any reason, we want to ensure that the job is retried a reasonable number of times before giving up (let's say 3 retries per job).

Importantly, jobs can fail within a worker for one of two reasons:

1.  **Visible failure**: The job fails visibly in some way. Most likely this is because of a bug in the task code or incorrect input parameters.
    
2.  **Invisible failure**: The job fails invisibly. Most likely this means the worker itself went down.
    

Let's start with the easier of the two cases: **Visible failures**.

For visible failures, we are careful to wrap the task code in a try/catch block so that we can log the error and mark the job as failed. We can then retry the job a few times with exponential backoff before giving up. Upon failure, we'll write to the Executions table to update the job's status to RETRYING with the number of attempts made so far.

We can then put the job back into the Message Queue (SQS) to be retried again in a few seconds (exponentially increasing this delay with each failure). If a job is retried 3 times and still fails, we can mark its status as FAILED in the Executions table and the job will no longer be retried.

Fortunately, SQS even handles exponential backoff for us out of the box, making the implementation of this pattern quite simple.

Now let's consider the more interesting case: **Invisible failures**. When a worker crashes or becomes unresponsive, we need a reliable way to detect this and retry the job. Let's examine different approaches to handling these failures:

### 

Bad Solution: Health Check Endpoints

###### Approach

One solution is to implement a health monitoring system where each worker exposes a health check endpoint like GET /health. A central monitoring service continuously polls these endpoints at regular intervals, typically every few seconds. When a worker fails to respond to multiple consecutive health checks, the monitoring service marks it as failed and its jobs are reassigned to healthy workers.

###### Challenges

Health checks don't scale well in a distributed system with thousands of workers, as the monitoring service needs to constantly poll each one. Network issues between the monitoring service and workers can trigger false positives, leading to unnecessary job retries. The approach also requires building and maintaining additional infrastructure for health monitoring, and complex coordination is needed between the health checker and job scheduler to handle race conditions. Not to mention the issues that arise if a monitoring service itself goes down.

### 

Good Solution: Job Leasing

###### Approach

Job leasing implements a distributed locking mechanism using a database to track ownership of jobs. When a worker wants to process a job, it first attempts to acquire a lease by updating the job record with its worker ID and an expiration timestamp.

For example, if Worker A wants to process Job 123, it would update the database: "Job 123 is leased to Worker A until 10:30:15". While processing the job, the worker must periodically extend its lease by updating the expiration timestamp. If Worker A is still processing at 10:30:00, it would update the lease to expire at 10:30:30, giving itself another 15 seconds. If Worker A crashes or becomes unresponsive, it will fail to renew its lease, and when the expiration time passes, another worker can acquire the lease and retry the job.

###### Challenges

This pattern requires careful implementation to handle distributed system complexities. The database must process frequent lease renewal operations, which at our scale of 10k jobs per second could mean 50k lease updates per second (assuming 5-second lease intervals).

Clock synchronization between workers becomes important too - if Worker A's clock says 10:30:00 but Worker B's clock says 10:30:20, Worker B might steal the job while Worker A is still processing. Network partitions create additional complexity: if Worker A can't reach the database to renew its lease but is still processing the job, another worker might start processing the same job, leading to duplicate execution.

### 

Great Solution: SQS Visibility Timeout

###### Approach

Amazon SQS provides a built-in mechanism for handling worker failures through visibility timeouts. When a worker receives a message from the queue, SQS automatically makes that message invisible to other workers for a configurable period. The worker processes the message and deletes it upon successful completion. If the worker crashes or fails to process the message within the visibility timeout period, SQS automatically makes the message visible again for other workers to process.

To optimize for quick failure recovery while still supporting longer-running jobs, we can set a relatively short visibility timeout (e.g. 30 seconds) and have workers periodically "heartbeat" by calling the ChangeMessageVisibility API to extend the timeout. For example, a worker processing a 5-minute job would extend the visibility timeout every 15 seconds. This way, if the worker crashes, other workers can pick up the job within 30 seconds rather than waiting for a longer timeout to expire.

###### Benefits

This approach handles worker failures without requiring any additional infrastructure or complex coordination. The combination of short visibility timeouts and periodic heartbeating gives us fast failure detection while still supporting long-running jobs. Failed messages are automatically moved to a dead-letter queue after a configurable number of retries, and the heartbeat mechanism ensures we maintain precise control over job ownership without adding complexity.

Retries

Lastly, one consequence of at-least-once execution is that we need to ensure our task code is idempotent. In other words, running the task multiple times should have the same outcome as running it just once.

Here are a few ways we can handle idempotency:

### 

Bad Solution: No Idempotency Controls

###### Approach

The simplest approach is to just execute the job every time it's received. For example, if the job is "send welcome email", we would send the email each time the job is processed, potentially resulting in duplicate emails.

###### Challenges

This can lead to serious data consistency issues. A job that transfers money could transfer it multiple times. A job that updates a counter could increment it too many times. This is clearly not acceptable for most real-world applications.

### 

Good Solution: Deduplication Table

###### Approach

Before executing a job, we check a deduplication table to see if this specific job execution has already been processed. We store each successful job execution with a unique identifier combining the job ID and execution timestamp. If we find a matching record, we skip the execution.

###### Challenges

This adds database operations to every job execution and requires maintaining another table. The deduplication table needs to be cleaned up periodically to prevent unbounded growth. There's also a small race condition window between checking and writing to the deduplication table.

### 

Great Solution: Idempotent Job Design

###### Approach

Design jobs to be naturally idempotent by using idempotency keys and conditional operations. For example, instead of "increment counter", the job would be "set counter to X". Instead of "send welcome email", we'd first check if the welcome email flag is already set in the user's profile. Each job execution includes a unique identifier that downstream services can use to deduplicate requests.

This is the most robust approach, essentially offloading idempotency concerns to the task's implementation.

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

### Mid-level

Mid-level candidates are expected to be able to produce a reasonably high-level design with a clear and cohesive data model. From there, they should be able to respond appropriately to interviewer inquiries about durability, precise scheduling, and retries. I don't expect them to know the right answers to these questions without guidance, but I do expect to see strong problem-solving skills and an ability to work with me, the interviewer, to arrive at reasonable solutions.

### Senior

For senior candidates, I expect much more autonomy. They should lead the conversation around two deep dives and show some hands-on experience with any of the relevant technologies. They should specifically be able to articulate the need for a two-phased architecture, even if they choose Redis or a different option for the priority queue. I want to see that they proactively identify issues and quickly adapt to scenarios I throw their way, like "what happens if a worker goes down mid-job execution."

### Staff+

Staff candidates are the cream of the crop. I expect they breeze through the setup and the high-level design and spend most of their time leading deep dives. They should proactively lead the discussion and understand the major bottlenecks of the system, proposing well-justified solutions. Typically, they'll have a deep understanding of at least one technology in question and be able to explain, justified with hands-on experience, why one approach is better than another. As always, while not a strict requirement, the best staff candidates have the ability to teach the interviewer something, no matter how small.

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

A

ArtisticHarlequinKiwi700

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3n3c6mc0071o5pb4kqhv1ue)

I do not understand the use of execution time as the partition key for the Executions table since it is unclear how this key is hashed into underlying partitions (meaning that you will likely end up reading from all shards anyways)

Show more

5

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3n87ju8008so0ypzjjkxr6n)

Great question. You can use the rounded Unix time as the partition key so that all jobs for, say, the same hour are hashed together. Then, have the actual execution time be the sort key.

Show more

6

Reply

A

ArtisticHarlequinKiwi700

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3nb9vcf00bwkminqfhvzpya)

Yeah, I that's a valid point. I'd probably lean towards a more explicit sharding scheme (eg just mysql) and then process in parallel by shard.

Show more

1

Reply

![Yiju Fang](https://lh3.googleusercontent.com/a/ACg8ocJX8uhUuyIomhJZOI_ptJvGgmuPkVz28EomlfpLnZo01mIGdAfa=s96-c)

Yiju Fang

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3zh2hat015efy899wxkefbw)

But, wouldn't the current hour partition become hot partition since all new created job in the the current hour will write to that partition?

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm403fi3x008bc18cn73sg0ey)

Good question, maybe. It depends on what our write throughput is, which would depend on how many jobs are recurring versus single execution. If it becomes a problem, we could batch our writes with a message queue.

Show more

1

Reply

![Yiju Fang](https://lh3.googleusercontent.com/a/ACg8ocJX8uhUuyIomhJZOI_ptJvGgmuPkVz28EomlfpLnZo01mIGdAfa=s96-c)

Yiju Fang

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm406bniu00b8vvr5zuhh3e0u)

Yeah, think of a Black Friday flash sales in the 1st hours. We will probably see a spike of place order jobs created during that hour and all these jobs will write to the same hour key/partition. But you are right, we can use a queue to batch write these jobs.

Show more

5

Reply

![Arunprasaath S](https://lh3.googleusercontent.com/a/ACg8ocIsFb8OQWBCy7F3UjpYP8e_gwbRxgFsYc1706I-EQXYBt5CdwD5=s96-c)

Arunprasaath S

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cma5g0898008lad08gd4s4id7)

Each and every order should not be a job. These are async workflows. Probably one execution should process multiple orders. There will be spikes but generally Job schedulers should not be having same number of increase corresponding to site traffic.

Show more

1

Reply

![indavarapu aneesh](https://lh3.googleusercontent.com/a/ACg8ocLx77-thRGA5bldZDZhNF8MbwtxB4dZmFZ3zHzbk_Xu4IB-og=s96-c)

indavarapu aneesh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmbp3xp5r009q07ad6k84eixe)

we could instead create the partition key on a 15 minute window basis. for example: 0-14 -> 1, 15-29 -> 2, 30-44 -> 3, 45-59 -> 4 so the partition key would 177474747-1 (unix timestamp in hour + window key).

Show more

0

Reply

![foram mehta](https://lh3.googleusercontent.com/a/ACg8ocLylUhCzglsOCakPGM4mS48a10ZRe3tJMJTnfnUrRBins40kIcw=s96-c)

foram mehta

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9kobb4t00c6ad08ewc26i96)

True it can become a hot partition. I had same question we can work around that by adding a random to generate the partition\_key. example :- partition\_key = time\_bucket + "#" + hash(job\_id)

Show more

1

Reply

I

ImmediateIvoryAngelfish117

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmajjew6f01z2ad08b7f9sel0)

How would you fetch all jobs for the current hour? there is no prefix matching in get by partition.

Show more

3

Reply

![Laura Kingsley](https://lh3.googleusercontent.com/a/ACg8ocJ9q5efeIAV9-4baMUJzECFWa5KvYItsnN0sXMFrwjCpGa8vQ=s96-c)

Laura Kingsley

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3njeb6u00kummna4zwtn7oq)

Evan, you are the best :)

Show more

24

Reply

F

FitBlackPython669

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcgoc2ga00gkad08k8mdcik6)

+1

Show more

1

Reply

C

CulturalAmethystTrout404

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3ntd22s00y1o5pbt0sjk0o4)

Thanks for the post! For job leasing part, when the lease time expires, how does another worker know and go grab the task? Especially we have a queue so that task expired needs to enqueue again I guess? So if we use Redis, can Redis inform the Execution service to do the enqueue at the time of lease expiry?

Asking this because I have encountered a variation of this problem where each task has up to 10s to execute. With this constraint, this option is appealing as lease renewal is not required

Show more

4

Reply

C

CulturalAmethystTrout404

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3nter1r00w5kminp6h4tush)

Essentially can Redis do something like a callback function at the time of ttl expiry, or can the execution service know that a entry in Redis is expired and do something?

Show more

2

Reply

![Anirudh Kaki](https://lh3.googleusercontent.com/a/ACg8ocJnh2a8FYhXHXBBH9NOrnr_OXXIPpM3ux4a_ZuCuuIQo8dQcnPF=s96-c)

Anirudh Kaki

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4pyfy0w02lelkt9orz5tzly)

Redis offers a feature called Keyspace Notifications that can notify clients about certain events, including key expirations. Clients subscribe to notification channels.

or Redis Streams with TTL to create a queue-like system

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4qb41gr00do10dl88pfslaf)

Be careful with this, the keyspace notifications for expirations don't work like you might think.

Show more

3

Reply

![Apoorv Gupta](https://lh3.googleusercontent.com/a/ACg8ocLtDeHnvwoH7ycKpx5vFTieLUlafrPEWSsW3hJuIaEfLIQXaA=s96-c)

Apoorv Gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdgx1lpk02goad09v2t1kfj9)

Can someone expand on using keyspace notifications to implement a leasing solution? It looks like a very good solution to me. You can create a key in Redis for each job that is currently being executed. The host that is running the jobs will set TTL = 30 seconds on that key and refresh it every 10 seconds. If that host fails, the key will get deleted acc to TTL.

Other hosts can use a wildcard to subscribe to key deletion notifications. The first host to respond and re-add the key to Redis will own the job.

Show more

0

Reply

A

abrar.a.hussain

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmayhzwb001wead08gfceqceb)

Bad idea no? It's at most once semantics so if you're relying on that to "unexpire" something you're going to be stuck in limbo

Show more

0

Reply

![indavarapu aneesh](https://lh3.googleusercontent.com/a/ACg8ocLx77-thRGA5bldZDZhNF8MbwtxB4dZmFZ3zHzbk_Xu4IB-og=s96-c)

indavarapu aneesh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmbp40dv5009s07ad6my12gf0)

I think the kafka consumer group protocol is great way to model the retry mechanism for the jobs. If the consumer doesn't commit the offset in the stipulated time (configured value), then the consumer is longer the leader of the topic partition and all of its partitions are handed off to another consumer.

Show more

0

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmc65pt8604laad08u4jp75ma)

if we want to use redis, can't we do like: watcher puts the job execution instance to redis sorted set. consumer worker pulls out the job\_execution from the sorted set and put it back to sorted set with (cutrenttime+15s) as score and make job\_execution invisible to other workers, so it can be picked up later. worker also updates itself in the worker field and update last heartbeat time in db every 5 sec.

now consumer worker executes the execution and updates the last heartbeat. if now lets say that worker is still executing the job and record in redis picked up again by some other worker it will only pick the job for execution if the job execution is in progress state and last heartbeat was atleast 3 cycles (15 sec) before current time otherwise just insert back to redis sorted set with currenttime + 15sec

Show more

0

Reply

![Ranjith Bodla](https://lh3.googleusercontent.com/a/ACg8ocKLJMxrLwhHozGjRCDrGk37yaJo6NmxaWnwfsdCuQGHHO52dZ1hIw=s96-c)

Ranjith Bodla

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3pels7u0295o0yp715ermhg)

"Second, when a recurring job completes, we can easily schedule its next occurrence by calculating the next execution time and creating a new entry in the Executions table." -> Given in our dive deep, if our watcher is polling every 5mins, if we have a job with å CRON expression to run say every 5 secs, we need to have 60 (300s/5s) job execution rows inserted in the executions table and into the message queue.

We can split this responsibility between job worker to insert 60 rows in execution table and watcher queuing the jobs with the appropriate delay time.

The challenge with the job worker's new responsibility is what if the row insertion fails after n (n < 60) insertions.

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3pfgzxf02a2o0ypbbti8r09)

Agree, but you lost me on the last sentence. Can you say it differently?

Show more

0

Reply

![Ranjith Bodla](https://lh3.googleusercontent.com/a/ACg8ocKLJMxrLwhHozGjRCDrGk37yaJo6NmxaWnwfsdCuQGHHO52dZ1hIw=s96-c)

Ranjith Bodla

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3qrmjem00g5ahfg9mpdr8s2)

Sure. Let me elaborate. Job worker need to now maintain the state of the # of rows inserted and if it fails OR worker dies after partial insertion, we need to somehow ensure the remaining rows gets inserted. Otherwise, we will miss out on those jobs being executed. This adds complexity to the job worker logic.

Show more

0

Reply

Y

YammeringTealBovid101

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4ereea9042i1i3rvnm358dr)

Wondering how to ensure durability, especially the potential of execution lost, when creating new execution records. If we are to insert 60 records to DB, need to ensure they are in 1 db transaction. But what if DB insert failed? Should worker be responsible for retry? Or should we alert the client? Or some auto-healing job to check for missing jobs. Durability & Disaster Recovery seems a challenging topic for deep dive. And might probably worth mention in the article.

Show more

1

Reply

A

aniu.reg

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4vjovey006x11628o9u3pxq)

Why is this a problem? Every 5 minutes, watcher inserts 60 rows into the execution table and sends 60 messages to the SQS. It is the watcher's responsibility to persist 60 rows into the Execution table. Workers do the execution and update the Execution table.

Show more

0

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmc663r2j04miad08wiaup75e)

should we have separate workers for the generation of execution instance of job, which runs in background may be once a day and generates all the execution instances for next day. when the job is submitted at that time generates at execution instances for next 24 hours and then runs regularly as well.

Show more

0

Reply

![Rich Davis](https://lh3.googleusercontent.com/a/ACg8ocL6NUQvatXIxmcCfR3FlXwcTq8lv1X1y24KlSP8_fVtg5ujQr4W=s96-c)

Rich Davis

[• 20 days ago• edited 20 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdw7ld6e03qdad08mtovrotr)

I would yes. I'd opt for something like a have a proto registry where teams can first configure a profile, which would be an umbrella for potentially several tasks they'd own and then task definitions. On pull request, this would create a kubernetes service object up to n replicas that we'd force them to specify (keep costs low) with deployment labels like profile={profile} and task={task} if a new task.

Then we could isolate dedicated worker pools for each task and have great costing metrics for the team owning the task. This would avoid noisy neighbors (large issue here imo) by isolating capacity to each profile and task while unlocking offering batch tiers (best effort, clumped together for workloads that aren't critical) and live tiers (individual worker pools and brokers per profile) for more critical workloads.

This natively distributes query latency in a simple manner while providing significant performance and stability enhancements.

Show more

0

Reply

I

IndividualGrayToucan573

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3rztu5f01brhft2yykbcj07)

Does it make sense that job creating service sends one-time requests directly to message queue bypassing cron jobs? one-time requests need to be scheduled within 2 seconds. If sending them to database and letting cron query db once every 5 minutes, how can 2s scheduling be guaranteed?

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3s29f4401ephft2slloocb0)

Yah exactly! Those go right to the message queue (still persist in DB, but in parallel)

Show more

1

Reply

![Yilun Zhu](https://lh3.googleusercontent.com/a/ACg8ocJSv0R51SdCPNA1RtUiiulhYR8DPiIJxUYyYc4_xZNHjjYTDw=s96-c)

Yilun Zhu

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8ttojs700eqad08siswsds0)

What if the newly created job is set to be executed not "immediately" but something like after 2 minutes(or any time < 5 minutes && not immediately)? I think ZSET could handle it as it's a PQ while not sure what happened to delayed message queues

Show more

0

Reply

![udit agrawal](https://lh3.googleusercontent.com/a/ACg8ocLEGap_XwS1Mcu4vZkpJXuJxMhH6Ely6OgAoxbvOhxGeRkRQzQD=s96-c)

udit agrawal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmc668slj04n0ad082uxuf39n)

any job which needs to run in next 5-10 min must also be put directly in the queue as well to avoid missing the execution and parallel putting next instance in execution db as well

Show more

0

Reply

X

XenialCoffeeBedbug734

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm3va0ugc01dzmrifabku4n4o)

I see some potential race conditions happening when the cron job is running to collect the scheduled jobs.

Say at 12PM, the cron jobs scans the DB to collect all jobs to be run in the next 5min.

At that very same instance, a task is submitted, scheduled to run at 12.05PM. We do not put it in the message queue since our threshold to do that is for tasks in less than 5 min, and only persist to the DB.

At 12.05PM, the cron job runs again. Even though it captures the task specified to run at that minute, our 2 second guarantee is then breached.

How do we handle edge cases like that?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm403e8lm006kg910c2a0pd2q)

Just have some overlap.

If you scan every 5 minutes, then have new jobs scheduled to be run within the next 10 minutes just go right into the queue.

Then you just need some way to dedup.

Show more

7

Reply

![Umesh](https://lh3.googleusercontent.com/a/ACg8ocLmLJZblgX0-Lfb0a9dl8XvVnJhTYOCosXWSlRga4qQ9o7n6OE=s96-c)

Umesh

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmapmeqzx00r9ad08gp1zt4vp)

I have one doubt here the polling service need to pull approx 3 million rows from DB in every run and need to check push to sqs & need to create next jobs for the same ,so it have to fetch in batches and create next jobs for the same , so single thread will be able to able to handle the same like how much time it will take in finishing the same and we need to maintain a flag or something till how much time jobs have been already scheduled so if the current service crashed next can pick from the flag and start from there only or we will have multiple polling services in that case how do we handle do partion reading ?

Show more

3

Reply

![Ujjwal Gulecha](https://lh3.googleusercontent.com/a/ACg8ocKPaMQoLjGzF4AyTATjyaihMVxW8Gnv8ejRrA9uEN9ICxvCT3q-UQ=s96-c)

Ujjwal Gulecha

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcqzxkbp01hxad07on0n7urd)

same question. i think we could have multiple instances of polling service and they each query a small subset of the entries in the database. how? we could use zookeeper to assign a "shard" to each polling server. then the query it will make is for a smaller subset. say we have 3 million rows, could have say 2k shards and each polling instance could query maybe 5 shards. the shard could be on jobId

just thinking out loud

Show more

0

Reply

H

hellointerview

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm41ngol5002yu5ybrclaf58x)

In the Executions table example, wouldn't there be a key collision if two executions were scheduled for the exact same time?

{
  "time\_bucket": 1715547600,  // Partition key (Unix timestamp rounded down to hour)
  "execution\_time": 1715548800,  // Sort key (exact execution time)
  "job\_id": "123e4567-e89b-12d3-a456-426614174000",
  ....
}

If so, perhaps a solution could be to concatenate the job\_id to the end of the execution\_time so that it is still sorted by execution\_time in a partition but avoids a collision. Any thoughts on this?

{
  "time\_bucket": 1715547600,  // Partition key (Unix timestamp rounded down to hour)
  "execution\_time\_job\_id": "1715548800#123e4567-e89b-12d3-a456-426614174000",  // Sort key (execution time + job ID)
  "execution\_time": 1715548800,  // Exact execution time for querying
  "job\_id": "123e4567-e89b-12d3-a456-426614174000",  // Unique job ID
  ...
}

Show more

5

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm41nkogy0027nugzs7opsm0g)

Yeah, good catch. So the partition key is used in DDB to determine the shard, so that needs to remain the time bucket, but the sort key can be time-jobId, and then we can use begins\_with when searching.

Show more

7

Reply

G

GiganticApricotSwordfish902

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm43dlwqp016ay518e34kyayz)

How would you capture job dependencies in this design? For example if job A runs when job B and job C are completed how would we best track this for each job run if they are all recurring jobs?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm43esi0c0190e3kq2v9vp28w)

At a very high level, you'd add a dependencies field to the Jobs table that lists which jobs need to complete first, and then track the specific execution IDs of completed dependencies in the Executions table. When parent jobs complete, you'd update a list of completed dependencies, and once all dependencies are met, the dependent job would be scheduled.

Show more

0

Reply

F

FlutteringCrimsonKite231

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4xq21js01psu8w5nm3foqlp)

why not combine them into one job with multiple steps? we don't care what's inside a job anyways why bother take over this application level dependencies

Show more

1

Reply

L

LooseAquaJaguar382

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6y4x4er0125cv3wqro01yr1)

We can't combine jobs from a user perspective. Users define jobs intentionally to be executed as separate units of work, probably with different dependencies, resource needs, and other configurations like retry policies, caching, etc. Even if we did, 1 big job is poor design. Much greater likelihood to fail, costly to retry a longer running job, and hard to customize the runs of the steps, each with probably different requirements.

Show more

1

Reply

P

PrivateMaroonSwordfish354

[• 11 hours ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmep2ynfs03aiad08vm7tbcl8)

There is an issue with write amplification here. I'd rather go with an event driven route.

Evan if you can specficially make a video around DAG Job scheduling though that'd be great!

Show more

0

Reply

L

LooseAquaJaguar382

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6y48wcb010dcv3wp7aa1137)

This is a complex question that can be its own system design question. Just look into any workflow orchestrator (Airflow, etc) and how they build and manage DAGS. There are "workflows" composed of tasks (jobs), and then workflow instances (executions). There are many considerations, like whether the DAG is built at compile time or determined at runtime (dynamic).

Show more

1

Reply

![indavarapu aneesh](https://lh3.googleusercontent.com/a/ACg8ocLx77-thRGA5bldZDZhNF8MbwtxB4dZmFZ3zHzbk_Xu4IB-og=s96-c)

indavarapu aneesh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmbp492fl009v07adloelbvf9)

In that case we're no longer looking jobs, instead we're looking at DAG workflows (pipelines). This would be something similar to Gitlab pipelines. have a yaml spec for listing the dependencies and use topological sort to process them. You can parallize the jobs at the same level in topo sort. Btw most job scheduler in the outer world don't promise an sla of 2 secs. Rather they promise that the job is executed after the scheduled time.

Show more

0

Reply

![Liliiia Rafikova](https://lh3.googleusercontent.com/a/ACg8ocLhKI61yr7h7HM0rSPvHT8QjFtJteOZbg84lT2Kk6f-YRWWSw=s96-c)

Liliiia Rafikova

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm44e27eg00ds3bhccybc2uuc)

Why we are saying above that we want to process 10K jobs per second and later we mention about retry and invisible timeout to 30 seconds?

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm44uo5bt00x53bhcfdccg3mx)

10k jobs is just to learn the scale at which we want to operate and how backed up our queue could get, thus we need way more workers to process these. the 30 sec invisibility is to ensure only one worker picks any given job at a time. some of these jobs could fail, we still want to retry them- not immediately but with exponential back off. the post doesn’t talk much about how back off is handled with sqs

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm44rsfi000upulx83zkndxtb)

execution table is partitioned by time. wouldn’t that cause hot partitions during some specific hours when most jobs are scheduled?

Show more

4

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm44rv8sc00v6siamoaryhoou)

in hindsight, this is a actually good since we know how many workers to schedule upfront for certain hours.

Show more

0

Reply

A

aniu.reg

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4vjxuf8007yy3n12in376gf)

The access pattern here is the watcher needs to read executions within a specific time range. So patitioning data by time avoids reading from all shards in a single watcher run. It is somehow "hot partition" by design.

Show more

2

Reply

D

DeliciousScarletFelidae259

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm44zc0pz014nulx8rllsmthp)

When scaling up the Amazon SQS the partition is using jobs execution timestamp. Does it mean only one of these partitions will be consumed at time?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm46lr3wv01q73o1hm37ppukl)

Actually, SQS doesn't use partitions in the same way as Kafka or other message brokers. When we scale SQS, we're typically creating separate queues (not partitions) for different time windows, and multiple workers can consume from each queue simultaneously. The delay parameter on messages controls when they become visible, not which partition they go to.

So with SQS, we don't actually need to manually shard at all - it's one of its biggest advantages. AWS automatically handles the distribution of messages and scaling under the hood. So it can handle nearly unlimited throughput by automatically partitioning internally across their infrastructure.

The only time you might want multiple queues is for functional separation (like having different queues for different job types or priorities), not for scaling purposes. I'll update the article to remove the sharding discussion for SQS as it's misleading - that's more relevant for solutions like Kafka or Redis.

Show more

2

Reply

![Shiladitya dey](https://lh3.googleusercontent.com/a/ACg8ocKI7ltwlBz_zSVkLIV-8JTTTwX6vLrvPrHv7emsysZ_R_zK1g=s96-c)

Shiladitya dey

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9i5zc1100w0ad08e2iwhn6m)

I believe the whole argument in favor of SQS is predicated on the simple premise that it supports delayed delivery. Can I request your opinion on the below design while I'm still skewing with Kafka.

Expectation: The message queue should be durable with the ability to support cross region fault tolerance. Once a job\_instance (let's call it a 'run') has been "enqueued" the queue should be my source of truth even in the face of a region level outage.

Challenge with SQS: whilst it supports DC / AZ level replication, cross region replication is not supported and hence not 'durable' in that sense.

Short delay handling with Kafka: ◦ Challenge: Kafka can’t delay messages natively, and 2 seconds is too short for external schedulers. ◦ Solution: In-memory buffering with immediate requeue: ▪ Workers hold messages with exec\_time > now in memory (max 2-second buffer). ▪ Use a timer (e.g., Java ScheduledExecutorService) to recheck every 100ms. ▪ When exec\_time arrives (within 2 seconds), process the job. ▪ Fallback: If a worker crashes, unprocessed messages remain in Kafka (offset not committed), picked up by another worker. Alternatively - I can design the workers as replicated state machines with ZK tracking the Leader (active worker). Once the leader crashes, one of the followers with replicated state would start processing with nearly no delay.

DR Strategy: Kafka: {enqueued jobs} Mirrormaker to replicate at near real time to the DR region. DB: {pending jobs} Standard bin-log or snapshot based DB replication to DR with measurable RPO / RTO.

Show more

0

Reply

![Abhimanyu Seth](https://lh3.googleusercontent.com/a/ACg8ocLumt5NND_D0T909iqztGu4UWWRoIgzKkZKNjr8UB5zwNARqCKlzQ=s96-c)

Abhimanyu Seth

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm46kh11901q9129ijz8x00sk)

Overall, the docker approach sounds great, but I have some doubts about how it works in practice.

Each task code has to be stored somewhere. We can't assume they'll be trivial and we could store them in our DB and execute them on generic docker containers. Most likely, they'll need to have a docker image. With such a high scale of 10K jobs per second, maybe we can assume there are 100K unique tasks. So the k8s node will need to prefetch all 100K docker images on every node in the node group. I'm not sure how practical this is?

Also, if we have X workers reading from SQS, how do they trigger the k8s pods? Using the kubernetes client APIs? I'm not sure how that would scale for triggering 10K pods per second.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm46lxktc004s4of9b7mrropl)

This much detail in job execution is typically out of scope for a job scheduler interview question, but these are fair points.

This often works with a single "worker" Docker image containing a job execution framework. Making the assumptions engineers just extend some class in a common language to add new tasks. This worker then:

1.  Pulls task code from a secure artifact store/repo
2.  Uses a factory pattern to instantiate the right task class with the input params
3.  Runs as a long-lived container that processes multiple jobs

The containers poll SQS continuously - we're not creating new pods per job. Security and isolation considerations are important but typically handled at the organization level through authentication, code signing, and proper error handling with the retry mechanisms we discussed.

Show more

4

Reply

![Abhimanyu Seth](https://lh3.googleusercontent.com/a/ACg8ocLumt5NND_D0T909iqztGu4UWWRoIgzKkZKNjr8UB5zwNARqCKlzQ=s96-c)

Abhimanyu Seth

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm46nire201u5129ivu71sneu)

Thank you! That sounds good. In the final diagram, there are separate boxes for worker consume job -> worker execute job, which made me think the worker was not the docker container.

Yeah, agree this is beyond the scope of interview, but if I don't understand it in-depth, I am not able to speak confidently about it during interview.

Show more

1

Reply

R

ResponsibleIvoryAlbatross601

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm81atvin00083j88vd4eeroy)

What is the benefit of using containers over VMs here?

Show more

0

Reply

![Shiladitya dey](https://lh3.googleusercontent.com/a/ACg8ocKI7ltwlBz_zSVkLIV-8JTTTwX6vLrvPrHv7emsysZ_R_zK1g=s96-c)

Shiladitya dey

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9ih1bnu0150ad08u5gvkl6r)

In a system similar to this I built earlier, the job\_definition table had a a path to the image. The executor subsystem would just have to pull the image and run in a container (some container orchestrator such as mesos, k8s etc. would do the job).

Show more

0

Reply

L

LinguisticRedViper140

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4bqaacq00deurisvwwzfdrw)

Can you please explain this a bit more clearly: "The system should be highly available (availability > consistency)." Like you say it is easy to mention the words but it should be in relation to this design. :) Does it mean available for the user to access the system but not consistent in terms of job status being instantaneous.

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4bqcsnr00crqfetciaw50kk)

Eventually, consistency for scheduling jobs is fine. If I add a job here in the US and someone doesn't see it for a couple of seconds in Europe—who cares? :)

Similarly, if a job finishes and its status is updated to DONE in the DB, but it takes a little bit for users to see that, that's totally okay.

Show more

6

Reply

![Shiladitya dey](https://lh3.googleusercontent.com/a/ACg8ocKI7ltwlBz_zSVkLIV-8JTTTwX6vLrvPrHv7emsysZ_R_zK1g=s96-c)

Shiladitya dey

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9nvbtww00iead08z347m8q9)

Evan, When I talk about consistency model in the context of databases, I tend to focus on ordering, consistency, determinism and replication (re-construction of state).

what you've described above - US -> Europe sounds more like a DR use-case. It is generally said that 'eventual consistency' is acceptable however, in a strict sense eventual consistency doesn't guarantee ordering. For a DR use-case I'd prefer Total order + Replayable Log

1.  Implies sequential consistency.
2.  Stronger than causal consistency because it forces a total order on all operations, even concurrent ones.
3.  Enables deterministic recovery and replication. Not concerned with client reads, only write and state convergence.

Can I request your views please.

Show more

0

Reply

G

GoldenCoralGazelle967

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4cdvvhg012q1i3rym7j9nxt)

Not sure who writes the first entry for a job in the Execution table. My guess would be the scheduler service when it receives the job and schedule from the user. Is my understanding correct?

If workers/consumers are inserting the next executions, do they insert new ones on success? How can we ensure next execution instance in in the DB if the worker fails?

Show more

4

Reply

A

aniu.reg

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4vk73up008phjxbmyesz38d)

IMO workers should not do insertion but only update existing Execution.

Show more

0

Reply

B

BraveGreenMackerel841

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5n5prlj009911a205gtia5x)

the scheduler service should create the first entry, the worker can create the next execution regardless of the current job status I think.

Show more

0

Reply

R

ResponsibleIvoryAlbatross601

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm822g1e300wkimsrxupa1ji8)

I think there should a separate job that pre-populate the future executions, lets say for the next 5 days. It runs every 5 days. The first set of executions can be created by the scheduler service when the job is created.

Show more

2

Reply

![Neeraj jain](https://lh3.googleusercontent.com/a/ACg8ocJAzrMQriHY2NYkdaOuS5_-r2zdrYIXz-3bABYqasyyHNOvtik=s96-c)

Neeraj jain

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm97dqh5b003qad08kltmqioc)

You can easily set a trigger or a CDC upon state change of a job which jobExecutors are doing, then within the trigger you can add the next schedule(and instead of adding just 1 schedule try to add multiple next schedules of the job to avoid re-calculation everytime)

Show more

0

Reply

![Ganesh kudva](https://lh3.googleusercontent.com/a/ACg8ocK2C_oCmIZR-ZZo3-ZXUbYU1kIag-S3N8bwyt31pfoWz-nRvkF3=s96-c)

Ganesh kudva

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4d7ufov01s6urisc1oxzx21)

Why do we want watcher to poll jobs from DB ? Can we not use CDC to directly enqueue jobs created in DB into SQS ?

Show more

0

Reply

N

NeutralAmethystUnicorn745

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4dqura202auurisqbucju7c)

I think it is because we only want to enqueue jobs that are close to the due time. If we enqueue jobs whose due time is very far in the future, it will be a great burden for SQS to have too many items that won't be dequeued anytime soon.

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4qbhivn00e610dl51xgkh31)

Correct. Also, SQS delivery delay has a cap. I actually thought this cap was 12 hours, but a Google search just suggested it may actually be only 15 minutes. In either case, you'd poll accordingly.

Show more

4

Reply

![Jose](https://lh3.googleusercontent.com/a/ACg8ocI0U4FzBIZeE_jngEBfFE3NF4Tj7WyqSOZo_DC7kBBEDA=s96-c)

Jose

[• 30 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdirip9t055vad08o5ab7hu1)

Could we just create Executions that are due now and CDC them to Workers instead of SQS?

Show more

0

Reply

N

NeutralAmethystUnicorn745

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4dqrqdc029gqfetufc6ns0s)

What happens if the watcher fails/restarts? First of all, the watcher needs to be distributed to avoid single point failure. And most likely it will be a lead-follower set up as only one watcher instance should be doing the polling. Now the leader goes dead or restarted/redeployed, we either need to wait for leader election to complete or wait for the restart/redeployment to finish. If the wait is longer than 5 minutes, we will miss "The system should execute jobs within 2s of their scheduled time" for the jobs that are scheduled to happen in the past 5-min window.

Show more

6

Reply

A

aniu.reg

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4vkjb7k0093hjxbmzpydhor)

1.  If a watcher fails, a stand by watcher can re-do the poll in the most recent 5 minute window and insert all the messages into the SQS. SQS can dedupe potential messages from old and new watcher (some dedupe logic needs to be added to SQS)
2.  How long is a typical leader election? Is 5 minutes enough for a leader election? If this is a concern the watcher extend its window to match the election SLA.

Show more

0

Reply

![Nick Elia](https://lh3.googleusercontent.com/a/ACg8ocLbmJbN7105wYizxpgZDhRwt48RoCiMKh6KWn71NXPMia2-9s3x=s96-c)

Nick Elia

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4gbvhml00c78jlrft2gjjs5)

Minor observation: the diagrams in the "Job Creation" section says "visibility timeout until execution time". I believe this was intended to say "Delay time until execution". Visibility timeouts are discussed later in the context of retries.

Also, "we'll need to request a quota increase from AWS as the default limit is 3,000 messages per second with batching." I don't think this is entirely accurate. This limit applies to FIFO-type SQS queues, but in this case we'd be using a Standard-type SQS queue (since FIFO does not support message-level delays, which we relied on in a previous section)

Big fan of the content, keep it up =)

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4qbfnfz00dy10dlkq4qehap)

Good catch! Updated this :)

Show more

0

Reply

C

ChemicalOrangeBaboon761

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4no2o0o00q8wlufwkdrspmf)

RE "GET /jobs?user\_id={user\_id}&status={status}&start\_time={start\_time}&end\_time={end\_time} -> Job\[\]", I remember you said the "user\_id" should not be directly passed in the query parameter. Could you explain why you did it here?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4qbfff800dv10dlistkx39r)

Fair enough. In this case, its used as a query param to filter for a set of jobs. More likely it would be &team or something like that. In either case, you still need to validate authorization.

Passing in the userId in the url isn't, on it's own, always bad. Its that you can't trust it. You always needs to validate the user via a more secure means.

Show more

0

Reply

![Sumit Gupta](https://lh3.googleusercontent.com/a/ACg8ocJn2KM4lh2TbluHMhvvzYK3t0a6m7JVVJIE6zNaS2hpKEHFFg=s96-c)

Sumit Gupta

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4rm3fn700e65ows2rkehtfs)

hi Evan, are you planning to add videos for the remaining questions?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4u5112r00qde4jkb5cpb78d)

Should have a couple more in the coming couple weeks. No guarantee there will be a video for all breakdowns though.

Show more

0

Reply

R

RealCyanHoverfly491

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4v9l64j00361162mm1g9zzv)

Hi! awesome post, thanks a lot!

Q: Let's assume we would like to add a requirement so users could edit/cancel already sent jobs. Meaning, assume a user's job already in our queue - what will be a valid approach for editing before it started? or cancel after it started?

Show more

1

Reply

F

FlutteringCrimsonKite231

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4xp4p9n01qzy3n1zfxy1jtt)

You can add a redis set for the edited/cancelled job, before executing the job, workers check if this job exists in the set.

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4zsvcp400r013vukxamfazz)

yup

Show more

1

Reply

A

aniu.reg

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4vkpxik008gy3n1ulvu4fkn)

At the "High Level Design",

"With the current design, querying jobs by userId would require an inefficient two-step process:"

Why does it need to query 2 tables to get executions of a user? Executions table already has user\_id you can just query one table ("Execution") by use\_id, right?

Show more

2

Reply

W

WoodenIvorySheep410

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm516s6z601rq13vuixmjr3es)

I had exactly the same doubt !!

Show more

0

Reply

F

FluffyAmberTarsier784

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6k0mvjl01r2jgz3qnfp2jcw)

I had the same thought. my conclusion was that since the Executions table has the job\_id as a partition key, it would be faster to query by that as opposed to user\_id.

So the train of thought for "how can I get all executions for a user" could be:

1.  Query Executions table by user\_id. This is slow because of table scan. Instead, what if I query by job\_id instead, since that's the partition key?
    
2.  In order to query by job\_id I need to get them from the Jobs table. I do that by querying the Jobs table for all job\_ids by user\_id. Now I have another table scan problem.
    

And that's how I think we land on the "inefficient two-step process".

Querying by user\_id might be worse since you may have much more Executions by a user than a Job by a user.

Show more

1

Reply

F

FreshCrimsonTick387

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4zcq4h100m8r5q61mnjn8q3)

Hello,

I had a question on Idempotency. Basically, are we saying clients (whoever has written the job) needs to handle it and we, as a system /service say we will process it at least once?

SQS also does that. It guarantees at least once delivery. Just checking my understanding.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm4zsuymg00seub4zzdpkt4g1)

Yeah, exactly, we can't guarantee that we won't run a job twice, especially if it failed, say, 90% of the way in. So if the task is to send an email, the task logic needs to handle idempotency.

Show more

0

Reply

F

FreshCrimsonTick387

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm509ipl5014owbx3hbmarq3n)

Agreed and makes sense. Thank you for getting back quickly.

Show more

0

Reply

I

IndividualGrayToucan573

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5093nn9014613vud4t3y3oo)

Is it simpler to remove "Worker consumer jobs" and let workers subscribe from SQS directly?

Show more

0

Reply

W

WoodenIvorySheep410

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm516qcau01rk13vu5ysstwe5)

"With the current design, querying jobs by userId would require an inefficient two-step process: Query the Jobs table to find all job\_ids for a user (an inefficient full table scan)" . A bit of clarification here the executions table also has a job\_id user\_id relationship - so why would it ever require the JObs table to be queried at first place ?

Show more

2

Reply

![Neeraj jain](https://lh3.googleusercontent.com/a/ACg8ocJAzrMQriHY2NYkdaOuS5_-r2zdrYIXz-3bABYqasyyHNOvtik=s96-c)

Neeraj jain

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm97dwo3r003zad08jyb3alyj)

I am not sure how going to 2 tables help, but why you can't fetch records from Executions table directly is because In DynamoDB, since userId is not the partition key of the executions table (which uses execution\_bucket as the partition key and execution\_time as the sort key), you cannot directly filter or query by userId unless it's part of the key schema or indexed via a secondary index. DynamoDB requires that queries be based on the partition key (and optionally the sort key), so to efficiently retrieve items by userId, you would need a separate table or a Global Secondary Index (GSI) where userId is the partition key.

Show more

0

Reply

W

WoodenIvorySheep410

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm51ada7v01tlr5q66c0w4mxz)

"But, like I said, Postgres or MySQL would work just as well, you'd just need to pay closer attention to scaling later on" . We need updates at the rate 10k/s(?) As I have understood in terms of Read write speed and as a thumb rule RDBMS cant support more then 5k/s read/write throughput . Shall we not omit the need of RDBMS at the first place as part of this conversation ?

Show more

0

Reply

![Mike Choi](https://lh3.googleusercontent.com/a/ACg8ocIiFetDZy5JBdoKw8jLl-fHkIC-pJpZhimcDzQH480L5rXr4Si1=s96-c)

Mike Choi

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm525wjgn02p9ub4zx00ldgnd)

Hi Evan, do you think its reasonable to mention some sort of DAG implementation for chained tasks with dependencies on children tasks?

Or would this be a bit too granular for the interview duration

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm52pn1uh02tp13vu4axtt4a3)

Just depends on whether or not you and the interviewer decide it's in the system requirements :)

Show more

1

Reply

W

WelcomeYellowAntelope965

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm59efcdn01d0c4lgzmp0o1z2)

For deduplication, could we set a deduplication table like redis cache that record of running job and the worker has the heartbeats? The record can still stay in the cache if it get successfully processed. Otherwise, it will be removed if Visible or invisible failure is confirmed by a heartbeat check.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5ev0ezc001zny91waxtemaw)

Sure, this works to help ensure the job isn't run multiple times. Certain task implementations still want to enforce idempotency on their own. Think of an email task. You don't want to send the same user the exact email more than once, even if you accidentally schedule two distinct jobs.

Show more

0

Reply

W

WelcomeYellowAntelope965

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5a9hqmm022pke2risgaa1e4)

Do we need to narrow down nonfunctional requirements with scalabilities for different task-type jobs with different lengths of running times? Users may schedule a lot of running jobs (e.g., 30 minutes long for a single run) and start their execution in several seconds, and hence, the executors cannot be scaled out in time. Do we need to add a controller checking execution table to scale the executor in advance?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5ev2rs1001aiassn3la5pfa)

In an interview, I'd just introduce the constraint that jobs need to be able to finish in under a few minutes. Have a 5-minute timeout or something. A durable workflow engine is a bit of a different question with different requirements.

Show more

0

Reply

![Vishal Kumar](https://lh3.googleusercontent.com/a/ACg8ocIKdtJG0-8M6TsIZBGvXateygnBmE_4b5EdRXBOx54I5yV4Zx4r=s96-c)

Vishal Kumar

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5arqeim02lwke2rfpxwarlz)

Given the scale of 10K executions/sec, wouldn't the DB partition by timestamp(hourly/minute) granularity lead to a hot partition issue as most of the reads and writes (multiple status updates) happen on the same partition at a given time? Can someone please elaborate on DB partitioning?

Also, since the execution worker will require job details for execution, should we cache job details to avoid additional load on DB (cache setup: CDC to redis or DynamoDAX)

Show more

5

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5ev8vds001vm6eclxrq9z3p)

If this is an issue we can have a compound partition key like {time\_bucket}-{shard\_id}

Show more

1

Reply

T

TraditionalBlueGibbon622

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5d7qote000we80ec3egdf0d)

Task definition and management should be also tackled in this design as it influences the execution service implementation.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5ev9frj002nny9152s3shwh)

Maybe. But only so much someone could discuss in a 35 minute interview!

Show more

0

Reply

R

ResponsibleIvoryAlbatross601

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5dq7qih007369izlt2bcbhl)

Hi Even, Thanks for the great content! I am a bit confused about the exponential backoff for retries, and at the same time setting the visibility timeout window to 30 seconds with periodic extensions in order to pick up the failed jobs faster. These are two different approaches for handling failed jobs retries. If the 30 second visibility timeout and periodic extension is used for all jobs, then how the exponential backoff is used?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5evan4e001liass2y7t9e4r)

Good question! The visibility timeout (30s) is about detecting worker failures (crashes, hangs), while exponential backoff is for handling visible task failures (errors, exceptions). When a task fails with an error, we explicitly use SQS's retry policy with backoff. When a worker crashes, the visibility timeout ensures quick recovery. These are two separate mechanisms handling different types of failures.

Show more

1

Reply

R

ResponsibleIvoryAlbatross601

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5ey1wt0005eny91qcs0y7p4)

Thanks for your response. Since we don't know how a job might fail, then I guess we always have 30 sec visibility window for all jobs. In case a job fails with an error (visible failure), it still will be visible after 30 seconds and technically the workers can pick it up again. With exponential backoff in place with this type of failure, at this point is it the worker who waits for the period of exponential backoff before reading the message again? Does the job remain visible in the queue during this time? I am wondering how the visibility is handled in this scenario.

Show more

0

Reply

![adithya r](https://lh3.googleusercontent.com/a/ACg8ocLNcYZCD7QFAUoQjbbHeJ9wq7tSj4PJgr35-tryIJqkC1xwwQ=s96-c)

adithya r

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5epgcfn00d6yag8qcf67keq)

I can't find the explicit reason, why we split into two services i.e.,schedule svc and query svc..can you point that out?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5evcg37002nf933l7lak2wf)

The watcher runs periodically to hydrate the queue. The scheduled service will have heavy write throughput. Thus, we can scale these independently by keeping them separate.

Show more

1

Reply

![Jiatang Dong](https://lh3.googleusercontent.com/a/ACg8ocKfQgaYilpR7RBKGa8_AXqyhuDM2GA6B29pLwiJomT1-dI5c0tTlg=s96-c)

Jiatang Dong

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5hbaxmv00v8a8vvlihg4esm)

Can you give us some deep dive on the "worker consume jobs" box? I encountered this problem today and I think I failed to explain about that part. From the high level design to the deep dive, I changed the "task executer" to the work watcher -> queue -> execute job model. But I missed the "worker consume jobs" part and the interviewer was not satisfied. He keep asking me questions about "how to dispatch the jobs". I though it's a producer and consumer model already, so the consumer will be able to pull the jobs by himself. I didn't get the question.

Show more

0

Reply

![Jiatang Dong](https://lh3.googleusercontent.com/a/ACg8ocKfQgaYilpR7RBKGa8_AXqyhuDM2GA6B29pLwiJomT1-dI5c0tTlg=s96-c)

Jiatang Dong

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5hbfyjt00uc5sp8jishx9y1)

Is the "worker consume jobs" supposed to work as a coordinator to dispatch the jobs under some strategies like round robin, LSU, LRU, and control the automated scaling of the task executers?

Show more

0

Reply

![Jiatang Dong](https://lh3.googleusercontent.com/a/ACg8ocKfQgaYilpR7RBKGa8_AXqyhuDM2GA6B29pLwiJomT1-dI5c0tTlg=s96-c)

Jiatang Dong

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5k2stie03ir5sp8v1nwzk4m)

As I made a deeper reflection on this problem, I found that it may be a different model we are talking about: push vs pull. In push model, we need a dispatching service that actively send the jobs to the job worker. In this case, the dispatching service needs a strategy like round robin, LSU, LRU to load balance the workers. In a pull model, we don't need to worry about that. It's the worker it itself proactively coming to the queue to fetch jobs when it is ready. Instead of load balancing to avoid a worker piling up the job, we should focus on the status of the queue and auto scale the job workers.

Show more

0

Reply

![Liliiia Rafikova](https://lh3.googleusercontent.com/a/ACg8ocLhKI61yr7h7HM0rSPvHT8QjFtJteOZbg84lT2Kk6f-YRWWSw=s96-c)

Liliiia Rafikova

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm735a6qh002palw2e2b665pt)

There are multiple ways jobs can be dispatched to workers: 1. Polling Model (Pull-Based) • Workers actively poll the job queue (e.g., using SQS.ReceiveMessage(), Kafka.consume(), Redis.BLPOP()). • Each worker asks: “Do you have a job for me?” • If a job is available, the worker locks it (visibility timeout in SQS, leasing in DB) and starts processing. • If no jobs are available, the worker sleeps for a short duration before polling again. • This is common in event-driven systems (e.g., AWS Lambda, Kafka consumers). 2. Push Model (Event-Driven) • The job queue actively pushes jobs to workers (e.g., WebSockets, gRPC streaming, SQS-to-Lambda trigger). • Workers don’t need to poll; they just listen for new jobs. • This model is used in low-latency, high-efficiency workloads (e.g., trading systems, high-speed job execution).

Show more

2

Reply

P

PayableIndigoFox324

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5j36rfi02qf5sp8c8h7z5yd)

This is from the section - Potential Deep Dives "This two-layered approach provides significant advantages by decoupling the database querying from job execution. By running database queries just once every 5 minutes, we reduce database load while maintaining precision through the message queue."

Q: Does this not mean that some job that was scheduled to execute at 4:57 PM, could execute earlier if it was picked by the watcher that scanned the DB at 4:55 PM. Just asking is this expected/tolerable behavior for this question

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5kj7kat00iltpf2elng4k98)

Oh no, when the watcher at 4:55 PM picks up a job scheduled for 4:57 PM, it doesn't execute immediately. We are increasing precision. Phase 1) crudely get upcoming jobs 2) precisely schedule them.

Show more

2

Reply

S

SpatialBlueAntlion614

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5nm2lwe000q13hr287br5fc)

In the jobs table, can we use user\_id as partition key and job\_id as sort key? This way querying per-user jobs would be more efficient, though we may potentially have issue with hot keys (where a single user creates many many jobs) -- is that a big issue in practice?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm62f20zn013vu73x12alilqi)

This would make our main access pattern (finding due jobs) require a full table scan. Better to use job\_id as PK and add a GSI on user\_id if we need user queries.

Show more

0

Reply

![Vashu Gupta](https://lh3.googleusercontent.com/a/ACg8ocIx4oMNeEGl0o1nz5_OTXLLM3i_iIavVW5wrh0wyIpnEP2BDmwU=s96-c)

Vashu Gupta

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5pp96yc01arqsjketms7xy9)

Hey Evan, Thanks for the awesome content! I have some questions about the above design:

1.  Do we need a worker to consume jobs? We can simply implement the SQS poller in the ECS.
2.  How does the watcher look like? Is it Lambda, ECS, etc.? Also, I am assuming the watcher won't be running on a single node to avoid the single node failure, and we have a scale requirement of 10K jobs/sec. So, when we have multiple processes looking at jobs table, how do we ensure we do not end up reading the same batch of jobs in two different processes?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm62f0o66018acf2adb4ktsr3)

1.  Correct. Trying to be not overly dependent on a single cloud provider, but in AWS world this is what you would do.
2.  Probably ECS better than lambda here given the steady, predictable workload (cheaper) but either works.

Show more

0

Reply

![Hemanth R](https://lh3.googleusercontent.com/a/ACg8ocICEo7vYYXTwSoi9ffWMxVjNpzYbpifyqVEUjtWWFds9vQ_=s96-c)

Hemanth R

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9xrr54000a7ad08p5en3zo9)

I'm also curious about this part in 2nd question. Anyone know the best way to handle this ? "So, when we have multiple processes looking at jobs table, how do we ensure we do not end up reading the same batch of jobs in two different processes?"

Show more

1

Reply

Y

yingdi1111

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5pq2stu01apwu5h25j0tpth)

SQS automatically handles scaling and message distribution across consumers, which is one of its biggest advantages. We don't need to worry about manual sharding or partitioning - AWS handles all of that for us under the hood. The service can handle our throughput requirements of 10,000 messages per second without any special configuration, though we'll need to request a quota increase from AWS as the default limit is 3,000 messages per second with batching

The 3000 limit is for FIFO

Show more

1

Reply

H

HotChocolateSpider690

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5q4qeg501uywu5hcxzgz0b1)

I am not clear , why do we need the first queue \[ producer->Queue part\] . I blv you are referring producer as the REST call handler; for job creation and its putting the job in the queue? But schedule service should autoscale based on number of requests or other metrics. How does adding another Queue help here? For 10K job creation, we should focus on scaling to the writing to the DB part.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm62ey0af0185cf2aifof10j7)

Yeah, you don't. Your understanding is correct. If your initial service starts to do a lot, then the durability provided is a plus, but this is minor and arguably over-engineering.

Show more

0

Reply

F

FederalHarlequinLamprey277

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5rzc0vb00ffxszbcdqzkurw)

Well done, Evan! Question:-

-   Why is Kafka not considered in the job prioritization options?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm62ewluf013ou73xq34xmluf)

it doesn't support message reordering or priority-based delivery

Show more

0

Reply

![Tom Oh](https://lh3.googleusercontent.com/a/ACg8ocL_zg_F9-J6vR8eza8MdsPWwRfQlwUUTYY5HA0PPSkPGaXt1A=s96-c)

Tom Oh

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5w3t4xm01l7a62zwfjvqwbb)

As-is: We can then put the job back into the Redis sorted set to be retried again in a few seconds (exponentially increasing this delay with each failure). To-be: We can then put the job back into the Message Queue (SQS) to be retried again in a few seconds (exponentially increasing this delay with each failure).

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm62ew1e8013iu73xq597v4h9)

Thank you, updating!

Show more

0

Reply

![Aditya Jain](https://lh3.googleusercontent.com/a/ACg8ocJjHh-eky22nfbV7YSOwPct6ROk615alDtDMGafjdNJV1gQcQ=s96-c)

Aditya Jain

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm5x83pqd00i4123dslp1ny27)

How does the system ensure that jobs are executed exactly at the time specified by the users? For example, suppose User A and User B have both scheduled jobs to run at the same time. User A’s job takes 6 minutes to complete, while User B’s job takes 8 minutes. If the system uses SQS and sets jobs to remain invisible until their scheduled execution time, both jobs will become visible in the queue simultaneously. Does this mean that multiple workers will pull and execute these jobs in parallel? How does the system guarantee timely execution without conflicts or delays?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm62evg7m013fu73xxtlb5ffk)

We have many parallel workers pulling and executing jobs, yes

Show more

0

Reply

C

CuddlyCopperWildebeest704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm61si12c00rrhqkjkhuriqot)

"execution\_time": "1715548800-123e4567-e89b-12d3-a456-426614174000",

I wonder if using snowflake id would be better here

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm62eu7l50169hqkjd6ve7xjf)

yah that would work well actually

Show more

0

Reply

C

CuddlyCopperWildebeest704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm61sl06d00siguduvoindopq)

Do we need userID on both Jobs and Executions tables?

With the current design, querying jobs by userId would require an inefficient two-step process: Query the Jobs table to find all job\_ids for a user (an inefficient full table scan) Query the Executions table to find the status of each job

Why not just query Executions table which has userId?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm62esi2i0167my0807nahu35)

Yes, we need userId in both tables - Jobs table (with a GSI on userId) for job ownership/access control, and Executions table for efficient status history querying.

Show more

1

Reply

![Baloch Singh](https://lh3.googleusercontent.com/a/ACg8ocJptFRf90Pa_nyWXvarq8PHZSPEt_z-j_ePAwSvn6m0SWwhgQ=s96-c)

Baloch Singh

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm66uacf402h0tbmrownfevnr)

If Worker A is still processing at 10:30:00, it would update the lease to expire at 10:30:30, giving itself another 15 seconds. 10:30:00 --> 10:30:15 Evan King, please fix it.

Show more

1

Reply

![Misha Borodin](https://lh3.googleusercontent.com/a/ACg8ocITt8_C-XimHao0Gj-BqF28IKe3WXyA8ppWstGFMnewgZtPMQ=s96-c)

Misha Borodin

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm68420tk00hfe7bzrb60q1um)

it resets the lease from 30:15 to 30:30, so it's 15 additional seconds

Show more

0

Reply

![Misha Borodin](https://lh3.googleusercontent.com/a/ACg8ocITt8_C-XimHao0Gj-BqF28IKe3WXyA8ppWstGFMnewgZtPMQ=s96-c)

Misha Borodin

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm683zbzk00hbe7bzp1qujgd1)

Regarding Job Leasing for at-least-one guarantee, don't we need a strongly consistent storage there? Otherwise, same job may be executed multiple times (which is OK for the goal of at-least-one, but still a drawback)

Show more

0

Reply

E

EagerScarletAardwolf955

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm69wry1o000e6hpqnoyf0y17)

I have to say this is the most elegant and practical system design I have read -- exactly same as what our team did for the scheduler refactoring! Design King Evan LOL

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm69wssgc000usn4h5l837uha)

😂 that's epic!

Show more

0

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm69yp915003u6hpqta9hopjk)

For "Executions" table what is the partition key? Is it job\_id or user\_id?

Show more

0

Reply

![Tommy Loalbo](https://lh3.googleusercontent.com/a/ACg8ocKJIn8OPXYOxiFFjMUkH5UDjWWCbOFuGt2Srsu9sGECWCgexFCq=s96-c)

Tommy Loalbo

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6do0uzn03la6hpq1giw22lf)

Few things to mention here.

In regards to: "We can optimize this setup by using spot instances to reduce costs" spot instances can be reclaimed causing the job to need to be processed potentially breaking the 2s window requirement. Care to speak to that? Are we bending the requirement there?

I think it should be noted that EKS has a clear advantage over ECS in this situation. EKS has a cluster autoscaler which allows you to autoscale based on pod utilizaiton. ECS doesn't natively support that. For ecs you have to use autoscaling groups (asg) and manage the the connection between the ASG and the tasks manually. Thoughts?

On the lambda front, it is important to mention turning on provisioned concurrency. That means you configure the number of lambda instances AWS will prewarm for you at all time. Otherwise you have to deal with lambda cold start issues which can add seconds of overhead before it gets started breaking the 2s window requirement. Thoughts?

I think another interesting aspect of this is the choice of batching jobs every 5 minutes. I guess we are assuming that jobs will all run in under 2 seconds? because a job could take longer. wouldn't it make sense to get everything all loaded and ready to commit so that way you only need to commit the result within the 2 second window and not do the processing if possible.

Also, would it be worth mentioning intelligent retries? Like depending on the type of error and job we can use different retry strategies to potentially save on compute in some cases and improve fault tolerance in others.

For the deduplication table, couldn't this be optimized by checkpointing progress? Couldn't jobs critically fail and leave "partial progress" wouldn't it make sense to pick up where we left off. On that note, I would think it makes a lot of sense to do a hybrid approach as the best solution. Certain jobs could be such a pain to make idempotent to where you might just wish to put it in a deduplication table. Best of both worlds. THoughts?

Show more

0

Reply

![Rajat Mishra](https://lh3.googleusercontent.com/a/ACg8ocK1VeQLni_QF_BrSquYJkb2LFFdlEUZHKhT9t6o5JuSTmt_OwU=s96-c)

Rajat Mishra

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6f1ymvw05i96hpqogc3y7ii)

Why will a postgres work in this case? Write throughput is 10k/s whereas postgres can handle only 5k/s.

Show more

0

Reply

![Tommy Loalbo](https://lh3.googleusercontent.com/a/ACg8ocKJIn8OPXYOxiFFjMUkH5UDjWWCbOFuGt2Srsu9sGECWCgexFCq=s96-c)

Tommy Loalbo

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6f3aef805jj6hpqghnnxguv)

I think its because you would have to shard it.

Show more

0

Reply

J

jasonliu0499

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm75ssxyx02sjalw26rmmb8c3)

Based on this benchmark, it should be able to handle that https://www.citusdata.com/blog/2017/09/29/what-performance-can-you-expect-from-postgres/

Show more

1

Reply

A

adamlawson1337

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6f68oeo05p98twykeeiygnh)

Will the system work for jobs that are scheduled to run with a frequency under 5 minutes? If we have a job scheduled to run every minute, and it completes just after the watcher finished it's last query of the DB, it wouldn't get scheduled until 5 minutes later, missing the 2s execution window.

One solution could be letting the workers directly place items in the execution queue if their next scheduled execution time is less than the 5 minute interval we have set for the watcher.

Show more

1

Reply

![Rajat Mishra](https://lh3.googleusercontent.com/a/ACg8ocK1VeQLni_QF_BrSquYJkb2LFFdlEUZHKhT9t6o5JuSTmt_OwU=s96-c)

Rajat Mishra

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6fey036062yiq3nl5y8dfoa)

Lets say at t = 0 to t = 5min couple of users scheduled jobs to be executed at t = 10min. Now at t = 5min the watcher will have to pull like 5 \* 10k \* 60 = 3000000 jobs. Fetching this much data is not directly supported by DDB.

Show more

0

Reply

C

ChristianPlumTakin500

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7h2ktps01n6mbkq5fxto4m6)

But you'll have 5 minutes to do that, and that's more than enough for 300k items retrieval

Show more

0

Reply

![hitesh gupta](https://lh3.googleusercontent.com/a/ACg8ocKlYqVxysla9QkT1XSlxNB21d-rPrUJQvWmQTxZ92c56h0IqAd3Qg=s96-c)

hitesh gupta

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6iavmaf00tljx73tk58vi3q)

Evan, I think you did not mention when and how we are inserting multiple rows into the Execution table with the time of execution. This can be done in various ways:

1.  When the request is received by the Scheduler Service, there will be multiple rows that need to be inserted which can lead to insert infinite rows.
2.  By the Watcher, which will have to fetch the Job Table and perform a full table scan.
3.  When a task is executed, it updates the database with the next task execution, resulting in at most 10,000 write requests to the database.

Let me know if I am missing anything?

Show more

0

Reply

S

SoleBlushSilverfish368

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6jhtuzc014klhflxu0n6xln)

On the final solution " workers periodically "heartbeat" by calling the ChangeMessageVisibility API to extend the timeout", how is this different from the leasing solution? i.e. extend the lease at periodic intervals. Except the fact that SQL handles the visibility natively.

Show more

0

Reply

Z

ztan5362

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6kawb5r02abjgz3xtyumf64)

GET /jobs?user\_id={user\_id}&status={status}&start\_time={start\_time}&end\_time={end\_time} -> Job\[\]

I though you are not suppose to pass in user\_id in query params (instead in JWT)?

Show more

0

Reply

![Nikhil Desai](https://lh3.googleusercontent.com/a/ACg8ocL7PtE3QPnse_xzmT_iPOk3OAN4lZ7G9m2_nnSyWsfWFehtdbtD=s96-c)

Nikhil Desai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6nsd4b500f641j33x5teirm)

This is what I would think:

Depends on who is querying. If the admin wants to get a list of all jobs submitted by a specific user then userId needs to be passed in. If the query is for him/ her then it can be extracted from the JWT as you rightly pointed out.

Show more

0

Reply

![Carlos](https://lh3.googleusercontent.com/a/ACg8ocJWZfPk9o9sB_EQafj88F3BCV-8uz_i_TRSaQSn578Nv-KkkA=s96-c)

Carlos

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6o45fme010wv55hjvjf34pb)

If a new job is created with a scheduled time < 5 minutes from the current time, it's sent directly to SQS with the appropriate delay.

I dont see in the chart you posted where we would post directly to SQS or am I missing something?

Show more

0

Reply

F

FrontYellowPheasant390

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6t1dukf036y14j6a6k0581r)

If in a case, using technology like SQS is not allowed. We need to clarify whether it essentially functions as a priority queue or something like a 'sorted Kafka.' I’m still not fully clear on how priority queues work(not sorted Kafka). For new jobs that come in within 5 minutes, we could try putting them directly into the message queue (bypassing the database), but if we’re not using sorting, how do we guarantee the order? Or how to set the priority queue? (e.g if 10:01 is a queue, and 10:03 is a queue, then when 10:02:30 coming, it may be put after 10:02:50? )" Is sorted the only choice if no SQS?

Show more

0

Reply

K

kushaq.2021

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm6uevqph0179y7solaqh0k4q)

I do not understand why are we passing the user id in the GET call, it should be retrieved from JWT ?

Show more

2

Reply

T

ToughGreenVulture620

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm72gbmk5000d134lbmzai6lb)

Can you give a comparison between Kafka and SQS, and at what time we should choose one vs the other?

From this write up seems SQS has more functionalities than Kafka (and other message queue solutions), which seems to me we should always choose SQS if we want some more complex functionalities, is that true?

Show more

1

Reply

C

ChristianPlumTakin500

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7h1zhz501m7mbkq0sk94xx6)

Kafka can maintain ordering per partition

Show more

0

Reply

![pankit thapar](https://lh3.googleusercontent.com/a/ACg8ocKJll4bhmNhPwIJxULLMR1ojZS3PdkYsMx7oJifmNm1cKVl1g=s96-c)

pankit thapar

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm78glnau02it4nd5g7pinogc)

Thank you for sharing the design. I have some experience with distributed schedulers for online and batch workloads especially without using public cloud services. Based on that, I went with an approach of using a high QPS key value store like etcd for machine or server related metadata, heartbeat as well as job/task ownership and status tracking. This moves the problem of scaling the cluster to the problem of scaling etcd key value store. At that point, job scheduler or allocator is spending most time in reading the messages in the job queue and then creating a plan based on available resources on the machine. Since, this kind of interview is open ended, I ended up spending some time on run time side of things more.

What are your thoughts on that kind of design as well as spending time on runt time?

Show more

0

Reply

![Neeraj jain](https://lh3.googleusercontent.com/a/ACg8ocJAzrMQriHY2NYkdaOuS5_-r2zdrYIXz-3bABYqasyyHNOvtik=s96-c)

Neeraj jain

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm97ejoj9004jad08t9ixtyn4)

Could you please throw an excalidraw drawing about how you have implemented this in real world. It's easier to corelate in HLD

Show more

3

Reply

G

GrandApricotRabbit449

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm78mugac02sir9w5t8o55uei)

What should we do if a job runs too long past a specific threshold that we set? Should it be cancelled or we should just notify the user maybe? And we shouldn't let it affect the next recurring execution then if one is set, right

Show more

0

Reply

K

k.87.sharma

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm79vx4lc04j92azpswpuvnnc)

Great content but many solutions are using too much AWS products. In an interview setting this might not be acceptable since many interviewers prefer having a generic solution since AWS is not used in the company.

Show more

0

Reply

![Priyangshu Roy](https://lh3.googleusercontent.com/a/ACg8ocJXi2S6LLHV4HR59WPr_PKRcpuZtBGgrBG7-HsFT24DMocISQ=s96-c)

Priyangshu Roy

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7aw8ge3003c19rhkolzy7cu)

I have a basic doubt. Our executors are picking up the jobs that should run in the next X mins. If we have mutiple instances of our workers running, how do we prevent the same set of jobs from being picked up by more than one worker. With a SQL db we could always fetch tasks WITH LIMIT 1 and lock that specific row, the other workers could just skip the locked rows and proceed to the next task. How do we handle this here?

Show more

3

Reply

![Neeraj jain](https://lh3.googleusercontent.com/a/ACg8ocJAzrMQriHY2NYkdaOuS5_-r2zdrYIXz-3bABYqasyyHNOvtik=s96-c)

Neeraj jain

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm97dlcy5003bad08wgudg32j)

I think you read it wrong, it's the Watcher who is fetching what to run in next X minutes, then SQS due to DeliveryDelay is providing execution\_time priority, workers are just fetching it one by one which is how they should have been executed. Am i missing anything ?

Show more

0

Reply

D

DisciplinaryFuchsiaTuna318

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7fi1yzs007yozj8jpyoi4dr)

hello, thanks for the clean and detailed write-up of this job scheduler design. You provide the job status get API: GET /jobs?user\_id={user\_id}&status={status}&start\_time={start\_time}&end\_time={end\_time} -> Job\[\]

I am curious why status is a parameter in this API. User does not need to have prior knowledge about a job's status, I think. Isn't it?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7fi3vbm008fdk3f1b4vlr2u)

This API is for listing jobs. It can be useful to list completed jobs, or jobs that are scheduled, etc.

Show more

1

Reply

D

DisciplinaryFuchsiaTuna318

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7fie4bg008pdk3fihn0kfb5)

This makes sense now.

Show more

0

Reply

![Yali Xu](https://lh3.googleusercontent.com/a/ACg8ocKi-2R8_-_cEAHfWNB3B3ig8_CU9lIp2V5Ni8XPPi8RjUCFQd4=s96-c)

Yali Xu

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7p4k8cd00h92ebzi1r3z4kz)

Oh, after I posted the comment, I saw this one. I think it resolved my confusion previously. I was thinking user would see the status of jobs whether they are completed or scheduled for a certain task. But thanks for clarification. It's really helpful.

Show more

0

Reply

M

MarkedMoccasinTern443

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7hn668w026mtygedqvvtdh6)

A little bit lost at how do we creation the execution table? The watcher will pull the executions need to be executed in next five minutes, which component is responsible for create the executions needed to be run? what if client change their job schema, do we need to remove all entries from executions table and rewrite?

Show more

3

Reply

P

PositiveEmeraldParakeet596

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7q2pclb01jrrpj4nrrijuev)

+1

Changing or Cancelling jobs is out of scope, but I also feel like we over abstracted this non-trivial component, as it won't be distributed as well (if several "Watchers" would handle the same job we will end up creating multiple executions mistakenly)

Show more

1

Reply

![Alexandre Martins](https://lh3.googleusercontent.com/a/ACg8ocJEsabuREJPbDKyiyUiZ71vj1mJZmub-fgNozm-xyXei7ISDR88Tg=s96-c)

Alexandre Martins

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7m7mg1200iv4y43f8f8l78n)

How would you make the TaskScheduler highly available? Is that a cron job? Given that every 5 minutes we have 3 million records returned from the database (a couple hundreds of MBs), is one instance enough? How to avoid that becoming a single point of failure? How to scale if needed?

Show more

2

Reply

![Yali Xu](https://lh3.googleusercontent.com/a/ACg8ocKi-2R8_-_cEAHfWNB3B3ig8_CU9lIp2V5Ni8XPPi8RjUCFQd4=s96-c)

Yali Xu

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7p3x2o000f9joxcdvzb0zqi)

The example of the endpoint which was supposed to queries the status of jobs leads to a confusion to me: GET /jobs?user\_id={user\_id}&status={status}&start\_time={start\_time}&end\_time={end\_time} -> Job\[\] --> this one actually queries the jobs with given status.  
From my point of view, in order to query the status of jobs or specific job, we can either use:

1.  GET /jobs/{job\_id}?start\_time={start\_time}&end\_time={end\_time} for single job status query, where the useId should be gotten from the header, which is log in user.
2.  GET /jobs?task\_id={task\_id}&start\_time={start\_time}&end\_time={end\_time} for multiple job status query. With log in user.

Show more

0

Reply

I

IntegralBeigeSloth435

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7qyj8mu02pqrpj4fqjxglk5)

It is said as - "But how can we query for the status of all jobs for a given user? With the current design, querying jobs by userId would require an inefficient two-step process:..." However, the execution table has fields - job id, user id, so we can just run one query on the execution table, won't that suffice? Why do we need GSI?

Show more

0

Reply

![Rajat Bhatnagar](https://lh3.googleusercontent.com/a/ACg8ocLl3BhThTLIfuwlDCrrPdb1c0NtbHq5pyyDDm4xNIVdMKoyW4Lw=s96-c)

Rajat Bhatnagar

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm7zc4ru60303e77fm906b4qo)

How will we create entries in the execution table for a cron which doesn't have an end time? For example, if the schedule is to execute each day at 1 PM then how will we create Entries for all the days? I am assuming that there might be another process running which can run each day and create entries in the execution table for recurring jobs for the whole next day. However in that case we'll be querying the jobs table for fetching the active jobs which would require the whole scan of DDB's jobs table which is expensive.

Show more

1

Reply

F

FinancialGreenTick391

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm832jm2c00jk10qjb4z36pkt)

I believe a more detailed discussion of the watcher service is needed. In other design problems, we address concerns about cron jobs due to their potential for failure or crashes. In this scenario, the watcher plays a very pivotal role. What would happen if it were to fail? Furthermore, how could it be scaled?

Show more

2

Reply

![Yash Shukla](https://lh3.googleusercontent.com/a/ACg8ocKlbomzbIWDTFkGY-SJEEdy4jwqoSpK0iVpB1ARZZl4qLzVnQ=s96-c)

Yash Shukla

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm87h1jl101qykzp5o9sriths)

"18. Distributed Job Scheduler"

Show more

0

Reply

F

FederalHarlequinLamprey277

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8ad1o8r00kl853otlgy2kks)

@Evan, in the final diagram. Does the execute job represent a process running on a separate machine? Or represents threads running on a worker consumer jobs?

Show more

1

Reply

![Alexander Berezovsky](https://lh3.googleusercontent.com/a/ACg8ocKm-E1_KG1rpWBVGtkBbfgSTIajqzMCdoC3zbbRZJkA3rvV4QZ7=s96-c)

Alexander Berezovsky

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8h0u6a4015bp0hfptiqnpwc)

Is't better instead of watcher to pull from DB, to use a DDD stream and lambda? This way we can avoid directly pushing to SQS from Scheduler service and get better separation of concerns. WDYT ?

Show more

0

Reply

O

OrthodoxAquaLamprey145

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8pllu6700l613jeuny26k2h)

Not quite, while there is some grounds to your point, there is a limit on how long (I think its 900 secs) you can delay your message to become visible to consumers. If a user schedule a job that is more than 900 secs later, then the stream you captured will be useless, since you cannot put the scheduled job to the queue with a correct delayed time attribute.

Show more

0

Reply

![Alexander Berezovsky](https://lh3.googleusercontent.com/a/ACg8ocKm-E1_KG1rpWBVGtkBbfgSTIajqzMCdoC3zbbRZJkA3rvV4QZ7=s96-c)

Alexander Berezovsky

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8plq3mp00i56dzq4xbutwjc)

Inserting. Will think about it :-)

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8jtke8m000yv3ctf3yel4tk)

Shouldn't At-least-Once delivery can be covered ACK mechanism which supported in most queue service

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8kmbirz00gg1297yte2upz1)

"If a job is retried 3 times and still fails, we can mark its status as FAILED in the Executions table and the job will no longer be retried"

Who did this? The retries times is internally to SQS and worker doesn't know the retry times. (1) The retry time is passed into worker. When execution failed, worker check the retry times. But this doesn't work for invisible failures that the worker went down all the times, then the status cannot be updated (2) Does SQS (or general message queue) support custom code to handle max retried event. (or event in the dead queue)

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8kmienk00go1297hmhjvkr8)

This is achieved by having another worker to handle the dead letter queue.

Show more

0

Reply

E

ExcessBlackAardvark647

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8l60tri00f27z0xlgakn9vk)

The heavy usage of amw sqs makes the 2nd half of this document becomes a promotion from Amazon. we cannot tell our interviewer: " hey, we're gonna use sqs here as it's the crystal ball, next question. "

Show more

1

Reply

![Ram Jashnani](https://lh3.googleusercontent.com/a/ACg8ocKKNPx6h8ueo9q0-AX2Ja8UKHb8j5I0BPCrdW96qg3yUcLn9Q=s96-c)

Ram Jashnani

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8lhq9q400tpmp8jzojvcusn)

As per CAP theorem, we can only give priority to two and here we are giving priority to Availability and Partition Tolerance. What would consistency mean in the context of this problem? Does it mean that every job run would have latest data whether it's been run for a particular time and since we are giving priority to Availability we are fine with a task running multiple times for the same time?

Show more

0

Reply

O

OrthodoxAquaLamprey145

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8pm3fcr00ly13jeb8i78av2)

@Evan King, I think the statement "Fortunately, SQS even handles exponential backoff for us out of the box, making the implementation of this pattern quite simple." in the Deep Dive 3) is incorrect. SQS does have retry mechanism in-place for worker failures (worker fails to delete the message from a SQS queue), and will make the message visible again to other consumers after visibility timeout. However, SQS does not handle application errors with exponential backoff retry. You explained correctly that we could put the message back to the queue with increased delay, but that statement following your explanation was kind of misleading.

Show more

0

Reply

![Jeremy Shi](https://lh3.googleusercontent.com/a/ACg8ocJ2sb0qOH8kcQXxe0Cn0yJ_g4LR3JiCmQXrdpnSwcy39kYvlNlm=s96-c)

Jeremy Shi

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8q52jwx00vgs2abc5tsqu5f)

if it uses rounded hour as the partition key, would multiple jobs be hashed into the same row? Let's say we have 3 jobs need to be executed in the next hours. in the execution table, it'll only have one row with the rounded hour. How can the poller find out 3 executions then?

Show more

0

Reply

O

OrthodoxAquaLamprey145

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8q9sz9f01g713jeg11b5t0i)

Yes and no, depending on your configuration. In DynamoDB, when you define a primary key (partition key) + a sort key, you get what is called a composite primary key (partition key), in which case only both values of the primary and the sort key are the same would they be mapped to the same row. So in our case here, different jobs executed with the same rounded-down timestamp would be different items in the table (since they have different execution\_time value). If you only define a primary key without defining a sort key, then only one item can exist in the table given the same primary key value.

Show more

0

Reply

![Jeremy Shi](https://lh3.googleusercontent.com/a/ACg8ocJ2sb0qOH8kcQXxe0Cn0yJ_g4LR3JiCmQXrdpnSwcy39kYvlNlm=s96-c)

Jeremy Shi

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8qgg10q000iad08r6scc3qs)

yeah, that makes sense. But still, it's possible to have different jobs having same rounded hours AND execution time which can be mapped into the same row, right?

Show more

0

Reply

O

OrthodoxAquaLamprey145

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8ql4iyq006pad087jtyjkif)

Not really, if you look at how the sort key is generated:

"execution\_time": "1715548800-123e4567-e89b-12d3-a456-426614174000"

it's a exact scheduled timestamp concat with the jobId, and the jobId is unique already. If you are talking different jobs, that means they have different jobId in the first place.

Show more

1

Reply

![Jeremy Shi](https://lh3.googleusercontent.com/a/ACg8ocJ2sb0qOH8kcQXxe0Cn0yJ_g4LR3JiCmQXrdpnSwcy39kYvlNlm=s96-c)

Jeremy Shi

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8ql7r0q007cad08bc9wxhi5)

you are right! thanks!

Show more

0

Reply

A

AcademicIndigoRhinoceros636

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm8xxva47007oad08314u25ut)

"Users should be able to schedule jobs to be executed immediately", with a 5 mins polling interval, immediate execution is not possible. If the SLA for task scheduling is within a minute, it's probably worth adding the job to a queue that can the watcher is always listening to. Watcher reading every 5mins only makes sense if there are more scheduled tasks than immediate execution.

Show more

0

Reply

![Shubham Sharma](https://lh3.googleusercontent.com/a/ACg8ocIOhMIi1_plG6Gtw-e68fmaWK_Xt-YmmEBjXVehf6UC-HffbrcL=s96-c)

Shubham Sharma

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm96jmbe2000mad07nwie5rep)

There will be multiple instances of executor running. How will we ensure that all the instances don't end up picking up the same set of jobs to be sent for execution to the message queue?

Show more

0

Reply

P

ParallelIvorySwallow280

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm991fzau00bnad08c4n08pss)

Jobs table: Partitioned by job\_id to distribute load across jobs This means there would be one job per partition. Let's say we have millions of total jobs so we will have millions of partitions. Is that fine? Wanted to know if the interviewer will probe on that?

Show more

0

Reply

![mohith medasani](https://lh3.googleusercontent.com/a/ACg8ocJjAAyklkcXJ7rxKCkcyoVxYZCOfoklSQ7NrB6DIXuGafw5zTo=s96-c)

mohith medasani

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9aui39c00koad075y19yw2s)

Instead of continuous polling on Dynamo DB, Service can add job metadata in ddb with status as scheduled and trigger an event bridge rule with cron expression and Task can update the status once completed, I think this is simpler approach

Show more

2

Reply

P

PreciseBlackPanther893

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9f1gihc00duad08cla50r1k)

will eventbridge scheduler scale for more than 10k jobs/sec https://docs.aws.amazon.com/scheduler/latest/UserGuide/scheduler-quotas.html

Show more

0

Reply

P

PreciseBlackPanther893

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9f0st7n00dgad08az51nrd7)

when a recurring job completes, we can easily schedule its next occurrence by calculating the next execution time and creating a new entry in the Executions table. How will this be done if there is a recurring every day job. Who will add this recurring run to the execution table? when you add a new job I am assuming you are only adding the next run to the execution table but what about the runs after that

Show more

0

Reply

X

XerothermicBlueGibbon404

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9gqu47902dyad08c7hxw8pd)

11:55pm - scheduler picks jobs due for execution within 11:55pm - 12:00am. 12:00am next batch query starts. Aren’t we cutting it too close for a job scheduled for 12:00:01am?

What if instead the query picks a 5 min window that starts after 5 min? So 11:55pm query picks jobs between 12-12:05am

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9rxd7up002nad08eapk0iow)

You need to always be running ahead of schedule. So at time T you query for jobs to be executed between T+5 and T+10. The jobs between T and T+5 should already be in the queue

Show more

1

Reply

D

Daisy

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9iyglkv019mad08udk55t0t)

We would need task information to run any job but that is not fetched anywhere in this design. Is it fetched by the worker for each job?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9kf3mga000yad089togj6yr)

The task information is stored in the Jobs table and fetched by the worker when it receives a job from SQS. The worker gets the job\_id from SQS, then queries the Jobs table to get the task\_id and parameters needed to execute the job.

Show more

0

Reply

![Yufei Liu](https://lh3.googleusercontent.com/a/ACg8ocL6zgyXPiswb9f6X92vcJq6xWgU_lqXm03LV6uyzQfube_cOg=s96-c)

Yufei Liu

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9kf08eb000xad08uv9jh1n6)

Love the content, I have 2 questions 1.

time\_bucket = (execution\_time // 3600) \* 3600  # Round down to nearest hour

Is this formula wrong ? Isn't this just returns execution\_time ?

1.  Lots of design decisions are made based on SQS, for people who has lots of hands on experience with SQS, it's good, but for people (like me) who don't have experience with SQS at all. I'd rather not talk about it. I suggest that in the solutions, we do not overly rely on a single technology (unless it's widely popular) and instead, I'd love to learn more on how to solve it using native solutions. Personally, SQS part is not very useful to me because most likely I won't mention it in my interviews due to lack of experience/understanding in it.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9kf34ky000uad08mb1urukc)

For #1: The formula is correct. Integer division (//) in Python drops decimals, so 3605 // 3600 = 1, then \* 3600 = 3600. This effectively rounds down to the nearest hour. For #2: everything SQS can do you could make Kafka do. Just going to take more engineering effort since some things aren't native. That said, the technology isn't the important bit, its how you use it and what problem it solves. Replace "SQS" with message queue throughout and you'll be fine.

Show more

0

Reply

![Yufei Liu](https://lh3.googleusercontent.com/a/ACg8ocL6zgyXPiswb9f6X92vcJq6xWgU_lqXm03LV6uyzQfube_cOg=s96-c)

Yufei Liu

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9kn72so00ciad09vqosbx87)

Ahhh I got it, because the execution\_time is in epoch seconds, that makes a lot of sense now. Thanks!

For #2, I see the point, my main concern was that if I mention SQS and say SQS can take care of X,Y,Z for me, the interviewer might ask me deeper questions to test if I'm just throwing out words there. I guess I could instead say, use MQ that supports Delayed Delivery feature, or build it ourselves.

Thanks for the reply!

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9ko4fg700e0ad088ix95g2w)

Exactly! Saying "SQS will solve it" is never a good answer, even if you know SQS. It's always about the how & why. Sounds like you have a clear understanding

Show more

1

Reply

C

CleverAquaFowl531

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9lw2nbg009xad087vh2o82m)

Nice design! One question is that, if you do 10k jobs/s that could mean 36m execution/h and your exectuion table is parititoned on hourly basis, will that be a problem for dynamo/cassandra as they have partition size limit?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9rxbvp7001iad08o1tacnzi)

DynamoDB's partition size limit (10GB) isn't an issue here. With 36M executions/hour at ~200 bytes each, we're only looking at ~7GB per partition. If we did hit the limit, we could easily shard the time bucket further (e.g., 30-minute buckets instead of hourly)

Show more

0

Reply

L

LeftVioletHerring114

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9mt4vdx00mmad08s4unebvy)

The practice exercise for this problem has no functional requirement for recurring jobs - is it intentional or accidental discrepancy between the practice and write-up? The lack of this requirement changes the design a bit. Which version is more probable in a real interview?

Show more

0

Reply

MK

M K

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9p0rfgc01vpad08qcufr7a0)

I have a few thoughts:

1.  "While SQS messages can be up to 256KB in size, our messages will be quite small, around 200 bytes each containing just the job ID, execution time, and metadata." - so worker itself has to query Job table for job/task information on execution, correct? if we expect executing10k jobs per second, we might need a cache for job information to be queried by those workers.
2.  we also need a (daily) cron job to create near-future jobs to Execution table so that Watcher can be freed from querying Job table for the "next" job.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9rxamdb001ead08npo3gufp)

For #1: Not necessarily. At 10k/s, the database can handle those reads just fine (DynamoDB can do millions of reads/sec). Caching would add complexity without much benefit since job info is only read once per execution. For #2: Nope, that would actually make it worse. The watcher (cron) needs to query the Job table anyway to handle new jobs created in the last 5 minutes. Creating jobs ahead of time would just mean maintaining two separate processes that both need to query the Job table. Keep it simple - one process that handles both recurring and new jobs.

Show more

1

Reply

M

MonetaryVioletErmine187

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cm9rghlib00xaad08zj42ugzz)

I think the worker part is a bit over-simplified, it directly invokes a kubernetes/lambda function, but how can we know which target kubernets/lambda to trigger?

In reality, should this be divided to two parts, one for ochestration, the orchestrator also served as updating the JOB/Execution DB with latest status; the other for execution, for example, if this triggers a long running model training task, you want th execution only focus on training, the orchestor pulls the status to check whether the job indeed finished. Or if you have a multi-step job, that need to invoke several steps in a dependency DAG, the orchestrator can handle that too.

Show more

1

Reply

![Aris Santas](https://lh3.googleusercontent.com/a/ACg8ocIla1c4C2-3jdlpkaC0yVXtZJatGvLM5meAEN-RfBvsovtwxrWuXQ=s96-c)

Aris Santas

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cma37c0z600f2ad08naceapva)

Great vid as usual - but fyi GSI keys do not need to be unique, and "primary key" refers to the entire composite (or simple) key in DynamoDB, not just the partition key component.

Show more

0

Reply

![learning buddies](https://lh3.googleusercontent.com/a/ACg8ocKb3ulgDowmFom690cLN6oPaql424dPEEDpp0MVyU9nXrZFXg=s96-c)

learning buddies

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cma4rg1y400a6ad08cxj87lrm)

There is strong consistency for job execution. A job should not run twice due to race conditions. Your non functional requirements does not mention that.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmakpvxoz03tqad08w4nv4kw4)

Actually, the non-functional requirements explicitly state "at-least-once execution" which is the opposite of strong consistency. This is a deliberate choice - in distributed job scheduling systems, it's generally better to risk running a job twice than not at all. That's why we emphasize making tasks idempotent rather than trying to guarantee exactly-once execution.

Show more

0

Reply

2

22parthgupta18

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cma5a7qxz0056ad08wa9wob8o)

Great article, Evan! been learning a lot through these. I have a scaling related question - by default dynamo offers around 6k rps for a partition. given we are dealing with 10k jobs per sec, wouldn't storing the jobs by hour based partitioning lead to throttling?

We can use a hash to store data across multiple shards - ex <time\_bucket>-hash(<job\_id>)%10. This would split the traffic across possibly 10 shards. and we can scale watchers horizontally then. each watcher can handle querying a particular hashId. Ex <time\_bucket>-1, <time\_bucket>-2 etc

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmakpv0io03tmad08vzsys9ob)

yah you'd need to further shard. could just be that we partition by 10s buckets instead of 60s as well

Show more

1

Reply

![Aditya Rohan](https://lh3.googleusercontent.com/a/ACg8ocJe-7y5dWw2FJQBidbc24y_P9ud1cJDaHi_lXFhcG_xsg5Bwp_B=s96-c)

Aditya Rohan

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cma5cqfjh006uad083nool2bt)

Generally, in microservice architecture, each service should access only its own database, but here in this design every service which requires database access is accessing job store. Is job store a service over dynamo db which has all the APIs to access the database?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmakprrzi03tbad08mos1qdgi)

https://www.youtube.com/shorts/jg-Dy0GNPvQ

Show more

1

Reply

![Aditya Rohan](https://lh3.googleusercontent.com/a/ACg8ocJe-7y5dWw2FJQBidbc24y_P9ud1cJDaHi_lXFhcG_xsg5Bwp_B=s96-c)

Aditya Rohan

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmaoa7ogp002tad08bzc6cjq6)

Thanks for sharing the video. I have gone through it and I agree what is explained. But video also mentions that big companies like amazon can prefer separate d/b per service. So in such case if we are interviewing for a tech giant, then what should be our approach? should we design it using separate d/b per service and if yes then how should we do that.

Show more

0

Reply

F

FavourablePlumBird366

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdbr3b5207jpad09aibk1hbr)

There isn't a straight answer to your question and it shouldn't be a straight answer. It depends on the specific situation. Say you have a e-commerce platform. You'd keep your products, catalogue, payment/txns, users etc in separate databases primarily because each of those systems are going to have interaction patterns and data-model built for specific problem. Those services aren't going to talk to each others' databases.

But say we talk about catalogue system for a second. Catalog searching might happen through an elasticsearch but it totally makes sense to have separate services for reading and writing but to the same database. Keeping diff dbs might be an overkill and overcomplication of the problem and keeping diff services is good because we have diff query pattern.

Say about a payment system- you have live traffic being handled through a transaction-service which is updating and reading the payment database, you almost always build a recon service which is going to update transaction statuses in the same databases. Makes sense to keep the services separate (because of diff query pattern) but same database is the good sense here.

There are obviously more factors to decide if we need different databases or not but laregly depends on traffic, nature of traffic, data-modelling and at times dbs capabilities.

For the context of interviews- you are not trying to showcase what an interviewer might prefer. You want to talk about different scenarios, different choices and your reasons to make those choices is what matters here. You are showcasing your thoughtprocess that you are making choices not because you are following some standard principles but because you are thinking about the use-case at hand wisely.

Moreover- I have worked a tech giant for 6 years and have taken those interviews. They are not looking for straight answers too.

Show more

0

Reply

![Rachit Saxena](https://lh3.googleusercontent.com/a/ACg8ocIIaOUXcoVmKWT2htqAtUDBwTlrhyRaL8dj0PVHwDfIHlQXqt6i=s96-c)

Rachit Saxena

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cma5uy9cy007qad07ds0q7omm)

Use Case of invisible failure where worker crashed : Task is to send the files to FTP server and update the flag in DB . Worker got crashed, while file has been sent but couldn't update DB. Now job will get retried and send the file again, which we want to avoid. How can we do so ?

Show more

0

Reply

![GlamorousMaroonEchidna536](https://lh3.googleusercontent.com/a/ACg8ocKWlkuv8bEMmY-qHU6dV2BcUxEt31C__yIOdjnxw8bx98K0_vzs=s96-c)

GlamorousMaroonEchidna536

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmc8dye3r01czad080utz0e9i)

You can't avoid this unless the job's itself is idempotent - which in this case is hard when it's user-defined.

Show more

0

Reply

E

EvolutionaryFuchsiaRooster944

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cma782q3e015kad073060i34l)

When putting jobs directly into the queue, doesn't this introduce another problem where we would need to do something like a 2 phase commit to make sure that the job was successfully written to the database and the job was successfully put into the queue?

Would this be an acceptable solution to say during an interview if the interviewer pointed this out?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmakppzf603t7ad081e4n9dh1)

You're right about the 2PC concern but it's actually not a big deal here. The database is our source of truth. if the queue write fails we can just retry, and if the DB write fails we never attempt the queue. No need for distributed transactions since we can make the operation idempotent through the job ID. Even in the worst case where we get duplicates in the queue, our workers are already built to handle that since we need at-least-once semantics anyway.

Show more

1

Reply

![Dewan](https://lh3.googleusercontent.com/a/ACg8ocL4JI5POtYZ6kS5WwPv3AzL__vRZUOpIvv2Mj7lNSwaOIIGUA=s96-c)

Dewan

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmad821bf01d3ad089uerdhr0)

The writing and explanation about DynamoDB's primary key, on and after 24:30, should be corrected. "time\_bucket" is the partition key(not primary key), which along with the sort key makes up DDB's primary key.

Show more

4

Reply

![Joe Li](https://lh3.googleusercontent.com/a/ACg8ocK-pPGLqHJvPbKNH4V238CavBP2kDA5mYon8TPfngs_HakslwvC9g=s96-c)

Joe Li

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmaf0mfud00ekad07nt5ix7zv)

The current approach relies on job execution to schedule next occurrence. However, the job execution could go wrong and it may never schedule next one. In that case, the job will never run again. So I think there could be something to avoid that or the design of scheduling could be different.

In my design, instead of querying the execution table to get all jobs that are due to run, it queries the jobs table for jobs that are due in next 5 min. I'd maintain a "next\_run\_datetime" field with each job record so that it's easy to query. Every time a job is picked up, that field will be updated to next occurrence time.

What do you think?

Show more

0

Reply

![GlamorousMaroonEchidna536](https://lh3.googleusercontent.com/a/ACg8ocKWlkuv8bEMmY-qHU6dV2BcUxEt31C__yIOdjnxw8bx98K0_vzs=s96-c)

GlamorousMaroonEchidna536

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmc8e2o3201ebad08c8xy7dy9)

Which component would keep the next\_run\_datetime updated for each job? Seems like this would either need to be the execution anyway, or a new cron job -- either of which could also just insert a new execution. So it would net out to complicating things without too much benefit. The main benefit would be an improvement to cold start (e.g. we lost all state of current executions due to a serious outage, and need to start scheduling from zero using Jobs).

Show more

0

Reply

P

PleasantBronzeCougar977

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmahqziv6007bad08dd13ilz3)

Seems like there are some windowing edge cases with the Watcher. E.g. say time now is 00:00, and the task is scheduled at 05:01, with cron every 5 minutes. We won't pick up the task in the first window, and the second one has only 1 second of buffer at 5:00. Possible it still makes it but very at risk.

The challenging part is the only way I see around this is with overlapping windows, but then you need to deduplicate.

Show more

1

Reply

S

StraightFuchsiaBee905

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmakoiqy303thad08mhlv40b8)

Thanks for the content!

Question regarding the watcher: isn't fetching 3M entries from the DB at once be compute intensive on both the DB and the watcher and prone to failure?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmakpngpe03t3ad08wedew38v)

You're right that fetching 3M entries at once would be problematic, but that's not what's happening here. The watcher queries for jobs in 5-minute windows using the time\_bucket partition key, so it's only fetching ~50k jobs at a time (10k/s \* 300s = 3M total, but split across 60 partitions). Even this can be further optimized by processing in smaller batches if needed.

Show more

0

Reply

S

StraightFuchsiaBee905

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmamosskr00joad07qnfoi61h)

Sorry for the additional questions, I feel I'm missing something obvious... Example: it is 12pm, the watcher queries jobs for 12:00pm-12:05pm. Isn't the hourly time\_bucket the same for all the jobs in this case? Even if the data is split across partitions, the watcher will eventually have to manage the 3M entries at once, won't it? Also, could you clarify how we are getting 60 partitions? Thanks

Show more

1

Reply

![Yipeng Wang](https://lh3.googleusercontent.com/a/ACg8ocLBBuTvgR40gHmDLjvicbmJEXu9PbBuLOGi2fanyw5YNUak_DtTGQ=s96-c)

Yipeng Wang

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmb42s6zn01stad07f0twoekd)

I’m also a bit confused here. Since the time bucket is rounded down to the hour and used as the partition key, I would expect these 3 million rows to share one partition key, or possibly two if the 5-minute span crosses an hour boundary.

Show more

2

Reply

![Vu Le](https://lh3.googleusercontent.com/a/ACg8ocLYsLfyO1bJmwEmsxb4mwrHZKiCebGX7Hoj3QMhkb1IWAyG5A=s96-c)

Vu Le

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcibdfqy00cpad08q1fiy3ik)

I think the time bucket is rounded down to minute not hour

Show more

1

Reply

![Yipeng Wang](https://lh3.googleusercontent.com/a/ACg8ocLBBuTvgR40gHmDLjvicbmJEXu9PbBuLOGi2fanyw5YNUak_DtTGQ=s96-c)

Yipeng Wang

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcicjyv800mzad08k2pa03s6)

> By using a time bucket (Unix timestamp rounded down to the nearest hour) as our partition key, we achieve efficient querying while avoiding hot partition issues.

This is the statement I refer to.

Show more

1

Reply

D

duakaran96

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcd79uzt04ygad089fluu9ga)

if we are round down to nearest hour, the parition key still stays the same for all 5-minute windows in an hour, so for a query fired at 12:00PM - we will have all 3M executions lying in the same partition since the partition key is hour. Wouldn't that result into a hot partition issue and too much for a single instance to handle? I didn't understand how this (60 partitions) is achieved -

> The watcher queries for jobs in 5-minute windows using the time\_bucket partition key, so it's only fetching ~50k jobs at a time (10k/s \* 300s = 3M total, but split across 60 partitions)

Show more

1

Reply

U

UsefulMoccasinLadybug263

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmalfg068049bad08atbm2yds)

How can we justify the standard numbers which we are putting in the non functional requirements? Like here, we have mentioned 2 seconds for low latency? How can we justify that this number is acceptable for low latency?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmapuf82x0005ad09imr5ns2j)

Ask/confirm with your interviewer!

Show more

0

Reply

D

Danny

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmapu35ou0034ad08dgb7gweg)

"I don't know why I feel I can't swear on here..." 🤣 Great video!

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmapueizm0001ad09o0tofg2m)

🙊

Show more

0

Reply

Q

QuietPinkOpossum868

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmar39zoc00n3ad08q7v1tqms)

If I want to use redis sorted set for the job queue, that implies the key is a single "queue" key with all execution ID's inside that key as a sorted set by timestamp, is that correct? Then if I want to lock an execution ID, I have separate key value pairs inside that redis instance where I lock using the execution ID.

If I want to use a redis cluster to avoid single points of failures, how can I ensure all the execution ID's in a redis instance's sorted set goes into same redis instance for the locked execution ID's?

Show more

0

Reply

![vahid Saber](https://lh3.googleusercontent.com/a/ACg8ocLLmrDzDAp-96SD7gPTdv4jCGNW20lW-ng8mEuVz6ysn-Q7VQ=s96-c)

vahid Saber

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmas4xv8300l8ad07b1jyyu3f)

Hi, Any chance we can get the link to the excalidraw ?

Show more

0

Reply

E

EnvironmentalYellowCarp592

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmatrvh0j011wad08654uwz6s)

To prevent stale job statuses in a message queue when execution fails, we use a TTL (Time-To-Live) timeout. However, each job has different execution time requirements—for example, Job A might take 20 minutes to process a dataset, while Job B could finish in just 1 minute (e.g., sending an email). Our execution worker updates the job status only after completion, but if the TTL is set to 30 seconds, the job becomes visible to other workers before it actually finishes. This can lead to duplicate execution, even though the job and worker are still active.

Show more

0

Reply

A

abrar.a.hussain

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmazesypd02tlad07q12p5tte)

In the video we talked about using Zookeeper for our distributed lock for leasing. Isn't this super expensive? My mental model for using Zookeeper znodes it's useful for things that are "rare but critical." Not sure if that's the right way to think about it. With the 10k RPS requirements these fine-grained locks are hiting on some limits of typical Zookeeper use aren't they? Every ephemeral znode creation/deletion is a full quorum write.

Could we instead use Redis with two ZSETs? One for pending jobs (score=exec\_ts) and one for in-flight leases (score=expiry) and a single Lua script that evicts expired leases and atomically moves due jobs between them? Am I missing any pitfalls here?

Show more

1

Reply

R

RobustBrownCicada921

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmb3w361q01tgad0725bxcc8e)

How do you handle the situation when a task is run early due to the cron running every five minutes so if the list is short, a task can run five minutes early?

Show more

0

Reply

![Yipeng Wang](https://lh3.googleusercontent.com/a/ACg8ocLBBuTvgR40gHmDLjvicbmJEXu9PbBuLOGi2fanyw5YNUak_DtTGQ=s96-c)

Yipeng Wang

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmb42k5xl01snad07thzuw207)

I think we also want to discuss how to scale out the watcher. It pulls 3M rows, creates 3M events in the MQ and inserts 3M new rows into the DB. This cannot be done in a single machine. If we have multiple instances for the watcher, how do we coordinate the instances to ensure they don't pull the same rows?

Show more

0

Reply

![Vaibhav Kushwaha](https://lh3.googleusercontent.com/a/ACg8ocK4isAlQpVzLMyd57ThUQKk0SswFa1qLFeqNEw3e5mu3ogJigEU=s96-c)

Vaibhav Kushwaha

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdgzavdu038aad08qvrdl3lb)

I think what we can do here is leverage the partition on the DB itself and assign each partition to each individual instance of the watcher (assuming it's horizontally scaled) - this way we'll know they are not pulling the same rows.

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmbb25fr10079ad081jyck835)

Hi Evan Here producer or watcher , how many instance of that we should have , how many records should it bring from DB at once. please comment on that part also if I talk about NFR Non-Functional Requirements Now is a good time to ask about the scale of the system in your interview. If I were your interviewer, I would explain that the system should be able to execute 10k jobs per second. Core Requirements The system should be scalable to support up to 10k jobs per second. just curious how 10 k records will be pulled from system and then how they will be pushed to

Show more

0

Reply

![Devendra sutar](https://lh3.googleusercontent.com/a/ACg8ocL6-WS0IaOAt5rs2NtfZASM3q5aWvkk57aQG5x7pPBfL5whomI5=s96-c)

Devendra sutar

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmbb99z4k00l4ad07bfmibs1e)

{ "job\_id": "123e4567-e89b-12d3-a456-426614174000", // Partition key for easy lookup by job\_id "user\_id": "user\_123", "task\_id": "send\_email", "schedule": { "type": "CRON" | "DATE" "expression": "0 10 \* \* \*" // Every day at 10:00 AM for CRON, specific date for DATE }, "parameters": { "to": "john@example.com", "subject": "Daily Report" } } Then, our Executions table tracks each individual time a job should run: { "time\_bucket": 1715547600, // Partition key (Unix timestamp rounded down to hour) "execution\_time": "1715548800-123e4567-e89b-12d3-a456-426614174000", // Sort key (exact execution time and the jobId since partition key and sort key must be unique) "job\_id": "123e4567-e89b-12d3-a456-426614174000", "user\_id": "user\_123", "status": "PENDING", "attempt": 0 }

The execution table shows job\_id, user\_id. How can it be treated as bucket of list of jobs that will be executed in certain window? I assume it should be list of job\_ids that needs to be executed in next 5 mins or window?

Show more

1

Reply

M

MarriedVioletMarten889

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmbqdgkw200jp08ad6gktpw5c)

"Second, when a recurring job completes, we can easily schedule its next occurrence by calculating the next execution time and creating a new entry in the Executions table. "

What happens if the current run hasn’t completed yet, but the scheduled time for the next occurrence has already arrived and we allow parallel executions of the same job?

Show more

0

Reply

![Aashima A](https://lh3.googleusercontent.com/a/ACg8ocJDjLQArHZ5jqceZ1gn1WZJMaZui4uqNr4KfS5GX1z9vYdLwA=s96-c)

Aashima A

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmc060d7a050708adff2fpotc)

Kids watching !! :D

Show more

0

Reply

![Yezhou Chen](https://lh3.googleusercontent.com/a/ACg8ocJFDpsD41xP_Hnw4b5hu4cCdBbi-1kBfwV1ohK_gcieOUxt_Q=s96-c)

Yezhou Chen

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmc3xt0cw00inad08zmw3g0g0)

Partition Key: user\_id Sort Key: execution\_time + job\_id

Why should we make a partition key on the user\_id?

It makes the message queue to reorder all jobs after query them from DB?

Show more

0

Reply

![Alex Bloomberg](https://lh3.googleusercontent.com/a/ACg8ocKgzZFA58tlTLDdyq8pVjpwnoA3WtKYaR1BWKJQjfuwmmIypTin=s96-c)

Alex Bloomberg

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmc5fvgwt00uead08uo73koy2)

Shouldn't users be creating tasks instead of jobs ? The API should also be /tasks instead of jobs. Jobs must be deduces from tasks, for instance a task to send an email to 100 users converts into 100 jobs of sending email to 100 users.

Show more

0

Reply

![Ashwani Kumar](https://lh3.googleusercontent.com/a/ACg8ocKnTRz8N7N2Qk-wtRpTexiIsP-zVCI57kQ3CQCWs-YnMQpH7cLf=s96-c)

Ashwani Kumar

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcgzfong02u4ad08udkrcwel)

Thanks for the article. I have one doubt though:

how the service that is pulling executions from db and pushing to message queue for execution avoiding duplicates executions i.e. multiple instances reading the same execution instance?

Show more

0

Reply

![Sarthak Koherwal](https://lh3.googleusercontent.com/a/ACg8ocKpZp3eiAQTJ2LWQRH4TkEljg-vsm4usTpHiCI5Ed0Nnw3Cilfb=s96-c)

Sarthak Koherwal

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmchuuje608qoad08t5aonrpr)

Why not use a worker manager to assign jobs to worker? This way SQS becomes free and worker manager can take care of retries

Show more

0

Reply

![Vu Le](https://lh3.googleusercontent.com/a/ACg8ocLYsLfyO1bJmwEmsxb4mwrHZKiCebGX7Hoj3QMhkb1IWAyG5A=s96-c)

Vu Le

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcib3dpu009pad08n9orv23a)

Why we need have distributed lock? instead of using visibility timeout from message queue. Message still there in queue until we commit that we processed it

Show more

0

Reply

![Илья Чубаров](https://lh3.googleusercontent.com/a/ACg8ocKMKBARZ4IvyH-htJ7nExn4tg9PKE5BeF7qm4qjhTjdB9AVRfo=s96-c)

Илья Чубаров

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmck4wued03pnad08tkkyl4u7)

When it comes to talk about idempotency I think we don't consider some payment cases.

How can we resolve this when we make a payment and the network goes down but 3rd party server accepts it request and make a payment? I assume we are able to use the idempotency key or ask 3rd party service by API only for this error group only.

Is there any way presented in this article that can resolve this issue?

Show more

0

Reply

![Jinchi Zhou](https://lh3.googleusercontent.com/a/ACg8ocJlgfSIsqLZQci3oolk9otak9Gkg7Hrg0FZdg33CqKSAoPg9g=s96-c)

Jinchi Zhou

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcp663oi01faad08at4q5612)

When are the execution rows created? If I wanted to schedule 1 execution per hour, it might be hard for the scheduler service to create 24 rows in the table on the synchronous path. It seems suggested that we can create the earliest execution, then rely on the worker to create the next execution. But what if we want the executions to be independent. Or the previous execution can fail, causing the next execution to not be scheduled. I'm thinking we can have a separate forward looking cron job that creates these executions for the next few days. The downside is that if the user changes the time of these executions, the worker would need to invalidate the execution before executing.

Show more

1

Reply

![Shirish Maheshwari](https://lh3.googleusercontent.com/a/ACg8ocJTpYid_rif8klUqYOPtU0d-fkg2lkBkwfKwL0hYxCgUo5G2Q=s96-c)

Shirish Maheshwari

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcpii0iy049iad08xafipz3u)

I recommend we allow the end user to define the type of worker needed for their job, as they are in the best position to assess their infrastructure requirements. During job registration, we can provide configurable options such as container sizes (e.g., S, M, L, XL) to let them choose based on resource needs.

Additionally, the user should specify the base image required for job execution—such as the runtime environment (e.g., JDK, Go, Python) and operating system (e.g., Linux, Windows). Maintaining a wide variety of runtime environments and OS images internally would become increasingly complex and unsustainable over time. It’s more scalable and maintainable to have the user supply this information upfront.

Thoughts?

Show more

0

Reply

D

ding.zhaowei.ding

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcqptsjj01csad08g6b7qz8j)

There's a lot of great things here thank you so much, it's very helpful! But I have a couple concerns hope you can clear up.

1.  **NextExecutionTime** is **crutial** for the query efficiency. This was mentioned only as a side note and does not show up any where in the entities.
    
      - Follow up, it's not clear where this Next Execution Time is set nor by what process.
      - I would argue that setting it on execution records is not a good idea. 
     		 - Because it becomes self-sustaining. If I were to delete an execution record or it becomes corrupt, that Job would be lost untill the execution records are re-seeded from the Jobs table. 
     		 - Much better to always build from the Jobs table with the NextExecutionTime set in that record. 
     		 
    
2.  Heavy reliance on SQS features, but how to effectively recreate the functionality in standard message queue is not clearly mentioned.
    
     - If Watcher polls 5 minutes ahead, how does a MQ without a Delay handle this? A multi teir funnel? (as in a queue for each minute, and works consume and puts the execution into the next minute until execution?)
     - alternative only seems to be memcache with a heap structure. Needs multiple heaps to scale.
    
3.  Distributed Lock with MQ. Please correct me if I'm wrong. If you don't consume the message from the que it blocks all messages behind it. If your worker holding the lock crashes and you have a time out TTL policy of > 2 seconds. You just broke your SLA because no workers can consume any messages in that queue during that time.
    

Thanks!

Show more

0

Reply

![Elvis C](https://lh3.googleusercontent.com/a/ACg8ocKGbP0iy-I8YeffxOQOypbjV93lEoEQTuTH-kz94AsdSHvxGQ=s96-c)

Elvis C

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcr4hvzs02dbad07eqt51e2w)

in deep dive 1, how can we make sure the message ordered by execution time if it is FIFO and append only mecahnism?

Show more

0

Reply

S

shaileshgupta0803

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmd95c50e00fgad088txdpva0)

It cant be fifo, since fifo sqs queue doesnt give delivery delay support at a msg level. It will be a standard queue. However standard queue dont guarantee ordering.

Show more

0

Reply

![shashank reddy](https://lh3.googleusercontent.com/a/ACg8ocKcTZoqzG50xSB3Xx92aKjdcRQceLX4q68B4gWS_wlT21Azejoy=s96-c)

shashank reddy

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcu4clul01gdad08tnflphdw)

By Job Scheduler, I thought we were going to create a system similar to Cron Job. But realized that we are using Cron Job itself to design a Cron Job like scheduler. I maybe missing something here, but would the cron job (here watcher) that queries our 'Jobs' table execute continously - quering for Jobs that run within the next x amount of time? Or would it run every x seconds/milliseconds to query the DB for the jobs to run?

If the latter is true, isn't this the one we're supposed to design?

Show more

1

Reply

D

DistinctAmberHarrier825

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmcugrxmy03gcad08riixgxmd)

Would it be ok to suggest a dedicated queue for immediate requests. so seperate instances of job scheduler for scheduled vs immediate requests.

Show more

0

Reply

![Sai Nikhil](https://lh3.googleusercontent.com/a/ACg8ocLn9Zn2QRZJuYG2tXubcCVGjaKoiWtJYFn0nBxLe1nWcS5auQ=s96-c)

Sai Nikhil

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmd1u0xbc0aynad081czbcnvj)

I was initially planning on using a RDBMS like Postgres/MySQL because there are many inplace updates with execution status at every stage, I felt cassandra wouldn't be a good choice as it would mean lots of tombstone deletes, not sure if it's the same case with DynamoDB, any thoughts on this?

Show more

0

Reply

Z

zbgscz

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmd486gbr07pkad08sq2mzxyv)

The design seems relay on Amazon SQS too much, do we have alternative "great" answers with other open source middlewares? Leveraging SQS's existing features look like a blackbox to me

Show more

0

Reply

![Sourabh Upadhyay](https://lh3.googleusercontent.com/a/ACg8ocKpj06uaBaRfJhVsAJL98n9F7-IyL3NsYEkFdZuG1m_9wYa4Q=s96-c)

Sourabh Upadhyay

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmd85xvjk01x9ad083vuta3wm)

Instead of a poller polling for next jobs - can't we just use CDC streams? Create two tables: ExecutedTask and ScheduledTask. When a user wants to schedule a job, they can call a ScheduleJob API to schedule a job. The API will create the first scheduled task with job id in the metadata (possibly a GSI) and a TTL matching the next execution time. When TTL expires, the job details come to the processing service via CDC + Kafka. The processing service calls the main scheduling service's PutTaskInExecution which will add the task to ExecutedTask table with InProgress. The processing service triggers the task and updates the status. The same CDC can be subscribed by another lambda that essentially goes and creates the next scheduled task.

Show more

0

Reply

![Shubham Sharma](https://lh3.googleusercontent.com/a/ACg8ocIOhMIi1_plG6Gtw-e68fmaWK_Xt-YmmEBjXVehf6UC-HffbrcL=s96-c)

Shubham Sharma

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmd8aq2yv03bjad08pvegfy1h)

I was asked this exact question yesterday. There are a couple of things that stood out from the interview:

1.  The design decision to create the next entry for the recurring job execution when the previous execution completes is completely dependent on the fact that the job will finish in less time than the recurring schedule. Ex: if a cron runs every 5 minutes and the job takes more time than the same (let's say 10 mins) - then the above design won't work properly and we'll miss out on jobs or lag behind.
2.  The above design completely relies on the fact that we'll do at least once execution. What if we were required to handle exactly once execution
3.  If let's say the scale is very high i.e 50k RPS, keeping a single instance of watcher service just won't work. How would you handle different instances of watcher getting different job executions to be able to scale this up
4.  The interviewer said that our design relies on SQS a lot completely - what if the company doesn't have the budget or familiarity with SQS

Show more

1

Reply

![Sourabh Upadhyay](https://lh3.googleusercontent.com/a/ACg8ocKpj06uaBaRfJhVsAJL98n9F7-IyL3NsYEkFdZuG1m_9wYa4Q=s96-c)

Sourabh Upadhyay

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmd8w6vsz06y2ad08w5zwvm8n)

I think the next entry needs to be created as soon as the latest entry has started processing and not once it has completed since each execution is independent.

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 29 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdj336k600uwad081jv383xl)

Hi , can we discuss more about these questions? send me your e mail

Show more

0

Reply

R

RelatedCrimsonAlpaca687

[• 23 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdrlzo9w062zad08jzoa0r0z)

Very interested in this as well since I was also recently asked this, and was asked to support up to 1M jobs scheduled at any given time as part of the NFR which would create an extremely hot partition and/or thundering herd type problem.

I'm wondering if you can partition on time and add some sort of coordinated consistent hashing to it. Then Watcher instances register with ZooKeeper and you can effectively have each instance query a manageable range. So if you have 1M or more jobs all scheduled for exactly noon then you can still distribute the poll load across multiple watchers and you get better durability on Watcher. Admittedly, I'm not 100% sure how crashes mid-poll would work (one of the watchers would need to catch up on the crash instance's range).

For #4 - you can use Kafka or RabbitMQ here, but Apache Pulsar actually has a lot of similar SQS type semantics needed for some of these problems (ack timeouts, delays).

Show more

0

Reply

F

FavourablePlumBird366

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdboghez0724ad09vn8mpwbf)

Maybe a solution to handle the jobs on priority is to introduce another topic in kafka message queue i.e. priority\_scheduled\_jobs. Realistically this topic isn't going to be used a lot so it should be fine when it comes to fast execution.

JobService can decide if the execution run is going to be in next 5-10 minutes and publish to the priority\_scheduled\_jobs topic as opposed to scheduled\_jobs topic to which the watcher is publishing to.

How does this sound?

Show more

0

Reply

![Vaibhav Kushwaha](https://lh3.googleusercontent.com/a/ACg8ocK4isAlQpVzLMyd57ThUQKk0SswFa1qLFeqNEw3e5mu3ogJigEU=s96-c)

Vaibhav Kushwaha

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdgz451y033mad08390wb269)

Assuming that we have a calendar like approach for job schedules where we see each individual instance of jobs, how would we handle jobs which take like 10 minutes to run each and are scheduled to run every 5 minutes? We cannot rely on the completion of one job to put the execution of the next one in SQS. How would we handle this?

Also, I am guessing, if that's handled somehow, we'll also need to ensure that if two similar jobs are running in parallel (which can happen with the above example), then they should be inherently designed like that so that there can be multiple instances of same job running in parallel.

Additionally, I have one more question. Consider this more of a generic question for all idempotency-handling cases. Let's say I have a job which is supposed to send an email to X, the job finishes and sends an email, but just before updating the idempotency-key, it fails. So when it will be retried, the email would be sent again and this time the key would be updated successfully. This would work fine as getting an email twice is alright, but what about the cases where critical data is involved? If I am thinking that the answer is somewhere along the lines of transactions and implementing ACID properties, am I thinking in the correct direction?

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 30 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdhm8mrw07qhad082e6vz7h4)

deep dives: job should execute in 2s Phase 1: Query the database: Just like in our current design, we will query the Executions table for jobs that are due for execution in the next ~5 minutes - here if we are fetching all records , wouldnt it load the database? Also how doe sit make sure jobs get executed in 2,s, so we pick josb to be run and send them one by one to message queue ?

Please explaiin how do we efficiently fetch the jobs from database and how do we put (given we are fecthing lot many jobs\_ - this can be one of patterns also I beleive

Show more

0

Reply

F

FlyingTomatoHeron745

[• 29 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdjytkvy06onad08oipth6x4)

Because we use idempotency check combined with retries for failures, shouldn't this be "exactly once" job execution and not "at most once" execution here?

Show more

0

Reply

![Prerana K](https://lh3.googleusercontent.com/a/ACg8ocJshNL6ReiMywhZOBnFdecLMYxpWu9zg4kqNK_tC46kIZVFoYLK0g=s96-c)

Prerana K

[• 21 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdunsgsh0a9wad08n53uy01v)

what if for peak hours/secs, we have 100K jobs per minute, would Watcher pick up that many updates and put it into SQS. Watcher also needs to be further sharded/distributed and data in Executions table should be sharded like having segment along with executionTime to avoid hot partitions here.

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 20 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmdwlbpw900l1ad07emn7q340)

I might have missed this in the article but is there a way to scale the Watcher?

Show more

0

Reply

C

CorrectSapphireGorilla213

[• 17 days ago• edited 17 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cme0l9qxp00w5ad08x8df8gix)

Shouldn't the core entities here be:

1.  Jobs
2.  Executions
3.  Users/Teams

From what I understand the core entities are essentially the tables in the data store? I don't see how these are meaningful entities in this problem.

1.  schedule
2.  task

Meaning, I don't see us having separate tables for the above two entities so maybe they shouldn't be listed in Core Entities?

or maybe my understanding of "Core Entities" is misaligned.

Show more

0

Reply

A

Abhi

[• 16 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cme2age3q02ycad082v592ycm)

Great tutorial once again! I think in the video you switched between primary key and partition key, I believe primary key is made up of partition and sort key but you mention partition key is made of primary key and sort key.

Show more

0

Reply

H

HandsomeIvoryCrow799

[• 15 days ago• edited 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cme43ihg305kuad07xdg0amf8)

Instead of Watcher, why don't we implement a CDC system on a new table called NextExecution? Basically, the NextExecution record will have TTL set to 10 minutes before the execution's scheduled time. Once the NextExecution record expires, it will trigger a CDC system which do two things:

1.  Put the execution request on the message queue.
2.  Create a new NextExecution record and Execution record.

I understand that the NextExecution solution sounds like a hack but it offers major advantages:

1.  It does not consume the table's read throughputs.
2.  We can easily scale the CDC system horizontally easily, whereas in the article, a single watcher instance would need to handle 3 million records every 5 minutes (assuming 10k jobs / second).

DynamoDB's CDC system also supports back-off retries and dead letter queue. :)

Show more

1

Reply

U

UnderlyingApricotGoldfish326

[• 8 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmedvx2cc00b5ad086kz0291o)

This is a very interesting idea! I was reading somewhere that TTL typically happens within 48 hours though?

Show more

0

Reply

H

HandsomeIvoryCrow799

[• 6 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmegc640f07cnad076d8hp7cn)

Per my knowledge, there are no current limit on the TTL. I used to set TTL for xx days for my past projects.

Show more

0

Reply

A

Abhi

[• 14 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cme53rnps0e8gad08p59cp0ji)

Since our workers are querying the database everytime for the taskId won't that overwhelm the database as well?

Show more

0

Reply

![Kangze Jia](https://lh3.googleusercontent.com/a/ACg8ocJxhLMNhjXqpuZ_IxTP42ew9Oj3IRX1UEe_ksM7h02aj6qc0g=s96-c)

Kangze Jia

[• 7 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmef042hb047mad08he9im3k3)

Hello Evan, Under the job leasing section, you mentioned "Network partitions create additional complexity: if Worker A can't reach the database to renew its lease but is still processing the job, another worker might start processing the same job, leading to duplicate execution." I think this is also a concern to SQS visibility timeout approach. No?

Actually, I saw "Here are a few ways we can handle idempotency" section. Will we rely on these solutions to avoid duplicate execution? If yes, the job leasing solution also can rely on these solutions, right?

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 7 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmefjkcv8016vad0703pg7xhp)

Deduplciation table: This approach is not clear , as we already have execution table , wht not check status there , and even if we create dedub table , what extra advatage for dedup it provides

Show more

0

Reply

![Kapil Arora](https://lh3.googleusercontent.com/a/ACg8ocIAD73tffG2lbTTyMHh6hvA9XgYESLQphOwC6zOMHrjELRrTdo=s96-c)

Kapil Arora

[• 7 days ago• edited 7 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmefnlxn701pzad081255l5u0)

How you are handling the data center failure here, let's say worker consuming the messages from queue are not able to connect to queue?

Show more

0

Reply

C

ComplicatedYellowLlama491

[• 12 hours ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmep0r90l030had08d7tcxma6)

IMO, handling data center failure, beyond extending to multiple AZs, is outside of usual system design. we should add simple monitoring on oldest message age, and that should be it

Show more

0

Reply

T

TartAzureAlbatross975

[• 3 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmeka1exq0106ad08mmqoko9i)

One bit that confuses me in this design is having the workers schedule the next instance of the job execution. Are we executing this in a transaction along side the job status update to COMPLETED? If not, how do we handle failure modes where the executor has finished the job but failed to create the next instance of the job?

In my mind it makes more sense to have a service dedicated to scheduling job executions, rather than coupling this behavior with the job execution itself. Admittedly this adds its own complexity with considering whether the job is already executing for high frequency jobs, but overall seems like a better high level design.

Show more

0

Reply

![kandha guru](https://lh3.googleusercontent.com/a/ACg8ocKgGsKmi2OFIfRaagNPh-Lidy_jdpyU99tg4fXsVagDeZpJTiN3CQ=s96-c)

kandha guru

[• 1 day ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/job-scheduler#comment-cmeo4obba00o1ad09rvgw2sas)

Hello, Are we running a single instance of a watcher polling jobs from DB? If so, if the watcher crashes and takes a minute or less to come back, performs the poll, in that case, don't we lose the jobs to execute in the minute that is used by the instance to recover?

Show more

0

Reply

