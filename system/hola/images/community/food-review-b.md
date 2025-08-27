##### Design a Review App for Food Items

by UnderlyingApricotGoldfish326• Senior• 25 days ago

Design a system to write reviews, where viewers can rate the reviews, and the best reviews get rewards each month.

3

11

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdrytnav003dad3a23ss1o8b)

Nice design! You've created a well-thought-out system with good scalability considerations, particularly your approach to handling hot reviews through partition key randomization and your use of pre-computed rating counts. The circuit breaker pattern for the third-party rewards provider shows mature thinking about external dependencies, though there are some consistency concerns and missing details around the monthly rewards calculation that need addressing.

1.  **Positives**
    
    -   Smart use of pre-computed rating counts in the Review table to avoid expensive aggregations during read operations, which will significantly improve performance for the read-heavy workload
    -   Excellent handling of hot reviews through partition key randomization (review123#1, review123#2) to distribute write load across multiple Kafka partitions and prevent bottlenecks
    -   Thoughtful circuit breaker implementation for the third-party rewards provider to fail fast and alert engineers when the external service is down
2.  **Potential Issues** **Race Condition in Rating Updates**
    
    -   Multiple workers processing different partitions of the same hot review (review123#1, review123#2) can simultaneously update the totalRating count
    -   This causes lost updates where the final count may be less than the actual number of ratings submitted
    -   This violates the requirement to accurately identify top 100 reviewers for monthly rewards
    
    **Inefficient Top 100 Query Design**
    
    -   Using totalRating as a sort key (SK) in DynamoDB doesn't allow efficient querying of top 100 reviews globally
    -   DynamoDB sort keys only work within a partition key, so you'd need to scan all partitions and sort client-side
    -   This becomes extremely expensive and slow with millions of reviews, potentially taking hours for the monthly job
    
    **Missing User Aggregation Logic**
    
    -   The design tracks individual review ratings but doesn't explain how to aggregate ratings per user for rewards
    -   A user might have multiple reviews with different ratings, but the reward goes to users, not individual reviews
    -   Without proper user-level aggregation, you can't identify the top 100 users as required
3.  **Follow-up Questions**
    
    -   How do you handle the consistency issue when multiple workers update the same review's totalRating count from different partitions?
    -   What happens if a user rates the same review multiple times - do you track and prevent duplicate ratings?
    -   How do you aggregate all reviews for a user to calculate their total rating score for the monthly rewards?
    -   What's your strategy for handling the DynamoDB limitation where sort keys only work within a partition?
    -   How do you ensure the monthly cron job completes within reasonable time when scanning millions of reviews?
    -   What happens to pending ratings in Kafka if the system crashes - are they persisted and replayed?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdryzjxf02wyad07ha9n2vcp)

Hey! Do you mind just updating the original one? It's the same problem, right?

Was your intension here to trigger another bot response? If so, good to know, I can add a way to manually trigger after changes.

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdrz266s030lad07o4tmvklx)

Hi Evan, this is a different problem. I think I still need to make some changes to it actually. The other problems I worked on were Design Notification System and Design Donation App which I think are a bit different?

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 25 days ago• edited 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdrz8f8s02snad087leobhfv)

I think there was someone else who wrote a solution to a very similar problem haha

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdrzaien02syad08vic8ajh9)

Aw yes I'm just going crazy. These anon names get me all twisted!

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdrzde2n02u5ad08kw1tu2fi)

No worries! Thanks for checking in, I really appreciate it!

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdrzbqfo02thad08o66qn6vu)

Do you mind updating the title actually to Design a Review App for Food Items? That way we can later group by problems to show different peoples solutions to different problems which will be pretty great!

Show more

1

Reply

U

UnderlyingApricotGoldfish326

[• 25 days ago• edited 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdrzcj9802tpad08hs9osct7)

Will do!

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmdrzctyh02tvad08rzp34rzj)

Appreciate it!

Show more

1

Reply

![Qiming Chen](https://lh3.googleusercontent.com/a/ACg8ocK3t51q_NuHh22cWtE2YTjf2AGknKx-IkDw2NRewfJTfZssaw=s96-c)

Qiming Chen

[• 9 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmeekr0ab0623ad07y2kabwj4)

This is a thorough design. One followup: How do you prevent the double rating problem to prevent bad users to keep thumbs up on a same review to cause inaccurate rating?

Show more

0

Reply

![Manasi](https://lh3.googleusercontent.com/a/ACg8ocI7UXL3f0aW-ZDAb0RPYCgcXfhirvXGD2ZEKHcFJ9kup84jcCo=s96-c)

Manasi

[• 6 days ago](https://www.hellointerview.com/community/submissions/cmdryt78802osad08omi8wz88#comment-cmejj99ii03vead088jqess7d)

You should take a deeper look at how totalRating gets updated. The real bottleneck is that multiple workers may attempt to update the same field at once. Even with optimistic locking, this becomes an issue and ideally, you’d add a version field to the reviews table to handle it properly. One approach is two-layered: Use optimistic locking on the reviews record itself. Go a step further by partitioning work, have each worker subscribe to a specific reviewId (or choose a more efficient cardinality). This way, as your worker pool scales, only one worker is ever responsible for updating a given review, eliminating direct contention on the same record.

Show more

0

Reply