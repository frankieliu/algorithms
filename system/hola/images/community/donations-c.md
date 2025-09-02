##### Design a Donations Website

by Akshay• Senior• 16 days ago

In 6 months from now, we are going to sponsor a charity event by releasing a new app that allows people to make a donation to one of 10 charities.

Assume we already developed the iOS, Android and web UI frontends. The UI has a form which contains fields for first name, last name, email, payment card info, a dropdown to pick a charity, a dollar amount and finally a big donate button on the bottom. The app will only really be used for 3 days because the charity event starts on a Friday and will only be online that Friday through Sunday. During those 3 days, millions of donations are expected. We estimate we might collect $100m in donations for the 10 charities we are working with. We negotiated with a 3rd party payment provider who’s got a very simple REST API - you POST the payment card info to their endpoint and they return a 201 created response. We really want to use them because they are giving us a nice low transaction fee for each donation since this is a special event for non profits. The application does not need to disburse money into the 10 non-profit bank accounts - we will dump all of the funds collected in a single account and our CFO will manually write checks out of that account after the donations event is over.

Collapse Description

1

2

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 16 days ago](https://www.hellointerview.com/community/submissions/cme63wiqh0ocwad08ync6r9af#comment-cme63xkst000gad3bsjfezfr0)

Nice design! You've created a well-thought-out architecture for handling high-volume donations during a short event window. Your use of SQS for fault tolerance and idempotency keys to prevent duplicate payments shows good understanding of distributed systems challenges. The design demonstrates solid thinking about scale calculations (6600 TPS) and includes important features like reconciliation workers and notification services. However, there are some critical issues around DynamoDB write capacity limits and the complexity of your aggregation approach that need addressing.

1.  **Positives**
    
    -   Excellent use of SQS for fault tolerance and decoupling - this provides natural backpressure during traffic spikes and ensures no donation requests are lost even if downstream services fail
    -   Smart implementation of idempotency using Redis with TTL - this effectively prevents duplicate payments while allowing legitimate retries after a reasonable timeout period
    -   Good separation of concerns with dedicated services for donations, notifications, and reconciliation - this makes the system more maintainable and allows independent scaling of components
2.  **Potential Issues** **DynamoDB Write Capacity Bottleneck**
    
    -   You're updating the charity table's moneyRaisedSoFar field for every donation, which at 6600 TPS could mean 660+ writes/second per charity during peak times
    -   DynamoDB has a limit of 1000 WCU per partition, and with hot charities this will cause write throttling and failed donations
    -   Your Lambda aggregation solution adds significant complexity and lag, making the "current amount raised" feature less real-time than users expect
    
    **Missing Payment Provider Failure Handling**
    
    -   The design doesn't specify how to handle 3rd party payment provider outages or rate limits
    -   Without circuit breakers or fallback mechanisms, the entire donation flow could fail if the payment provider has issues
    -   This single point of failure could prevent millions in donations during your critical 3-day window
    
    **Unclear User Authentication Strategy**
    
    -   The API mentions JWT/session cookies but there's no user registration or login flow described
    -   Without proper user identification, the "view your donations" feature won't work correctly
    -   This also makes tax receipt generation problematic as you need verified user information
3.  **Follow-up Questions**
    
    -   How do you handle the scenario where the payment provider accepts the charge but your system fails before recording the donation?
    -   What happens if a charity becomes extremely popular and receives 80% of all donations, creating a severe hot partition?
    -   How do you ensure the reconciliation worker doesn't accidentally double-charge users for failed transactions?
    -   Why use both Redis for idempotency and DynamoDB's optimistic concurrency control - couldn't one solution handle both concerns?
    -   How long do you retain donation records, and what's your data archival strategy after the event?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

A

Akshay

[• 16 days ago](https://www.hellointerview.com/community/submissions/cme63wiqh0ocwad08ync6r9af#comment-cme654q090ovtad08msp8tr2f)

2.1 For DDB write bottlenecks we can distribute the charity id over N keys, by adding this key randomly when we're writing we can distribute the write load over many partitions overcoming the 1000 WCU limitation. For the totals I mentioned that we're okay with eventual consistency, if we want more real time updates, we can use Redis and increment charity totals in a cache that we can reconcile with the DDB amounts on a periodic basis. Client's can poll the charity server to get the updates from our redis cache. 2.2 If the payment provider fails we will put the donation request back on the queue. If it fails after multiple retries we can mark the donations as failed and move the message to a dead letter queue. Our reconciliation worker will retry the payment at a later point with exponential backoff and jitter to overcome rate limitations 2.3 I put it BTL

3.1 The SQS strategy would allow us to retry the payment, using the 3rd party transaction id we can check whether the payment completed on the 3rd party side. 3.2 The sharding strategy I mentioned above for the partition keys should help remediate this issue. 0.8\*6660 = 5300TPS, if we use N = 10, we can distribute the loads to 10 partitions at 530 TPS each. 3.3 It will check the status of the payment with the 3rd party provider using the 3rd party transaction id 3.4 DDB could do it, but I wanted to mention an alternative to show breadth. 3.5 Move the data to AWS glacier or similar cold storage :)

Show more

1

Reply