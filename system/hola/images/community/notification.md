##### Design a Notification Service

by UnderlyingApricotGoldfish326• Senior• 26 days ago

Design a notification service that takes in notification requests from client apps to send to iOS, Android, email, and SMS. There will be about 10 million notifications a day.

4

19

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdr6cwbn001tad39xa4ll9kj)

Nice design! You've created a well-structured notification service with good separation of concerns through channel-specific queues and proper worker patterns. Your approach to handling user preferences and preventing duplicate notifications shows solid distributed systems thinking, though there are some critical gaps around data consistency and failure handling that would need addressing before production deployment.

1.  **Positives**
    
    -   Using separate SQS queues for each notification channel (iOS, Android, SMS, Email) provides excellent isolation and allows independent scaling of workers based on channel-specific load patterns
    -   The duplicate prevention strategy using notification ID lookups before sending prevents erroneous reprocessing, which is crucial for maintaining user trust in a notification system
    -   Including circuit breakers for third-party provider failures shows good defensive programming and will prevent cascading failures when external services experience issues
2.  **Potential Issues** **Race Condition in Duplicate Prevention**
    
    -   Your duplicate check reads the notification status from the database, then sends to the provider, then updates the status to success
    -   Two workers processing the same notification simultaneously could both read "pending" status and proceed to send duplicate messages
    -   This violates your requirement to avoid duplicate messages and could annoy users with multiple notifications
    
    **Missing User Preference Lookup Caching**
    
    -   Every notification requires a database lookup for user preferences, creating unnecessary load on the User Database
    -   With 10M notifications/day (116/second average, likely 500+ at peak), this creates significant database pressure
    -   User preferences change infrequently, making them ideal candidates for caching to reduce database load
    
    **No Retry Limit or Dead Letter Queue**
    
    -   Failed notifications are put back on the queue with an incremented attempt counter, but there's no maximum retry limit mentioned
    -   Without a retry limit and dead letter queue, permanently failing notifications will cycle indefinitely
    -   This wastes resources and prevents identification of systemic issues with specific notification types
3.  **Follow-up Questions**
    
    -   How do you handle the scheduled notification feature mentioned in requirements? SQS delay is limited to 15 minutes, but users might schedule notifications days in advance
    -   What happens when a user updates their notification preferences while messages are already in the queues?
    -   How do you ensure the 1-second delivery SLA when third-party providers like APNS or FCM might have variable latency?
    -   How does your system handle partial failures where some channels succeed but others fail for the same notification?
    -   What's your strategy for handling notification templates beyond email? Push notifications often need titles, bodies, and custom data

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

U

UnderlyingApricotGoldfish326

[• 26 days ago• edited 25 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdr6o5gj05aiad086wyun9ss)

-   SQS delay is limited to 15 minutes, but users might schedule notifications days in advance

Totally forgot about that. I'd say for scheduling we can use the method outlined in the Job Scheduling article on Hello Interview.

A watcher polls for notifications that will run in the next five minutes and sends it on its way to one of the SQS queues.

Show more

2

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdrj91e907ntad08kkb8gdvm)

Yeah, that's exactly right! A super similar implementation can be used here with the two-tier technique.

Show more

1

Reply

U

UnderlyingApricotGoldfish326

[• 25 days ago• edited 25 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdrny6r608ybad08fhvodcj6)

> Two workers processing the same notification simultaneously could both read "pending" status and proceed to send duplicate messages

In Kafka, workers are assigned partitions so that no one message can get handled by more than one subscriber. I was thinking the same idea applied here? A worker gets assigned only one unique notification? And we also check the Notifications DB to see if the notification is in a pending state? Would a distributed lock help here to show that a worker is being assigned work on one notification?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 25 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdro7hxn090xad08xckewo4k)

No you should be fine here. Your understanding is correct and with SQS when a worker reads a message it sets a visibility timeout while its processing, so other works can't read it. I would not be concerned with a potential race here.

Show more

1

Reply

H

HomelessAquamarineTrout494

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdynfilo088pad07dt9ytcvc)

But your design is using SQS only right? I couldn't find Kafka setup anywhere on the board. What am I missing?

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdzcbfoi05wiad08ftq29uls)

I was more familiar with Kafka, just wanted to check that SQS has similar functionality to not send the same message to multiple consumers.

Show more

0

Reply

![Naveen Maurya](https://lh3.googleusercontent.com/a/ACg8ocL03z03XE6dUenyCZMv685dLnVqwTrpmZAbIm0v-gTiL47SpSI6=s96-c)

Naveen Maurya

[• 21 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdxekxla08e4ad07ydgru0o6)

How can we ensure the Notification has been successfully delivered to the users? Will there be any service which ensures the successful delivery? What will happen if failure happened after the 3rd Party service handover?

Show more

2

Reply

U

UnderlyingApricotGoldfish326

[• 21 days ago• edited 21 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdxhy6mn09nsad09myay8s2c)

Hi, thanks for your message! I guess the third party provider will give a status of where the notification is at. If there is some failure then we'll have to put the notification message back onto the SQS queue for further processing, and perhaps increment the number of attempts.

Show more

1

Reply

H

HomelessAquamarineTrout494

[• 20 days ago• edited 20 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdym45xv080yad07zlwpj7g3)

Great design with lot of details, few clarifications:

Requirements:

1.  Client app send the notifications (Is client here 3rd party?)
2.  How are users signing up or their state managed for receiving particular notifications from a particular client? Or how is this system redirecting a client's notification only to a subset of users?

HLD 3. Which DBs choice are we proposing for both User and Notifications and why? Why did we go for separate DBs for 2 tables?

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdzceyt005xdad08sndy23vb)

Thanks for your response. I think the client app would be an internal app within the same company that wants to send notifications. In this design, each message will have an attached recipientUserId so we'll know which user to forward the message to. I suppose everything could be in the same database. If we go with DynamoDB, we could have a partition key on the notificationId for when a worker updates the status of a notification. I would imagine we would have many more notifications than users.

Show more

0

Reply

H

HomelessAquamarineTrout494

[• 20 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cmdzq24gf0annad08p3to305d)

Thanks for the reply. The thing that's unanswered for me is even though there would be 10-1000x times more notifications than users, for every notification workers are still querying the recipientUser's preference, etc so the request load for both User and Notifications DB are similar. I am looking for strong reasonings or technology choices towards designing the storage components of this design and what strategies we're applying to handle failure scenarios.

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 19 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cme06uyem00nsad08z90y6phe)

Good point. I think adding a cache layer to store user preferences would help since they might not change that frequently. They can be updated via CDC if the user decides to update their notificationPreferences, phoneNumber, email, etc.

Show more

2

Reply

H

HomelessAquamarineTrout494

[• 19 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cme07it6a00zaad08e6ize6hw)

sounds good.

Show more

0

Reply

S

shktrigonometry

[• 19 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cme0ednhk03jgad08t22on6o3)

Should we USe CDC in notification Creation and pushing it to sqs Queue flow as well?

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 19 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cme0ei7ak000yad084fc24ab3)

I think that would work too if we want to decouple the creation of a new record and message publishing.

Show more

1

Reply

C

ConsiderableScarletCrocodile260

[• 15 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cme64rt1h0p0qad083l2dpsn9)

what triggers the successful response to the client app on send notification? Is is only after the message has been committed to all relevant queues & the notifications DB? how do you handle only one of the DB or Queue write going through but not the other?

Show more

1

Reply

U

UnderlyingApricotGoldfish326

[• 15 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cme6lvmqk0vfaad08dl9kcpfk)

I think for this design the client could poll the Notification Service or another service to get the current status of the notifications that they sent out. Alternatively a webhook design could be implemented to send a response on successful push to the desired channel(s). I'd say success means that the third party providers say that the message was pushed successfully, though I'm limited in my knowledge of how those work. Any issues with writes through through SQS would have retries on visible errors (such as errors with 3rd party proviers) and invisible errors such as workers going down would have the notification message from SQS re-sent to another worker.

Show more

0

Reply

C

ConsiderableScarletCrocodile260

[• 15 days ago](https://www.hellointerview.com/community/submissions/cmdr5zd2i0520ad09h31b8zk6#comment-cme6nd7nq00huad08exro82nf)

But specifically on the incoming send notification request to the notification service, there's 2 data responsibilities on the online path it seems:

1.  Create notification in the notification DB
2.  Add the message to all of the relevant queues

If these are write operations to multiple data stores, how do you handle only some of these writes being successful? Even if using the Kafka's transactional API, in the scenario where these are multiple topics in the same cluster which can be written to in a single transaction.

Show more

0

Reply