##### Design a Donations Website

by UnderlyingApricotGoldfish326• Senior• 27 days ago

In 6 months from now, we are going to sponsor a charity event by releasing a new app that allows people to make a donation to one of 10 charities.

Assume we already developed the iOS, Android and web UI frontends. The UI has a form which contains fields for first name, last name, email, payment card info, a dropdown to pick a charity, a dollar amount and finally a big donate button on the bottom. The app will only really be used for 3 days because the charity event starts on a Friday and will only be online that Friday through Sunday. During those 3 days, millions of donations are expected. We estimate we might collect $100m in donations for the 10 charities we are working with. We negotiated with a 3rd party payment provider who’s got a very simple REST API - you POST the payment card info to their endpoint and they return a 201 created response. We really want to use them because they are giving us a nice low transaction fee for each donation since this is a special event for non profits. The application does not need to disburse money into the 10 non-profit bank accounts - we will dump all of the funds collected in a single account and our CFO will manually write checks out of that account after the donations event is over.

Collapse Description

4

7

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 26 days ago• edited 26 days ago](https://www.hellointerview.com/community/submissions/cmdquzcch01wxad09rneix4es#comment-cmdquzs8g001nad3ayp3emslj)

Nice design! You've built a solid foundation for a donation system with good attention to fault tolerance and security. Your use of idempotency keys to prevent double payments and the webhook pattern for handling payment status updates shows thoughtful consideration of real-world payment processing challenges. However, there are some critical issues around database consistency and scaling that need addressing, particularly around how you're updating charity totals without proper concurrency controls.

1.  **Positives**
    
    -   Excellent security approach using iFrames and payment tokens instead of storing credit card details directly - this properly delegates PCI compliance to the payment provider
    -   Smart use of idempotency keys (donationSessionId) to prevent double payments during network failures or user errors
    -   Good webhook pattern for handling asynchronous payment confirmations, which gracefully handles timeouts between services
2.  **Potential Issues** **Race Condition in Charity Total Updates**
    
    -   You mention wrapping updates in a transaction, but this doesn't prevent concurrent updates to the same charity's total
    -   Multiple simultaneous donations will read the same total, add their amount, and write back, causing lost updates
    -   This violates your requirement that "donation amounts for a charity are accurate" - you could lose millions in tracked donations
    
    **Database Scaling Limitations**
    
    -   A single Postgres instance handling both donations and charity totals creates a bottleneck at your peak load
    -   With 10-20 TPS and all writes going to one database, you'll hit contention issues during traffic spikes
    -   Read replicas help with viewing but don't solve the write bottleneck for recording donations and updating totals
    
    **Missing Payment Failure Handling**
    
    -   No clear mechanism for handling partial failures between recording the donation and updating charity totals
    -   If the system crashes after payment succeeds but before database update, donations could be lost
    -   The webhook updates donation status but doesn't trigger charity total recalculation for failed initial attempts
3.  **Follow-up Questions**
    
    -   How do you handle concurrent updates to the same charity's total donation amount without losing data?
    -   What happens if the webhook arrives but the original donation record was never created due to a database failure?
    -   How do you ensure the system can handle traffic spikes beyond 20 TPS during peak donation periods?
    -   What's your strategy for reconciling charity totals if discrepancies are discovered after the event?
    -   How do you prevent the donation service from becoming overwhelmed if the payment provider is slow to respond?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdquzcch01wxad09rneix4es#comment-cmdqx3owl02pgad08cerr29ak)

I definitely disagree with "Database Scaling Limitations." 10-20 TPS is nothing, this is fine.

Show more

2

Reply

U

UnderlyingApricotGoldfish326

[• 26 days ago• edited 26 days ago](https://www.hellointerview.com/community/submissions/cmdquzcch01wxad09rneix4es#comment-cmdqvejp6020nad09levcfd4h)

Hmm I wonder if I should update charity totals as a background task instead through CDC to a worker...

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 26 days ago• edited 26 days ago](https://www.hellointerview.com/community/submissions/cmdquzcch01wxad09rneix4es#comment-cmdqx106902ofad0885bj5p7i)

Depending on how up-to-date the totals need to be, the easiest thing is probably just to update the total via a transaction like you do. Each time you receive a new donation, write that new donation and update the charity's total amount atomically. You just want to make sure you lock or have proper isolation level too.

Show more

1

Reply

U

UnderlyingApricotGoldfish326

[• 26 days ago• edited 26 days ago](https://www.hellointerview.com/community/submissions/cmdquzcch01wxad09rneix4es#comment-cmdqxf3oa00tjad08czwbqvna)

Thanks for your response Evan! The bot seems to think that the load here might be an issue if we update the new donation and charity's total amount atomically in a transaction so I was thinking of updating a charity's total in a separate process. I do think this feature would prioritize availability over consistency for total donations for a charity though.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 26 days ago](https://www.hellointerview.com/community/submissions/cmdquzcch01wxad09rneix4es#comment-cmdqy2vsf02zbad08dr437v41)

Yah I replied to the bot. It got that wrong. 10-20 tps is nothing!

Show more

1

Reply

U

UndergroundJadeMeadowlark921

[• 22 days ago](https://www.hellointerview.com/community/submissions/cmdquzcch01wxad09rneix4es#comment-cmdxr6i160cnxad08pbuun1gt)

Thanks for sharing. I really like your thought process!

Show more

4

Reply