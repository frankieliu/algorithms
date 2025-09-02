##### Design an Auction System

by Tomer Amir• Manager• 1 month ago

In this design I had a few deep dives I went into:

1.  Reliably not losing bids - thus Kafka
    
2.  Scaling the main PG db to 50TB of data per year, thus sharding + cleanup to S3 and Athena
    
3.  Real time updates + "hot auctions". the updates are in the design using SSE and a single consistently sharded pub-sub redis cluster, but I "verbally" explained that for hot topics I could manage a separate pub-sub cluster with a channel per auction (this might have a better solution, not sure)
    
4.  Allowing for analytics cross shards. this is why I introduced the SQS, analytics service and snowflake. I think that SQS might be a bad choice here because of the scale, and I didn't dive into how many events I am writing to it, but that was the last question and I was trying to keep to some reasonable amount of time.
    

Collapse Description

6

7

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

H

HomelessAquamarineTrout494

[• 1 month ago• edited 28 days ago](https://www.hellointerview.com/community/submissions/cmdik06y1042qad08fqzlklxc#comment-cmdj9nzrz019had08f4usvdtx)

Wow, neat diagram!

Show more

2

Reply

P

ParentalRedSole744

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdik06y1042qad08fqzlklxc#comment-cmdnz6sx9038jad08j5o2ulf2)

Looks great! With this design and those deep dives, I think you've nailed it.

Show more

2

Reply

![Kathir Kalimuthu](https://lh3.googleusercontent.com/a/ACg8ocILsDgvb89tb_WSjSClZcktRPNRK7lfXfqNd_DHQUYqBbcEiw=s96-c)

Kathir Kalimuthu

[• 12 days ago](https://www.hellointerview.com/community/submissions/cmdik06y1042qad08fqzlklxc#comment-cmeau64ky02a8ad072ekru56y)

what is the purpose of connection manager and using SSE to put some events to client?

Show more

0

Reply

![Tomer Amir](https://lh3.googleusercontent.com/a/ACg8ocJ_u6mXy5P0HLb4MsPN8aW0EvXHCi3jB2Kas1nLJEK8OOhdvcj_QQ=s96-c)

Tomer Amir

[• 10 days ago](https://www.hellointerview.com/community/submissions/cmdik06y1042qad08fqzlklxc#comment-cmedwwxla00f9ad07wm6g7z13)

yes, like last/highest bid

Show more

0

Reply

![Niilesh Raaje](https://lh3.googleusercontent.com/a/ACg8ocLpFPjUAiYziqFLvF41hrg4lc2anmBMP6kuv2H2_3-7szjt127u=s96-c)

Niilesh Raaje

[• 7 days ago](https://www.hellointerview.com/community/submissions/cmdik06y1042qad08fqzlklxc#comment-cmehrcd7z00biad077cnvwhqk)

Question - Whats the logic behind Auction service talking to Athena instead of the PG DB ? What about the Atomicity , Transactions in Athena . I understand that Athena provides a way to access and analyze data stored in S3 using familiar SQL commands. And curious why auction db points to clean up and then to s3 ? Did you clear the interview?

Show more

0

Reply

![Tomer Amir](https://lh3.googleusercontent.com/a/ACg8ocJ_u6mXy5P0HLb4MsPN8aW0EvXHCi3jB2Kas1nLJEK8OOhdvcj_QQ=s96-c)

Tomer Amir

[• 5 days ago](https://www.hellointerview.com/community/submissions/cmdik06y1042qad08fqzlklxc#comment-cmel1mjhi010vad07dpad0zfb)

It wasn't a real life interview :) I used Athena to clear historical data from PG, as it can accumulate and become pretty big. Since this is used just to read historical data, I don't care about ACID behavior.. the Auctions Service just uses it as a read-only source

Show more

0

Reply