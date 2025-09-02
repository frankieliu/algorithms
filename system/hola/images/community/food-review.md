##### Design a Review App for Food Items

by Viswa• Senior• 29 days ago

Design a system where users can review individual food items they have ordered. Other users can view these reviews and upvote or downvote them. Once the upvote to downvote ratio reaches a certain threshold, the user who wrote the review receives a payout.

Collapse Description

6

8

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdmcmixp05a0ad07gax6yhwe#comment-cmdnmvm8z01kiad08m7anvk87)

Nice design! Just some thoughts:

1.  I'm curious where the non-functional requirements around < 1 min viewing of the average rating come from. With many such services, they aggregate on a periodic basis, like every N hours or days, in order to prevent low float items from having massive swings in rating. Certainly, there is no right or wrong answer, though it can be argued either way.
2.  Nit: n+1 query params should be & not ?, i.e., /api/items/?restaurantId&page&pageSize.
3.  The deep dive one is interesting to me. Redis incr is reasonable, but I don't follow the assumption that we would see a large surge in upvoted/downvoted reviews at any given point. I'd expect this volume to always be pretty low, actually.
4.  For #2, you'll want to be careful to make sure that this update is via a transaction with proper synchronization controls! Either lock the rows or use OCC. Otherwise, you could overwrite two concurrent reviews.
5.  Can the reward also be assigned within the single, synchronous transaction? It might be over-engineering to do everything you've done in deep dive 2, unless I'm misunderstanding.

Show more

1

Reply

F

FairLimePenguin653

[• 22 days ago](https://www.hellointerview.com/community/submissions/cmdmcmixp05a0ad07gax6yhwe#comment-cmdvwlb190nmxad087mb7ytp9)

Thanks Evan for your review.

1.  I added that non-functional requirement what if the rating should be reflect asap. As you mentioned, it can be updated every N hours/days.
2.  Should avoid the restaurantId param. It is added by mistake. Should not include n+1 query param.
3.  Large surge is based on the assumption only. But I agree with your volume.
4.  Yeah, can be done using database lock or redis lock on review id
5.  Its not synchronous transaction, once the feedback received, feedback service will publish the event and the worker will calculate the reward eligibility and grant async.

Show more

0

Reply

U

UnderlyingApricotGoldfish326

[• 24 days ago](https://www.hellointerview.com/community/submissions/cmdmcmixp05a0ad07gax6yhwe#comment-cmdtir00w01gkad08yibnexrh)

Hi, this is a nice design! I was wondering if the Items table should also have total\_rating as well?

Show more

0

Reply

F

FairLimePenguin653

[• 22 days ago](https://www.hellointerview.com/community/submissions/cmdmcmixp05a0ad07gax6yhwe#comment-cmdvwmtsw0nnfad08ytetyg9i)

Items table will have the average\_rating and num of reviews. Total rating can be calculated (average\_rating \* num\_of\_reviews).

Show more

0

Reply

O

OfficialBeigeTern763

[• 23 days ago](https://www.hellointerview.com/community/submissions/cmdmcmixp05a0ad07gax6yhwe#comment-cmdv0wnbn0fdyad08qx58syjb)

For reward calculation, what if the cron job is dead in the middle? How do you make sure retry is idempotent?

Show more

0

Reply

F

FairLimePenguin653

[• 22 days ago](https://www.hellointerview.com/community/submissions/cmdmcmixp05a0ad07gax6yhwe#comment-cmdvwukek0nprad08i7gkrz7d)

We can enforce a unique key on the rewards table using the reviewId to ensure that only one reward is granted per review. This way, if the cron job attempts to insert a duplicate reward, the operation will fail gracefully.

Alternatively, we could maintain a separate table to track reviews that have already been rewarded. The cron job would check this table before inserting a new reward, ensuring idempotency. However, this approach requires an additional database query and storage. We’d also need a mechanism to purge old data from the tracking table periodically.

Show more

0

Reply

![Manasi](https://lh3.googleusercontent.com/a/ACg8ocI7UXL3f0aW-ZDAb0RPYCgcXfhirvXGD2ZEKHcFJ9kup84jcCo=s96-c)

Manasi

[• 6 days ago](https://www.hellointerview.com/community/submissions/cmdmcmixp05a0ad07gax6yhwe#comment-cmejix4uj03q8ad084wknn9ir)

Why do you have consistency over availability? It seems bit unnatural for a system like this one

Show more

1

Reply

F

FairLimePenguin653

[• 5 days ago](https://www.hellointerview.com/community/submissions/cmdmcmixp05a0ad07gax6yhwe#comment-cmejydhkb06ctad07pjwn5tna)

For this system design, availability is preferred over consistency. That was a typo. Thanks for catching that.

Show more

0

Reply