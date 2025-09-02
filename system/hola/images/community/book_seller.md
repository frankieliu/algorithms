##### Design a Book Seller Platform

by DominantScarletBoa987• Senior Manager• 30 days ago

Design a service where customers can submit book purchase requests with details such as customer information, book info, maximum price willing to pay, and payment information. The service should check with registered book sellers to find books priced at or below the maximum price. It should place a buy request with the seller offering the minimum price within the customer's limit. The design should include APIs for book seller registration and adapt to various seller APIs, supporting up to 10,000 queries per second.

Collapse Description

8

17

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdnmxc0401lbad088hjr7xxz)

Curious, where did this question come from? Was it asked at a company? Almost like a market limit style question in a stock app.

Show more

0

Reply

D

DominantScarletBoa987

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdnn5ttg01i4ad07vobn83kz)

This was in the Databricks list of questions on hellointerview real experiences. It didn’t translate directly to an online auction system like ebay so I assume whatever the customer pays will be processed as long as there’s a seller that fits the criteria.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdnn9oy401q3ad0840b9txmu)

Sweet design! Some thoughts from me

1.  nit: GET /orderStatus?orderId={id} should just be GET /orders/:id
2.  Dealing with CC info here would be a nightmare. Unless not allowed, you'd want to just use Stripe or other payment processor. otherwise you need to adhere to PCI compliance which is a headache which distracts from the purpose of this design (i assume)
3.  What happens if a seller posts a book for sale AFTER a buy request which would match?
4.  Choice of Kafka partition could create a hot partition for best selling books. I dont see why order would matter, so just hash(orderId) seems fine to me to evenly distribute.
5.  I'm not following the OCC usage here? You want to make sure you don't oversell a book. An easy way is via a transaction with locking in a ACID compliant DB. But, you'd already introduced Kafka which naturally serializes, so OCC (or any further contention mechanism) may even just be redundant

Show more

4

Reply

H

HomelessAquamarineTrout494

[• 28 days ago• edited 28 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdno4ons01zqad08467bxqxw)

I got this for my Databricks interview too, a prominent usecase here for inventory concurrency control that stood out to me is that the Seller can sell the book off this platform as well on their own website. Any suggestions for this usecase, Evan? Now, we need to manage inventories between our platform and sellers' shops.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdnobqqs01pkad08l7dxt2ke)

Yeah this is like ticketmaster syncing with venues. The core challenge is distributed inventory across multiple systems that each have their own source of truth.

For push vs pull: Push (webhooks) is more efficient but requires seller integration - they notify you of changes. Pull means polling their APIs which is simpler but wasteful. Reality is you'll need both since large sellers (Amazon) might support webhooks while small bookshops only have basic REST APIs.

The big issue is timing and race conditions. A book shows 5 copies in your cache, but seller just sold 3 on their site. By the time your customer clicks buy, it's gone. You need to decide: optimistic (show availability, handle failures) or pessimistic (real-time checks, slower UX). The most practical approach is reservation-based: when customer starts checkout, immediately reserve with seller API for 5-10 minutes. If seller is down, either fail fast or proceed with risk. TBH though, large marketplaces often just eat the cost of occasional oversells rather than building complex distributed transactions (had it happen to me plenty with ticketmaster grr)

Show more

1

Reply

H

HomelessAquamarineTrout494

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdo7tefb06e5ad08qe2lkag2)

Thanks for sharing these insights. I assume the reservation approach with fail-fast works best when the requirement is that the customers are waiting in checkout session with a loading icon and they need their bid response synchronously, right? What if the external Payment Processor is unavailable at that time - should we just return a Successful response to customer and later fail their order asynchronously if Payment doesn't go through (since Payments inherently is async in nature)?

Another thing I didn't understand in your previous question,

> Choice of Kafka partition could create a hot partition for best selling books. I dont see why order would matter, so just hash(orderId) seems fine to me to evenly distribute.

I see in the diagram, author has mentioned they want to ensure "user fairness" or FCFS (first come first serve) to ensure the limited quantity of books gets sold to whoever placed orders first - this seems to make sense for me, why do you think ordering doesn't matter?

Dealing with the hot partitions of particular bookName while not losing out the customer "order sequence" - can't this be solved by using a composite key like bookName+customerId as the Kafka partitionKey?

Show more

0

Reply

C

ConfidentOrangeGull774

[• 17 days ago• edited 17 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cme3qjy0t03acad07upbn5lzv)

> Dealing with the hot partitions of particular bookName while not losing out the customer "order sequence" - can't this be solved by using a composite key like bookName+customerId as the Kafka partitionKey?

Wouldn't this create too many partitions? If m users purchase n different books - it would be O(mn). In my opinion, we can rather identify hot partitions and append random number in book titles before pushing to kafka. But i would argue why we want to maintain fairness? If i own such book selling platform i would like to optimize for higher bid on competing offers.

Show more

1

Reply

D

DominantScarletBoa987

[• 28 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdnwdagl0275ad087t9ovczr)

What happens if a seller posts a book for sale AFTER a buy request which would match?

-   In that case, it would not impact any customers who are currently in the checkout process (i.e., those that have already marked an item in DynamoDB as pending).

I'm not following the OCC usage here? You want to make sure you don't oversell a book. An easy way is via a transaction with locking in a ACID compliant DB. But, you'd already introduced Kafka which naturally serializes, so OCC (or any further contention mechanism) may even just be redundant

-   Your right kafka will order it within the same partition anyway and with a hash(itemName), they'll all go to the same BookService node. However, if we chose hash(orderId), that would mean two book service nodes could process the same itemID in parallel and then we'll need OCC. Right?

Show more

0

Reply

H

HomelessAquamarineTrout494

[• 28 days ago• edited 28 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdo7wscs06enad088959rfwg)

1.  Minor comment on the Schema design: the ApiKey should be in Seller table not User for the SellerRegistration. Why would a User need an APIKey for placing an order?
2.  Curious if we considered a distributed lock (e.g. Redlock) in this case (presuming the customer is waiting for order status in checkout session)? Why not?

Show more

1

Reply

![Manasi](https://lh3.googleusercontent.com/a/ACg8ocI7UXL3f0aW-ZDAb0RPYCgcXfhirvXGD2ZEKHcFJ9kup84jcCo=s96-c)

Manasi

[• 21 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdxqipjn0cv8ad07etgwzove)

I have faced this question and answered correctly. Here's my 2 cents;

1.  The APIs in this section need to be divided into 3 sections, 1) Customer facing APIs 2) Seller facing APIs 3) Internal/Platform APIs
2.  Draw parallel from the payment system as well. Considering you are already taking credit card details, you need to reflect that intent somewhere, so perhaps BookRequest and Payment can be their own entities. That does make the amount of entities in this question larger, but it's the nature of the question.

Show more

1

Reply

![Manasi](https://lh3.googleusercontent.com/a/ACg8ocI7UXL3f0aW-ZDAb0RPYCgcXfhirvXGD2ZEKHcFJ9kup84jcCo=s96-c)

Manasi

[• 21 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmdxqx5780d06ad07ooiarhbs)

I made a version of this in excalidraw couple months ago when I was practicing. You can check it out here. I will submit to hello interview too when I get a chance. - https://drive.google.com/file/d/1e9bQRUNZ\_18oRVYRKXFDOHRm3MNdGDLQ/view?usp=sharing

Show more

3

Reply

G

GastricTomatoTakin400

[• 16 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cme569fl10f6qad08nud1u6ua)

The access is locked.

Show more

2

Reply

![Ankit Jain](https://lh3.googleusercontent.com/a/ACg8ocLs0gtOi6adPYLRI93VuNkgPDbd6pYARi5qTi-mHrc3wWY6Bw=s96-c)

Ankit Jain

[• 11 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmebsme6c01ghad08hjgt0l93)

@Manasi can you provide the view access to all as default. I send you access request separately too.

Show more

0

Reply

![Manasi](https://lh3.googleusercontent.com/a/ACg8ocI7UXL3f0aW-ZDAb0RPYCgcXfhirvXGD2ZEKHcFJ9kup84jcCo=s96-c)

Manasi

[• 11 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmebsw0d801e6ad072vt3psj1)

I updated access for everyone by default. Check now

Show more

0

Reply

![Ankit Jain](https://lh3.googleusercontent.com/a/ACg8ocLs0gtOi6adPYLRI93VuNkgPDbd6pYARi5qTi-mHrc3wWY6Bw=s96-c)

Ankit Jain

[• 11 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cmebsxr0601kmad08ghv2losu)

Yes got it, thanks for quick response.

Show more

0

Reply

S

sarthuaksharma

[• 14 days ago](https://www.hellointerview.com/community/submissions/cmdl5267802yxad08mffua6am#comment-cme6v7d7e02qaad08867g6syf)

One of my friends was asked the same question in Databricks

Show more

1

Reply