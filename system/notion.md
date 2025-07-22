[

![Javarevisited Newsletter](https://substackcdn.com/image/fetch/$s_!WbL3!,w_80,h_80,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb9abffb-bd2d-4fe4-a166-4f88de0a3cba_512x512.png)



](https://javarevisited.substack.com/)

# [Javarevisited Newsletter](https://javarevisited.substack.com/)

SubscribeSign in

#### Share this post

[

![](https://substackcdn.com/image/fetch/$s_!Jid0!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93eb2532-5fb2-49c8-9ff8-b3d2aa84c320_1200x630.png)

![Javarevisited Newsletter](https://substackcdn.com/image/fetch/$s_!WbL3!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb9abffb-bd2d-4fe4-a166-4f88de0a3cba_512x512.png)

Javarevisited Newsletter

How Notion Handles 200+ BILLION Notes (Without Crashing)









](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes#)

Copy link

Facebook

Email

Notes

More

# How Notion Handles 200+ BILLION Notes (Without Crashing)

### How Notion uses horizontal scalability and sharding to handle 200+ billion notes without crashing.

[

![javinpaul's avatar](https://substackcdn.com/image/fetch/$s_!9bIo!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5663d1cb-2e66-4a0d-8f76-8a3aad3f2382_48x48.png)



](https://substack.com/@javinpaul)

[

![Konstantin Borimechkov's avatar](https://substackcdn.com/image/fetch/$s_!BeGt!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F532e6c77-82f4-44ff-b207-c5f7ce5acb27_620x620.jpeg)



](https://substack.com/@konstantinmb)

[javinpaul](https://substack.com/@javinpaul)

and

[Konstantin Borimechkov](https://substack.com/@konstantinmb)

Jul 18, 2025

31

#### Share this post

[

![](https://substackcdn.com/image/fetch/$s_!Jid0!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93eb2532-5fb2-49c8-9ff8-b3d2aa84c320_1200x630.png)

![Javarevisited Newsletter](https://substackcdn.com/image/fetch/$s_!WbL3!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb9abffb-bd2d-4fe4-a166-4f88de0a3cba_512x512.png)

Javarevisited Newsletter

How Notion Handles 200+ BILLION Notes (Without Crashing)









](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes#)

Copy link

Facebook

Email

Notes

More

[

3

](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes/comments)

7

[

Share

](javascript:void\(0\))

Hello guys, if you are preparing for system design interview or just want to level up your Software architecture skills then there is no better way to learn than reading case studies and how others are doing.

Whenever we talk about scale we talk about Google scale or Amazon scale but today I am going to share another great story of scaling software which we use every day, Yes, I am talking about **Notion**, one of the fastest growing software and app.

Javarevisited Newsletter is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.

Subscribe

Their story of scaling to 200 BILLION notes without crashing has many great insignts and learning about how system design fundamentals like [sharding](https://javarevisited.substack.com/p/system-design-basics-database-replication) and [horizontal scalability](https://javarevisited.substack.com/p/horizontal-vs-vertical-scalability) can do wonders.

Earlier I have talked about common system design concepts like [Rate Limiter](https://javarevisited.substack.com/p/what-is-rate-limiter-how-does-it), [Database Scaling](https://javarevisited.substack.com/p/system-design-basics-database-scaling), [API Gateway vs Load Balancer](https://javarevisited.substack.com/p/difference-between-api-gateway-and) and [Forward proxy vs reverse proxy](https://javarevisited.substack.com/p/system-design-basics-reverse-proxy) as well common [System Design problems](https://javarevisited.substack.com/p/8-system-design-problems-for-interview) and concepts like [Single Point Failure](https://javarevisited.substack.com/p/what-is-single-point-of-failure-spof), and in this article we will talk about how Notion used horizontal scalability and sharding to handle 200 billion notes without crashing.

For this article, I have teamed up with [Konstantin Borimechkov](https://open.substack.com/users/145622751-konstantin-borimechkov?utm_source=mentions), a passionate Software Engineer and who has contributed some of the pieces you all have loved like his piece on **[ACID and transaction](https://javarevisited.substack.com/p/system-design-basics-acid-and-transactions)**

With that, I hand over to him to take you through the rest of the article.

> By the way, if you are preparing for System design interviews and want to learn System Design in a limited time then you can also checkout sites like **[Codemia.io](https://codemia.io/?via=javarevisited)**, **[ByteByteGo](https://bit.ly/3P3eqMN)**, **[Design Guru](https://bit.ly/3pMiO8g)**, **[Exponent](https://bit.ly/3cNF0vw)**, **[Educative](https://bit.ly/3Mnh6UR)** and **[Udemy](https://bit.ly/3vFNPid)** which have many great System design courses. These will help you prepare better.

---

While searching for my next YouTube video to watch while studying system design, a video **of how Notion scaled their system popped up**

and as a Notion fan, I couldn’t just skip it.

[

![](https://substackcdn.com/image/fetch/$s_!Jid0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93eb2532-5fb2-49c8-9ff8-b3d2aa84c320_1200x630.png)



](https://substackcdn.com/image/fetch/$s_!Jid0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93eb2532-5fb2-49c8-9ff8-b3d2aa84c320_1200x630.png)

if you haven’t used Notion, that’s how a general page looks like in a team

safe to say, it was worth the watch (link at the end) and inspired me to write-up this hopefully valuable article 💪

that said, let’s start with conceptualizing [sharding](https://javarevisited.substack.com/p/the-complete-guide-of-database-sharding) and then moving to the Notion’s story 👇

..

i.e. how they went from a single server to handling hundreds of billions of content blocks, while preserving excellent UX 🔥

> > note: no prior database wizardry required. I’ve included a glossary, going over each more complicated term used 🤘

---

## **so, what’s sharding and why does it matter?**

_**Sharding means breaking a large dataset into many smaller partitions (shards) and spreading them across multiple machines (nodes/servers)**_

- **parallel reads/writes** - each shard has its own CPU, memory, and disk, so the cluster’s total throughput grows almost linearly with node count
    
- **reduced hotspots** (_if done correctly_) - queries touch only a slice of the table, keeping index pages hot in cache and vacuums shorter
    
- **fault isolation** - if one node fails, only its shard is affected; the rest of the application stays online
    
- **operational elasticity** — need more capacity? Add nodes and move shards - no “big‑bang” vertical upgrade
    

> > **tl;dr:** Sharding turns a single‑server ceiling into a horizontal beast for scale - at the price of extra routing logic and cross‑shard constraints

---

with the that out of the way let’s focus on Notion’s story. I am covering more sharding concepts in the storytelling itself 🤫

---

## **The Early Pain at Notion**

**2016 — a single Postgres Primary + Read Replica**

that was great for beta, but terrible for growth..

**but, what were the symptoms?**

✔️ full‑table vacuums paused writes

✔️ index scans slowed once the `blocks` table exceeded 50 M rows

✔️ large workspace imports queued millions of inserts in hours (_if you’ve used Notion and imported some document or a big chunk of text, that’s spread across multiple **blocks** and inserted independently_**)**

👉 the takeaway: one hot B‑tree and vacuum cycle became the pain.. quick

---

#### **but why** `workspace_id` **became the shard key ??**

> > **what even is a shard key??** that’s the field based on which you separate a single DB into many different shards. It depends on the business case, but as an example:
> 
> > you have a table holding _**users**_ information. you wanna shard it into multiple DBs across multiple data centers around the globe to have the user info closer geographically to each one of them.
> > 
> > So you pick the _**users.’country\_of\_origin’**_ as shard key. The result is, you have one shard (DB node) in Bulgaria with Bulgarians and another in Germany with Germans 😅

[

![](https://substackcdn.com/image/fetch/$s_!jbXa!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd3f2d7ee-41e1-46a2-a67c-ce7ad0116472_626x421.png)



](https://substackcdn.com/image/fetch/$s_!jbXa!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd3f2d7ee-41e1-46a2-a67c-ce7ad0116472_626x421.png)

random pic i found to visualize a shard baed on some key like the user\_id 👆

on to the question:

1. that’s what fit their business rules (i.e. one company having one workspace means no cross-tenant joins).
    
2. They can spread the data across shards evenly (small & large teams balancing between hash buckets)
    
3. workspaces are immutable. Users can’t merge workspaces, so the keys (_workspace\_id_) never changes
    

**🔑 a key point to take out from this is:** stable key ⇒ router logic stays simple and re‑shards are predictable

---

## **Evolution Timeline (Numbers Approximate)**

did my best in illustrating the evolution of Notion’s engineering progress throughout the years 👇

[

![](https://substackcdn.com/image/fetch/$s_!qYJB!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F76e39a49-2215-40a7-9675-6c9ccea0e582_2844x5462.png)



](https://substackcdn.com/image/fetch/$s_!qYJB!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F76e39a49-2215-40a7-9675-6c9ccea0e582_2844x5462.png)

i mean, come on.. don’t I deserve a sub or like for this diagram? 😆 pretty proud of it, ngl

### **briefly going through each year**

#### **2018**

- partition tables **inside** one DB by `workspace_id`
    
- **what was the impact?** Vacuums run per‑partition now; p95 WRITE latency down  40 %
    

#### **2020**

- this year Notion ran **32 physical Postgres servers**, each one carrying **15 logical shards**, for a total of 480 logical shards across the fleet
    
- **what was the impact?** multi-year head-room; lower p-latencies & faster pages; balanced resource util; fault isolation..
    

#### **2022**

- **the Great Re‑Shard** happen (i.e. live split of hottest partitions) **more in this YouTube vid:**
    

- **what was the impact?** CPU & I/O head-room restored;
    

#### **2023**

- Notion offloads the Big Analytics query to a separate data lake, developed using open-source software (Apache Hudi on S3)
    
- **what was the impact?** the heavy analytical queries didn’t bother the business/transaction ones
    

#### **2024**

- **hitting 200 Billion Blocks** & implementing in-house weekly auto‑rebalance
    
- **what was the impact?** shard skew < 1.4× - keeps query plans stable
    

---

## **Lessons You Can Apply at 1/1000‑th Scale**

1. **be brave when picking a shard key** - changing it later hurts
    
2. **separate analytics** sooner; cold queries age the primary. Better for analytics as data folks can use more tools, better for the business logic and servers as they don’t get locked by big analytics queries
    
3. **implement robust monitoring of shards & more specifically - shard skew.** Add alerts when max/min size >  2X
    
4. **abstract connections** behind a router library; apps call `getConn(id)`
    
5. **automate re‑sharding** (cron + pg\_dump beats manual midnight ops)
    

---

## **A lil glossary in case you don’t know some used terms 🤝**

- **shard skew -** the ratio between the busiest shard (by disk or QPS) and the quietest
    
- **router -** tiny service/library that looks at a key and points to the right database
    
- **hot shard** - one slice is busier than the rest, causing CPU/disk spikes
    
- **re-shard** - split or move data to make shards even again, live and online
    
- **consistent hash** - hashing trick so adding a new DB moves only a few keys, not everything
    

---

## **Real‑World Sharding Examples Beyond Notion**

- **[Instagram Feeds](https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c)** [- hash on](https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c) `user_id` [to scatter photos; celebrity feeds get special shards](https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c):
    

[

![](https://substackcdn.com/image/fetch/$s_!NrBT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3aa2c9a8-d27b-4948-9144-ad3656dc1797_872x450.png)



](https://substackcdn.com/image/fetch/$s_!NrBT!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3aa2c9a8-d27b-4948-9144-ad3656dc1797_872x450.png)

- **[Shopify Stores](https://shopify.engineering/mysql-database-shard-balancing-terabyte-scale)** [- each merchant is a shard in MySQL. A re‑shard tool “Kiseki” live‑moves busy stores](https://shopify.engineering/mysql-database-shard-balancing-terabyte-scale)
    
- **[Stripe Ledger](https://stripe.com/blog/ledger-stripe-system-for-tracking-and-validating-money-movement)** [-](https://stripe.com/blog/ledger-stripe-system-for-tracking-and-validating-money-movement) `customer_id` [shard key across Postgres ranges for strict balance consistency](https://stripe.com/blog/ledger-stripe-system-for-tracking-and-validating-money-movement)
    

they use mostly the same playbook, just on different scale and based on different business needs & spec!

---

Hope you found this valuable! The video and going deeper with the blog-article really helped me to master the theory behind sharding!

Let’s crush it this week!

oh.. here’s the link to [the YouTube vid](https://www.youtube.com/watch?v=NwZ26lxl8wU&ab_channel=CodingwithLewis) as promised at the start as well 🤘

Drop a ❤️ to help me spread the knowledge & to let me know you’d like more of this!

_Enjoyed this? Consider subscribing to **[The Excited Engineer](https://theexcitedengineer.substack.com/?r=a1ck9)** for weekly deep dives just like this 👇_

[![](https://substackcdn.com/image/fetch/$s_!XWBP!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F61ac2deb-a882-4496-b4c6-9d0cbcc2c554_500x500.png)The Excited Engineer

Learn something new and exciting every Sunday! Digest hard software engineering concepts with ease and joy!

By Konstantin Borimechkov

](https://theexcitedengineer.substack.com/?utm_source=substack&utm_campaign=publication_embed&utm_medium=web)

Other **System Design and AI articles** you may like

[](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering)

[

![The Complete AI and LLM Engineering Roadmap: From Beginner to Expert](https://substackcdn.com/image/fetch/$s_!UE20!,w_140,h_140,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe8daef35-b630-4f56-90d0-7b27ed261823_1200x1200.png)

](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering)

[

#### The Complete AI and LLM Engineering Roadmap: From Beginner to Expert

](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering)

[](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering)

[](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering)[javinpaul](https://substack.com/profile/16859097-javinpaul) and [Paul Iusztin](https://substack.com/profile/110559689-paul-iusztin)

·

Jun 19

[

Read full story

](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering)

[](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for)

[

![11 Must-Read AI and LLM Engineering Books for Developers in 2025](https://substackcdn.com/image/fetch/$s_!DjCG!,w_140,h_140,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa6f62868-22d1-418a-8389-35d7532172da_640x480.webp)

](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for)

[

#### 11 Must-Read AI and LLM Engineering Books for Developers in 2025

](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for)

[](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for)

[](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for)[javinpaul](https://substack.com/profile/16859097-javinpaul)

·

Jun 7

[

Read full story

](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for)

[](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind)

[

![Scaling to Millions: The Secret Behind NGINX's Concurrent Connection Handling](https://substackcdn.com/image/fetch/$s_!-g1Z!,w_140,h_140,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F377ca779-5d63-4995-9541-b37008700a8c_800x1073.jpeg)

](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind)

[

#### Scaling to Millions: The Secret Behind NGINX's Concurrent Connection Handling

](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind)

[](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind)

[](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind)[javinpaul](https://substack.com/profile/16859097-javinpaul) and [Animesh Gaitonde](https://substack.com/profile/14776687-animesh-gaitonde)

·

Apr 5

[

Read full story

](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind)

[](https://javarevisited.substack.com/p/system-design-interview-question)

[

![System Design Interview Question: Design URL Shortener](https://substackcdn.com/image/fetch/$s_!YjYZ!,w_140,h_140,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F483ada1b-1ce2-4d93-826e-d21e283bb12e_800x450.png)

](https://javarevisited.substack.com/p/system-design-interview-question)

[

#### System Design Interview Question: Design URL Shortener

](https://javarevisited.substack.com/p/system-design-interview-question)

[](https://javarevisited.substack.com/p/system-design-interview-question)

[](https://javarevisited.substack.com/p/system-design-interview-question)[javinpaul](https://substack.com/profile/16859097-javinpaul) and [Hayk](https://substack.com/profile/145622767-hayk)

·

Jul 5

[

Read full story

](https://javarevisited.substack.com/p/system-design-interview-question)

Javarevisited Newsletter is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.

Subscribe

[

![Ayman Issa's avatar](https://substackcdn.com/image/fetch/$s_!x9V0!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc1d226c4-f30a-4a29-af1d-6e02b9896052_96x96.jpeg)



](https://substack.com/profile/14128052-ayman-issa)

[

![Shubhanshu upadhyay's avatar](https://substackcdn.com/image/fetch/$s_!h2Lw!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0b3fede4-3a88-4b8a-914a-182cc262260c_96x96.jpeg)



](https://substack.com/profile/344998035-shubhanshu-upadhyay)

[

![KISHORE D's avatar](https://substackcdn.com/image/fetch/$s_!z5H_!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0c7dbbf-0442-44b8-ba2d-dce0c00a7f31_96x96.png)



](https://substack.com/profile/325781072-kishore-d)

[

![javinpaul's avatar](https://substackcdn.com/image/fetch/$s_!9bIo!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F5663d1cb-2e66-4a0d-8f76-8a3aad3f2382_48x48.png)



](https://substack.com/profile/16859097-javinpaul)

[

![Jordan Newman's avatar](https://substackcdn.com/image/fetch/$s_!Bf3B!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fa4dc0a64-f631-4fe7-a979-64b0f519a8fd_1166x1168.jpeg)



](https://substack.com/profile/89866471-jordan-newman)

31 Likes∙

[7 Restacks](https://substack.com/note/p-168686689/restacks?utm_source=substack&utm_content=facepile-restacks)

31

#### Share this post

[

![](https://substackcdn.com/image/fetch/$s_!Jid0!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93eb2532-5fb2-49c8-9ff8-b3d2aa84c320_1200x630.png)

![Javarevisited Newsletter](https://substackcdn.com/image/fetch/$s_!WbL3!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb9abffb-bd2d-4fe4-a166-4f88de0a3cba_512x512.png)

Javarevisited Newsletter

How Notion Handles 200+ BILLION Notes (Without Crashing)









](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes#)

Copy link

Facebook

Email

Notes

More

[

3

](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes/comments)

7

[

Share

](javascript:void\(0\))

PreviousNext

<table cellpadding="0" cellspacing="0" class="post-contributor-bio-table"><tbody><tr class="post-contributor-bio-table-row"><td class="post-contributor-bio-userhead-cell post-contributor-bio-userhead-cell-web"><div class="profile-hover-card-target profileHoverCardTarget-PBxvGm"><div class="user-head"><a href="https://substack.com/profile/145622751-konstantin-borimechkov"><div class="profile-img-wrap"><picture><source type="image/webp" srcset="https://substackcdn.com/image/fetch/$s_!BeGt!,w_104,h_104,c_fill,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F532e6c77-82f4-44ff-b207-c5f7ce5acb27_620x620.jpeg"><img src="https://substackcdn.com/image/fetch/$s_!BeGt!,w_104,h_104,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F532e6c77-82f4-44ff-b207-c5f7ce5acb27_620x620.jpeg" sizes="100vw" alt="" width="104" height="104" class="img-OACg1c pencraft pc-reset"></picture></div></a></div></div></td><td class="post-contributor-bio-body-cell post-contributor-bio-body-cell-web"><div class="post-contributor-bio-body"><div class="post-contributor-bio-preamble">A guest post by</div><table cellpadding="0" cellspacing="0" class="post-contributor-bio-body-table"><tbody><tr class="post-contributor-bio-body-table-row"><td class="post-contributor-bio-copy-cell"><div class="pencraft pc-display-flex pc-gap-4 pc-paddingBottom-4 pc-alignItems-center pc-reset"><div class="profile-hover-card-target profileHoverCardTarget-PBxvGm"><a href="https://substack.com/@konstantinmb?utm_campaign=guest_post_bio&amp;utm_medium=web" native="true" class="post-contributor-bio-title no-margin">Konstantin Borimechkov</a></div></div><div class="post-contributor-bio-text">🚀 Passionate Software Engineer | Tech Blogger &amp; Enthusiast</div></td><td class="post-contributor-bio-controls-cell post-contributor-bio-controls-cell-web"><div class="post-contributor-bio-controls"><a href="https://theexcitedengineer.substack.com/subscribe?" native="true" class="post-contributor-bio-subscribe-button button primary"><span class="post-contributor-bio-subscribe-button-label">Subscribe to Konstantin</span></a></div></td></tr></tbody></table></div></td></tr></tbody></table>

#### Discussion about this post

CommentsRestacks

![User's avatar](https://substackcdn.com/image/fetch/$s_!TnFC!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fdefault-light.png)

[

![Subramanya Chakravarthy's avatar](https://substackcdn.com/image/fetch/$s_!CKr0!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F912e32d6-003b-4d3d-b2b2-d5de258c92f3_800x800.jpeg)



](https://substack.com/profile/8413616-subramanya-chakravarthy?utm_source=comment)

[Subramanya Chakravarthy](https://substack.com/profile/8413616-subramanya-chakravarthy?utm_source=substack-feed-item)

[2d](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes/comment/137051613 "Jul 20, 2025, 12:40 AM")

Liked by Konstantin Borimechkov

What tool did you use for diagrams??

Expand full comment

[

Like (1)



](javascript:void\(0\))

Reply

Share

[1 reply](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes/comment/137051613)

[

![Konstantin Borimechkov's avatar](https://substackcdn.com/image/fetch/$s_!BeGt!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F532e6c77-82f4-44ff-b207-c5f7ce5acb27_620x620.jpeg)



](https://substack.com/profile/145622751-konstantin-borimechkov?utm_source=comment)

[Konstantin Borimechkov](https://substack.com/profile/145622751-konstantin-borimechkov?utm_source=substack-feed-item)

[3d](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes/comment/136750175 "Jul 18, 2025, 9:40 PM")

Author

That was an interesting one to do research on and write! Thanks for sharing through the Java Revisited newsletter javinpaul 💪🔥

Expand full comment

[

Like



](javascript:void\(0\))

Reply

Share

[1 more comment...](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes/comments)

TopLatestDiscussions

[The Complete AI and LLM Engineering Roadmap: From Beginner to Expert](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering)

[From Raw Data to LLM Fine-Tuning, RAG and LLMOps](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering)

Jun 19 • 

[javinpaul](https://substack.com/@javinpaul)

 and 

[Paul Iusztin](https://substack.com/@pauliusztin)

204

#### Share this post

[

![](https://substackcdn.com/image/fetch/$s_!UE20!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe8daef35-b630-4f56-90d0-7b27ed261823_1200x1200.png)

![Javarevisited Newsletter](https://substackcdn.com/image/fetch/$s_!WbL3!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb9abffb-bd2d-4fe4-a166-4f88de0a3cba_512x512.png)

Javarevisited Newsletter

The Complete AI and LLM Engineering Roadmap: From Beginner to Expert









](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes#)

Copy link

Facebook

Email

Notes

More

[

8

](https://javarevisited.substack.com/p/the-complete-ai-and-llm-engineering/comments)

[](javascript:void\(0\))

![](https://substackcdn.com/image/fetch/$s_!UE20!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe8daef35-b630-4f56-90d0-7b27ed261823_1200x1200.png)

[11 Must-Read AI and LLM Engineering Books for Developers in 2025](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for)

[Software Engineer to AI Engineer - 11 Books to Learn AI and LLM Engineering](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for)

Jun 7 • 

[javinpaul](https://substack.com/@javinpaul)

176

#### Share this post

[

![](https://substackcdn.com/image/fetch/$s_!DjCG!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa6f62868-22d1-418a-8389-35d7532172da_640x480.webp)

![Javarevisited Newsletter](https://substackcdn.com/image/fetch/$s_!WbL3!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb9abffb-bd2d-4fe4-a166-4f88de0a3cba_512x512.png)

Javarevisited Newsletter

11 Must-Read AI and LLM Engineering Books for Developers in 2025









](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes#)

Copy link

Facebook

Email

Notes

More

[

3

](https://javarevisited.substack.com/p/11-ai-and-llm-engineering-books-for/comments)

[](javascript:void\(0\))

![](https://substackcdn.com/image/fetch/$s_!DjCG!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa6f62868-22d1-418a-8389-35d7532172da_640x480.webp)

[Scaling to Millions: The Secret Behind NGINX's Concurrent Connection Handling](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind)

[The Magic of NGINX: Event-Driven Architecture and Million-Scale Connections](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind)

Apr 5 • 

[javinpaul](https://substack.com/@javinpaul)

 and 

[Animesh Gaitonde](https://substack.com/@engineeringatscale)

76

#### Share this post

[

![](https://substackcdn.com/image/fetch/$s_!-g1Z!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F377ca779-5d63-4995-9541-b37008700a8c_800x1073.jpeg)

![Javarevisited Newsletter](https://substackcdn.com/image/fetch/$s_!WbL3!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb9abffb-bd2d-4fe4-a166-4f88de0a3cba_512x512.png)

Javarevisited Newsletter

Scaling to Millions: The Secret Behind NGINX's Concurrent Connection Handling









](https://javarevisited.substack.com/p/how-notion-handles-200-billion-notes#)

Copy link

Facebook

Email

Notes

More

[

1

](https://javarevisited.substack.com/p/scaling-to-millions-the-secret-behind/comments)

[](javascript:void\(0\))

![](https://substackcdn.com/image/fetch/$s_!-g1Z!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F377ca779-5d63-4995-9541-b37008700a8c_800x1073.jpeg)

See all

Ready for more?

Subscribe

© 2025 javinpaul

[Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start writing](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer)[Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com/) is the home for great culture