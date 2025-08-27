##### Design a Pastebin System

by Abhi• Senior• 1 month ago

Looking for critique and thoughts of my 30 mins pastebin system design interview at a random startup (I hadn't studied this particular design question before so just winged it): [https://excalidraw.com/#json=KGMIDdOUrbxCli8NK5KxI,Md\_SRWsmc\_2iVKzLXsDsgw](https://excalidraw.com/#json=KGMIDdOUrbxCli8NK5KxI,Md_SRWsmc_2iVKzLXsDsgw)

  Functional requirements from the interviewer: Store text: Allows users to input and save text. Share text: Enables users to share the stored text with others. The system should generate a unique URL for each piece of stored text, which can be shared with others for viewing.

Collapse Description

2

5

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 29 days ago](https://www.hellointerview.com/community/submissions/cmdj9jj6i0187ad08ief7yyrx#comment-cmdnniw8l01tbad084eigi2ux)

Nice stuff. Really fun to see! Some quick thoughts from me on this:

1.  The dual write problem when writing to cache and dynamo stands out. How to ensure consistency between the two? Since using DDB, the easy answer would be to just use DAX. Alternatively, rely on the cache as a cache-aside so you write only to DB, let cache populate on read miss.
2.  Why separate URL service? Just generate the short ID in the Pastes Service directly (nanoid, UUID, etc). The Redis Streams between them adds latency and complexity for no benefit - this isn't async work (unless I'm misunderstanding)
3.  Could consider a CDN for caching popular pastes.
4.  No mention of paste size limits, content validation, or private/public paste access control which often comes up in an interview like this.

I hope the interview went well!

Show more

1

Reply

A

Abhi

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdj9jj6i0187ad08ief7yyrx#comment-cmdpv6hul01ypad08urvygy98)

Thanks for the review Evan!

1.  Agreed, should have used cache-aside mode (I made a mistake specifying Redis in "write-through setting". Even DAX would be a bad answer probably? I'm assuming that DAX is suitable as a write-through/read-through cache? Let me know.
2.  So I realize that "minifying url" is misplaced and should've been written inside "Pastes Service box". The separate "Pastes URL service" has a "get(pasteId" arrow and I separated it as reads are much more frequent than writes.
3.  CDN is a good idea!
4.  Paste size limits told by interviewer was 2kB max when I tried introducing S3 Blob storage, so then we decided it could be stored directly inside DDB as per record size in DDB is 400kB. Thoughts?
5.  One important aspect I did want your feedback on is - I used DDB's native TTL feature to manage the paste's expiration time of 30 | 60 | 365 day - is this a satisfactory choice here?

Thanks, I do seem to have progressed to further rounds but yes this design can use some re-work.

Show more

0

Reply

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdj9jj6i0187ad08ief7yyrx#comment-cmdqn60wg000bad3ah6af1bf2)

Nice design! You've created a solid foundation for a Pastebin system with good separation of concerns between URL shortening and paste management services. Your use of Redis for caching frequently accessed pastes and DynamoDB with TTL for automatic expiration shows thoughtful consideration of the requirements. However, there are some critical issues around data consistency and storage architecture that would need to be addressed before this system could handle production load at the specified scale.

1.  **Positives**
    
    -   Smart use of Redis caching for hot pastes reduces database load and achieves the <500ms latency requirement for reads
    -   DynamoDB with built-in TTL elegantly handles the 365-day retention requirement without manual cleanup jobs
    -   Separating paste creation and URL services allows independent scaling of write-heavy vs read-heavy operations
2.  **Potential Issues** **Conflicting Storage Architecture**
    
    -   You show pastes stored in both DynamoDB (with User as partition key) and S3 (with paste content)
    -   This creates confusion about where paste data actually lives and how consistency is maintained
    -   The system would fail to retrieve pastes correctly without a clear primary storage location
    
    **DynamoDB Schema Bottleneck**
    
    -   Using User ID as partition key means all of a user's pastes hit the same partition
    -   Power users creating many pastes would create hot partitions exceeding DynamoDB's 3000 RCU/1000 WCU limits
    -   This violates the scalability requirement of 10k writes/sec as traffic concentrates on popular users
    
    **Missing URL Generation Logic**
    
    -   No explanation of how unique paste IDs/URLs are generated
    -   Without proper ID generation, you risk collisions or predictable URLs
    -   This could lead to security issues where users can guess other paste URLs
    
    **Undefined Message Broker Purpose**
    
    -   Redis Streams is shown but never used in any data flow
    -   Adding unnecessary components increases complexity and failure points
    -   This suggests incomplete thinking about asynchronous processing needs
3.  **Follow-up Questions**
    
    -   How does the Paste URL Service generate unique IDs to avoid collisions at 10k writes/second?
    -   What happens when a paste is requested - does it come from DynamoDB or S3?
    -   How do you handle cache invalidation when a user deletes a paste?
    -   Why use both DynamoDB and S3 for paste storage instead of picking one?
    -   What's the purpose of Redis Streams in your architecture?
    -   How would you shard the data if User ID partitioning creates hotspots?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply