##### Design a Distributed File System

by ControlledSapphireCapybara717• Senior• 19 days ago

1

6

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 19 days ago](https://www.hellointerview.com/community/submissions/cme1j7dln09u3ad089dyy0sxq#comment-cme1j7qjn004cad3bkvyxw3od)

Nice design! You've created a solid foundation for a distributed file system with good separation of concerns between metadata and block storage services. Your approach to handling data distribution and fault tolerance through consistent hashing and health monitoring shows thoughtful consideration of key distributed systems challenges.

1.  **Positives**
    
    -   Separating metadata service from block storage is a proven architectural pattern that allows independent scaling of metadata operations and data storage
    -   Using consistent hashing for DataNode distribution minimizes data movement when nodes fail or are added, which is crucial for petabyte-scale systems
    -   Supporting parallel block downloads from multiple DataNodes enables high throughput for large file transfers and better utilizes available network bandwidth
2.  **Potential Issues** **Single Database Bottleneck**
    
    -   You're using a single database for both file metadata and block mappings without any sharding or replication strategy
    -   This creates a critical single point of failure and performance bottleneck - a single database instance can only handle ~10k TPS writes
    -   For a petabyte-scale system with millions of files and billions of blocks, this violates both the scalability and fault tolerance requirements
    
    **Missing Replication Strategy**
    
    -   The design doesn't specify how blocks are replicated across DataNodes for durability
    -   Without replication, a single DataNode failure would result in permanent data loss for all blocks stored on that node
    -   This violates the durability requirement and makes the system unsuitable for production use
    
    **Undefined Consistency Model**
    
    -   There's no clear consistency mechanism between metadata updates and actual block storage operations
    -   If a client uploads blocks to DataNodes but the metadata update fails, you'll have orphaned blocks; if metadata updates succeed but block uploads fail, you'll have corrupted files
    -   This can lead to data corruption and violates basic file system integrity requirements
3.  **Follow-up Questions**
    
    -   How do you ensure the 200ms latency requirement for list file operations when the metadata database grows to billions of entries?
    -   What happens when a DataNode fails during an active file upload - how do you handle partial writes and ensure atomicity?
    -   How do you handle the split-brain scenario where the block manager loses connectivity to some DataNodes but not others?
    -   What's your strategy for garbage collection of orphaned blocks when file deletions or upload failures occur?
    -   How do you prevent hot spots when multiple clients are uploading/downloading the same popular files?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

T

TightAquamarinePlatypus676

[• 18 days ago](https://www.hellointerview.com/community/submissions/cme1j7dln09u3ad089dyy0sxq#comment-cme35mjtw003gad08t07xhoda)

I was asked about file distribution resource estimation. like the bandwidth and how soon the files will reach other servers

Show more

0

Reply

S

SocialBlackFly760

[• 14 days ago](https://www.hellointerview.com/community/submissions/cme1j7dln09u3ad089dyy0sxq#comment-cme7wgrjj03p2ad084rplu1u4)

would use of dynamodb be a problem? based on description, distributed storage system should be assumed not available in the design.

Show more

0

Reply

A

AdorableGrayHawk775

[• 7 days ago](https://www.hellointerview.com/community/submissions/cme1j7dln09u3ad089dyy0sxq#comment-cmein6ne200ryad085np6js9c)

Should be fine as the question says existing databases are allowed. The point of the question isn't to re-implement a database anyways. This is assuming you're using dynamodb to store the metadata.

What's more interesting would be the discussion of if the system is strongly consistent or not. What semantics do you want to offer? Should a user be able to immediately get/list a file after creation, or is it eventually consistent? "Read after write"

Some file systems aim for strong consistency as it's easier to build on top of, developers don't have to reason about and account for eventual consistency.

Show more

0

Reply

S

sarthuaksharma

[• 3 days ago](https://www.hellointerview.com/community/submissions/cme1j7dln09u3ad089dyy0sxq#comment-cmeo5uwgc00v9ad090mnmzqu1)

What about directories and subdirectories? How are you storing hierarchical structures?

Show more

0

Reply