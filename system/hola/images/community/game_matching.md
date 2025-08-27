##### Design Game Matching System

by AmusedFuchsiaAsp382• Senior• 16 days ago

1M DAU, many games. After the user chooses to join a game, they enter a waiting room. The system arranges matches with other users of similar skill levels; once a match is successfully formed, the game starts immediately. The user’s skill can be obtained through an API, with a score range of 0–100.

We need to define some constraints ourselves, such as:

User wait time should not be too long.

The required number of players to start a game may be dynamic (e.g., reduced minimum players when there are fewer active users).

Previously we also mentioned:

The matchmaking system needs to handle a large number of games in parallel.

It must be able to scale to match thousands of games.

A message queue capable of handling bursts of data should be added.

We need to consider region, with server clusters in different regions.

We also need to discuss how the server connects to the client, and this feedback loop must be addressed.

Collapse Description

2

1

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 16 days ago](https://www.hellointerview.com/community/submissions/cme5bueg50hd7ad08ydtgjo73#comment-cme5bv89w0013ad3bhcgs1oka)

Nice design! You've created a solid foundation for a game matching system with good separation of concerns between player evaluation, skill tracking, and game allocation. Your use of Redis sorted sets for queue management and the consumer-based matching approach shows thoughtful consideration of the real-time matching requirements. However, there are some critical issues around consistency guarantees and scalability that need addressing to ensure players aren't double-matched and the system can handle your stated scale requirements.

1.  **Positives**
    
    -   Using Redis sorted sets for the matching queue is an excellent choice - it naturally supports skill-based ordering and efficient range queries for finding similar-skilled players
    -   Separating skill service from player service provides good modularity and allows independent scaling of skill calculations
    -   The consumer-based matching with configurable intervals (100-500ms) gives good control over matching speed vs quality trade-offs
2.  **Potential Issues** **Missing Distributed Locking for Match Assignment**
    
    -   Multiple consumer workers can pull the same players from Redis and attempt to match them simultaneously
    -   This will cause players to be assigned to multiple games, violating your strong consistency requirement
    -   Without coordination between workers, the system cannot guarantee a player is only in one game
    
    **Single Redis Instance Bottleneck**
    
    -   All 1M DAU players joining queues will hit a single Redis instance for queue operations
    -   At peak times with 100k+ concurrent players, this creates a single point of failure and performance bottleneck
    -   Redis can handle high throughput but needs clustering/sharding for this scale and availability
    
    **Unclear Player Notification Mechanism**
    
    -   The design mentions "WS notification" but doesn't show how players receive match notifications
    -   Without persistent connections or a proper notification service, players won't know when matches are found
    -   This breaks the core requirement of notifying users when matching completes
3.  **Follow-up Questions**
    
    -   How do you prevent a player from being matched by two different consumer workers processing the queue simultaneously?
    -   What happens when the matching service needs to scale beyond what a single Redis instance can handle?
    -   How do you ensure players don't wait forever - what's your strategy for relaxing skill requirements over time?
    -   How does the WebRTC component connect to the matching flow for establishing game connections?
    -   What's your strategy for handling region-based matching when you mention both "nearest/selected region"?
    -   How do you handle partial failures when some players accept a match but others disconnect?

_Warning: This feedback was AI generated and may contain errors._

Show more

2

Reply