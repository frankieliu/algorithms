##### Design Online Chess

by ShortAmethystPuma422• Mid-level• 15 days ago

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

[• 15 days ago](https://www.hellointerview.com/community/submissions/cme7btk2p00fnad08zi67y580#comment-cme7bug5e000pad3blwbplu2s)

Nice design! You've created a solid architecture for an online chess platform with good separation of concerns between game management, gameplay, and spectating. Your use of WebSockets for real-time moves and Redis Pub/Sub for broadcasting updates shows strong understanding of real-time system patterns. However, there are some critical consistency issues around concurrent move validation and undo operations that need addressing, and the leaderboard update mechanism could create performance bottlenecks.

1.  **Positives**
    
    -   Excellent use of WebSockets for real-time game moves and spectating, providing the low-latency bidirectional communication needed for chess gameplay
    -   Smart separation of Game Play Service from Spectate Service, allowing independent scaling based on the ratio of players to spectators
    -   Good choice of Redis Pub/Sub for broadcasting game state updates to multiple spectators efficiently
2.  **Potential Issues** **Race Condition in Move Validation**
    
    -   Multiple Game Play Service instances can read the same game state and validate moves concurrently
    -   This allows both players to move simultaneously, violating chess turn rules
    -   Without distributed locking or optimistic concurrency control, the strong consistency requirement is broken
    
    **Inconsistent Undo Operation**
    
    -   The undo operation has no mechanism to check if the opponent has already made their move
    -   Race condition exists between undo request and opponent's move being processed
    -   This violates the functional requirement that undo should only work if the other player hasn't played
    
    **Leaderboard Performance Bottleneck**
    
    -   Using CDC to update Redis sorted set for every game completion will create lag
    -   The <500ms real-time update requirement conflicts with CDC's eventual consistency model
    -   At 10K concurrent games, game completions could overwhelm the single leaderboard worker
3.  **Follow-up Questions**
    
    -   How does the system ensure only one player can move at a time when multiple Game Play Service instances handle the same game?
    -   What happens when a player tries to undo while their opponent is simultaneously making a move?
    -   How do you handle WebSocket reconnections to ensure players don't miss game state updates?
    -   How does the system validate complex chess rules like en passant, castling, or check/checkmate?
    -   What's the strategy for horizontally scaling the Game Play Service while maintaining game state consistency?
    -   How do you ensure the leaderboard updates within 500ms when using CDC from PostgreSQL?

_Warning: This feedback was AI generated and may contain errors._

Show more

3

Reply