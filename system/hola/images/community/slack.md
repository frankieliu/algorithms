##### Design Slack

by SpecialMaroonSparrow268• Mid-level• 27 days ago

This was the OpenAI architecture interview question I was asked.

3

4

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdpledex0066ad08h723x9zs#comment-cmdples3o0003ad3ibyu76dck)

Great work on this design! You've covered the core functionality well and made some solid architectural choices. The use of sockets for real-time messaging and the overall service-oriented architecture provide a good foundation to build on.

1.  **Positives**
    
    -   Separating services (chat, user, org) enables independent scaling and development
    -   Using sockets is appropriate for the real-time messaging requirements
    -   Storing messages in a separate database from metadata supports scaling
    -   Considering both relational DB and DynamoDB shows thoughtful trade-off analysis
2.  **Potential Issues** **Lack of message queueing**
    
    -   Sending messages directly between services couples them tightly
    -   If the receiving service is down, messages will be lost
    -   Graceful handling of backpressure or failures is difficult
    
    **Unclear how offline messaging works**
    
    -   No mechanism described for persisting messages for offline users
    -   Risk of losing messages if a user is offline when a message is sent
    -   Doesn't fulfill requirement of delivering messages on reconnection
    
    **Insufficient detail on message storage**
    
    -   Schema for storing messages is not defined
    -   Unclear if one-to-one chats vs group chats are stored differently
    -   Difficult to assess if storage will meet access pattern needs
    
    **Missing rate limiting and abuse prevention**
    
    -   High-volume senders could overwhelm the system
    -   Risk of spam or abuse without rate limiting per user
    -   System vulnerable to DoS attacks that drain resources
3.  **Follow-up Questions**
    
    -   How are you handling replication and consistency for the databases?
    -   What's the strategy for distributing load across the chat service instances?
    -   How do you handle service discovery as instances scale up and down?
    -   What's the expected latency for message delivery?
    -   How are you securing the socket connections?
    -   What's the failover strategy if a chat service instance goes down?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply

S

SpecialMaroonSparrow268

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdpledex0066ad08h723x9zs#comment-cmdplh7lj007cad08agam4rbx)

We discussed WebSocket, database, and service scaling.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdpledex0066ad08h723x9zs#comment-cmdq5hgsg04fqad08qnfm3tzh)

Nice stuff. Were there any particular areas your OpenAI interviewer was really probing into? This is definitely one of their most popular questions.

Show more

0

Reply

S

SpecialMaroonSparrow268

[• 27 days ago](https://www.hellointerview.com/community/submissions/cmdpledex0066ad08h723x9zs#comment-cmdqdlwri078lad08wpsftfgm)

Most of the focus seemed to be on scaling this system (websocket scaling and database scaling). I think my approach to this problem was well-structured, but I recognize I was a bit vague on many aspects.

I think I did not go through enough details, but the interviewer, tbh, did not say much. I was driving the interview, and they simply suggested a few things here and there.

Show more

0

Reply