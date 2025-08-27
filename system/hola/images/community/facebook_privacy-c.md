##### Design Facebook Privacy

by Ankit Thakur• Staff• 4 days ago

1

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

[• 4 days ago](https://www.hellointerview.com/community/submissions/cmemg8t1u03izad081t0nu2dj#comment-cmemg9j3g001bad3a4gdn0o7o)

Good job on defining clear functional requirements and API endpoints for the privacy system! However, this submission is fundamentally flawed as it lacks any actual system architecture - the high-level design section is completely empty with no components or connections specified. Without understanding how you plan to store privacy data, perform authorization checks, handle the massive scale of 100M concurrent viewers, or ensure consistency during privacy updates, it's impossible to evaluate whether this design would work in production. The API design shows good thinking about the operations needed, but a system design interview requires architectural decisions about databases, caching, service topology, and how components interact to meet the stringent latency and scale requirements.

1.  **Positives**
    
    -   Clear API design with appropriate REST endpoints for creating posts with privacy settings, viewing posts with authorization checks, and updating privacy settings
    -   Good identification of core entities (User, Post, PrivacySetting) that capture the essential data model for a privacy system
    -   Well-defined non-functional requirements including specific latency targets (<100ms) and scale requirements (1bn users, 100M concurrent viewers) that would guide architectural decisions
2.  **Potential Issues** **No System Architecture**
    
    -   The high-level design section is completely empty with no components, databases, caches, or services defined
    -   Without any architecture, there's no way to handle 100M concurrent viewers checking privacy permissions
    -   This violates the basic requirement of a system design interview which is to design an actual system
    
    **Missing Privacy Check Implementation**
    
    -   No explanation of how privacy checks would be performed when 100M concurrent users try to view posts
    -   Without a caching strategy or efficient data storage, checking permissions for every view request would be impossible at scale
    -   The <100ms latency requirement cannot be met without a concrete implementation strategy
    
    **No Consistency Strategy**
    
    -   The requirement states need for consistency to prevent unauthorized access during privacy changes, but no mechanism is proposed
    -   Without distributed locking, caching invalidation, or consistency protocols, users could access content after privacy settings change
    -   This creates a critical security vulnerability where private content could be exposed
3.  **Follow-up Questions**
    
    -   How would you store and retrieve privacy settings to handle 100M concurrent permission checks per second?
    -   What happens when a user changes a post from public to private - how do you ensure no unauthorized viewers can still access it?
    -   How do you handle the friend list lookups needed for "private" posts when checking if a viewer is authorized?
    -   Where would you implement the actual privacy check logic - at the API gateway, in a separate service, or at the data layer?
    -   How would you cache privacy decisions while still maintaining consistency when settings change?
    -   What database would you use to store billions of posts with their associated privacy settings?

_Note: This evaluation focuses on technical design quality, not presentation. A complete high-level architecture is essential for system design interviews._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply