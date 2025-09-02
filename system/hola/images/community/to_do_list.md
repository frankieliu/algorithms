##### Design A To-Do List App With Multi-User Collaboration

by shivadeepthi toopran• Senior• 18 days ago

The system should handle concurrent editing. Allow users to share and collaboratively edit tasks. The application should allow a single user or team of users to keep track of a list of tasks to do in the future. Example: John creates a list named "Grocery shopping list." John adds task "Buy Tomatoes" to the Grocery shopping list. John adds task "Buy Onions" to the Grocery shopping list. John completes the task "Buy Onions" from the Grocery shopping list. John now has 1 TODO list with 2 tasks remaining. Design a checklist application that allows multiple people to share and collaborate. It needs to support: adding a list, adding items, deleting items, marking items as completed, and sharing the list. The assessment focuses on details like data model, API design, protocol, and scalability, even including some SQL queries.

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

[• 18 days ago](https://www.hellointerview.com/community/submissions/cme2hc3at056lad08wb453du0#comment-cme2hck4r001kad3bne3gao5q)

Nice design! You've created a solid foundation for a collaborative to-do list application with good separation of concerns between list management, task operations, and real-time collaboration. Your use of event-driven architecture with Kafka for broadcasting changes and the inclusion of versioning for concurrent edits shows thoughtful consideration of the multi-user requirements. However, there are some critical issues around concurrent edit handling and scalability that need to be addressed to make this production-ready.

1.  **Positives**
    
    -   Separating collaboration logic into a dedicated service (Collaborationsvc) is a clean architectural choice that isolates the complexity of real-time updates and makes the system more maintainable
    -   Using Kafka for event broadcasting between services provides reliable message delivery and allows multiple consumers (collaboration service, notification service) to react to task changes asynchronously
    -   Including version numbers on tasks for optimistic concurrency control shows good understanding of concurrent edit challenges and provides a foundation for conflict resolution
2.  **Potential Issues** **Concurrent Edit Race Conditions**
    
    -   Your version-based approach only checks versions at submission time without any distributed locking mechanism
    -   This allows two users to read version 1, both make edits, and both try to update to version 2, causing one user's changes to be silently lost
    -   This violates the requirement for proper concurrent editing support and will frustrate users who lose work
    
    **Single Database Bottleneck**
    
    -   All services appear to share the same database instances without any sharding or partitioning strategy
    -   With billions of lists and millions of users, a single database will hit write limits around 10k TPS
    -   This creates an impossible scaling ceiling that prevents meeting the stated scale requirements
    
    **Missing WebSocket Details for Real-time Updates**
    
    -   The collaboration service maintains a map of list\_id to user\_id but doesn't specify how it pushes updates to clients
    -   Without persistent WebSocket connections or a proper real-time protocol, the <100ms latency requirement cannot be met
    -   This breaks the core requirement of users seeing each other's changes in real-time
3.  **Follow-up Questions**
    
    -   How does your system handle network partitions when the collaboration service can't reach some clients but Kafka is still receiving updates?
    -   What happens when two users simultaneously mark the same task as complete - which user gets credit in the activity log?
    -   How do you ensure the in-memory map of list\_id to user\_id in the collaboration service survives service restarts?
    -   What's your strategy for sharding the database when you reach billions of lists - by user\_id, list\_id, or something else?
    -   How do you handle permission changes (like revoking access) for users who have active WebSocket connections?
    -   What's the maximum number of collaborators you'd support per list given the 100 concurrent edit limit?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply