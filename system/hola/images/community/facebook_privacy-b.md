##### Design Facebook Privacy

by RemainingAmaranthBlackbird581• Mid-level• 11 days ago

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

[• 11 days ago](https://www.hellointerview.com/community/submissions/cmecsxrl305c2ad08tzzqp63d#comment-cmecsyjsv0097ad39xoaehgs7)

Nice design! You've identified some important components for handling privacy updates in a social media system, particularly the use of CDC for tracking privacy changes and maintaining a separate privacy updates cache. However, there are critical security and consistency issues that need addressing, especially around client-side privacy enforcement and the lack of a clear mechanism for propagating privacy changes to existing feeds.

1.  **Positives**
    
    -   Using CDC (Change Data Capture) to track privacy updates is a smart approach for capturing privacy changes asynchronously without impacting write performance
    -   Maintaining a separate privacy updates cache with key=userId:postId is a good optimization for quickly checking privacy levels during feed generation
    -   Using SSE (Server-Sent Events) for real-time feed updates allows efficient push-based notifications when privacy changes occur
2.  **Potential Issues** **Client-side privacy enforcement**
    
    -   The design states "use UI code to hide it" for privacy enforcement
    -   This allows any malicious user to bypass privacy by modifying client code or API calls
    -   Privacy must be enforced server-side before data leaves your backend to ensure security
    
    **Unclear privacy propagation mechanism**
    
    -   The queue/worker setup doesn't clearly explain how privacy changes propagate to existing feeds
    -   When a post changes from public to private, all affected user feeds need updating
    -   Without a clear propagation path, users may continue seeing posts they shouldn't have access to
    
    **Inconsistent data model**
    
    -   Privacy level is stored in both the Post table and a separate privacy updates cache
    -   No consistency mechanism ensures these stay synchronized during updates
    -   This could lead to situations where the cache and database disagree on a post's privacy level
3.  **Follow-up Questions**
    
    -   How do you ensure privacy is enforced server-side in the feed service rather than relying on UI code?
    -   What happens to cached feeds when a post's privacy changes from public to private?
    -   How do you handle privacy changes for users with millions of followers without overwhelming the system?
    -   How does the system maintain consistency between the privacy level in the Post table and the privacy updates cache?
    -   What's the exact flow when a privacy update triggers the queue - what message format and what do workers do?

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply