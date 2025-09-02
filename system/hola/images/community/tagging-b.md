##### Design a tagging system

by Mynameisdon• Senior Manager• 24 days ago

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

[• 24 days ago](https://www.hellointerview.com/community/submissions/cmdtsmkjo03oead08lik6c0hv#comment-cmdtsn0jq004dad3bhsimm15s)

Nice design! You've created a comprehensive tagging system with good separation of concerns between Tag, Post, and Search services. Your use of Elasticsearch for search functionality and CDC for keeping it synchronized shows solid understanding of modern search architectures. However, there are some critical consistency issues with the denormalized tag arrays in posts and missing details around how trending tags are calculated that need addressing.

1.  **Positives**
    
    -   Using Elasticsearch with inverted indexes for tag search and autocomplete is an excellent choice that will easily meet your <100ms search and <200ms autocomplete latency requirements
    -   Separating Tag, Post, and Search services provides good modularity and allows independent scaling of each component based on their specific load patterns
    -   Including CDC (Change Data Capture) to keep Elasticsearch synchronized with the primary database ensures your search index stays up-to-date without impacting write performance
2.  **Potential Issues** **Data Consistency Between Posts and Tags**
    
    -   You store tag\_ids as an array in the Post table while also maintaining a separate tag\_post junction table
    -   This denormalization creates potential for inconsistency where the array and junction table can get out of sync during concurrent updates
    -   This violates your high consistency requirement for "read your own write" as users might see different tags depending on which data source is queried
    
    **Missing Trending Tags Implementation**
    
    -   The design mentions trending tags as a functional requirement but provides no mechanism for calculating or storing trending data
    -   Without a time-series data structure or aggregation pipeline, you cannot determine which tags are trending over specific time windows
    -   This completely misses one of your five core functional requirements
    
    **Unclear Sharding Strategy for Billions of Tags**
    
    -   The design mentions "billions of tags" in scalability requirements but doesn't specify how the database will be partitioned
    -   A single database instance cannot efficiently handle billions of rows with the query patterns you've described
    -   Without a clear sharding key (by tag\_id, tenant\_id, or tag name prefix), you'll hit scaling limits well before reaching billions of tags
3.  **Follow-up Questions**
    
    -   How do you maintain consistency between the tag\_ids array in the Post table and the tag\_post junction table during concurrent tag additions/removals?
    -   What happens when multiple users try to add/remove the same tag from a post simultaneously - how do you prevent race conditions?
    -   How do you calculate trending tags - what time windows do you support and where is this aggregated data stored?
    -   What is your sharding strategy for the tag and tag\_post tables when you reach billions of tags?
    -   How do you handle tag search across multiple tenants/scopes - does each tenant get isolated Elasticsearch indices?
    -   What happens to posts when a tag is deleted - how do you cascade this deletion across all affected posts?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply