##### Design a Global IP Address Blocking System

by YearlingHarlequinMite643• Senior• 1 day ago

Functional Requirements Block request by IP Globally - must be denied at the edge before reaching backedns systems. Support multiple government blocklists - Accept from different contries/agensces. Enforce scope (global or country specific) Efficient update to blocklist - Support both real time updates and periodic batch updates. Apply deta changes and not full reloads Audit & compliance - Store versioned copies of blocklist. Provide logs for enforcement decisions.

  Non Functional Requirements. Scale : Handle 100+M DAU and millions of request/sec. Edge enformcence must be logaritichim time Latency : Lookup overhead per request < 10ms. Propagation of new block entries - critical <5s. Normal advisory < 1 hours Availability vs Consistency - Availability priortized . Eventual consistency for updates but must converge. Reliability & Durability : 99.99% succes sin enforcing blocks . 11 nines durability for audig logs. Security : All lists mut be cryptographicaaly singed by trusted issuers.

Collapse Description

1

2

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Hello Interview Bot](https://hellointerview-files.s3.us-west-2.amazonaws.com/public-media/hilogo.jpg)

Hello Interview Bot

[• 1 day ago](https://www.hellointerview.com/community/submissions/cmeqxpid701vyad080wa7897z#comment-cmeqxq5yf00c3ad3bkxmghzgi)

Nice design! You've created a solid foundation for a global IP blocking system with good separation between the management layer and enforcement layer. The horizontally scaled publisher component shows good thinking about update distribution, though there are some critical issues around edge enforcement and data propagation that need addressing to meet the stringent latency requirements.

1.  **Positives**
    
    -   Good separation between the Blocklist Management Service and enforcement components, which allows independent scaling of write and read paths
    -   Horizontally scaled Blocklistpublisher component shows understanding that update distribution needs to scale with the number of edge locations
    -   Including a BlocklistCache layer recognizes the need for high-performance lookups to meet the <10ms requirement
2.  **Potential Issues** **Missing True Edge Enforcement**
    
    -   Your design shows clients connecting to both API Gateway and CDN, but IP blocking happens at the API Gateway level
    -   This means malicious requests still reach your infrastructure instead of being blocked at the CDN edge
    -   This violates the requirement that requests "must be denied at the edge before reaching backend systems"
    
    **Unclear Data Structure for Logarithmic Lookup**
    
    -   The BlocklistCache component doesn't specify what data structure enables logarithmic time lookups
    -   With potentially millions of blocked IPs, a simple list or hash table won't meet the performance requirement
    -   Without a proper data structure like a radix tree or CIDR trie, you can't achieve the required lookup performance
    
    **No Global Update Propagation Mechanism**
    
    -   The design shows Blocklistpublisher updating BlocklistCache, but doesn't explain how updates reach all global edge locations
    -   Critical updates must propagate globally in <5 seconds, which requires a sophisticated push mechanism
    -   The current design could have significant lag between different regions receiving updates
    
    **Missing Audit Log Storage**
    
    -   The design only shows BlocklistDB but requirements specify 11 nines durability for audit logs
    -   Enforcement decisions need to be logged separately from the blocklist data itself
    -   Without dedicated audit log storage with proper replication, you can't meet the durability requirement
3.  **Follow-up Questions**
    
    -   How does the CDN enforce IP blocks when clients connect directly to it without going through your API Gateway?
    -   What data structure does the BlocklistCache use to achieve logarithmic lookup time for IP ranges?
    -   How do you ensure cryptographic verification of blocklists from government agencies before enforcement?
    -   What mechanism pushes critical updates to all global edge locations within 5 seconds?
    -   How do you handle CIDR ranges and IP subnets efficiently in your blocking logic?
    -   Where are enforcement decision logs stored to achieve 11 nines durability?
    -   How does the system handle country-specific blocklists vs global blocklists at the edge?
    -   What happens when a BlocklistCache instance fails - how do you maintain 99.99% enforcement success?
    -   How do you apply delta updates without full reloads across thousands of edge locations?

_Remember that true edge enforcement typically requires integration with CDN providers' edge computing capabilities or deploying your own edge infrastructure globally._

You can respond to this comment with questions and I'll do my best to answer them!

_Warning: This feedback was AI generated and may contain errors._

Show more

0

Reply

Y

YearlingHarlequinMite643

[• 1 day ago](https://www.hellointerview.com/community/submissions/cmeqxpid701vyad080wa7897z#comment-cmeqxq61o01w2ad08fs3rych2)

Give feedback

Show more

0

Reply