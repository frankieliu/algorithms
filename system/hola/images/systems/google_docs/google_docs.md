# Design Google Docs

Real-time Updates

[![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.ef667baa.png&w=96&q=75&dpl=e097d75362416d314ca97da7e72db8953ccb9c4d)

Stefan Mai

Ex-Meta & Amazon Sr. Manager

](https://www.linkedin.com/in/stefanmai/)

hard

Published Jul 13, 2024

* * *

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

## Understanding the Problem

**📄 What is [Google Docs](https://docs.google.com/)?** Google Docs is a browser-based collaborative document editor. Users can create rich text documents and collaborate with others in real-time.

In this writeup we'll design a system that supports the core functionality of Google Docs, dipping into websockets and collaborative editing systems. We'll start with the requirements (like a real interview), then move on to complete the design following our [Delivery Framework](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery).

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements)

**Core Requirements**

1.  Users should be able to create new documents.
    
2.  Multiple users should be able to edit the same document concurrently.
    
3.  Users should be able to view each other's changes in real-time.
    
4.  Users should be able to see the cursor position and presence of other users.
    

**Below the line (out of scope)**

1.  Sophisticated document structure. We'll assume a simple text editor.
    
2.  Permissions and collaboration levels (e.g. who has access to a document).
    
3.  Document history and versioning.
    

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements)

**Core Requirements**

1.  Documents should be eventually consistent (i.e. all users should eventually see the same document state).
    
2.  Updates should be low latency (< 100ms).
    
3.  The system should scale to millions of concurrent users across billions of documents.
    
4.  No more than 100 concurrent editors per document.
    
5.  Documents should be durable and available even if the server restarts.
    

Some non-functional requirements make your job _easier_! In this case, limiting the number of concurrent editors per document is a great constraint to have. It means we can avoid worrying massive throughput on a single document and instead focus on the core functionality.

For what it's worth, this is a choice that Google docs also made - beyond a certain number of concurrent users, everyone new can only join as readers (implying a bit about their architectural choices).

Here's how it might look on your whiteboard:

Requirements

## Set Up

### Planning the Approach

Before we start designing the system, we need to think briefly through the approach we'll take.

For this problem, we'll start by designing a system which simply supports our functional requirements without much concern for scale or our non-functional requirements. Then, in our deep-dives, we'll bring back those concerns one by one.

To do this we'll start by enumerating the "nouns" of our system, build out an API, and then start drawing our boxes.

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

First we'll define some core entities. These help to define terms between you and the interviewer, understand the data central to your design, and gives us a clue as to our APIs and how the system will fit together. The easiest way to do this is to look over your functional requirements and think about what nouns are involved in satisfying them. Asking yourself deeper questions about the behavior of the system ("what happens when...") will help you uncover the entities that are important to your design.

For this problem, we'll need just a few entities on our whiteboard:

Core Entities

We'll explain these to our interviewer as we go through the problem.

-   **Editor**: A user editing a document.
    
-   **Document**: A collection of text managed by an editor.
    
-   **Edit**: A change made to the document by an editor.
    
-   **Cursor**: The position of the editor's cursor in the document (and the presence of that user in a document).
    

### Defining the API

Next, we can move to the APIs we need to satisfy which will very closely track our functional requirements. For this problem, we probably want some REST APIs to manage the document itself. We also know we're going to need lots of bi-directional communication between each editor and the document they're collectively editing. In this case it makes sense to assume we'll need some sort of websocket-based approach for the editor experience, so we'll define a few of the messages that we'll send over the wire.

Each interviewer may put more of less emphasis on this step. Some interviewers may want to really understand the intricacies of the messages (veering into a low-level design question) while others may be happy to know that you have a few messages defined opaquely and can move on. We'll assume the latter here!

`POST /docs {   title: string } -> {   docId }  WS /docs/{docId}   SEND {     type: "insert"     ....   }    SEND {     type: "updateCursor"     position: ...   }    SEND {      type: "delete"     ...   }    RECV {      type: "update"     ...   }`

When your API involves websockets, you'll be talking about _messages_ you send over the websocket vs endpoints you hit with HTTP. The notation is completely up to you, but having some way to describe the protocol or message schema is helpful to convey what is going over the wire.

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes)

With a sketch of an API we'll move to our high-level design. Tactically, it can be helpful to tell your interviewer that you're sure there are more details of the API to work through but you'll come back to those later as you flesh out the problem.

While it's _awesome_ to be able to polish a perfect answer, you're in a time crunch! Acknowledging issues that you might get to later and leaving space to adjust your design as you learn more is a great way to keep the interview moving productively.

### 1) Users should be able to create new documents.

Our first step in our document editor is to allow users to create new documents. This is a simple API that takes a title and returns a document ID. There's a lot of interesting metadata that we may want to hang off of the document (e.g. permissions, history, tags, etc.) so it's sensible for us to assume there will be some separation of the document itself and the metadata. For this API, we'll tackle the first part.

From an interviewing perspective the pattern we're using here of a simple horizontally scaled CRUD service fronted by an API gateway is so common you should be able to sling them out really quickly! Don't spend too much time setting the stage.

Basic Metadata

Our interviewer is likely to ask us what database we're going to use here. For this problem, let's assume a simple Postgres database for now because it gives us flexible querying and, if we need to scale, we can always partition and replicate later.

### 2) Multiple users should be able to edit the same document concurrently.

For our next requirement, we'll have to deal with writing to the document itself which has both consistency and scaling problems. This is where things start to get interesting. In a collaborative editor multiple users are making high frequency edits to the same document at the same time — a recipe for consistency problems and contention!

We're going to defer the scaling concerns for our deep dive later in the interview, so we'll make a note to our interviewer that we'll come back to that so we can focus on consistency.

#### Collaborative Edits Breakdown

First, let me explain why achieving consistency in a collaborative editor is not easy by starting with a deficient solution and then building up to a more correct one.

I'm going to walk through the fundamentals here to explain how we can solve the consistency problems here but, like before, this is a place where it's good to have the background knowledge to be able to field questions from your interviewer but you probably won't use the interview time to teach these concepts to your interviewer. Skip ahead if you already understand them!

##### Sending Snapshots (Wrong)

Let's pretend on each edit the users send their entire document over to a Document Service. The Document Service will then store the document in a blob storage system like S3.

Deficient Solution

Simple enough, right? Not so fast. First, this design is incredibly inefficient: we're transferring the entire document over the wire for every edit. For a fast typer, this could mean 100s of KBs per keystroke, yuck. But a more serious problem looms.

Assume User A and User B are editing the same document concurrently. The document starts with a single paragraph of text:

Hello!

-   User A adds ", world" to produce Hello, world! which they submit.
    
-   User B deletes the "!" to produce Hello which they submit.
    

Both submit their edits at the same time. The actual document that gets stored to our blob storage is entirely dependent on which request arrives first. If User A's edit arrives first, the document will be Hello, world! If User B's edit arrives first, the document will be Hello. The user experience is terrible, we're losing all concurrent edits!

##### Sending Edits (Getting Warmer)

We can take a crack at solving these problems by recognizing that we're making **edits** to the document. Instead of transmitting the entire document over the wire, we can transmit just the edits themselves.

-   User A adds ", world" and sends INSERT(5, ", world")
    
-   User B deletes the "!" and sends DELETE(6)
    

We solved our first problem! We're no longer transmitting the entire document over the wire, so now we don't need to send 100s of KB on every keystroke. But we have a new problem.

User B's deletion assumes that the character at position 6 is an exclamation mark. If User A's edit arrives after User B's deletion, we'll end up with Hello, world - ok, good. But if User B's edit arrives after User A's edit, we'll end up with Hello world! - we deleted the comma instead of the exclamation mark!

##### Collaborative Editing Approaches

The critical missing piece here is that each edit is contextual: it's an edit based on a specific state of the document. Dealing with a stream of edits presents a bunch of consistency problems! We have two options to solve this problem:

### 

Good Solution: Operational Transformation (OT)

###### Approach

One way to thread the needle is to reinterpret or _transform_ each edit before it's applied. The idea here is that each user's edit can be adjusted based on the edits that came before it.

We'll collect all of the edits to the document on a single server. From there, we can, in batches, transform each edit based on the edits that came before it. While an exhaustive treatment of OT is beyond the scope of this design (you can read more [on the wikipedia page](https://en.wikipedia.org/wiki/Operational_transformation)), let's consider a simple worked example.

User B's DELETE(6) is trying to delete the character at position 6, which for them was a ! at the time the edit was created. The problem occurs when User A's edit INSERT(5, ", world") arrives first. If we don't transform User B's edit, we'll end up with Hello, world instead of Hello, world!.

The OT approach is to transform the operations before they're applied (and, later, before they're sent back to the client). In this case, if User A's edit arrives before User B's deletion, we can transform User B's DELETE(6) to DELETE(12) so that when User A's edit is applied, it deletes the exclamation mark instead of the comma.

OT is low memory and fast, but comes with a big tradeoff.

###### Challenges

OT requires a central server which can provide a final ordering of operations. This allows us to scale to a small number of collaborators, but not an enormous number. Our non-functional requirements help us here. OT is also tricky to implement correctly and easy to get wrong.

Operation Transforms

### 

Good Solution: Conflict-free Replicated Data Types (CRDTs)

##### Approach

The alternative to reinterpreting every edit is to make every edit commutative or, able to be applied in any order. This is the approach taken by Conflict-free Replicated Data Types (CRDTs).

CRDTs allow us to represent edits as a set of operations that can be applied in any order but still produce the exact same output.

For simple text operations, CRDTs do this with two tricks:

1.  CRDTs represent positions using real numbers rather than integers. These positions _never change_ even in light of insertions. If we need to create more "space", real numbers allow us to do so infinitely (like adding 1.5 between 1 and 2).
    
2.  CRDTs keep "tombstones" for deleted text. The text is never actually removed from the document, but we remember that it is deleted before displaying the document to the user.
    

Note: We won't be able to go into too much depth to explain CRDTs (a deep research topic!) though you're free to read more on [the wikipedia page](https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type). But to give you some intuition, let's consider a simple worked example.

Assume we have two users looking at "Hello!" where each character has a position number under it:

H  e  l  l  o  !

0  1  2  3  4  5

Both users want to insert something after "Hello". User A wants to insert "," and User B wants to insert " there". This would be tricky in OT because we'd need to transform one edit based on the other. But with CRDTs, we just need to make sure each new character gets its own unique position number between 4 and 5.

When User A's wants to insert "," she picks an arbitrary number between 4 and 5 as the position for her comma, in this case she chooses 4.6:

H    e    l    l    o    ,    !

0    1    2    3    4    4.3    5

User B does the same (at the same time), assigning arbitrary numbers between 4 and 5 for each of the characters he wants to insert:

H     e     l     l     o         t     h     e     r     e    !

0     1     2     3     4    4.1  4.2   4.4   4.5   4.7   4.8  5

The magic is that these position numbers create a total ordering - no matter what order the edits arrive in, every client will sort the characters the same way and see the same final text.

In this case, the resulting string is "Hello t,here!" which is a consequence of both users editing in the same position. There are a lot of tricks to prevent conflicts like this, but the core guarantee of a CRDT is that no matter what order the edits arrive in, every client will converge on the same document.

CRDTs are elegant in that they remove the need for a central server. As long as all updates eventually reach all clients, the clients will converge on the same document. This makes it a good solution for peer-to-peer applications. CRDTs are also more easily adaptable to offline use-cases - you can just keep all of the edits in local storage and add them to the bag of updates once the editor comes back online.

A great commercial implementation of CRDTs is [Yjs](https://github.com/yjs/yjs) which I would highly recommend for lightweight collaborative applications.

###### Challenges

The benefits of CRDTs come with some big downsides. The memory usage is higher because we need to remember every edit, including tombstones for text that has long been deleted. This means our document only grows in size, never shrinks. CRDTs are also less efficient computationally, though there are workarounds! And finally the handling of conflicts can be inelegant. For most real-time use cases this isn't as much a problem, simply having a cursor indicator is enough for you and I to avoid editing the same part of the same document, but as clients need to buffer or go offline it can become a major issue.

For this problem, we're going to use the Operational Transformation (OT) approach. This is the approach that Google Docs takes and it benefits from requiring low memory, being more adaptable to text in particular, and making use of a centralized server.

If we needed to support a larger number of collaborators, or a Peer-to-Peer system (maybe using WebRTC!), we could use CRDTs. There are plenty of examples of industrial CRDTs which cut some corners (Figma being a notable one) - there's a healthy amount of research and development happening on both approaches.

With that in mind, we can update our design to solve the collaborative edit problem. Our updated design sends edits to a central server as operations, which are transformed by the server before being recorded in a database.

Document Service with Operational Transformation

For our new Document Operations DB, we want to use something that we can write very quickly in an append-only fashion (for now). Cassandra should do the trick. We'll partition by documentId and order by timestamp (which will be set by our document service). Once the edit has been successfully recorded in the database, we can acknowledge/confirm it to the user. We satisfied our durability requirement!

### 3) Users should be able to view each other's changes in real-time.

Ok, now that we have a write path let's talk about the most important remaining component: reads! We need to handle two paths:

-   First, when a document is just created or hasn't been viewed in a while, we need to be able to load the document from our database and transfer it to the connected editors.
    
-   Next, when another editor makes an edit, we need to get notified so that our document stays up-to-date.
    

#### When the Document is Loaded

When a user first connects, they need to get the latest version of the document. In this case when the Websocket connection is initially established for the document we can push to the connection all of the previous operations that have been applied. Since everyone is connecting to the same server, this allows us to assume that all connections are starting from the same place.

Loading a Document

#### When Updates Happen

When an edit is made successfully by another collaborator, every remaining connected editor needs to receive the updates. Since all of our editors are connected to the same server, this is straightforward: after we record the operation to our database, we can also send it to all clients who are currently connected to the same document.

The next step might surprise you: On the client, we **also** have to perform the operational transformation. Let's talk briefly about why this is.

When users make edits to their own document, they expect to see the changes _immediately_. In some sense, their changes are always applied first to their local document and then shipped off to the server. What happens if another user lands an edit on the server _after_ we've applied our local change but _before_ it can be recorded to the sever?

Our OT gives us a way of handling out of order edits. Remember that OT takes a sequence of edits in an arbitrary order and rewrites them so they consistently return the same final document. We can do the same thing here!

So so if User A submits Ea to the server, and User B submits Eb (which arrives after Ea), the perceived order of operations from each site is:

Server: Ea, Eb
User A: Ea, Eb
User B: Eb, Ea

Regardless of the ordering, by applying OT to the operations we can guarantee that each user sees the same document!

Sending Updates to Clients

### 4) Users should be able to see the cursor position and presence of other users.

One of the more important features of a collaborative editor is "awareness" or being able to see where your collaborators are editing. This helps to avoid situations where you're simultaneously editing the same content (always a mess!) or repeating work. Awareness is inherently transient, we only really care about where our collaborators' cursor is **now** not where it was an hour ago. We also don't care about where the user's cursor is when they aren't currenty connected - it doesn't make sense. This data is **ephemeral** with the connection.

These properties help us to decide that we don't need to keep cursor position or the presence of other users in the document data itself. Instead, we can have users report changes to their cursor position to the Document Service which can store it in memory and broadcast it to all other users via the same websocket. When a new user connects, the Document Service can read the properties of other users out of memory and send them to the connecting user. Finally, the Document Service can keep track of socket hangups/disconnects and remove those users from the list (sending a broadcast when it happens to any remaining users).

And with that, we have a functional collaborative editor which scales to a few thousand users!

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes)

With the core functional requirements met, it's time to cover non-functional requirements and issues we introduced.

### 1) How do we scale to millions of websocket connections?

One of the weakest assumptions we made as we built out the system is that we'll be able to serve everyone from a single Document Service server. This won't scale to millions of concurrent users and introduces a single point of failure for availability.

Wwe need to scale the number of Document Service servers to the number of concurrent connections. When I connect to the Document Service, I either (a) need to be co-located with all other users connected to the same document or (b) know where all the other users are connected. (See [Whatsapp](https://www.hellointerview.com/learn/system-design/problem-breakdowns/whatsapp#1-how-can-we-handle-billons-of-simultaneous-users) for a similar problem.)

###### Pattern: Real-time Updates

This pattern of needing push updates to clients across a scaled backend service is a quintessential example of the real-time updates pattern.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates)

Websocket Connections

When you introduce websockets into your design, you're probably doing it because they are _bi-directional_, you can send messages back to your clients. With a traditional client-server architecture you're mostly talking about left-to-right arrows: how can clients connect to the servers to send messages and receive a response. The jump that many candidates fail to make is to think about the **right-to-left** arrows on their diagrams: how do your internal services talk to the clients?

Because the statefulness of websockets is a pain, it can be useful to handle them at the "edge" of your design. By terminating websockets early and exposing an "internal" API to the rest of our system, other systems can retain statelessness and don't need to concern themselves with the details of websocket connections.

The solution here is to horizontally scale the Document Service and use a consistent hash ring to distribute connections across servers. Each server in our system joins a the hash ring, with each server responsible for a range of hash values. This means we always know both which server is responsible for the document and all connections go there. We use Apache [ZooKeeper](https://www.hellointerview.com/learn/system-design/deep-dives/zookeeper) to maintain the hash ring configuration and coordinate between servers.

When a client needs to connect:

1.  They can initially open an HTTP connection to any of the document servers (potentially via round robin) with the document ID requested.
    
2.  That server will check the hash ring configuration for the given document ID and if it's not responsible for that hash range, it will respond with a redirect to the correct server's address. This way, clients eventually connect directly to the right server without an intermediary.
    
3.  Once the correct server is found, the HTTP connection is upgraded to a websocket connection. The socket server maintains a map from the document ID to the corresponding websocket for use later.
    
4.  We load all the stored ops (if they aren't already loaded) from our Document Operations DB and are ready to process operations.
    

Since all of the users of a document are connecting to the same server, when updates happen we can simply pass them to all connected clients.

The beauty of [consistent hashing](https://www.hellointerview.com/learn/system-design/deep-dives/consistent-hashing) is that when we add or remove servers, only a small portion of connections need to be redistributed. Servers periodically check Zookeeper for ring changes and can initiate connection transfers when needed.

Architecture with Consistent Hash

The downside of this approach is that in these scaling events we need to move a lot of state. Not only do we need to move websocket connections for displaced users (by disconnecting them and forcing them to reconnect to the right server), but we also need to ensure document operations are moved to the correct server.

In an interview, we may be asked to expand on this further:

**Scaling complexity**

-   Need to track both old and new hash rings when adding servers
    
-   Requests may need to be sent to multiple servers during transitions
    
-   Server failures require quick redistribution of connections and hash ring updates
    

**Connection state management challenges**

-   Need robust monitoring for server failures and connection issues
    
-   Must implement client-side reconnection logic for server unavailability
    

**Capacity planning requirements**

-   Must ensure servers have sufficient resources for their connection load
    
-   Need to monitor connection distribution to prevent server hotspots
    

### 2) How do we keep storage under control?

With billions of documents, we need to be thoughtful about how we manage storage. By choosing OT vs CRDTs, we can already reduce the amount of storage we need by a factor. Still, if each document is 50kb, we're looking at 50TB of storage! Worse, if we have a document with millions of operations, each of these operations will need to be transferred and applied to each client that joins the document!

Remember also that all _active_ documents need to be held in memory in our Document Service. We need to keep that space small in order to avoid being memory bottlenecked.

One of the most natural solutions here is we'll want to periodically snapshot/compact operations. The idea here is that we don't need all of the operations in our Document Operations DB to be stored (exception being if we want to manage some sort of versionining), and we can collapse many operations into one to save on both processing and space.

Lots of options to pull this off!

### 

Good Solution: Offline Snapshot/Compact

##### Approach

One approach we can take is to introduce a new Compaction Service which periodically reads operations out of the Document Operations DB and writes a snapshot back to the DB collapsing those instructions. The exact schedule has lots of potential parameters: we may look for large documents, that haven't been recently compacted, and probably haven't been written recently.

Because this necessarily changes the operations that follow, we need to make sure that what we're writing isn't impacting a live document. This is a tricky distributed transaction that we need to pull off and, because we're using Cassandra for the Document Operations DB, we only have row-level transactions to work with.

To ensure document-level atomicity, we'll introduce a new documentVersionId to the Document Metadata DB. Before loading a document, we'll grab this documentVersionId out of the Document Metadata DB so we know which document operations to retrieve.

Whenever we want to change operations, we can write new operations and the _flip_ the documentVersionId. We'll make sure all _flips_ go through the Document Service so we don't have (yet another) race condition.

The Compaction Service can:

1.  Read the document operations out for a given document out of the DB.
    
2.  Compact these operations into as few operations as possible (probably a singular insert operation!).
    
3.  Write the resulting operations to a new documentVersionId.
    
4.  Tell the Document Service to _flip_ the documentVersionId.
    

If the Document Service has a loaded document, it will discard the flip command to defer compaction which might corrupt existing operations.

Compaction Service

##### Challenges

### 

Great Solution: Online Snapshot/Compact Operations

##### Approach

A different approach we can take is to have our Document Service periodically snapshot/compact operations. Since the Document Service has exclusive ownership of a document when we're not scaling up the cluster, we can safely compact operations without worrying about coordination/consistency. In fact, a natural time to do this is when the document is idle and has no connections - something the document service will know about immediately. We also already have all of the existing document operations in memory at that time, making our job easier.

When the last client disconnects we can then:

1.  Take all of the existing operations and offload them to a separate (low CPU nice) process for compaction.
    
2.  Write the resulting operations to the DB under a new documentVersionId.
    
3.  Flip the documentVersionId in the Document Metadata DB.
    

Online Snapshot/Compact Operations

##### Challenges

Moving compaction into the document service risks increasing latency for document operations (especially at the tail P99), we need to make sure they're not hogging our CPUs unnecessarily. One option for this is to run them in a separate process with lower CPU priority.

### Some additional deep dives you might consider

Google Docs is a beast with hundreds to thousands of engineers working on it, so we can't cover everything here. Here are some additional deep dives you might consider:

1.  **Read-Only Mode**: Google Docs has a "read-only" mode where users can view documents without interfering with others. It's also considerably more scalable, millions of readers can be viewing the same document. How might we implement this?
    
2.  **Versioning**: Google Docs allows users to revert to previous versions of a document. How can we extend our snapshot/compact approach to support this?
    
3.  **Memory**: The memory usage of our Document Service can be a bottleneck. How can we further optimize it?
    
4.  **Offline Mode**: How might we expand our design if we want to allow our clients to operate in an offline mode? What additional considerations do we need to bring into our design?
    

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

### Mid-level

Personally, I probably wouldn't ask this question of a mid-level candidate. That's not to say you won't get it, but I wouldn't. With that out of the way, I'd be looking for a candidate who can create a high-level design and then think through (together with some assistance) some of the issues that you'll encounter in a high concurrency problem like this. I want to see evidence that you can think on your feet, spitball solutions, and have a small (but growing) toolbox of technologies and approaches that you can apply to this problem.

### Senior

For senior candidates, I would expect them to immediately start to grok some of the consistency and durability challenges in this problem. They may start with a very basic, inefficient solution but I expect them to be able to proactively identify bottlenecks and solve them. While I would not expect a candidate to necessarily know the distinction between OT and CRDT, I would expect them to be able to talk about database tradeoffs and limitations, and how they might impact our design. We might have time for an additional deep dive.

### Staff

For staff engineers, I'm expecting a degree of mastery over the problem. I'd expect them to be loosely familiar with CRDTs (if they're not, they'll make it up somewhere else) but intimately familiar with scaling socket services, consistency tradeoffs, serialization problems, transactions, and more. I'd expect to get through the deep dives and probably tack on an extra one if we're not too deep in optimizations.

## References

-   [Real-Time Mouse Pointers](https://www.canva.dev/blog/engineering/realtime-mouse-pointers/): How Canva enables collaborative editing using WebRTC.
    

Mark as read

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

Z

ZealousSilverClownfish609

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3b3t0h6016aotaaqf9dl82v)

If the applied Ops are OT transformed before writing to the document database (Cassandra) then why during read again OT is required at client side?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3eukvo800d2130robs64ylj)

Added some clarity about this to the final design which will be published today.

Show more

1

Reply

S

SubstantialHarlequinHerring381

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3fk9jl800ntq5z99839yaiq)

Hey Stefan, where is the actual document stored? It's a little unclear from the diagrams, but is it safe to assume that every known version of a document is stored in S3, and when a client connects for the first time, we grab the most recent version from S3 and load the latest operations from Document Ops DB and send this to the client?

Show more

6

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3fkd22900oxr95rmrg1wypj)

In this design, the document is represented as a sequence of operations in the operations DB. After compaction that's just 1 operation "insert {the document contents}".

Having some sort of interweaving with blob storage is a good extension, but remember we're not snapshotting the document here! We're recording a sequence of edits.

Show more

8

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8cnaecn00gpl2byaj58daor)

In Dropbox design, i believe we store the file in S3 due to the file can be very large like 50G. In reality, google doc can also be large. What storage strategy does goold use for large file with collaborative edit

Show more

3

Reply

![Majid Parvin](https://lh3.googleusercontent.com/a/ACg8ocLNlEyq83JWBHnKhb6tKU3BpX_288_j-QtoQu-9g_aPygzjm8ko=s96-c)

Majid Parvin

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma3mkaw10108ad08517so2fm)

Hi Stefan,

Imagine we have a document that initially contains the word “hello,” and a couple of users users go on to make roughly 10,000 edits. The next day, a new team member joins—how can we reconstruct the document’s current state if yesterday’s updates were never synced to S3?

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma43f22801c6ad08xfwyyhzt)

The operations are stored in our Document Operations DB (Cassandra), not S3. When loading the document, we'll fetch the latest compacted state plus any subsequent operations - likely far fewer than 10k ops since our compaction service would have collapsed those into a minimal set of operations overnight.

Show more

2

Reply

![deepak gupta](https://lh3.googleusercontent.com/a/ACg8ocJnydfF5BGePvAFEvXhWoY73HjvUZHcic0azbs9_10t75ujBw=s96-c)

deepak gupta

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcw8685901i5ad072xqf15sm)

I think this design should be updated with this information. It was unclear to me as well. I would assume the same. i.e. store the document in S3 and then apply operations on this document when we do compaction. Whenever some user connects we send the document from S3 and then apply operations with timestamp > timestamp of S3 document. Something like lambda architecture.

Show more

2

Reply

![Amir Tugendhaft](https://lh3.googleusercontent.com/a/ACg8ocI7XhTBSwnHabWF_66q2qSHU8kQmTt4RVIWk3qlGdUsOdqzlO0OxQ=s96-c)

Amir Tugendhaft

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3hk9n4c0096ofz0oig5kzt6)

Something doesn't sits well with me: You mention in the CRDT the fact that user A adds , to the text, and it chooses 4.3 randomly as the number between the characters. But then user B randomly chooses for their edit numbers from 4.1 - 4.8 excluding 4.3.

If CRDT nodes are supposed to be independent from each other - how come they know not to conflict like that with each other?

And what happens if 4.3 would've been chosen for user B?

Show more

8

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3hkbp4x008411zarfm2611m)

I'm using round numbers for the illustration, but in reality you'd have some random floating point number like 4.3423442346234 which have an significantly lower collision probability. Typical implementations have a tiebreaker scheme based on the site author as well.

Show more

6

Reply

![Amir Tugendhaft](https://lh3.googleusercontent.com/a/ACg8ocI7XhTBSwnHabWF_66q2qSHU8kQmTt4RVIWk3qlGdUsOdqzlO0OxQ=s96-c)

Amir Tugendhaft

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3hkxs1i00astp6v5uaw7a8q)

Nice. Thanks :)

Show more

0

Reply

D

DecisiveOlivePtarmigan394

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3uibczh00t6foxaqkr2yxxw)

I believe each client also uses some jitter so that both are not adding\\editing 4.3 , instead they would add jitter so that it is 4.344, 4.44 etc...

Show more

2

Reply

![Amir Tugendhaft](https://lh3.googleusercontent.com/a/ACg8ocI7XhTBSwnHabWF_66q2qSHU8kQmTt4RVIWk3qlGdUsOdqzlO0OxQ=s96-c)

Amir Tugendhaft

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3hkwtbc0090zs3efackseb8)

When talking about scaling web-sockets, why not use the same method as in the Live comments answer key: Redis pub/sub with layer 7 load-balancer to direct you to the relevant server? And then to topic the pub/sub by the docID.

Are web-sockets and SSE connection different in that aspect?

Here's the link to the live-comments sections: https://www.hellointerview.com/learn/system-design/answer-keys/fb-live-comments#1-how-will-the-system-scale-to-support-millions-of-concurrent-viewers

Show more

16

Reply

I

IntellectualCoralWhitefish366

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3kkelqq022pp0z0oiguozsr)

+1, seems like the framing in this problem is that we want the ability to broadcast specific events to specific users. Would it be simpler and more efficient to broadcast the same set of events for each document instead?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm47etg2e02ek129iypg2kpao)

Yeah good point. The "Great Solution" actually uses docId but the pub/sub variant doesn't. Let me fix this!

Show more

1

Reply

T

ThoughtfulEmeraldPheasant808

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm49kl3m3000uk9acxwplmawi)

Why can't we use the layer 7 LB for the consistent ring approach as well. That way we have the doc id and the LB can reach out to zookeeper to point to the server that has the respective doc. based on the hash of the doc id. The server can than create a websocket connection with that user and keep the websocket and the userid info in memory.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm49lh6hy0014sp7uo7ek8o11)

That’s what’s happening in the great solution? Is the point you’re trying to press on the load balancer? Which L7 load balancer did you have in mind?

Show more

0

Reply

T

ThoughtfulEmeraldPheasant808

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm49zfwsi006gbrwjhv2o9i41)

I was confused with these steps in the great solution “That server will check the hash ring configuration for the given document ID and if it's not responsible for that hash range, it will respond with a redirect to the correct server's address. This way, clients eventually connect directly to the right server without an intermediary.” Why would the server check this. This could be checked by the load balancer by talking to zookeeper before connecting to the server.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4a0t1ez0073wkmaapbo8m64)

It’s relatively uncommon to have a load balancer implement logic like this. I can’t name one that does. But if you can that’s great!

Show more

0

Reply

S

stevenjern

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8hr2o3a001qwoxabf6gw9t6)

When you wrote, "clients eventually connect directly to the right server without an intermediary", did you mean, the client and WebSocket server would connect directly without the API Gateway? Is that secured? also, would each websocket server have its public IPs in addition to private IPs? is that common? I'm asking because I'm used to architectures where client only connects with LBs created within the demilitarized zone (DMZ), while the backend servers reside in a protected network.

Show more

1

Reply

![Rashmi lalwani](https://lh3.googleusercontent.com/a/ACg8ocI1Ja31bZxqjVWsd8puRUxzVOKFZTbEbO0GIsFnxZwHPik2vA=s96-c)

Rashmi lalwani

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma4pjo0s0084ad08zbt1n75p)

The websocket should still be created via APIGateway, L7 load balancer can have a logic on which server to connect based on some server-id which can be passed in the request header. Client can do a http call to the service to get the server id (via Zookeeper) and then make a websocket request with the server-id in the header.

Show more

0

Reply

T

ThoughtfulEmeraldPheasant808

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4arq2ss00mw10dqaa7d9tcs)

I believe Traefik open source L7 and can talk to zookeeper but I am not 100% sure. You are correct that this is fairly uncommon.

Show more

0

Reply

![Mohit rao](https://lh3.googleusercontent.com/a/ACg8ocICNX8FVE9vC87vUVi-7icTi-h6FgzwBpgML3VmTlpxsfxKlA=s96-c)

Mohit rao

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm91iikhb00i7ad08d410bz4m)

Hey Stefan - not able to see multiple approaches for scaling web-sockets. Is it removed now?

Show more

6

Reply

I

IndividualGrayToucan573

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3qxdiw300klum797hmoj3qi)

" we'll be able to serve everyone from a single Document Service server. This won't scale to millions of concurrent users and introduces a single point of failure for availability."

With consistent hash ring approach, all of the users of a document are connecting to the same server. so, still the same server handles edits/view for a given document? If so, why the consistent hash ring approach solves the scalability problem?

Show more

0

Reply

R

ResponsibleIvoryAlbatross601

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5fsxbcl00dsrkl83eps4qhq)

I think the statement refers to having a single server for all of the requests of all of the documents. By consistent hashing we scale out and distribute requests among several servers based on DocumentId.

Show more

1

Reply

D

DisciplinaryMoccasinSnipe541

[• 9 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3u1ngbh00m6fx5sm3th72ng)

Hi Stefan, Thanks for the write-up, this is extremely helpful. I wanted to understand, for optimizing the memory of the document service for live documents, why store all operations in memory of document service server. Instead can't we only store the live document constructed from the operations in memory and send out all operations to be written into a time-series DB and use the Write-Ahead log of time series DB to build point in time snapshot as needed? Also in this case we are compacting when no users are online. We can use similar approach to build an old version of the document during compaction using time-series DB and Write ahead logging. This way we do not need a lot of memory for all operations to be stored in Document service and also this helps in avoiding construction of the document from all operations when user is trying to access the document? I guess we could use CRDT as well using this strategy as in-memory storage is just a document snapshot and we can scale out time series DB for every single edit that is showing up.

Show more

2

Reply

![Rajeev Ranjan](https://lh3.googleusercontent.com/a/ACg8ocIBEO7pBAXfB_OqF5nngGLiPiqMKUh55p5Z_ZU90W3o8hkOKw=s96-c)

Rajeev Ranjan

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm3ygg1t00068fy89q89mp887)

The difficulty level for this system design problem should be set to hard.

Show more

16

Reply

![Chua Chiah Soon](https://lh3.googleusercontent.com/a/ACg8ocKAr5qwe5r4pZczyZIOEKj7X8eyCxucQUgBaj50gyG5j8Wjig=s96-c)

Chua Chiah Soon

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm45b8sng00cv4dcrnsubjpk7)

Just a heads up, looks like the challenges under the Offline Snapshot/Compact are missing!

Show more

23

Reply

U

UniversalOlivePanda895

[• 7 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmefi7umv00npad07ppf253qc)

still missing

Show more

0

Reply

R

RoundAquamarineWasp973

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4752iyi022z129i7y6yju2c)

Is this less likely to be asked in a product architecture interview?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm49gr0rb00n5jtnoapjjg1ww)

Not impossible by any means. Less common though.

Show more

0

Reply

M

MobileTomatoAlpaca231

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4a7cn9a00dnk9acgr3bxs54)

In the great solution, is the Document service also handling socket connections by being responsible for a range of document ids or is there a seperate socket server in the consistent hashing ring that maintains the websocket connections? If it is the former, why is there a reference to socket servers in the great solution?Could you elaborate?

Show more

1

Reply

![Alonza Huang](https://lh3.googleusercontent.com/a/ACg8ocLWXptO6UbmQwUQJDjYE2DDP_XWsT7dUFo9CcLOuH70JZddbEM=s96-c)

Alonza Huang

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4d17tki01l9j6nbqs3skn58)

According to NGINX documentation\[1\], the NGINX Load Balancer has a built-in option to leverage the Ketama consistent hashing algorithm. But the hash ring can only be in memory. So, it seems that ZooKeeper is necessary.

My question is: do we always use ZooKeeper when we want to persistent consistent hashing?

\[1\] https://nginx.org/en/docs/http/ngx\_http\_upstream\_module.html?&\_ga=2.159878846.2005464618.1733505604-362919900.1733505604#hash

Show more

1

Reply

C

ConfidentialCrimsonFlamingo719

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4dtf09f02d3j6nb3da9y0mu)

Excellent explanation!

I'm curious why in this example, for optimizing webhooks, the consistent hash ring is better than the pub/sub solution, but in your Whatsapp writeup, the pub/sub is better than consistent hashing?

Show more

2

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4gb5bua00ai11jt4x9es1gp)

We need to keep some state around to apply the OT. If we're already doing that, we're halfway done with the consistent hash solution.

In most cases avoiding a stateful service is preferable - they're a pain to manage (e.g. deployments) and scale. But in this case we can't avoid it!

Show more

1

Reply

C

ConfidentialCrimsonFlamingo719

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4gevlr100ftbb1kx1ncvqh5)

Now I see I wrote "webhooks" instead of "websockets", but thanks for the great reply, it makes sense!

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8cnjco000hcl2byt6lww4oo)

"We need to keep some state around to apply the OT. If we're already doing that, we're halfway done with the consistent hash solution."

What state are we keeping here? Also, this implies each user<>document requires opening a ws connection. What if one user opens many document, there needs multiple ws connections create for the one user. Is that inefficient? One ws connections per user would be more efficient and we have to use pub/sub in this case, as connections is not per document

Show more

0

Reply

L

LittleCoralGalliform652

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4f4mth903ijamnhn3nh2xx5)

What does "flipping" the documentVersionId mean?

Show more

3

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4rx8rsj00odf66gbeit4fqw)

I assume it means setting the documentVersionId to point to the newly compacted row that has all the edits thus far. However, the corresponding diagram doesn't show any data for this row in Doc Ops DB, so, I'm a little confused. Also, this approach will require a distributed txn across 2 DBs, which is complicated. I don't understand why the compaction doesn't simply move all the old rows to cold storage, there's no need to update the Doc Metadata DB.

Show more

1

Reply

L

LuckyCoralChinchilla879

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4g9vimt0094bb1kgtqho8wd)

In the Redis pub/sub solution diagram, why is the websocket connection still in btwn API gateway and document service? shouldn't it be btwn API gateway and Document socket servers? in that case, would there be a request from the document socket servers to the document service to apply the OT algorithm before returning the response to the client? i don't understand the need for the document socket servers and the document service. why do we need both? why can't the document service be subscribed to Redis pub/sub?

Show more

1

Reply

F

FlyingTomatoHeron745

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4hlrjf601gl11jt4qxod54d)

For premium write-ups will there be videos released as well?

Show more

0

Reply

M

MarxistHarlequinMarsupial399

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4q3tjp7003q13kdc3knya3g)

Little typo here:

\`User A adds ", world" to produce Hello, world! which they submit.\`\`

The ! should not be there.

Show more

0

Reply

S

socialguy

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4qw2wcp012nxwp6m91hgiwm)

Why would the document service store all the active edits in memory? Is the document service serving as the OT central server?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4reu9ml005puek6pojsiocm)

Yep!

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm50k3cwl01fn13vurrtp1xxp)

for new users that connect to the document,why can’t we just seerialize the in memory representation of the document on server? i fail to understand why we need to send individual edits

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm50m3yvv01gsmqq6djfyeo4w)

Need durability!

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm50meolh01gxmqq6f9n5l5fi)

snapshotting and WAL for operations is a given. what i meant is for new users that start, why do we have to start from empty doc and send million (hypothetical) operations for them to build

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm50mgh1a01h5mqq6otvogg9h)

You won’t, compaction should kick in which would smush those down to a few ops.

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm50mxni001iowbx3tp4ekx7y)

thanks, but doesn’t compaction depend on getting an opportunity to do it? for a really popular file (say layoff updates) many people will connect to it at the same time, and there may not be opportunity to disconnect. if the server is the defacto source of truth, and keeps an in-memory version of correct state, we might as well pass it to the newly connected clients as the start state imo. do u see a problem with this approach

Show more

0

Reply

O

OfficialAmberGuppy264

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8aarmoz00jrsj0fls11vcg6)

I think we can extend compaction to also work on actively viewed files as well. We could compact for example the ops\[0:-1000\], and then replace all the old ops with this single compaction result. It could be a bit challenging due to active operations but definitely doable I think.

Show more

0

Reply

NB

Nick Babcock

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4vzk6st00lfpecvnnuwsjrt)

Thanks for the writeup, TIL about operational transforms.

In your opinion, is it ok if a candidate elects for a vendor specific solution as long as they detail the lock-in risk and outline an alternative? I've seen a couple deep dives callout Amazon SQS for delays/retries.

I ask here as I think Cloudflare Durable Objects would be a great fit for this use case and would replace the document service (keeping the metadata and operations DB untouched):

With each document having an associated durable object, we'd scale to millions of websockets without needing to introduce zookeeper and consistent hashing to ensure the same server handles all the document's connections. Another benefit is that Durable Objects can (optionally) be hosted in the region nearest to the creator. If we assume most collaborators are close to each other, this will cut down on cursor presence and edit latencies.

My hope would be to impart a way to simplify with an alternative solution, but if you think this is a bit too much of a deep dive curveball in an interview, let me know.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm4vzqmtt00llpecv87vvuwpi)

I like Durable Objects as a solution here! The upside is they're a pretty reasonable solution. The risk is that your interviewer may not be familiar with them or that they ask you details about how they're implemented.

Interviewers can get a bit weird when you oversimplify their question - most are banking on spending a certain amount of time in the session so if you flip their problem over like a turtle and simplify it too much they'll usually respond by either taking tools away from you ("how about if we can't use durable objects?") or by extending the question in odd ways.

Being able to present alternatives is always a net win, but I wouldn't bank on the availability of any less commonly known tech.

Show more

1

Reply

I

InclinedSapphireScallop143

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm54loys7044mwbx37ddomjrd)

one problem with the consisten hashing with documentId approach, usually we only want to maintain one websocket connection with the user, but the user can open and edit multiple documents at the same time. For example, the connection is built on server with hashing documentId1, now user opens up documentId2, how to find this connection when we want to deliver documentId2 changes now?

With this case, looks like the Redis pub/sub method is probably better

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm54lrilc043t13vu1fo7xa0u)

Yeah in this case we'll need a separate connection per document ID. This isn't particularly problematic because often for a document editor like Google Docs you're going to have a separate tab/window for each document so you _have_ to have a separate connection anyways.

Show more

2

Reply

I

InclinedSapphireScallop143

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm54m8jjg0458wbx3zqldnb0k)

Thank you, make sense! I didn't know that for sockets with web server, each tab has to have a separate connection.

Show more

0

Reply

R

ResponsibleIvoryAlbatross601

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5ewt8zg003em6ecuo760y1a)

Hi Stefan, thanks for this great write-up! I have a question about OT logic.

Based on the section that defines OT, it is mentioned that OT transforms an edit based on the edits that came before it. So it seems that OT cares about the order of the edits. As far as, the benefit of CRDT over OT is that with CRDT edits can be applied in any order.

However, later it is mentioned that the client should also apply OT, explaining that "Our OT gives us a way of handling out of order edits. Remember that OT takes a sequence of edits in an arbitrary order and rewrites them so they consistently return the same final document."

These two descriptions looks contradictory to me, regarding how OT cares about order of the edits.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5exh2v7004giassj23k7vkp)

You can read more about convergence [from the Wikipedia page](https://en.wikipedia.org/wiki/Operational_transformation#Transformation_properties).

The key thing is we need to be able to deal with concurrent edits. If A and B are racing, if A's edit a arrives before B's edit b, the server sees \[a, b\] (this is the ground truth!) and sends b to A - easy case. But B already applied b, so OT provides us a way to transform a to a. such that it can be sent to B and applied \[b, a.\] to yield the same document as \[a, b\].

The method cares about the ordering of operations (the canonical ordering is what the server is there for) but operates in such a way that all clients aren't required to apply the updates in the same order. A bit of a contradiction?

Show more

0

Reply

R

ResponsibleIvoryAlbatross601

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5f0td0e007pny91m35xvmpb)

Thanks for your detailed response.

In your example, my understanding is that if the server sees \[a, b\], it applies OT and transforms those two edits into a converged state 'c' and pushes the new state 'c' to both user A and B. In this case 'c' include the both changes from A and B. If client applies OT, then \[a, 'c'\] and \[b, 'c'\] will still result in 'c'. (Am I right?)

However, from your explanation I can imagine the case that the user has just made the edit 'a' and receives new update 't' from the server while the edit 'a' is still not received by the server. At this point the client can does OT \[a, t\]. But when the server receives the edit 'a', it applied OT for \[t, a\] (?) and pushes new converged state to all the user which should be the same final version for all.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5f11bkg0066m6ecj3qw6tei)

There's a misunderstanding here.

The key to OT is there's no converged state being transferred around. The state is inferred from the operations, but it's the operations that are pushed around (and the transform operates on them). This is important because if you made a 1 character change to a 1MB document, you don't want to push the full state around.

When the central server receives b after a, it needs to run the transform on b because b was created without any knowledge of a - if b is a deletion at position and a is an insertion, we have a problem that the transform has to resolve. We have a parallel problem when we try to transfer a over to B!

Show more

1

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8co2izp00iel2by44llub93)

That explains how OT operates on server. But when and how does OT apply on the client side?

Show more

0

Reply

L

LiveAzureSturgeon811

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5gijuai004va8vviu58vhbn)

Hi Stefan, can you please elaborate a bit more about the interaction between user, API gateway, and the Document Service? I thought users will connect directly to a server running Document Service for collaboration through WebSocket connection, but the diagrams show that users connect to the API gateway not the Document Service, and there is a WebSocket connection between API gateway and the Document Service, so I am a bit confused. Please let me know if I misunderstood anything. Thanks a lot!

Show more

4

Reply

T

ToughGreenVulture620

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm73ul1hb00o6t4ttf3gg30nh)

I have the same question, my understanding is user will connect to the API gateway for the initial request in order to find the corresponding Document Server. Once the Document Server is found the client will directly talk to the Document Server for further edits, but not sure whether that's the case.

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8co3ydb00iol2byk7nukdke)

I think the case is the web socket connection is maintained in user<>API gateway <> Document Service

Show more

1

Reply

C

CostlyAquaMite871

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5h3zh4f00j51osrcn2uyult)

I have questions about this paragraph in the 'Offline Snapshot/Compaction' block in deep dive 2;

"Because this necessarily changes the operations that follow, we need to make sure that what we're writing isn't impacting a live document. This is a tricky distributed transaction that we need to pull off and, because we're using Cassandra for the Document Operations DB, we only have row-level transactions to work with."

1.  Why would compaction of earlier operations affect following operations on the same document? If we compact a sequence of operations into a single op, don't both the compacted op and equivalent sequence of ops put the document into the same state?
2.  Can someone explain more about why a 'distributed transaction' is happening? i.e. which tables are involved in this transaction, and what are the entities that are being changed?

Show more

3

Reply

R

RacialTurquoiseRabbit968

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm9b7i4ek008aad09q1tgt42q)

I have the same doubts. @Stefan, can you please help clarify.

Show more

0

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5hfq33900z8ep4lg0o6airh)

I think, this is not correct statement. "Each socket server will, in the limit, connect to each node of our Redis cluster." With Redis sharded pub/sub, each shard in a Redis cluster could host different channels. Which means, socket servers would be connecting to the Redis instances (primary/replicas) of respective shards. Stefan, could you also please validate and comment on this?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5hgmba3010cep4ldsgqzydj)

The statement is correct, you’re thinking about a single user. With 10 users each connecting to 10 different documents, those topics will be spread across shards of the Redis cluster.

The way to avoid this is to assign users of a given segment of documents to specific servers … which forces us back into the consistent hashing solution.

Show more

1

Reply

F

FederalHarlequinLamprey277

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5q2i2zy01qgkzmh1r0m18yn)

This is really brilliant, thanks for the write-up :)! It seems durability is tradeoff for latency? Coupling these 3 things:-

-   web socket states
-   active document (in-memory)
-   document service(OT & compact functionalities)

Provides lots of opportunities for data loss due to :-

1.  deployment
2.  error in transformation/compact steps
3.  backpressure

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5q3md9701t7wu5hofhcmf6z)

The edits are stored in a durable store and not acknowledged until they're written, so the client can straightforwardly retry.

Show more

0

Reply

F

FederalHarlequinLamprey277

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm5qfjo7400ce83yoqzmoboln)

Got it, thank you Stefan!

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8co7pd900j7l2byqgcmr69y)

what does "active document (in-memory)" store here?

Show more

0

Reply

![Tom Oh](https://lh3.googleusercontent.com/a/ACg8ocL_zg_F9-J6vR8eza8MdsPWwRfQlwUUTYY5HA0PPSkPGaXt1A=s96-c)

Tom Oh

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm642mnfj00uhy43n4hl4bbde)

When User A's wants to insert "," she picks an arbitrary number between 4 and 5 as the position for her comma, in this case she chooses 4.6 -> isn't it 4.3?

Show more

0

Reply

D

DemocraticSalmonLemming370

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm66oo2od029tsdpmebcxl1rj)

I think the deep dive for scaling to millions of web sockets needs to be updated. I would appreciate if someone could explain the following:

-   The pub/sub work flow is that the socket server receives an update from the client and publishes it to the channel corresponding to that documentId. Who is the subscriber here ? If the document service is the subscriber, how does it about know websocket connections for that documentId. According to the writeup, the socket server is the subscriber of pub/sub. If that is the case, how is the update persisted to db ?
-   The second approach about using consistent hash ring, there is no socket server in this approach. I assume the socket connections are managed by the document servers here.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm66vruzz02jetbmrguayo6ix)

Agreed. Updated this in a release which will go out at 1pm today. Ignore the pub/sub workflow, I think it has more problems than I was acknowledging in the writeup.

For the second approach I've updated the language. Yes, the document servers will accept the connections.

Show more

1

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8coawv300jnl2by5hxgymvr)

It would be cleaner if we still put the pub/sub solution and compare cons and pros with consistent hashing in the article. It's easy to get confuse about why it uses consistent hashing instead of pub/sub as in the other channel until reading the comments

Show more

1

Reply

![Miriam Budayr](https://lh3.googleusercontent.com/a/ACg8ocLNSDCllgeEgXUo1RTUTh9m6HGThyWziYMfCrep7llr55LVkzQ=s96-c)

Miriam Budayr

[• 7 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm6797muc005h6dn804sau09u)

Looks like you are missing the "challenges" section under "Offline Snapshot/Compaction".

Show more

0

Reply

Y

yajas

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm6fcniol05s8sagczlo6bp0v)

Is there an issue with the example given for Operational Transforms. If User B wants to delete "!" at char 5, but User A's insert ", world" comes in first, wouldn't the resulting output be "hello world!" instead of "hello, world" ?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm6fj5kek0683iq3nxlqfohjq)

No, the benefit of OT is that it preserves this intent. User B's operation is based on the state when they created it and their intent is to delete the "!".

If we were doing raw index-based character operations what you're saying is indeed what would happen and from a document editing standpoint it's unusable.

Show more

0

Reply

![Zeyu Li](https://lh3.googleusercontent.com/a/ACg8ocKp59vDgUhM-pcS-4RjB_uzguAO58CvpX8GVBAZVEmVNZkLiLsT=s96-c)

Zeyu Li

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm6gqnhrb07bpsagc1ckth4s6)

Seems there are some formatting error and typos.

1.  "Wwe need to scale the number"
2.  "Server: Ea, Eb User A: Ea, Eb User B: Eb, Ea" not aligned and the line space has some issue.
3.  Challenges section missing for Good Solution: Offline Snapshot/Compact

Show more

0

Reply

I

ImmediateOliveBedbug325

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm6hau05x00cluo8si6f36bkc)

> Still, if each document is 50kb, we're looking at 50TB of storage!

How did you calculate this storage estimate for OT?

Oh, and there is a typo below in "Potential Deep Dives" Q1, Para 2.

> Wwe need to scale the number of Document Service servers to the number of concurrent connections.

Show more

0

Reply

![Tommy Loalbo](https://lh3.googleusercontent.com/a/ACg8ocKJIn8OPXYOxiFFjMUkH5UDjWWCbOFuGt2Srsu9sGECWCgexFCq=s96-c)

Tommy Loalbo

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm6jgpz88010k11hv8b2tki8j)

Don't we need some type of 2 phase commit to make sure that we don't fail to update the version Id after updating the snapshot?

Show more

0

Reply

V

VariousBrownNarwhal695

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm6xw2i7000lecv3wo0wtmy4p)

Where are we maintaining the mapping for active users (ones who have established a web socket connection) and the document which was shared with them ? is it this. : "The socket server maintains a map from the document ID to the corresponding websocket for use later." Doesn't it need an explicit table for mapping like in the whatsapp design ?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm6xx6xhj00pghomjkje58xwy)

It doesn't, no. These are transient connections. For the Whatsapp design we need to be able to send messages to offline users.

Show more

1

Reply

F

FormidableOrangePorcupine432

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm70recv904dmhomj48dgmmw5)

@ Stefan Mai: Is there a reason the “Read Mode” feature is not available for this page? I have eye issues, and the read mode significantly helps me. Could the moderators please assist me with this?

Interestingly, the read mode is also unavailable for other pages such as Online Auction, Distributed Cache, Strava, and Distributed Job Scheduler, while it is available on other pages.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm70tbogw01v1po2ukynk7mcm)

What is "Read Mode"?

Show more

0

Reply

F

FormidableOrangePorcupine432

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm70uaibf04hphomj1ov541ji)

Thank you for responding Stefan. This link explains it in detail — https://support.apple.com/en-az/guide/safari/sfri32632/mac

I attempted to use other browsers, such as Firefox, which also have this feature integrated, but unfortunately, these specific pages are not accessible in reader mode on those browsers either, unlike other pages on Hello Interview.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm70ujjoj04gocv3wdawd6exu)

If I had a guess this is related to inline SVGs and read mode's inability to process them. We haven't specifically enabled/disabled anything though!

Show more

0

Reply

F

FormidableOrangePorcupine432

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm70uqup701xppo2upt4iotza)

That makes sense. Is there a way we can reformat these pages so that read-mode can process them?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm70usa5y04hicv3w4015j6oa)

Not sure, let me know if you know of anything. Unfortunately there are dozens of extensions, devices, modes, etc. We try our best to support what we can but it's hard to ensure everything is working.

Show more

0

Reply

S

SocialistGoldVole970

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm74dhskb01bo14jj6bzen5b1)

Hi Stefan, the challenge of "Offline Snapshot/Compact" is missing. Could you somehow add it back? Thanks

Show more

0

Reply

![Misha Borodin](https://lh3.googleusercontent.com/a/ACg8ocITt8_C-XimHao0Gj-BqF28IKe3WXyA8ppWstGFMnewgZtPMQ=s96-c)

Misha Borodin

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm75aukjr026188mxi3r4xfft)

Thanks, Stefan! How would you go about optimizing latencies for geographically distributed users?

Show more

1

Reply

X

XenialIndigoGuanaco422

[• 6 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm75ltchm02lvalw24ztcpx8c)

Why the WebSocket is from Document service to APIGW not client directly?

Show more

0

Reply

O

OfficialAmberGuppy264

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8a795c400ct11p3qioa3i9d)

What parts would you change if the app was not Google Docs but Miro, or something like that? Less text oriented and more dashboardy?

Show more

0

Reply

Y

YearningBeigeWarbler287

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8ayaabm025msj0fdeiynw8a)

1.  Why can't we use transactions for Operational Transformation ? Because it would be slower or is it because it could be inaccurate ?
2.  Why are CRDTs more memory intensive and more computationally expensive ?
3.  Why can't API Gateway maintain a consistent hash ring ? Why do we need redirect from DocumentService ?
4.  What would the data model look like for Operational Transformation case ? Its not clear from the schemas ?
5.  Why does DocumentService need to hold all the operations in memory when the document is live ? Why can't it compact periodically while the document is live to the database ?

Show more

0

Reply

M

ManyAzureImpala710

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8bp7zm5008dtya2633r9yev)

This is great, but would you mind covering how they handle formatting, tables, and images for the Sr folks? Maybe a few points on how Google/MS/notion handles.. Example Google docs are stored as protobuf structured model as small deltas (deletions, insertions).. how it handles redo/undo,.. wanna share some thoughts

Show more

0

Reply

O

OkIndigoTiger161

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8coft3800k5l2bygw5eexga)

For OT, i think we already need a version like documentVersionId. Is this dup or conflict with the documentVersionId discussed in the snapshot?

Show more

0

Reply

![Anmol Gupta](https://lh3.googleusercontent.com/a/ACg8ocJsnWIrH3HYEMeCyV2qpCh0mBZu99hwnIavXAzJKXer3QjiOw=s96-c)

Anmol Gupta

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8epy9lr01phu192iyrbg1j4)

Hi Stefan, This was a great write up. Just a few follow up questions. Can this not be viewed similar to the problem FB Live comments with Kafka as pub sub system to replay the edit log for a new reader joining the document. The kafka partition can also be leveraged to do compaction using flink cluster by reading it from kafka and then updating the DB/S3 in 10-15 seconds.

That way, we will get rid of this additional DB. Do you think any additional tradeoffs with that approach which are not here?

Show more

0

Reply

C

ConservationYellowLeopard727

[• 5 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8kjaf9l008erq22p6hsx4je)

I realized how good that co-location with web-socket server scalability pattern is. It's applicable to the chess.com sort of questions as well.

One note, you cannot really upgrade a redirected web socket connect request, this is due to the web-socket standard constraints. Instead, you would need a dedicated REST API to identify web-socket server path to connect to. So, from the client perspective, the web-socket connection will become 2 separate calls instead of 1 individual call. And as usual there are tradeoffs to consider which go beyond my message :)

Show more

0

Reply

![Tripti Singh](https://lh3.googleusercontent.com/a/ACg8ocLzxC3WW-VQzWWi-w3pyPSxBP3MWF44GSDkQyXSJpUs7FuT8VkHZA=s96-c)

Tripti Singh

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8omx52e00i8ox0psygnqg97)

Instead of a Zookeeper, couldn't we have simply used a Layer 7 Load balancer for correct routing based on consistent hashing scheme?

Show more

0

Reply

![Tripti Singh](https://lh3.googleusercontent.com/a/ACg8ocLzxC3WW-VQzWWi-w3pyPSxBP3MWF44GSDkQyXSJpUs7FuT8VkHZA=s96-c)

Tripti Singh

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8on58fc00inox0p5w7ebzab)

Looks like this was answered somewhere on the implementation of Load balancers in real world. I was reading about HAProxy and was curious if that would be an alternative to Zookeeper.

Show more

0

Reply

P

PeacefulAmaranthTarantula827

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm8vr3lfn02gsad081knqoyoj)

Hey, there is nothing under the "Challenges" section of Compaction Service.

https://imgur.com/a/d5QKsjk

Show more

1

Reply

![Mohit rao](https://lh3.googleusercontent.com/a/ACg8ocICNX8FVE9vC87vUVi-7icTi-h6FgzwBpgML3VmTlpxsfxKlA=s96-c)

Mohit rao

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm91jhtfp00jead0806xy9qdz)

Hi Evan/Stefan - just a suggestion. You guys should add a pre-requisite section to design like this where we require information for technology like OT/CRDT which is usually not known.

TIA

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm91lhl2x00ktad09skzpmzjg)

Good suggestion. We're exploring a few ideas to better show dependency and structure, including some filling out the high-level patterns which we think will be broadly useful. Stay tuned!

Show more

0

Reply

C

chalasaninaveenz

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm9l2uhi9010bad08e6tuhkkv)

Can we use TTL also log size as a trigger for compaction. Set a TTL after few minutes check if document is idle and compact the document in another service altogether. That way document service can be decoupled with compaction.

Show more

0

Reply

S

SeriousOlivePeacock154

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm9mgzf8g00bqad08pkuwk8hi)

i think consistent hash should be done at LB level (envoy) that always match to the same server. the servers are in private network that are being reached through the same url, as a result you cannot really redirect.

Show more

0

Reply

![Nitesh Khandelwal](https://lh3.googleusercontent.com/a/ACg8ocIZTxVh_RKTNDR-qdfzGzacNhpEwe14iBjEa5yPnoZSr77dltMx=s96-c)

Nitesh Khandelwal

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm9nwjyhl00n7ad07iqd4q1f3)

looks like the great solution for web socket is removed?

Show more

1

Reply

Q

QuietPinkOpossum868

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmayofaun01xyad084a89k6n0)

+1, curious why it was removed?

Show more

0

Reply

![learning buddies](https://lh3.googleusercontent.com/a/ACg8ocKb3ulgDowmFom690cLN6oPaql424dPEEDpp0MVyU9nXrZFXg=s96-c)

learning buddies

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm9u9sbq800ssad08t7wee5bk)

Hi Stefan, question on consistency. For non-concurrent users document can have eventual consistency. For concurrent editors, document needs to have strong consistency, i.e. all of them see one state of document after all merges resolved. Can we mention both in nonfunctional requirements?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 4 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm9ubah7h001oad07vq0vjjyb)

Actually, both concurrent and non-concurrent users only need eventual consistency. Strong consistency would mean all users see updates in the same order immediately, which isn't practical (or necessary) for collaborative editing. What matters is that all users eventually converge to the same document state after edits are merged, which is exactly what eventual consistency provides. This is why we use OT/CRDTs - they guarantee convergence without requiring strong consistency.

Show more

0

Reply

![Sanjay Sahoo](https://lh3.googleusercontent.com/a/ACg8ocJRntPRTqVRJ6diTyQca3Is0PrzhJ0aEV27LsjEtlfjpRP9ow=s96-c)

Sanjay Sahoo

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm9ytce8n0106ad099nkohbhs)

When User A's wants to insert "," she picks an arbitrary number between 4 and 5 as the position for her comma, in this case she chooses 4.6: H e l l o , ! 0 1 2 3 4 4.3 5

shouldnt the , be placed on 4.6 but in the diagram we see it as 4.3

Show more

1

Reply

![Sanjay Sahoo](https://lh3.googleusercontent.com/a/ACg8ocJRntPRTqVRJ6diTyQca3Is0PrzhJ0aEV27LsjEtlfjpRP9ow=s96-c)

Sanjay Sahoo

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cm9ytdieb0108ad092qwgfuu7)

and during B's Turn User B does the same, assigning arbitrary numbers between 4 and 5 for each of the characters he wants to insert: H e l l o t h e r e ! 0 1 2 3 4 4.1 4.2 4.4 4.5 4.7 4.8 5

should there be comma in the illustration, the comma added by A?

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmara5pkw009rad08e3coibii)

Good catch on the 4.6 vs 4.3.

For User B, no — they're doing this concurrently! User B hasn't seen User A's edit yet. But we have an answer about where it will land and it's consistent between both A and B.

Show more

0

Reply

![Yue Zhang](https://lh3.googleusercontent.com/a/ACg8ocLGLVITlQMgNPWhq2AGWmebEtDIopou02FCzqrtOgII14GBMw=s96-c)

Yue Zhang

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma3gwqu800ubad08v8r7dzlz)

Hi Stefan! Great writeup as always. One question about the "horizontally scale the Document Service" part, how would the API Gateway work with the WebSocket connection? Are clients creating WebSocket connections directly with servers, or is it client <=> API Gateway <=> Server? In the latter, the API Gateway will be handling the consistent hashing and deciding which server to connect to.

Show more

0

Reply

![Vivek Tiwary](https://lh3.googleusercontent.com/a/ACg8ocJjflqocshnvEvuex4Xh43BNKPTUvLbEueHmmDNtlq6T8MuFNB-pw=s96-c)

Vivek Tiwary

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma52ybl2002iad08ksmzhmsb)

Hi Stefan, I just opened a Google Doc using a different account and was surprised to see that it uses APIs instead of WebSockets. Curious — is there a specific reason we chose the WebSocket approach in our case? May be video wil help for this design. Also

-   Could not relate to api defined in API section with API gateway paths ?
-   Is it AWS API gateway ? Because not all api gateway support out of box web socket connection ?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma5jl86l00asad08eisn6m69)

Google Docs shipped way before Websockets became stable. It doesn't look like the lift to upgrade is worth revisiting their legacy SSE-like solution.

Show more

0

Reply

I

ImprovedMagentaKoala455

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma5eqxcc007iad08bqs8p37a)

I don't understand why we need to apply OT at the client side. If there were conflicts can't we just override what we received from the server? Since the server will send OT applied result?

Show more

2

Reply

I

ImprovedMagentaKoala455

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma5facro00b3ad07hfsk6z9p)

"Remember also that all active documents need to be held in memory in our Document Service." I can't seem to find where this was mentioned explicitly. Does this happen when the first document load is sent to the client and we store it in memory? and when we have no active connection for the document, we remove this document from memory?

If thats the case, if we want to run compaction at a time there are no active connections, how will this happen? Do we do compaction right before document is evicted from memory?

Show more

0

Reply

![Anuj Acharya](https://lh3.googleusercontent.com/a/ACg8ocI32ztAOMQr-RkMlxPVkCD5uWmChNs8uMhBahKz_FWtGDH8ETMvPA=s96-c)

Anuj Acharya

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma9y5kcf02esad08cmvuqic5)

While practicing the problem, before looking at the answer. Based on other problems, I build a S3 notification to Document server such that it can scale. It would be good to know if we go with S3 notification route how, OT can be helped. Basically OT helps to solve the consistency problem and S3 notification solve the scalability problem. Regardless this problem requires a video explanation. I see lot of YouTube video from different people, but for premium subscriber a video on this problem would be really helpful

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cma9yr6b502duad08x048xshk)

Do you mind elaborating on what you're asking? You're saying you are writing edits to S3?

Show more

0

Reply

![KHAL DROGO](https://lh3.googleusercontent.com/a/ACg8ocIKoFmlqebbEU7dj_GAj4l97L6grS4XeBEurwQx5lNddmsmlw=s96-c)

KHAL DROGO

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmaht1bqt009fad08j56891qz)

Could you add a section that expands on the Defining API part of the flow? I've noticed that junior and mid-level interviews tend to spend a significant amount of time on API design. It would be helpful to include something that covers the general aspects of API definition, holistic considerations, and structural principles that can serve as a foundation for designing any API.

Show more

0

Reply

D

DigitalBeigePanda786

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmalsl8j0001wad0858ncekbb)

Is there a need for rate limiter in this setup? Say there are a lot of updates coming in

Show more

0

Reply

E

ElegantMoccasinSwan892

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmar58y4z0019ad09q6q2uhv9)

Hi! Thanks for the write-up. It is helpful.

A bit nit-picky but just to confirm- INSERT(5, ", world") uses 0-based indexing and the delete operation DELETE(6) uses 1-based indexing?

Show more

1

Reply

![Chinmay Bijwe](https://lh3.googleusercontent.com/a/ACg8ocKm1zES8pSU5lg0eScwtAfBlfVruwB1W5nTQuhnRRbmTqbWgba0fw=s96-c)

Chinmay Bijwe

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmaydojso01ktad08k6e1wwm3)

Another solid design and analysis. I'm curious how we'd be able to handle latency for a user editing in the US, and other in Asia. Where would the WS server be located? I'd imagine we don't have latency requirements as crazy as MMORP games, where we need to restrict users to their geo regions.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmaydre1o01rsad08o51f44q6)

For only 2 users, you can minimize _total_ latency by putting it anywhere on the line between them. In practice you'd probably want to minimize the max latency, and you'd do that by choosing a spot right in between them.

But with OT and CRDTs you can tolerate a bit of latency so long as users aren't trying to edit the same words. Awareness/cursors generally solves this problem from a UX perspective: users see that others are trying to edit the same place so they either stay out or expect some turbulence.

Show more

1

Reply

![Chinmay Bijwe](https://lh3.googleusercontent.com/a/ACg8ocKm1zES8pSU5lg0eScwtAfBlfVruwB1W5nTQuhnRRbmTqbWgba0fw=s96-c)

Chinmay Bijwe

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmayeh5w40222ad08dq987rq1)

Makes sense! Thanks for the quick reply ❤️

Show more

0

Reply

![Declan Zhao](https://lh3.googleusercontent.com/a/ACg8ocIqVEkfjElEb4U9UjtqJgiRBxBSM2b0UlGDZknNGhZnqTfjt-QE=s96-c)

Declan Zhao

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmb5x1wmc00fcad0809gjdky4)

Do we need to route via API Gateway to connect with WebSocket Server? I thought there should be direct connection.

Show more

0

Reply

M

magicmaker192

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmbf6qu4001dwad07ta0kvlkp)

Isn't cursor update costly operation hogging CPUs up? Almost as similar as single character update on the document? How does it work in real time?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmbf73i0200dtad07moc561lr)

Which CPUs, client? This is easily throttled to something manageable like every 30-100ms.

Show more

0

Reply

N

NobleRoseVulture249

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmbosftce01s608adle8z6333)

> we want to use something that we can write very quickly in an append-only fashion (for now). Cassandra should do the trick. We'll partition by documentId and order by timestamp

can you please explain why do we choose Cassandra as the DB here? doesn't seems to be very clear to me becuase

1.  How Cassandra is an append-only fashion db? it's just like any other no-sql db to me.
2.  Eventually we will need to query a huge amount of rows to construct the final version of doc. (snapshot job also needs this) and as far as I concern, Cassandra is definitely not good at it.
3.  Speaking of append-only, can Kafka or time-series db be a better choice here?

Show more

0

Reply

![Ahmad Awad](https://lh3.googleusercontent.com/a/ACg8ocIyavhH2pmR4YTSKjR5Cqor7NBvHjJ2N7Mgm-8_umh6DukkKq2D=s96-c)

Ahmad Awad

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmc0cbzdx05ja08ad4qpf2thp)

I have this explanation about scaling stateful connection horizontally. I'd be interested about your feedback.

**Chat App / Google Docs Design** Real-time communication requires immediate responses, making asynchronous processing unsuitable. We scale horizontally and use a Coordinator to ensure that users participating in the same session are connected to the same server. Since WebSocket connections remain open, there's no need for sticky sessions or complex load balancer (LB) logic after the connection is established. A load balancer is only required during the initial connection to route the client to the appropriate server.

**Ride-Sharing / Uber Design** For periodic location updates, no response is required. Therefore, we offload the data to a queue for asynchronous processing, enabling scalability and reducing load on real-time systems.

Show more

0

Reply

D

DoubleCrimsonGecko613

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmc5gobpi00z3ad08w38trczd)

Will each user store the document locally, or will it be saved in a blob storage?

Show more

0

Reply

F

FaintChocolatePheasant616

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmc7b8b5r0db6ad07vz6935ii)

I typically watch the video first and then read the write-up to confirm my understanding. Given the complexity of this topic, having a video—or even multiple videos—would be incredibly helpful for diving deeper into OT and CRDT and overall memory optimization.

Thanks again for incredible content.

Show more

0

Reply

![Mansi Agrawal](https://lh3.googleusercontent.com/a/ACg8ocKzp6cCjSF_8oAD2hWSQ1CY73CLrxC4WGNX7ecMc9qYFyxogOrH=s96-c)

Mansi Agrawal

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmch78d3g042ead08d95ub3yn)

Does the Document Operations DB, store the transformed operations, so that we transform only the latest operation to be inserted on the basis of previous operations?

Show more

0

Reply

![KK](https://lh3.googleusercontent.com/a/ACg8ocKLYy4a8XaThL-Nnj1QyGTaV4iZ2XTEw6SyGGAOimZyD5-wC6hd=s96-c)

KK

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcin6o9f03buad089yqfkg88)

Honestly the miss of how online snapshot / compact operations left so many questions unanswered. Based on the current design, follow up questions I can think of immideiately are:

1.  Document service became a single point of failure. **This service handles not only websocket connections, document operations database facade, but also snapshot compaction opeartions.** I don't think any production level service can bear with this design.
2.  All opeartions now saved in its memory before writing to database due to this compaction opeartions design, how to ensure availability and consistency? A cash can be fatal to the whole service. Whole document is now messed up and how to build it back?
3.  NoSql database is not unlimited storage. Especially if we partition by document id, if I recall this number correctly it's 16GB per partition, so how does a document with long editing history compact all of its opeartions? some sort of corn job needs to go to the database to do it. Even document service is doing the compaction work on write, it's just keep adding new things to the DB never cleaning it.
4.  Due to point 3, a snapshot service that involves s3 is unavoidable, but how does that work exactly need to be discussed as a whole new topic.

Show more

0

Reply

Y

ylucki

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcnu7mxr02w5ad08927ia9ad)

An old, but good example from Google: https://drive.googleblog.com/2010/09/whats-different-about-new-google-docs.html

Show more

1

Reply

A

anandmehrotra420

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcoqtlse05n8ad08a39zrj02)

Hey Stefan, Isn't there an inherent race condition in the **Great Solution: Online Snapshot/Compact Operations**? After we started compaction after the last user disconnected, but before we could write those compactions to DB, we got new connections, and they wrote their edits to the DB. Now if the compaction process wrote to DB, the versions will be out of sync.

Would we need to sent these compacted edits to the main document process and have it apply OT before writing the compacted edits to the DB? Particularly, assigning the correct document Version ID. Do we need to call out that IPC?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcoy2h5z06zdad09rts47tet)

No, we've made it atomic by using the documentVersionId as a pointer for compare-and-swap. If the document service gets a new request while a compact is in progress, it can be aborted.

Show more

0

Reply

Z

zhixuandai1991

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcp6sh3501o3ad07tbfcktfw)

I am still confused why the Client side also need to handle OT.

UserB: Insert(5, "!") -> Hello!

then receive Insert (5, ",world) from the Central Server. Can't userB simply apply it and will see "Hello, world!"? If it performs another OT -> Insert(6, ",world), would userB see "Hello!,world" instead? Because the central server has already applied all the OTs for us, shouldn't each user just apply as it?

Show more

0

Reply

D

DefiniteCopperRabbit337

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcqemz9x0aa1ad080xblpvqq)

Would it be incorrect to say I could use Redis pub/sub to help me keep track of who is online for a document and to figure out who to broadcast edits to? In the diagrams shown it seems to be abstracted away into the document service, if I haven't missed anything.

Show more

0

Reply

C

cygnets\_techs\_2z

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcs4601d0bhrad082n3oh64b)

A small typo fix in the text.

> When User A's wants to insert "," she picks an arbitrary number between 4 and 5 as the position for her comma, in this case she chooses 4.6: H e l l o , ! 0 1 2 3 4 4.3 5

Here the selected value is 4.3 Please update it.

Show more

0

Reply

Q

QuixoticYellowHoverfly945

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcsobpcl0hnuad0794ank8yo)

Why cant we use a pub/sub for real time publishing of cursor information across clients ?

Show more

0

Reply

![Sadhana Parashar](https://lh3.googleusercontent.com/a/ACg8ocKrjyS9cGMaLOf83G2l08SE53oMInCLvmowVfyH7VF6zVbogDlR=s96-c)

Sadhana Parashar

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcu15qli00f0ad08lk68zsks)

Where do I find the video tutorial for the same?

Show more

1

Reply

![Cesar](https://lh3.googleusercontent.com/a/ACg8ocIZr8G7-UnO4S_KrC3pQjrm0vxjhWoDzl5vxdrTKbZ81Fi_Hw=s96-c)

Cesar

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcwh8xum003rad08c97fzd2v)

Typo - "Wwe need to scale the number ..."

Show more

0

Reply

![Cesar](https://lh3.googleusercontent.com/a/ACg8ocIZr8G7-UnO4S_KrC3pQjrm0vxjhWoDzl5vxdrTKbZ81Fi_Hw=s96-c)

Cesar

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmcwie1xk00g6ad08hffy9vd7)

The solution with redirecting the websocket connection to the host that owns the doc has security implications. It would mean that our hosts would now have to be publicly facing instead of just the load balancer. Shouldn't we simply do consistent hashing at the ALB level? We can terminate SSL, hash the docId, and direct the request to the right backend server.

Alternatively, if the ALB doesn't support that, we can have a "notification service" layer that establishes connection with the client, and have the document service push updates through it to the client.

Show more

0

Reply

S

SquealingPeachKiwi982

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmdh4syuj04abad083loknr7x)

Please create the video walkthrough for this!

Show more

1

Reply

![Bharath Kumar Reddy Appareddy](https://lh3.googleusercontent.com/a/ACg8ocIeySLtCQMhaW7G53KkqIX_k0Te2ICZ48NXKBGa806OyaLBVg=s96-c)

Bharath Kumar Reddy Appareddy

[• 25 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmdptlu8c01mqad074hwx5ke0)

Can we get video reference of this, it would be a lot better

Show more

1

Reply

I

InstantVioletHorse636

[• 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cme3k2lsm01yvad07iuza24ee)

What does flip mean in the solution above for compaction operations. For e.g it says "Tell the Document Service to flip the documentVersionId.". Could you explain this a bit more clearly with examples?

Show more

1

Reply

![Akhil Mittal](https://lh3.googleusercontent.com/a/ACg8ocJ49Bjt-GUsJ_aGWMMxw7eU0KF1qPyVoGmDwNj3DXfd5PrJH8aQ=s96-c)

Akhil Mittal

[• 11 days ago• edited 11 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cme9voa9j0c7had08awjrrjko)

@stefan I do not see any challenges for offline snapshot/compact solution. Also what do you mean by flip the docVersionId - 1001 to 0110 flip bits?

Show more

0

Reply

M

mnaguib92

[• 8 days ago• edited 7 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmee8m0z302ncad07wc2yrpre)

Hi Stefan, I have a bit of naive question related to the OT VS CRDT. Can I suggest to the interviewer something simple to lock the line number where a user is updating so no other user can update? Quip does this but I am wondering if this is a different approach or am I missing something and it's actually one of the two approaches.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs#comment-cmeeo4wom00j7ad085h0905oo)

This will really depend on the interviewer, but it's worth a shot. Worst thing is they say no. So you might say "I could use something complex like OTs or CRDTs, but I think a line-level lock is acceptable for this use-case, is that ok?"

P.S. Love quip.

Show more

1

Reply
