# Real-time Updates

Learn about methods for triggering real-time updates in your system design

Real-time Updates

* * *

Hey, everyone. In this video, we're diving into one of the most

Real-Time Updates in System Design: Why They Matter

0:00

Play

Mute

0%

0:00

/

39:53

Premium Content

Closed-Captions On

Chapters

Settings

AirPlay

Google Cast

Enter PiP

Enter Fullscreen

**⚡ Real-time Updates** addresses the challenge of delivering immediate notifications and data changes from servers to clients as events occur. From chat applications where messages need instant delivery to live dashboards showing real-time metrics, users expect to be notified the moment something happens. This pattern covers the architectural approaches to enable low-latency, bidirectional communication.

## The Problem

Consider a collaborative document editor like Google Docs. When one user types a character, all other users viewing the document need to see that change within milliseconds. In apps like this you can't have every user constantly polling the server for updates every few milliseconds without crushing your infrastructure.

The core challenge is establishing efficient, persistent communication channels between clients and servers. Standard HTTP follows a request-response model: clients ask for data, servers respond, then the connection closes. This works great for traditional web browsing but breaks down when you need servers to proactively push updates to clients.

Unfortunately for many candidates, these problems (if they are faced at all) are often solved once by a specialized team. This means that many design challenges will require you to cross a bridge you may not have had the opportunity to build (I've spoken with dozens of candidates with 10+ years of experience for whom this was the case). Don't worry though, in this pattern we'll cover the important things you know to be able to make great decisions in your interview. And who knows, maybe one day you'll be the one building these pieces of your next project!

## The Solution

When systems require real-time updates, push notifications, etc, the solution requires two distinct pieces:

1.  The first "hop": how do we get updates from the server to the client?
    
2.  The second "hop": how do we get updates from the source to the server?
    

Two Hops for Real-time Updates

We'll break down each hop separately as they involve different trade-offs which work together.

### Client-Server Connection Protocols

The first "hop" is establishing efficient communication channels between clients and servers. While traditional HTTP request-response works for a startling number of use-cases, real-time systems frequently need persistent connections or clever polling strategies to enable servers to **push** updates to clients. This is where we get into the nitty-gritty of networking.

#### Networking 101

Before diving into the different protocols for facilitating real-time updates, it's helpful to understand a bit about how networking works — in some sense the problems we're talking about here are just networking problems! Networks are built on a layered architecture (the so-called ["OSI model"](https://en.wikipedia.org/wiki/OSI_model)) which greatly simplifies the world for us application developers who sit on top of it.

##### Networking Layers

In networks, each layer builds on the **abstractions** of the previous one. This way, when you're requesting a webpage, you don't need to know which voltages represent a 1 or a 0 on the network wire - you just need to know how to use the next layer down the stack. While the full networking stack is fascinating, there are three key layers that come up most often in system design interviews:

-   **Network Layer (Layer 3):** At this layer is IP, the protocol that handles routing and addressing. It's responsible for breaking the data into packets, handling packet forwarding between networks, and providing best-effort delivery to any destination IP address on the network. However, there are no guarantees: packets can get lost, duplicated, or reordered along the way.
    
-   **Transport Layer (Layer 4):** At this layer, we have TCP and UDP, which provide end-to-end communication services:
    
    -   TCP is a **connection-oriented** protocol: before you can send data, you need to establish a connection with the other side. Once the connection is established, it ensures that the data is delivered correctly and in order. This is a great guarantee to have but it also means that TCP connections take time to establish, resources to maintain, and bandwidth to use.
        
    -   UDP is a **connectionless** protocol: you can send data to any other IP address on the network without any prior setup. It does not ensure that the data is delivered correctly or in order. Spray and pray!
        
    
-   **Application Layer (Layer 7):** At the final layer are the application protocols like DNS, HTTP, Websockets, WebRTC. These are common protocols that build on top of TCP to provide a layer of abstraction for different types of data typically associated with web applications. We'll get into them in a bit!
    

These layers work together to enable all our network communications. To see how they interact in practice, let's walk through a concrete example of how a simple web request works.

##### Request Lifecycle

When you type a URL into your browser, several layers of networking protocols spring into action. Let's break down how these layers work together to retrieve a simple web page over HTTP. First, we use DNS to convert a human-readable domain name like hellointerview.com into an IP address like 32.42.52.62. Then, a series of carefully orchestrated steps begins:

Simple HTTP Request

1.  **DNS Resolution**: The client starts by resolving the domain name of the website to an IP address using DNS (Domain Name System)[1](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#user-content-fn-dns).
    
2.  **TCP Handshake**: The client initiates a TCP connection with the server using a three-way handshake:
    
    -   **SYN**: The client sends a SYN (synchronize) packet to the server to request a connection.
        
    -   **SYN-ACK**: The server responds with a SYN-ACK (synchronize-acknowledge) packet to acknowledge the request.
        
    -   **ACK**: The client sends an ACK (acknowledge) packet to establish the connection.
        
    
3.  **HTTP Request**: Once the TCP connection is established, the client sends an HTTP GET request to the server to request the web page.
    
4.  **Server Processing**: The server processes the request, retrieves the requested web page, and prepares an HTTP response.
    
5.  **HTTP Response**: The server sends the HTTP response back to the client, which includes the requested web page content.
    
6.  **TCP Teardown**: After the data transfer is complete, the client and server close the TCP connection using a four-way handshake:
    
    -   **FIN**: The client sends a FIN (finish) packet to the server to terminate the connection.
        
    -   **ACK**: The server acknowledges the FIN packet with an ACK.
        
    -   **FIN**: The server sends a FIN packet to the client to terminate its side of the connection.
        
    -   **ACK**: The client acknowledges the server's FIN packet with an ACK.
        
    

While the specific details of TCP handshakes might seem technical, two key points are particularly relevant for system design interviews:

1.  First, each round trip between client and server adds **latency** to our request, including those to establish connections before we send our application data.
    
2.  Second, the TCP connection itself represents **state** that both the client and server must maintain. Unless we use features like HTTP keep-alive, we need to repeat this connection setup process for every request - a potentially significant overhead.
    

Understanding when connections are established and how they are managed is crucial to touching on the important choices relevant for realtime updates.

It's less common recently in BigTech, but it used to be a popular interview question to ask candidates to dive into the details of "what happens when you type (e.g.) hellointerview.com into your browser and press enter?".

Details like these aren't typically a part of a system design interview but it's helpful to understand the basics of networking. It may save you some headaches on the job!

Show More

##### With Load Balancers

In real-world systems, we typically have multiple servers working together behind a load balancer. Load balancers distribute incoming requests across these servers to ensure even load distribution and high availability. There are two main types of load balancers you'll encounter in system design interviews: Layer 4 and Layer 7.

These will have some implications for decisions we'll make later, but let's briefly cover the difference between the two.

###### Layer 4 Load Balancers

Layer 4 load balancers operate at the transport layer (TCP/UDP). They make routing decisions based on network information like IP addresses and ports, without looking at the actual content of the packets. The effect of a L4 load balancer is as-if you randomly selected a backend server and assumed that TCP connections were established directly between the client and that server — this mental model is not far off.

Simple HTTP Request with L4 Load Balancer

Key characteristics of L4 load balancers:

-   Maintain persistent TCP connections between client and server.
    
-   Fast and efficient due to minimal packet inspection.
    
-   Cannot make routing decisions based on application data.
    
-   Typically used when raw performance is the priority.
    

For example, if a client establishes a TCP connection through an L4 load balancer, that same server will handle all subsequent requests within that TCP session. This makes L4 load balancers particularly well-suited for protocols that require persistent connections, like WebSocket connections. At a conceptual level, _it's as if we have a direct TCP connection between client and server which we can use to communicate at higher layers_.

###### Layer 7 Load Balancers

Layer 7 load balancers operate at the application layer, understanding protocols like HTTP. They can examine the actual content of each request and make more intelligent routing decisions.

Simple HTTP Request with L7 Load Balancer

Key characteristics of L7 load balancers:

-   Terminate incoming connections and create new ones to backend servers.
    
-   Can route based on request content (URL, headers, cookies, etc.).
    
-   More CPU-intensive due to packet inspection.
    
-   Provide more flexibility and features.
    
-   Better suited for HTTP-based traffic.
    

For example, an L7 load balancer could route all API requests to one set of servers while sending web page requests to another (providing similar functionality to an [API Gateway](https://www.hellointerview.com/learn/system-design/deep-dives/api-gateway)), or it could ensure that all requests from a specific user go to the same server based on a cookie. The underlying TCP connection that's made to your server via an L7 load balancer is not all that relevant! It's just a way for the load balancer to forward L7 requests, like HTTP, to your server.

While L7 load balancers can help us to not have to worry about lower-level details like TCP connections, we aren't able to ignore the connection-level reality if we want peristent connections to consistent servers.

The choice between L4 and L7 load balancers often comes up in system design interviews when discussing real-time features. There are some L7 load balancers which explicitly support connection-oriented protocols like WebSockets, but generally speaking L4 load balancers are better for WebSocket connections, while L7 load balancers offer more flexibility for HTTP-based solutions like long polling. We'll get into more detail on this in the next section.

Show More

Alright, now that we covered the necessary networking concepts, let's dive into the different approaches for facilitating real-time updates between clients and servers, our first "hop". As a motivating example, let's consider a chat application where users need to receive new messages sent to the chat room they are a part of.

#### Simple Polling: The Baseline

The simplest possible approach is to have the client regularly poll the server for updates. This could be done with a simple HTTP request that the client makes at a regular interval. This doesn't technically qualify as real-time, but it's a good starting point and provides a good contrast for our other methods.

A lot of interview questions don't _actually_ require real-time updates. Think critically about the product and ask yourself whether lower frequency updates (e.g. every 2-5 seconds) would work. If so, you may want to propose a simple, polling-based approach. It's better to suggest a less-than-perfect solution than to fail to implement a complex one.

That said, do make this proposal to your interviewer before pulling the trigger. If they are dead-set on you talking about WebSockets, SSE, or WebRTC, you'll want to know that sooner than later!

Show More

How does it work? It's dead simple! The client makes a request to the server at a regular interval and the server responds with the current state of the world. In our chat app, we would just constantly be polling for "what messages have I not received yet?".

`async function poll() {   const response = await fetch('/api/updates');   const data = await response.json();   processData(data); } // Poll every 2 seconds setInterval(poll, 2000);`

##### Advantages

-   Simple to implement.
    
-   Stateless.
    
-   No special infrastructure needed.
    
-   Works with any standard networking infrastructure.
    
-   Doesn't take much time to explain.
    

This last point is underrated. If the crux of your problem is _not_ real-time updates, you'll want to propose a simple, polling-based approach. You'll preserve your mental energy and interview time for the parts of the system that truly matter.

##### Disadvantages

-   Higher latency than other methods. Updates might be delayed as long as the polling interval + the time it takes to process the request.
    
-   Limited update frequency.
    
-   More bandwidth usage.
    
-   Can be resource-intensive with many clients, establishing new connections, etc.
    

##### When to use simple polling

Simple polling is a great baseline and, absent a problem which specifically requires very-low latency, real-time updates, it's a great solution. It's also appropriate when the window where you need updates is short, like in our [Leetcode system design](https://www.hellointerview.com/learn/system-design/problem-breakdowns/leetcode).

##### Things to Discuss in Your Interview

You'll want to be clear with your interviewer about the trade-offs you're making with polling vs other methods. A good explanation highlights the simplicity of the approach and gives yourself a backdoor if you discover that you need something more sophisticated. "I'm going to start with a simple polling approach so I can focus on X, but we can switch to something more sophisticated if we need to later."

The most common objection from interviewers to polling is either that it's too slow or that it's not efficient. Be prepared to discuss why the polling frequency you've chosen is appropriate and sufficient for the problem. On the efficiency front, it's great to be able to discuss how you can reduce the overhead. One way to do this is to take advantage of HTTP keep-alive connections. Setting an HTTP keep-alive which is longer than the polling interval will mean that, in most cases, you'll only need to establish a TCP connection once which minimizes some of the setup and teardown overhead.

#### Long Polling: The Easy Solution

After a baseline for simple polling, long polling is the easiest approach to achieving near real-time updates. It builds on standard HTTP, making it easy to implement and scale.

The idea is also simple: the client makes a request to the server and the server holds the request open until new data is available. It's as if the server is just taking really long to process the request. The server then responds with the data, finalizes the HTTP requests, and the client immediately makes a **new** HTTP request. This repeats until the server has new data to send. If no data has come through in a long while, we might even return an empty response to the client so that they can make another request.

For our chat app, we would keep making a request to get the _next message_. If there was no message to retrieve, the server would just hold the request open until a new message was sent before responding to us. After we received that message, we'd make a new request for the next message.

1.  Client makes HTTP request to server
    
2.  Server holds request open until new data is available
    
3.  Server responds with data
    
4.  Client immediately makes new request
    
5.  Process repeats
    

`// Client-side of long polling async function longPoll() {   while (true) {     try {       const response = await fetch('/api/updates');       const data = await response.json();              // Handle data       processData(data);     } catch (error) {       // Handle error       console.error(error);              // Add small delay before retrying on error       await new Promise(resolve => setTimeout(resolve, 1000));     }   } }`

The simplicity of the approach hides an important trade-off for high-frequency updates. Since the client needs to "call back" to the server after each receipt, the approach can introduce some extra latency:

Long Polling Latency

> _Assume the latency between the client and server is 100ms._

> _If we have 2 updates which occur within 10ms of each other, with long polling we'll receive the first update 100ms after it occurred (100ms of network latency) but the second update may be up to 290ms after it occurred (90ms for the first response to finish returning, 100ms for the second request to be made, and another 100ms to get the response)._

##### Advantages

-   Builds on standard HTTP and works everywhere HTTP works.
    
-   Easy to implement.
    
-   No special infrastructure needed.
    
-   Stateless server-side.
    

##### Disadvantages

-   Higher latency than alternatives.
    
-   More HTTP overhead.
    
-   Can be resource-intensive with many clients.
    
-   Not suitable for frequent updates due to the issues mentioned above.
    
-   Makes monitoring more painful since requests can hang around for a long time.
    
-   Browsers limit the number of concurrent connections per domain, meaning you may only be able to have a few long-polling connections per domain.
    

##### When to Use Long Polling

Long polling is a great solution for near real-time updates with a simple implementation. It's a good choice when updates are infrequent and a simple solution is preferred. If the latency trade-off of a simple polling solution is at all an issue, long-polling is an obvious upgrade with minimal additional complexity.

Long Polling is a great solution for applications where a long async process is running but you want to know when it finishes, as soon as it finishes - like is often the case in payment processing. We'll long-poll for the payment status before showing the user a success page.

##### Things to Discuss in Your Interview

Because long-polling utilizes the existing HTTP infrastructure, there's not a bunch of extra infrastructure you're going to need to talk through. Even though the polling is "long", you still do need to be specific about the polling frequency. Keep in mind that each hop in your infrastructure needs to be aware of these lengthy requests: you don't want your load balancer hanging up on the client after 30 seconds when your long-polling server is happy to keep the connection open for 60 (15-30s is a pretty common polling interval that minimizes the fuss here).

#### Server-Sent Events (SSE): The Efficient One-Way Street

SSE is an extension on long-polling that allows the server to send a stream of data to the client.

Normally HTTP responses have a header like Content-Length which tells the client how much data to expect. SSE instead uses a special header Transfer-Encoding: chunked which tells the client that the response is a series of chunks - we don't know how many there are or how big they are until we send them. This allows us to move from a single, atomic request/response to a more granular "stream" of data.

With SSE, instead of sending a full response once data becomes available, the server sends a chunk of data and then keeps the request open to send more data as needed. SSE is perfect for scenarios where servers need to push data to clients, but clients don't need to send data back frequently.

In our chat app, we would open up a request to stream messages and then each new message would be sent as a chunk to the client.

##### How SSE Works

1.  Client establishes SSE connection
    
2.  Server keeps connection open
    
3.  Server sends messages when data changes or updates happen
    
4.  Client receives updates in real-time
    

Modern browsers have built-in support for SSE through the EventSource object, making the client-side implementation straightforward.

``// Client-side const eventSource = new EventSource('/api/updates'); eventSource.onmessage = (event) => {   const data = JSON.parse(event.data);   updateUI(data); }; // Server-side (Node.js/Express example) app.get('/api/updates', (req, res) => {   res.setHeader('Content-Type', 'text/event-stream');   res.setHeader('Cache-Control', 'no-cache');   res.setHeader('Connection', 'keep-alive');   const sendUpdate = (data) => {     res.write(`data: ${JSON.stringify(data)}\n\n`);   };   // Send updates when data changes   dataSource.on('update', sendUpdate);   // Clean up on client disconnect   req.on('close', () => {     dataSource.off('update', sendUpdate);   }); });``

##### Advantages

-   Built into browsers.
    
-   Automatic reconnection.
    
-   Works over HTTP.
    
-   More efficient than long polling due to less connection initiation/teardown.
    
-   Simple to implement.
    

##### Disadvantages

-   One-way communication only.
    
-   Limited browser support (not an issue for modern browsers).
    
-   Some proxies and networking equipment don't support streaming. Nasty to debug!
    
-   Browsers limit the number of concurrent connections per domain, meaning you may only be able to have a few SSE connections per domain.
    
-   Makes monitoring more painful since requests can hang around for a long time.
    

##### When to Use SSE

SSE is a great upgrade to long-polling because it eliminates the issues around high-frequency updates while still building on top of standard HTTP. That said, it comes with lesser overall support because you'll need both browsers and and/all infra between the client and server to support streaming responses.

A very popular use-case for SSE today is AI chat apps which frequently involve the need to stream new tokens (words) to the user as they are generated to keep the UI responsive.

An example of an infra gap is that many proxies and load balancers don't support streaming responses. In these cases, the proxy will try to buffer the response until it completes - which effectively blocks our stream in an annoying, opaque way that is hard to debug! [2](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#user-content-fn-sse)

As an aside: most interviewers will not be familiar with the [infrastructure considerations associated with SSE](https://dev.to/miketalbot/server-sent-events-are-still-not-production-ready-after-a-decade-a-lesson-for-me-a-warning-for-you-2gie) and aren't going to ask you detailed questions about them. But if the role you're interviewing for is very frontend-centric be prepared in case they expect you to know your stuff!

##### Things to Discuss in Your Interview

SSE rides on existing HTTP infrastructure, so there's not a lot of extra infrastructure you'll need to talk through. You also don't have a polling interval to negotiate or tune.

Most SSE connections won't be super-long-lived (e.g. 30-60s is pretty typical), so if you need to send messages for a longer period you'll need to talk about how clients re-establish connections and how they deal with the gaps in between. The [SSE standard](https://html.spec.whatwg.org/multipage/server-sent-events.html) includes a "last event ID" which is intended to cover this gap, and the EventSource object in browsers explicitly handles this reconnection logic. If a client loses its connection, it can reconnect and provide the last event ID it received. The server can then use that ID to send all the events that occurred while the client was disconnected.

#### Websockets: The Full-Duplex Champion

WebSockets are the go-to choice for true bi-directional communication between client and server. If you have high frequency writes _and_ reads, WebSockets are the champ.

##### How WebSockets Works

Websockets build on HTTP through an "upgrade" protocol, which allows an existing TCP connection to change L7 protocols. This is super convenient because it means you can utilize some of the existing HTTP session information (e.g. cookies, headers, etc.) to your advantage.

Just because clients can upgrade from HTTP to WebSocket doesn't mean that the infrastructure will support it. Every piece of infrastructure between the client and server will need to support WebSocket connections.

Once a connection is established, both client and server can send "messages" to each other which are effectively opaque binary blobs. You can shove strings, JSON, Protobufs, or anything else in there. Think of WebSockets like a TCP connection with some niceties that make establishing the connection easier, especially for browsers.

1.  Client initiates WebSocket handshake over HTTP
    
2.  Connection upgrades to WebSocket protocol
    
3.  Both client and server can send messages
    
4.  Connection stays open until explicitly closed
    

For our chat app, we'd connect to a WebSocket endpoint over HTTP, sharing our authentication token via cookies. The connection would get upgraded to a WebSocket connection and then we'd be able to receive messages back to the client over the connection as they happen. Bonus: we'd also be able to send messages to other users in the chat room!

`// Client-side const ws = new WebSocket('ws://api.example.com/socket'); ws.onmessage = (event) => {   const data = JSON.parse(event.data);   handleUpdate(data); }; ws.onclose = () => {   // Implement reconnection logic   setTimeout(connectWebSocket, 1000); }; // Server-side (Node.js/ws example) const WebSocket = require('ws'); const wss = new WebSocket.Server({ port: 8080 }); wss.on('connection', (ws) => {   ws.on('message', (message) => {     // Handle incoming messages     const data = JSON.parse(message);     processMessage(data);   });   // Send updates to client   dataSource.on('update', (data) => {     ws.send(JSON.stringify(data));   }); });`

##### Extra Challenges

Because the Websocket is a _persistent connection_, we need our infrastructure to support it. Some L7 load balancers support websockets, but support is generally spotty (remember that L7 load balancers aren't guaranteeing we're using the same TCP connection for each incoming request). L4 load balancers will support websockets natively since the same TCP connection is used for each request.

When we have long-running connections we have another problem: deployments. When servers are redeployed, we either need to sever all old connections and have them reconnect or have the new servers take over and keep the connections alive. Generally speaking you should prefer the former since it's simpler, but it does have some ramifications on how "persistent" you expect the connection to be. You also need to be able to handle situations where a client needs to reconnect and may have missed updates while they were disconnected.

Finally, balancing load across websocket servers can be more complex. If the connections are truly long-running, we have to "stick with" each allocation decision we made. If we have a load balancer that wants to send a new request to a different server, it can't do that if it would break an existing websocket connection.

Because of all the issues associated with _stateful_ connections, many architectures will terminate WebSockets into a WebSocket service early in their infrastructure. This service can then handle the connection management and scaling concerns and allows the rest of the system to remain _stateless_. The WebSocket service is also less likely to change meaning it requires less deployments which churn connections.

WebSocket Reference Architecture

##### Advantages

-   Full-duplex (read and write) communication.
    
-   Lower latency than HTTP due to reduced overhead (e.g. no headers).
    
-   Efficient for frequent messages.
    
-   Wide browser support.
    

##### Disadvantages

-   More complex to implement.
    
-   Requires special infrastructure.
    
-   Stateful connections, can make load balancing and scaling more complex.
    
-   Need to handle reconnection.
    

##### When to Use WebSockets

Generally speaking, if you need **high-frequency**, bi-directional communication, you're going to want to use WebSocket. I'm emphasizing high-frequency here because you can always make additional requests/connections for writes: a very common pattern is to have SSE subscriptions for updates and do writes over simple HTTP POST/PUT whenever they occur.

I often find candidates too eagerly adopting Websockets when they could be using SSE or simple polling instead. Because of the additional complexity and infra lift, you'll want to defer to SSE unless you have a specific need for this bi-directional communication.

##### Things to Discuss in Your Interview

Websockets are a powerful tool, but they do come with a lot of complexity. You'll want to talk through how you'll manage connections and deal with reconnections. You'll also need to consider how your deployment strategy will handle server restarts.

Managing _statefulness_ is a big part of the conversation. Senior/staff candidates will frequently talk about how to minimize the spread of state across their architecture.

There's also a lot to discuss about how to scale WebSocket servers. Load can be uneven which can result in hotspots and failures. Using a "least connections" strategy for the load balancer can help, as well as minimizing the amount of work the WebSocket servers need to do as they process messages. Using the reference architecture above and offloading more intensive processing to other services (which can scale independently) can help.

#### WebRTC: The Peer-to-Peer Solution

Our last option is the most unique. WebRTC enables direct peer-to-peer communication between browsers, perfect for video/audio calls and some data sharing like document editors.

Clients talk to a central "signaling server" which keeps track of which peers are available together with their connection information. Once a client has the connection information for another peer, they can try to establish a direct connection without going through any intermediary servers.

In practice, most clients don't allow inbound connections for security reasons (the exception would be servers which broadcast their availability on specific ports at specific addresses) using devices like NAT (network address translation). So if we stopped there, most peers wouldn't be able to "speak" to each other.

The WebRTC standard includes two methods to work around these restrictions:

-   **STUN**: "Session Traversal Utilities for NAT" is a protocol and a set of techniques like "hole punching" which allows peers to establish publically routable addresses and ports. I won't go into details here, but as hacky as it sounds it's a standard way to deal with NAT traversal and it involves repeatedly creating open ports and sharing them via the signaling server with peers.
    
-   **TURN**: "Traversal Using Relays around NAT" is effectively a relay service, a way to bounce requests through a central server which can then be routed to the appropriate peer.
    

WebRTC Setup

In practice, the signaling server is relatively lightweight and isn't handling much of the bandwidth as the bulk of the traffic is handled by the peer-to-peer connections. But interestingly the signaling server does effectively act as a real-time update system for its clients (so they can find their peers and update their connection info) so it either needs to utilize WebSockets, SSE, or some other approach detailed above.

For our chat app, we'd connect to our signaling server over a WebSocket connection to find all of our peers (others in the chat room). Once we'd identified them and exchanged connection information, we'd be able to establish direct peer-to-peer connections with them. Chat messages would be broadcast by room participants to all of their peers (or, if you want to be extra fancy, bounced between participants until they settle).

##### How WebRTC Works

Ok, but how does it work?

1.  Peers discover each other through signaling server.
    
2.  Exchange connection info (ICE candidates)
    
3.  Establish direct peer connection, using STUN/TURN if needed
    
4.  Stream audio/video or send data directly
    

Pretty simple, apart from the acronyms and NAT traversal.

`// Simplified WebRTC setup async function startCall() {   const pc = new RTCPeerConnection();      // Get local stream   const stream = await navigator.mediaDevices.getUserMedia({     video: true,     audio: true   });      // Add tracks to peer connection   stream.getTracks().forEach(track => {     pc.addTrack(track, stream);   });      // Create and send offer   const offer = await pc.createOffer();   await pc.setLocalDescription(offer);      // Send offer to signaling server   signalingServer.send(offer); }`

##### When to Use WebRTC

WebRTC is the most complex and heavyweight of the options we've discussed. It's overkill for most real-time update use cases, but it's a great tool for scenarios like video/audio calls, screen sharing, and gaming.

The notable exception is that it can be used to reduce server load. If you have a system where clients need to talk to each other frequently, you could use WebRTC to reduce the load on _your_ servers by having clients establish their own connections. [Canva took this approach with presence/pointer sharing in their canvas editor](https://www.canva.dev/blog/engineering/realtime-mouse-pointers/) and it's a popular approach from collaborative document editing like [Google Docs](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs) when used in conjunction with CRDTs which are better suited for a peer-to-peer architecture.

##### Advantages

-   Direct peer communication
    
-   Lower latency
    
-   Reduced server costs
    
-   Native audio/video support
    

##### Disadvantages

-   Complex setup (> WebSockets)
    
-   Requires signaling server
    
-   NAT/firewall issues
    
-   Connection setup delay
    

##### Things to Discuss in Your Interview

If you're building a WebRTC app in a system design interview, it should be really obvious why you're using it. Either you're trying to do video conferencing or the scale strictly requires you to have clients talk to each other directly.

Some interviews will get cute and introduce unrealistic constraints to try to get you to think outside the box, like "the system must run on Raspberry Pi". These _might_ be a case where a peer-to-peer architecture makes sense, but tread carefully.

Having some knowledge of the infra requirements (STUN/TURN, signaling servers, etc.) will give you the flexibility to make the best design decision for your system. You'll also want to speak pretty extensively about the communication patterns between peer clients and any eventual synchronization to a central server (almost all design questions will have some calling home to the mothership to store data or report results).

#### Overview

There are a lot of options for delivering events from the server to the client. Being familiar with the trade-offs associated with each will give you the flexibility to make the best design decision for your system. If you're in a hurry, the following flowchart will help you choose the right tool for the job.

-   If you're not latency sensitive, **simple polling** is a great baseline. You should probably start here unless you have a specific need in your system.
    
-   If you don't need bi-directional communication, **SSE** is a great choice. It's lightweight and works well for many use cases. There are some infrastructure considerations to keep in mind, but they're less invasive than with WebSocket and generally interviewers are less familiar with them or less critical if you don't address them.
    
-   If you need frequent, bi-directional communication, **WebSocket** is the way to go. It's more complex, but the performance benefits are huge.
    
-   Finally, if you need to do audio/video calls, **WebRTC** is the only way to go. In some instances peer-to-peer collaboration can be enhanced with WebRTC, but you're unlikely to see it in a system design interview.
    

Client Updates Flowchart

But now that we have the first hop out of the way, let's talk about how updates propagate from their source to the server in question.

### Server-Side Push/pull

Now that we've established our options for the hop from server to client (Simple Polling, Long-Polling, SSE, WebSockets, WebRTC), let's talk about how we can propagate updates from the source to the server.

Server-Side Push/Pull

Invariably our system is somehow _producing_ updates that we want to propagate to our clients. This could be other users making edits to a shared documents, drivers making location updates, or friends sending messages to a shared chat.

Making sure these updates get to their ultimate destination is closely tied to how we propagate updates from the source to the server. Said differently, we need a **trigger**.

When it comes to triggering, there's three patterns that you'll commonly see:

1.  Pulling via Polling
    
2.  Pushing via Consistent Hashes
    
3.  Pushing via Pub/Sub
    

#### Pulling with Simple Polling

With Simple Polling, we're using a **pull-based** model. Our client is constantly asking the server for updates and the server needs to maintain the state necessary to respond to those requests. The most common way to do this is to have a database where we can store the updates (e.g. all of the messages in the chat room), and from this database our clients can pull the updates they need when they can. For our chat app, we'd basically be polling for "what messages have been sent to the room with a timestamp larger than the last message I received?".

Pulling with Simple Polling

Remember with polling we're tolerating delay! We use the poll itself as the trigger, even though the actual update may have occurred some time prior.

The nice thing about this from a system design perspective is that we've _decoupled_ the source of the update from the client receiving it. The line that receives the updates is interrupted (by the DB) from the line that produces them - data is not _flowing_ from the producer to the consumer. The downside is that we've lost the real-time aspect of our updates.

##### Advantages

-   Dead simple to implement.
    
-   State is constrained to our DB.
    
-   No special infrastructure.
    
-   Doesn't take much time to explain.
    

##### Disadvantages

-   High latency.
    
-   Excess DB load when updates are infrequent and polling is frequent.
    

##### When to Use Pull-Based Polling

Pull-based polling is great when you want your user experience to be somewhat more responsive to changes that happen on the backend, but responding quickly is not the main thing. Generally speaking, if you need real-time updates this is not the best approach, but again there are a lot of use-cases where real-time updates are actually not required!

##### Things to Discuss in Your Interview

When you're using Pull-based Polling, you'll want to talk about how you're storing the updates. If you're using a database, you'll want to discuss how you're querying for the updates and how that might change given your load.

In many instances where this approach might be used, the number of clients can actually be quite large. If you have a million clients polling every 10 seconds, you've got 100k TPS of read volume! This is easy to forget about.

#### Pushing via Consistent Hashes

The remaining approaches involve **pushing** updates to the clients. In many of the client update mechanisms that we discussed above (long-polling, SSE, WebSockets) the client has a persistent connection to one server and that server is responsible for sending updates to the client.

But this creates a problem! For our chat application, in order to send a message to User C, we need to know which server they are connected to.

Push Problems

Ideally, when an a message needs to be sent, I would:

1.  Figure out which server User C is connected to.
    
2.  Send the message to that server.
    
3.  That server will look up which (websocket, SSE, long-polling) request is associated with User C.
    
4.  The server will then write the message via the appropriate connection.
    

There are two common ways to handle this situation, and the first is to use **hashing**. Let's build up our intuition for this in two steps.

##### Simple Hashing

Our first approach might be to use a simple modulo operation to figure out which server is responsible for a given user. Then, we'll always have 1 server who "owns" the connections for that user.

To do this, we'll have a central service that knows how many servers there are N and can assign them each a number 0 through N-1. This is frequently Apache [ZooKeeper](https://www.hellointerview.com/learn/system-design/deep-dives/zookeeper) or Etcd which allows us to manage this metadata and allows the servers to keep in sync as it updates, though in practice there are many alternatives.

We'll make the server number n responsible for user u % N. When clients initially connect to our service, we can either: a. Directly connect them to the appropriate server (e.g. by having them know the hash, N, and the server addressess associated with each index). b. Have them randomly connect to any of the servers and have that server redirect them to the appropriate server based on internal data.

Connecting to the Right Server

When a client connects, the following happens:

1.  The client connects to a random server.
    
2.  The server looks up the client's hash in Zookeeper to figure out which server is responsible for them.
    
3.  The server redirects the client to the appropriate server.
    
4.  The client connects to the correct server.
    
5.  The server adds that client to a map of connections.
    

Now we're ready to send updates and messages!

When we want to send messages to User C, we can simply hash the user's id to figure out which server is responsible for them and send the message there.

Sending Updates to the Right Server

1.  Our Update Server stays connected to Zookeeper and knows the addresses of all servers and the modulo N.
    
2.  When the Update Server needs to send a message to User C, it can hash the user's id to figure out which server is responsible for them (Server 2) and sends the message there.
    
3.  Server 2 receives the message, looks up which connection is associated with User C, and sends the message to that connection.
    

This approach works because we always know that a single server is responsible for a given user (or entity, or ID, or whatever). All inbound connections go to that server and, if we want to use the connection associated with that entity, we know to pass it to that server for forwarding to the end client.

##### Consistent Hashing

The hashing approach works great when N is fixed, but becomes problematic when we need to scale our service up or down. With simple modulo hashing, changing the number of servers would require almost all users to disconnect and reconnect to different servers - an expensive operation that disrupts service.

[Consistent hashing](https://www.hellointerview.com/learn/system-design/deep-dives/consistent-hashing) solves this by minimizing the number of connections that need to move when scaling. It maps both servers and users onto a hash ring, and each user connects to the next server they encounter when moving clockwise around the ring.

Consistent Hash Ring

When we add or remove servers, only the users in the affected segments of the ring need to move. This greatly reduces connection churn during scaling operations.

Consistent Hash Ring

##### Advantages

-   Predictable server assignment
    
-   Minimal connection disruption during scaling
    
-   Works well with stateful connections
    
-   Easy to add/remove servers
    

##### Disadvantages

-   Complex to implement correctly
    
-   Requires coordination service (like Zookeeper)
    
-   All servers need to maintain routing information
    
-   Connection state is lost if a server fails
    

##### When to Use Consistent Hashing

Consistent hashing is ideal when you need to maintain persistent connections (WebSocket/SSE) and your system needs to scale dynamically. It's particularly valuable when each connection requires significant server-side state that would be expensive to transfer between servers.

For example, in [the Google Docs design](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs), connections are associated with specific documents that require substantial state to maintain collaborative editing functionality. Consistent hashing helps keep that state on a single server while allowing for scaling.

However, if you're just passing small messages to clients without much associated state, you're probably better off using the next approach: Pub/Sub.

##### Things to Discuss in Your Interview

If you introduce a consistent hashing approach in an interview, you'll want to be able to discuss not only how the updates are routed (e.g. a cooordination service like Zookeeper or etcd). Interviewers are usually interested to understand how the system scales: what happens when we need to increase or decrease the number of nodes. For those instances, you'll want to be able to share your knowledge about consistent hashing but also talk about the orchestration logic necessary to make it work. In practice, this usually means:

1.  Signaling the beginning of a scaling event. Recording both the old and new server assignments.
    
2.  Slowly disconnecting clients from the old server and having them reconnect to their newly assigned server.
    
3.  Signaling the end of the scaling event and updating the coordination service with the new server assignments.
    
4.  In the interim, having messages which are sent to both the old and new server until they're fully transitioned.
    

The mechanics of discovering the initial server assignments is also interesting. Having clients know about the internal structure of your system can be problematic, but there are performance tradeoffs associated with redirecting clients to the correct server or requiring them to do a round-trip to a central server to look up the correct one. Especially during scaling events, any central registration service may become a bottleneck so it's important to discuss the tradeoffs with your interviewer.

#### Pushing via Pub/Sub

Another approach to triggering updates is to use a **pub/sub** model. In this model, we have a single service that is responsible for collecting updates from the source and then broadcasting them to all interested clients. Popular options here include Kafka and Redis.

The pub/sub service becomes the biggest source of _state_ for our realtime updates. Our persistent connections are now made to lightweight servers which simply subscribe to the relevant topics and forward the updates to the appropriate clients. I'll refer to these servers as _endpoint_ servers.

When clients connect, we don't need them to connect to a specific endpoint server (like we did with consistent hashing) and instead can connect to any of them. Once connected, the endpoint server will register the client with the pub/sub server so that any updates can be sent to them.

Pub/Sub

On the connection side, the following happens:

1.  The client establishes a connection with an endpoint server.
    
2.  The endpoint server registers the client with the Pub/Sub service, often by creating a topic, subscribing to it, and keeping a mapping from topics to the connections associated with them.
    

Pub/Sub Message Sending

On the update broadcasting side, the following happens:

1.  Updates are pushed to the Pub/Sub service to the relevant topic.
    
2.  The Pub/Sub service broadcasts the update to all clients subscribed to that topic.
    
3.  The endpoint server receives the update, looks up which client is subscribed to that topic, and forwards the update to that client over the existing connection.
    

For our chat application, we'll create a topic for each user. When the client connects to our endpoint, it will subscribe to the topic associated with the connected user. When we need to send messages, we publish them to that user's topic and the Pub/Sub service will broadcast them to all of the subscribed servers - then these servers will forward them to the appropriate clients over the existing connections.

##### Advantages

-   Managing load on endpoint servers is easy, we can use a simple load balancer with "least connections" strategy.
    
-   We can broadcast updates to a large number of clients efficiently.
    
-   We minimize the proliferation of state through our system.
    

##### Disadvantages

-   We don't know whether subscribers are connected to the endpoint server, or when they disconnect.
    
-   The Pub/Sub service becomes a single point of failure and bottleneck.
    
-   We introduce an additional layer of indirection which can add to latency.
    
-   There exist many-to-many connections between Pub/Sub service hosts and the endpoint servers.
    

##### When to Use Pub/Sub

Pub/Sub is a great choice when you need to broadcast updates to a large number of clients. It's easy to set up and requires little overhead on the part of the endpoint servers. The latency impact is minimal (<10ms). If you don't need to respond to connect/disconnect events or maintain a lot of state associated with a given client, this is a great approach.

##### Things to Discuss in Your Interview

If you're using a pub/sub model, you'll probably need to talk about the single point of failure and bottleneck of the pub/sub service. Redis cluster is a popular way to scale pub/sub service which involves sharding the subscriptions by their key across multiple hosts. This scales up the number of subscriptions you can support and the throughput.

Introducing a cluster for the Pub/Sub component means you'll manage the many-to-many connections between the pub/sub service and the endpoint servers (each endpoint server will be connected to all hosts in the cluster). In some cases this can be managed by carefully choosing topics to partition the service, but in many cases the number of nodes in the cluster is small.

For inbound connections to the endpoint servers, you'll probably want to use a load balancer with a "least connections" strategy. This will help ensure that you're distributing the load across the servers in the cluster. Since the connection itself (and the messages sent across it) are effectively the only resource being consumed, load balancing based on connections is a great way to manage the load.

## When to Use in Interviews

Real-time updates appear in almost every system design interview that involves user interaction or live data. Rather than waiting for the interviewer to ask about real-time features, proactively identify where immediate updates matter and address them in your initial design.

A strong candidate recognizes real-time requirements early. When designing a chat application, immediately acknowledge that "messages need to be delivered instantly to all participants - I'll address that with WebSockets." For collaborative editing, mention that "character-level changes need sub-second propagation between users."

### Common Interview Scenarios

**[Chat Applications](https://www.hellointerview.com/learn/system-design/problem-breakdowns/whatsapp)** - The classic real-time use case. Messages must appear instantly across all participants. WebSockets handle the bidirectional communication perfectly, while pub/sub distributes messages to the right servers. Consider message ordering, typing indicators, and presence status.

**[Live Comments](https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-live-comments)** - High-volume, real-time social interaction during live events. Millions of viewers commenting simultaneously creates extreme fan-out problems. Hierarchical aggregation and careful batching prevent system overload while maintaining the live feel.

**[Collaborative Document Editing](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs)** - Character-level changes need instant propagation between users. WebSockets provide the low-latency communication, while operational transforms or CRDTs handle conflict resolution. User cursors and selection highlighting add additional real-time complexity.

Collaborative editing commonly requires us to not only deal with getting updates to clients with low latency and high frequency, but also to deal with conflicts and ensure that the state of the document is consistent. If users can be typing on top of one another, they often need to be able to deal with the conflicts that arise. We talk about two approaches for dealing with this in the [Google Docs](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs) breakdown: CRDTs and Operational Transforms.

Show More

**[Live Dashboards and Analytics](https://www.hellointerview.com/learn/system-design/problem-breakdowns/uber)** - Business metrics and operational data that changes constantly. Server-Sent Events work well for one-way data flow from servers to dashboards. Consider data aggregation intervals and what constitutes "real-time enough" for business decisions.

**Gaming and Interactive Applications** - Multiplayer games need the lowest latency possible. WebRTC enables peer-to-peer communication for reduced latency, while WebSockets handle server coordination. Consider different update frequencies for different game elements.

### When NOT to Use

Avoid real-time updates when you can get away with a simple polling model. If you're not latency sensitive, polling is a great baseline and minimizes complexity — a property highly valued in senior+ interviews. By polling you avoid both hops: you don't need to worry about the client->server protocols AND you don't have to handle propagation from the event source.

## Common Deep Dives

Interviewers love to probe the operational challenges and edge cases of real-time systems. Here are the most common follow-up questions you'll encounter.

### "How do you handle connection failures and reconnection?"

Real-world networks are unreliable. Mobile users lose connections constantly, WiFi drops out, and servers restart. Your real-time system needs graceful degradation and recovery.

The key challenge is detecting disconnections quickly and resuming without data loss. WebSocket connections don't always signal when they break - a client might think it's connected while the server has already cleaned up the connection. Implementing heartbeat mechanisms helps detect these "zombie" connections.

For recovery, you need to track what messages or updates a client has received. When they reconnect, they should get everything they missed. This often means maintaining a per-user message queue or implementing sequence numbers that clients can reference during reconnection. Using [Redis](https://www.hellointerview.com/learn/system-design/deep-dives/redis#redis-for-event-sourcing) streams for this is a popular option.

### "What happens when a single user has millions of followers who all need the same update?"

This is the classic "celebrity problem" in real-time systems. When a celebrity posts, millions of users need that update simultaneously. Naive approaches create massive fan-out that can crash your system.

The solution involves strategic caching and hierarchical distribution. Instead of writing the update to millions of individual user feeds, cache the update once and distribute through multiple layers. Regional servers can pull the update and push to their local clients, reducing the load on any single component. More details on this in the [Batching and Hierarchical Aggregation](https://www.hellointerview.com/learn/system-design/patterns/scaling-writes#batching-and-hierarchical-aggregation) section of the Scaling Writes pattern.

Hierarchical Aggregation

### "How do you maintain message ordering across distributed servers?"

When multiple servers handle real-time updates, ensuring consistent ordering becomes complex. Two messages sent milliseconds apart might arrive out of order if they travel different network paths or get processed by different servers.

Vector clocks or logical timestamps help establish ordering relationships between messages. Each server maintains its own clock, and messages include timestamp information that helps recipients determine the correct order.

For critical ordering requirements, you might need to funnel all related messages through a single server or partition. This trades some scalability for consistency guarantees, but simplifies the ordering problem significantly.

For most _product_\-style system design interviews, using a single server or partition is the way to go. There's a place for vector clocks and other techniques but they most often apply to deep infra rather than a question like "Design an Online Auction System". If all your messages make their way to a single host, stamping them with the correct timestamp and establishing a total order is straightforward.

## Conclusion

Real-time updates are among the most challenging patterns in system design, appearing in virtually every interactive application from messaging to collaborative editing. The key insight is that real-time systems require solving two distinct problems: client-server communication protocols and server-side update propagation.

Start simple and escalate based on requirements. If polling every few seconds meets your needs, don't jump to complex WebSocket architectures. Most candidates over-engineer real-time solutions when simpler approaches would suffice. However, when true real-time performance is required, understanding the trade-offs between protocols becomes crucial.

For client communication, SSE and WebSockets handle most real-time scenarios effectively. SSE works brilliantly for one-way updates like live dashboards, while WebSockets excel when you need bidirectional communication. Both are well-supported and understood by most infrastructure teams.

On the server side, pub/sub systems provide the best balance of simplicity and scalability for most applications. They decouple update sources from client connections, making your system easier to reason about and scale. Reserve consistent hashing approaches for scenarios where connection state management becomes a primary concern.

Overall real-time update applications bring a lot of interesting complexity to system design: low latency, scaling, networking issues, and the need to manage multiple services and state. By learning about the options available to you, you'll be able to make the best design decision for your system and communicate your reasoning to your interviewer. Good luck!

###### Test Your Knowledge

Take a quick 15 question quiz to test what you've learned.

Start Quiz

## Footnotes

1.  DNS technically can run over TCP or UDP, but we'll exclude that for simplicity in this illustration. [↩](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#user-content-fnref-dns)
    
2.  We use SSE extensively for Hello Interview and the amount of time we've spent dealing with networking edge cases is mind boggling. [↩](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#user-content-fnref-sse)
    

Mark as read

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

T

TrickyTanIguana957

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm41do9sl00grub2hzygyp739)

This is an awesome deep-dive!

Show more

68

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm42fjat700g3krsrixnf6pu9)

Thanks! Glad you like it.

Show more

9

Reply

![sahil chug](https://lh3.googleusercontent.com/a/ACg8ocLlYRqPtHso0b1gwGbbZzYvNFHhSSqCVhauGf89PfOeH4Rh9-KnWA=s96-c)

sahil chug

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm43cooiq014jkrsr232gjkot)

This is just amazing!!

Show more

1

Reply

M

manivannan.sivaraj

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm45296rm002k3o1hz4590kkn)

Wonderful article! Thanks for putting it all together in one :)

Show more

4

Reply

N

NetBlackMite212

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm45vs32100oo3o1hrndlbmeq)

Great deep dive! Qq: what are some common strategies to discuss pub/sub single point of failures?

Show more

2

Reply

N

NetBlackMite212

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm45vu0a000qt4dcrwflhocb4)

And also if we use redis pub/sub, what are some common ways to handle messages being dropped with redis’ at most one message sending

Show more

4

Reply

C

CurlyAmberAlpaca598

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm492n48s0007pi961l4tplys)

How to scale websockets & SSE to handle millions of connections simulateneously?

Show more

6

Reply

A

AdministrativePeachLocust788

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4anmd5900igsycj2fcn4rph)

Thank you for putting this together! I especially appreciate how the article is structured: it briefly touches on various key points while providing an in-depth analysis of the pros and cons of specific techniques. The focus on tradeoffs and tailored recommendations is really helpful.

This contrasts with the general, textbook-style system design materials. Those materials always make me feel like ‘if all you have is a hammer, everything looks like a nail.’ It’s really consuming to explore every system design topic to build a comprehensive knowledge map, but I think this deep dive distills the essence of that work for me. Great work!

Show more

4

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4jcivkt00rx12z8nf78izzz)

Glad it was helpful! Always trying to dial in the right level of detail and was a bit worried this got overtechnical.

Show more

1

Reply

A

aniu.reg

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4bohapp00b91i3rgrk3hm3t)

Thank you for your hard work. You mentioned twice "Makes monitoring more painful since requests can hang around for a long time." Can you explain a little bit more? Thank you!

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4jci4tb00tdfop8fegagyt5)

Most monitoring packages for webservers assumes that the time to deliver a request is latency to be eliminated - so having requests with 15s load times can distort measurement. You'll need different telemetry to be able to deal with streaming connections!

Show more

2

Reply

I

IndividualGrayToucan573

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4d3gglx01jjamnhy0af1p16)

Hi Stefan, Wondering what is the advantage of SSE over Web Socket? Like in Live comment design, SSE is mentioned as a better approach. I know SSE is one way from server to client, and web socket is duplex. If we simply use web socket for one-way case, what is the downside? Is it because web socket does not have reconnection? I guess if web socket is used, the lib should have that function and users do not need to care about it. Any suggestions?

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4resh080055f66gr1omk2u5)

Websocket requires infra support through the stack whereas SSE can be done over standard HTTP (with some exceptions). Other than that there's a lot of overlap between the approaches!

Show more

8

Reply

P

ParliamentaryOlivePelican346

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4deyk0j02vr1i3r5h2h6blu)

Thanks Stefan for this deep dive! One question, Maybe having a system design question which would use WebRTC like FaceTime etc would be helpful to have? As seems like, there is no any common problem, which discusses that.

Show more

3

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4retfsg005muek60tvxx4be)

I haven't seen a WebRTC question asked in the wild! Including it for completeness here, but generally speaking most interviews won't touch on it.

Show more

6

Reply

A

Abhi

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmcb1ggav01eiad08c0sria33)

Thanks for this deep dive, Stefan. Agreed that WebRTC questions are less common in interviews, but it’s better to be prepared than miss out on an opportunity simply due to a lack of knowledge about a single protocol based pattern. Including at least one WebRTC-based system design question in the catalog would be valuable (example WhatsApp Video calling , Skype, FaceTime, Google Meet, etc) with some possible deep dives on how these apps achieve end-to-end encryption, etc.

Case in point: It’s especially relevant for roles in communication-focused teams at big companies like Discord or AAA gaming companies (e.g., Riot Games, Blizzard Entertainment), Apple FaceTime (Apple's interview process is entirely team based unlike other FAANG!) where such expertise aligns closely with their work. Besides, it doesn’t hurt to learn! :)

Show more

2

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4ggdri200jq5gbo6685hl40)

I think it may be worth mentioning that L4 load balancers both Direct routing and NAT ways to connect to a backend server. The benefit of DR is that the client IP is preserved. In NAT, a new socket to the backend server is created but the NAT mapping allows packets to be transferred back and forth.

Show more

0

Reply

V

VocalTealMandrill585

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4gihm6e00jjehzqzrznzllk)

This was an informative deep dive as usual, keep them coming. I've read a couple sys design books and a lot of this still felt new to me, specifically the trade-offs involved in the different approaches.

Show more

0

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4jcecxx00rt12z8v3he1vkn)

I do see a mention of setting http keep-alive but i wanted to clarify that it is something that is implicit since http 1.1. By default connections are persistent and there is no need to set this header by the client as it is default. Reference is 31

From RFC 2616, section 8.1.2:

A significant difference between HTTP/1.1 and earlier versions of HTTP is that persistent connections are the default behavior of any HTTP connection. That is, unless otherwise indicated, the client SHOULD assume that the server will maintain a persistent connection, even after error responses from the server.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4jcgpfz00t8fop8b0vczx65)

This is a fair point but the comments are a bit broader than that. Keep-alive settings are valid at each point in the chain - most load balancers, application servers, etc. have a similar setting.

Show more

0

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4k3klvh0001cgw3a1dq2oho)

Point noted. Though I'd argue that pretty much entire pipeline of this flow either uses HTTP 1.1 or HTTP 2.0 so everything is already taken care of. There are separate timeouts (non-http) that may apply at various levels (such as LB, DB etc) and that need to be accounted for if one is pooling connections so that the connection is reused to the most extent. Are you somehow implying that part? Also, it may be worth calling out that the connection pooling is what helps here in a scenarios where creating http connections is expensive so creating a pool (browsers already do that for http 1.1) or leveraging http 2 (or HTTP3/QUIC) (that has multiplexing) would be something to think about.

Show more

2

Reply

![Ahmad](https://lh3.googleusercontent.com/a/ACg8ocLlzxBMlUXM67q7g-abAJlLMjfD_4Ruz08tmFxW-J3RlvAg61U=s96-c)

Ahmad

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4n2xlfw003tjs6sdv3p6drx)

Good stuff I learned a lot today! Thanks

Show more

1

Reply

C

caprariu.ap

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4pw5bh102oqw46g2yz6vm38)

Great deep dive! Btw the links to "deep-dive interview questions" (ctrl f for that) have wrong urls (ex: "system-design/deep-dives/google-docs" instead of "system-design/problem-breakdowns/google-docs")

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4qb1obg00esz3r0przjoh2x)

Ah thank you. Fixed in next release!

Show more

0

Reply

M

MedicalLimeEel446

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4r7y1ei01amxwp6ubpbuerc)

Such a good article! The quality of content you guys are producing is incredible.

Question about the WebSocket approach involving Zookeeper - since the clients have to connect to specific servers, am I right thinking that there would be no load balancers between the clients and the WS servers?

If yes, I'm wondering what could be an elegant approach to "discover" any of the servers in the first place so that they can tell the clients where to connect to. Have a DNS entry pointing to one or several of them? And then separate DNS entries for individual servers once the client knows which one to connect to?

Show more

2

Reply

A

aniu.reg

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm507yb2k014qr5q610a4umis)

I think from the context the WS server that the client first connects to needs to redirect the client to the correct WS server. There is no LB server.

Show more

1

Reply

![Alfred Gui](https://lh3.googleusercontent.com/a/ACg8ocJrrl0mvI_wtYdm8wn7lF6lcgdWZ1TOO3l91WPFN0UZSrFoROc=s96-c)

Alfred Gui

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4re7qp3004iuek6bu3csd7m)

For chat application, should we use a combination of polling and REDIS pub-sub subscription? As REDIS pub-sub doesn't guarantee message delivery, a new message may be lost if there is a temporary disconnection between websocket server and REDIS cluster. And a polling thread can collect the missed messages.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4rejij00058uek6cql2xidw)

We have a good breakdown of [Whatsapp](https://www.hellointerview.com/learn/system-design/problem-breakdowns/whatsapp) that you can use for reference. Using a consistent hashing approach or using pub/sub both are doable.

Show more

2

Reply

![Alfred Gui](https://lh3.googleusercontent.com/a/ACg8ocJrrl0mvI_wtYdm8wn7lF6lcgdWZ1TOO3l91WPFN0UZSrFoROc=s96-c)

Alfred Gui

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4reshri004z5owsbugcahoy)

Thanks for pointing me to that document. This line is exactly what I was looking for: "If clients aren't connected, they'll receive the message when they connect later and retrieve all of the undelivered messages (we can also periodically poll for undelivered messages for transient failures in between). "

Show more

2

Reply

![cst labs](https://lh3.googleusercontent.com/a/ACg8ocIN2ZMgNoHBb6RDKN2xJfh_zke9WDrTjB-JzVE8WV_00kU42g=s96-c)

cst labs

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4w5h0t400qspecvp5exfzy9)

I spotted an anomaly in the image of consistent hashing. May be it is just a type in the image but wanted to call out when you remove server n8, all the keys stored on n8 are going to move to n7 by moving counter clockwise from 87 and 81. None of the other keys should change everything otherwise it defeats the whole purpose of using consistent hashing since the affected keys = total\_keys/no\_of\_buckets.

The image in the page is also moving the keys 68 and 75 to n6 but probably that's only a wrong representation in the image. After n8 is removed, n7 will have all keys (68,75,81,87)

Show more

8

Reply

C

carlosft

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7np6me6001iaakwvteks3wb)

I had the exact same thought. All of the users that had been on n8 should move to n7.

I also might be worth mentioning that this is a very naive/simple approach to consistent hashing. To distribute the impact of a server being removed from the ring, each server would be listed multiple times in the ring.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7nrlbvg004xdd6l4rbh5uc2)

Yeah we're going to remove the detail here and reference the more authoritative [Consistent Hashing](https://www.hellointerview.com/learn/system-design/deep-dives/consistent-hashing) guide.

Show more

0

Reply

D

DeliciousCoffeePython561

[• 24 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdqpbemw006aad08hnvigdd1)

Thanks Stefan, but it looks like the faulty diagram is still here.

Show more

0

Reply

![Sarvesh Sawant](https://lh3.googleusercontent.com/a/ACg8ocL75Xz248mdN-_5Gx_jq9kEoFR-lXg2MCuqlWrjTKiPA6gN1A=s96-c)

Sarvesh Sawant

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm4yaik5b01zihjxbpx7ki0w4)

Any thoughts : Why can't we use AWS API Gateway (Serverless) with WebSockets, Lambda, and DynamoDB? This approach seems like a simpler solution to me compared to pub/sub.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5un0x3c01x1tixjake2mm52)

API gateway is a decent offering. The main limitations here are (a) it's not going to be useful for non-AWS companies, (b) it covers a lot of complexity for you so your interviewer might want to dig in more, and (c) serverless ends up being very expensive at scale.

So give it a shot? But good to have a backup understanding depending on the attitude of your interview.

Show more

0

Reply

A

aniu.reg

[• 8 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5083yq80146ub4zzwf672e3)

Regarding HTTP long polling,

"Keep in mind that each hop in your infrastructure needs to be aware of these lengthy requests: you don't want your load balancer hanging up on the client after 30 seconds when your long-polling server is happy to keep the connection open for 60 (15-30s is a pretty common polling interval that minimizes the fuss here)."

How to configure the LB to handle long connection? Here the LB could be either L4 or L7, right?

Show more

0

Reply

![Vipul Kaushik](https://lh3.googleusercontent.com/a/ACg8ocIRGMbFOB_yRbprYo7cob53fUPYU0uft7j9fT78Vfpf8Zqa1Q_wBQ=s96-c)

Vipul Kaushik

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm593jril018dke2rl4e5ip1x)

Most helpful article ... I am a big fan now!

Show more

0

Reply

![Nitin K](https://lh3.googleusercontent.com/a/ACg8ocL9P3gZZSrheedfA6DtO1UdtbusbQDsV8818csPk3aW6Ud9sQ=s96-c)

Nitin K

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5ht1h6e01gaa8vvz37y1i5a)

Thanks for the deep-dive. However I am a bit confused about the explanation in the pub/sub model. Concrete examples would immensely help. I was getting lost with the generic phrasing e.g. relevant topic", "registering a client" and possibly some terms being used interchangeably. e.g. client, topic, user, subscribers.

Show more

2

Reply

S

SoleBrownPanther252

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5id8mhi01qf10u870t4kfr4)

Very good article. I have a question regarding this statement "When we add a new server, we simply move the server's position clockwise around the ring. When we remove a server, we move the server's position counter-clockwise." Should the position of the server be fixed and only the keys will be reassigned upon adding or removing the server?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5imqp4f024aep4lkc8teyim)

I think this may be a question of how you think about it geometrically. You're either reassigning servers to topics or topics to servers but the net effect is that a small proportion of topics end up on new servers and the number of servers {increases, decreases}.

Show more

0

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5ugw4ru01lltixjmn1sc9pp)

pretty comprehensive, thx! one question on push mechanism: consistent hashing. Does it also require coordination service like Zookeeper similar to simple hashing? I found these statements contradictory: "Requires coordination service (like Zookeeper)", "All servers need to maintain routing information"

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5uha5xg01mdtixjncbk958a)

I guess not absolutely. You could theoretically hard code the participants. You just wouldn’t be able to scale without your clients doing a deployment or something goofy.

Show more

0

Reply

R

RadicalBlackBeetle554

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5umqohe0216t5d7g7kt4fqo)

"each endpoint server will be connected to all hosts in the cluster". As per my understanding, with Redis sharded pub/sub, the cluster can be divided into shards. Which means, the end point server will be connected to all hosts only in the respective shard rather than all hosts in the entire Redis cluster. Could you please validate my understanding?

This is from the Redis docs: "Sharded Pub/Sub helps to scale the usage of Pub/Sub in cluster mode. It restricts the propagation of messages to be within the shard of a cluster. Hence, the amount of data passing through the cluster bus is limited in comparison to global Pub/Sub where each message propagates to each node in the cluster. This allows users to horizontally scale the Pub/Sub usage by adding more shards."

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm5umxt2001zzp38vw62s3os6)

It can be sharded! but each endpoint server will invariably have users from each shard — that's the point this sentence is trying to make. You're not the first person to be confused by this, let me update the language.

Show more

1

Reply

C

CuddlyCopperWildebeest704

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm64kfa0h01r7y43nmkg959eu)

Amazing deep dive, wonder if it might be a good idea to add some info about wss vs ws

Show more

0

Reply

![Tom Oh](https://lh3.googleusercontent.com/a/ACg8ocL_zg_F9-J6vR8eza8MdsPWwRfQlwUUTYY5HA0PPSkPGaXt1A=s96-c)

Tom Oh

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm66y4ep001tda4uubjw08gez)

"e.g. a cooordination service like Zookeeper or etcd"

three "o"s

Show more

0

Reply

Y

yingdi1111

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6a6ria300f3sagc1vgl7wte)

Should we only use ""zoo keeper"" to store the configuration between and videoId and ip address? I am wondering if DynamoDB with DAX could be a good candidate here as well (DAX is not strong consistency though)? Since DynamoDB is scalable, low latency and managed service

what makes zoo keeper a good option here.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6a6x8h100fbsagc0vxjtp2c)

Zookeeper is purpose-built for this. You can roll your own with dynamo (not sure why you'd use DAX), but sometimes it's useful to have someone figure out the problems you'll run into ahead of time.

Show more

1

Reply

Y

yingdi1111

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6a7f2li00g58twyqqqrf49f)

DAX for fast retrieval so better performance and low latency.

\---EDIT--- If we have a local cache of the config we will not need the DAX. ---EDIT END ---

but sometimes it's useful to have someone figure out the problems you'll run into ahead of time

I believe what you mean is we should use zoo keeper since it is purpose built. Someone already build a solution to address some potential issues we may face using dynamodb.

Show more

0

Reply

Y

yingdi1111

[• 7 months ago• edited 7 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6a9ukmb00k0iq3ncazudcn2)

If you end up with an application where each connection requires a lot of resources and state: consider whether you would be better served with consistent hashing.

Why is this and when will a connection requires a lot of resources and state?

Besides, if we have MANY users/clients register for the SAME redis pub/sub channel is it usually better NOT to use it? because we are sending message to multiple servers instead of one (users with the same channel may connect to different servers). Or we can use zoo keeper here as well to make sure all users with the same channel will connect to the same server??

Show more

3

Reply

A

AppropriateBlueDove838

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6pyiy6r03omob0klhrtfzjh)

+1 not clear why this is - why amount of state/resources affects the decision here

Show more

1

Reply

![E Z](https://lh3.googleusercontent.com/a/ACg8ocIvzX0SuEb-25SaNWiD-Ye0PAgT4B_Bjg2gbwo6kHyq995G5U8=s96-c)

E Z

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6zqh7y10352homjeax9cspb)

For the statement, "If you end up with an application where each connection requires a lot of resources and state: consider whether you would be better served with consistent hashing.", I believe this is used for a reason when choosing between consistent hashing and simple hashing, not a reason when choosing between pub/sub and hashing.

Show more

0

Reply

![Ge Xu](https://lh3.googleusercontent.com/a/ACg8ocIUwlvTlvr8XDYlJr0mawAAAJrlIz-ROily9Zg_FOKgpjV_7mqN=s96-c)

Ge Xu

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6imlv02009b11hvor99iy1h)

Great article. There is no best technique, only the best approach for specific scenario

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6iudxlm00kyjgz3dp4nzbn5)

100%

Show more

0

Reply

A

AmbitiousTealSheep769

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6moj2m804qslhflonuo4n5u)

This deep dive made me go from hating System Design to loving System Design, thank you!

Show more

1

Reply

A

AllegedGoldQuokka417

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6pqskik03cl124qp9lt9q1t)

Fantastic article! It would be great if you could also talk a little about webhooks.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6prld9x03ci41j3sazeg9b7)

Webhooks sound similar to websockets but serve an entirely different purpose! You'd use webhooks when you want to send updates to a system you don't own. You basically are just registering an HTTP endpoint and committing to ping it (with some well-defined retry policy) when certain things happen.

Most webhooks operate on a >1s SLA - they're not really all that useful for delivering updates to end-user clients. Aside from "design a webhook server", you're less likely to see them in a system design interview.

Show more

0

Reply

A

AllegedGoldQuokka417

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6pt60o903e9ob0kn2zqvm5w)

Oh I see! I was totally confused then. Thank you for the quick clarification

Show more

0

Reply

M

MarineBlueFly138

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6sf4nq0020kcv0wqcmp7bkt)

I've seen some system design videos where server-side push utilized a directory service. IIRC, the connections were assigned to a server by the LB using a "least connection" strategy, which then updated the directory service with the server holding the connection. Is that common pattern, or would it be better to use one of the push patterns outlined in this article?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6sfc2g1025j14j66tz8r0zu)

Yeah this is very similar to the pub/sub model. The problem is there’s a lot of complexity lingering with the directory: e.g it needs to identify dead servers and handle scale out/in. Tradeoffs as usual!

Show more

0

Reply

D

dileepkratnala

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm6x6ag7n001bhomj3mcb031d)

Great deep-dive. Thank you. Silly question perhaps - Client establishes connection with load balancer/api gateway and backend server is generally in private network. How a backend server can directly communicate back with Client.

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm70uqyxf04gsun62wommyyn8)

They have an open TCP connection all the way through!

Show more

0

Reply

A

AddedLimeCarp593

[• 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm9z260o101i0ad08ffs4wx94)

Through the load balancer/api gateway? Or are the client and backend server communicating directly somehow?

Show more

0

Reply

S

StaticJadeLemur903

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm708o27603w8un620ul4cgmb)

Wondering if all the diagrams in "Network 101" section the IP->TCP->Http box positioning order are wrong at "Client" side.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm70ooqo00496cv3wzqiqja4m)

Which part do you think is wrong?

Show more

0

Reply

S

StaticJadeLemur903

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7trr4d2004qpso3cvdhvnxr)

In the diagram "Simple Http request with L4 load balancer", from left to right, you have "client" -> IP -> TCP -> layer 4 load balancer -> IP -> TCP server. I thought it shall be "client" -> TCP -> IP -> layer 4 load balancer -> IP -> TCP server. But I just realized that you are trying to omit the client's network stack but only drawing out the request receivers' network stacks (layer 4 load balancer, and the server).

Show more

0

Reply

SJ

Sam Jhon

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm70uc9xn04fzun62g140yeop)

Is 432 port right in "Sending Updates to the Right Server" picture?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm70ui0or04ggcv3w3x2i0iix)

Nah that's not a port — just a random ID for illustration.

Show more

0

Reply

H

hzota1042

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm742q5qe00xp14jjaf8hcbho)

This is GOLD!

Show more

1

Reply

K

karzhouandrew

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7534c4901sg14jjf1gixg6u)

Thanks for the amazing article! Could you please elaborate a bit more on why you didn’t consider gRPC here? It’s support bi-directional streaming that can be an alternative to WebSockets. Thank you in advance.

Show more

1

Reply

![Robert](https://lh3.googleusercontent.com/a/ACg8ocLzJOwmklmw_sU9vcF1R-hRXkUuvy4eEAaU4mc12lsZ0oK7Idw=s96-c)

Robert

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm75stlju02qv88mx52e3r1d1)

I am just curious is it standard practice to create 1 pub/sub topic per user + one or more hosts for fault tolerance. Would it not be overkill to have billions of pub/sub Redis topics for something like whatsapp with billions of users?

Show more

0

Reply

![Robert](https://lh3.googleusercontent.com/a/ACg8ocLzJOwmklmw_sU9vcF1R-hRXkUuvy4eEAaU4mc12lsZ0oK7Idw=s96-c)

Robert

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm75t3ug702sgt4ttnikbb3r0)

Great article, really enjoyed reading it. What would you recommend for youtube/twitch streaming? I am thinking for publishing we can use webRTC to send chunks of video + audio, transcode and upload to CDN. For consuming we can use HTTP Live Streaming (HLS) or Dynamic Adaptive Streaming over HTTP (DASH) where the client dynamically requests chunks from a CDN?

Show more

0

Reply

C

chiranjeeb.nandy

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm76kb8ui03ey88mxyl36zl5t)

thanks for this detailed write up. After reading this I am more confident to answer follow-up questions about real time updates for my upcoming interview!

Show more

1

Reply

![Ge Xu](https://lh3.googleusercontent.com/a/ACg8ocIUwlvTlvr8XDYlJr0mawAAAJrlIz-ROily9Zg_FOKgpjV_7mqN=s96-c)

Ge Xu

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm76of9q5000fl2b75qv6lhmm)

A little confused about long pulling. In dropbox design, in change sync, it says for hot file, use long pulling. But here it says long pulling is for low frequent update.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm76ohjou000t4nd55yxl79m5)

It's relative. Frequent here is referring to many times a second.

Show more

0

Reply

![Ge Xu](https://lh3.googleusercontent.com/a/ACg8ocIUwlvTlvr8XDYlJr0mawAAAJrlIz-ROily9Zg_FOKgpjV_7mqN=s96-c)

Ge Xu

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm76ouqmv0016l7b1icu2a1dw)

Gotcha. Now I have a clearer picture of the scenarios of choosing long polling over simple polling. Thanks.

Show more

0

Reply

![Sudhanshu Bansal](https://lh3.googleusercontent.com/a/ACg8ocISLhjg7YR3Nd9WTWPrYqJ6vgpMifWAM9QvTQK0xIfPPkDSDGr0=s96-c)

Sudhanshu Bansal

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7dfohon00b7hsip5cul24tu)

In Pushing via Consistent Hashes -> Can we use Load Balancer and implement Consistent hashing in Load Balancer to determine which chat server to connect to? Why are we not using Load Balancer in WebSockets + Consistent Hashing Solution?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7dpdjwy00obdbe0lbhw4uem)

Most load balancers aren't going to be able to handle this internally. If you have one that does, great!

Show more

0

Reply

![Sudhanshu Bansal](https://lh3.googleusercontent.com/a/ACg8ocISLhjg7YR3Nd9WTWPrYqJ6vgpMifWAM9QvTQK0xIfPPkDSDGr0=s96-c)

Sudhanshu Bansal

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7r4j93p02ve2ebz3ctesqln)

"Most load balancers aren't going to be able to handle this internally." Do you mean LBs don't handle WebSockets or Consistent Hashing?

Show more

0

Reply

C

carlosft

[• 6 months ago• edited 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7idg83n03iqimqnbz7jsctt)

> Most SSE connections won't be super-long-lived (e.g. 30-60s is pretty typical), so if you need to send messages for a longer period you'll need to talk about how clients re-establish connections and how they deal with the gaps in between.

Is the typically short life of SSE connections related to the common use cases associated to SSE, or is this more of an acceptance of the shorter configured timeouts the various http proxy servers you will have sitting between the client and application code? I am guessing the later.

I am also guessing that the fact EventSource will attempt to \[automatically reconnect\] (https://html.spec.whatwg.org/multipage/server-sent-events.html) is part of why running SSE over your existing HTTP infrastructure is just easier than trying to step up to websockets.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7idhs9r03iwimqn7wv6nsap)

They’re interrelated, but it’s primarily the latter.

Show more

0

Reply

C

carlosft

[• 6 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7idjpxu03m2mbkqtvmf9ion)

Stefan ... you replied before i could even edit the markdown for the link (i thought you might be using markdown). Are you a bot or do they have you chained to a keyboard?

Show more

0

Reply

I

InjuredYellowTick667

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7kqij3800cw8gbtgqr3542w)

Are we gonna make videos of these please.

Show more

1

Reply

![Mishu Goyal](https://lh3.googleusercontent.com/a/ACg8ocKRHncAF5KWU5b1lnWsUEmrUVhiWDY1oUNuNUfLbqfvbVgcAQ=s96-c)

Mishu Goyal

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7l8rfch01484scgdjat26d9)

"In many instances where this approach might be used, the number of clients can actually be quite large. If you have a million clients polling every 10 seconds, you've got 100k TPS of read volume! This is easy to forget about." - Could someone please elaborate this and what does easy to forget about mean?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7l94pc801538gbtkmk1k7py)

Polling seems like it doesn't require much resources, but it actually does. It's easy to forget how much resources it requires because it feels like a "drip drip".

Show more

0

Reply

M

me

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7mhfvwh00c9rqzdzmcvdowg)

I'm having a hard time parsing this sentence: "We'll make the server number n responsible for user n % N." N is the total number of servers, and n is the specific index? Shouldn't we modding the user ID against the total number of instances?

Show more

0

Reply

M

me

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7mj1ata00fuvfhm0sijgzb3)

Further down, "The server looks up the client's hash in Zookeeper to figure out which server is responsible for them." Presumably, we'd be looking up the connection information for a specific server number after modding our hash from earlier, NOT looking for a precomputed hash in Zookeeper?

Great content!

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7mj24sh00fyvfhmu2u46z34)

Changing the sentence to "server number n responsible for u % N!

Show more

1

Reply

![Briggs Elsperger](https://lh3.googleusercontent.com/a/ACg8ocLnH38djA-Rkrfs5-MFkv3_1SfDXRt1P9bOfdHp3YTBakY5xEgn=s96-c)

Briggs Elsperger

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7msq93000rjvfhm1l9jbh9w)

Hey! This is good stuff, but I've noticed when talking about pushing via consistent hashes, the diagrams show information for User A, but the description talks about User C. Is this intentional?

Show more

0

Reply

L

LexicalMagentaOrangutan372

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm7qxx5h102owrpj4lgn6ysqf)

"The endpoint server registers the client with the Pub/Sub service, often by creating a topic, subscribing to it, and keeping a mapping from topics to the connections associated with them." - What kind of mappings are stored in the pub/sub service? Is it the connection information between the user and the endpoint server? During the update stage for a specific topic how does the pub/sub service know which endpoint server it has to connect.

Show more

1

Reply

![learning buddies](https://lh3.googleusercontent.com/a/ACg8ocKb3ulgDowmFom690cLN6oPaql424dPEEDpp0MVyU9nXrZFXg=s96-c)

learning buddies

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm80vo6wk04wy5693rncq6w02)

Question: Chat App. I use websockets and L4 LB. User1(server1) is sending message to User2(server2). Web socket connection for user2 is associated with server2. Confusion to me, How L4 Load balancer can route web socket connection of user2 to server2, given L4 LB cannot inspect payload so it will not know it is web socket connection is meant for user2.

Or , for chat app scenario, L4 LB is not right thing to use?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm80vrfyb04uce77fc6p1nd3c)

TCP is stateful. L4 load balancers operate on this state. The inbound TCP connection of user2 is always associated with server 2, all packets received on the inbound connection are forwarded to the associated outbound connection.

Show more

0

Reply

![George Yarish](https://lh3.googleusercontent.com/a/ACg8ocKaWULPLDzaxkgBDyDHOkyvjcYLBx-dGQnOVofJ82KTzoLSiw=s96-c)

George Yarish

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm8hz9vk400ilv7ntz7kncpyc)

> SSE instead uses a special header Transfer-Encoding: chunked

is this actually true? I though it is based on text/event-stream content type (as it shown on example code)

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm8hzxasb00efwoxagm01ei8t)

Use both! You want to make sure intermediate proxies aren't batching up the responses.

Show more

1

Reply

![George Yarish](https://lh3.googleusercontent.com/a/ACg8ocKaWULPLDzaxkgBDyDHOkyvjcYLBx-dGQnOVofJ82KTzoLSiw=s96-c)

George Yarish

[• 5 months ago• edited 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm8ioty6500k7sn6x8c4n1k14)

Hi Stefan, thank you for the reply. Just for my understanding, could I use Transfer-Encoding: chunked aside of SSE, e.g. to stream response data for long polling?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm8ivxs7k00nysn6xdni3sr4q)

Yes! Although note that you're depending on middle boxes actually respecting the transfer-encoding header :)

Show more

0

Reply

A

AdorableTurquoiseCrane528

[• 4 months ago• edited 4 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm8pxd4gd00xofg39o95ie1yr)

Great article! Small suggestion - since most of the production level communication is not unencrypted(atleast not recommended), would be good to include a part on TLS handshake and HTTPS in initial HTTP section.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 4 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm8q3d1nv011jfg39e2ubf3mq)

It's a fair callout, but I didn't want to add complexity/noise here. The core is the same — you're going to have a fair bit of back and forth to establish connections and this needs to be considered during your design.

Show more

0

Reply

N

NuclearWhiteCrayfish589

[• 4 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm8u4tcwa00lbad08zj6ik698)

Hi Evan,

In the long polling section:

The idea is also simple: the client makes a request to the server, and the server holds the request open until new data is available.

How does the server know when new data is available? Does it constantly poll for the new data?

Let's take a concrete example, as you mentioned. In payment processing, we use long polling for payment status. Let's say Order1, which was placed by User1.

How does the write path (where some worker in the queue might change the payment status to "success") know that User1 is connected to a specific server, say ServerX, which is waiting for the status of Order1?

Show more

1

Reply

![Brad M](https://lh3.googleusercontent.com/a/ACg8ocLUmD-ls--6lbvU0pKdlg5Wc6ExLSFwPvlrX48hp_Tjez4UGA=s96-c)

Brad M

[• 4 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm8ueetac010wad079w7avkz2)

Amazing article.

To clarify my understanding, in the case of communication between two clients, WebSockets can work, but they're more complex. We can emulate this full-duplex communication by a mix of traditional HTTP post requests and then clients setting up an SSE connection to a pub/sub store if latency requirements aren't near real time?

Show more

1

Reply

![Aiwei Zheng](https://lh3.googleusercontent.com/a/ACg8ocJznVspMNiSLNKxG4QDo7lwVPd3HAdQzMc21V7YWUIv5_b5XdM=s96-c)

Aiwei Zheng

[• 4 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm9amnio7006ead08gd2rbh94)

This is a really helpful deep-dive! Thank you so much!

Show more

0

Reply

M

MeltedJadeImpala762

[• 4 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm9bwbc00001uad07zptpzwm4)

Hi,

Thanks for the amazing deep dive. I have a doubt that's bothering me for quite a while across different designs (related to persistent connections).

1.  For e.g. let's take the WhatsApp design, if we expose the chat servers through the web via DNS for web socket connections and have a chat registry (along with Zookeeper) to decide the right chat server for a user, how will the initial checks (like authentication, rate limiting, etc.) happen because there's no API Gateway any longer? Is it the case that these checks would be performed when the user first connects with the chat registry server via a L7 load balancer? And once the chat registry server assigns the chat server to connect to, it connects directly relying on the previously done authentication? If yes, what if the user abuses this connection and bombards the chat server? How will rate limiting be done?
2.  When a user establishes an SSE connection, does the server connect directly to the client (or) it happens via the API Gateway? Was curious if the former is even possible because clients (like browsers) are usually behind NATs and firewalls and they're assigned private IPs that are not directly reachable from the internet?

Would be very helpful and relaxing for my mind if I can get your opinion on these queries. Thanks in advance!

Show more

0

Reply

![Dominic Fannjiang](https://lh3.googleusercontent.com/a/ACg8ocJ4qAPPHW4Wik_3SR752jHgO8XmS5MeoPSusZhZwh6kJf9TYUw=s96-c)

Dominic Fannjiang

[• 4 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm9d7u8c900hfad07chsh4bpn)

"We don't know whether subscribers are connected to the endpoint server, or when they disconnect." Can you elaborate why this is the case? Won't the endpoint servers know when a connection has been established or disconnected and can't they take some action (e.g. call some API to another service) upon those events if necessary?

Show more

1

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cma1gfy8y01q8ad08zckp6wly)

Let me see if I can clarify this a bit more in the writeup. You're right, you could have a separate topic or explicitly create messages for connect/disconnect, but it's not a "connection" that's being managed.

If you have e.g. a message that happens when a client disconnects and you publish to a topic when that happens, there's a very real problem if they reconnect quickly if those connect/disconnect messages arrive out of order. In order to patch over this deficiency, you'll quickly converge on a solution that looks like a full connection.

Show more

0

Reply

M

MeltedJadeImpala762

[• 4 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cm9fl9d4200vnad08ymzajlng)

Hi,

In consistent hashing, do we need a separate registry service (that maintains the consistent hash ring) along with Zookeeper (to maintain server details, health checks, etc.)?

Or Zookeeper in itself is enough and it can also manage the consistent hash ring configurations and provide the right server on which request should be routed to? Not able to figure out if Zookeeper supports consistent hashing as well (or) just supports server discovery and their health checks?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cma1gckis01q4ad08l0ffmlvg)

Check out our [Zookeeper guide](https://www.hellointerview.com/learn/system-design/deep-dives/zookeeper) which was recently published. Zookeeper doesn't provide any first-class consistent hashing API, it will tell you about which servers exist and any parameters you'll use to your consistent hashing function. You'll rely on clients to have a shared library (and the parameters from zookeeper) to decide which server to connect to.

Show more

0

Reply

E

ElectronicMaroonChipmunk819

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cma1bol0s01m9ad084lg9pbms)

Is it correct to understand that pushing via Pub/Sub also requires some form of sticky connections based on user ID (i.e., a given user typically connects to the same server)? To achieve this, a hashing mechanism—or an L7 load balancer—is needed to route connections accordingly. Hash-based connection routing would ensure consistency, and to support scaling up or down, consistent hashing could be used

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cma1gafzf01pyad08706qnd2f)

Nope, pub/sub actually doesn't require sticky connections at all - that's one of its main benefits. Any endpoint server can handle any user's connection because the pub/sub service (like Kafka/Redis) maintains the mapping of topics to subscribers. When a client connects to an endpoint server, that server just subscribes to the relevant topic(s) and forwards messages. If the client disconnects and reconnects to a different server, that new server just subscribes to the same topics. The state lives in the pub/sub service, not the endpoint servers, which is why you can use simple load balancing strategies like "least connections" instead of needing consistent hashing.

Show more

1

Reply

E

ElectronicMaroonChipmunk819

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cma1m8k3a025rad08pcq3gky9)

Agree. However when reading the "FB: Live Comments" article, I was able to identify the Pub/Sub pattern being applied to deliver comments in near-real time. Specifically in the "Great Solution: Partitioned Pub/Sub with Layer 7 Load Balancer" section of the article, it is proposed, the users connect to the server which is responsible to a particular :liveVideoId in order to receive comments from it. Which is why wanted to confirm

Show more

0

Reply

P

pointless.decommission

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbdng4fh01cpad08k9sr9t6c)

It sounds clear enough with something like Redis as the pu/sub service. However, if we were to use Kafka, I don't think it would be as easy as the article makes it sound. How could we ensure that endpoint servers consume messages for users connected to them if user connection assignment would be a non-deterministic process? Simply partitioning topics by user\_id would not solve the problem since: a) we would likely have much more users that topic partitions b) each topic partition would likely have messages from other many different users c) each server endpoint would have connected users that span many different topic partitions

Show more

0

Reply

![Joseph Kunzler](https://lh3.googleusercontent.com/a/ACg8ocLjVi6B4H8Fgg5dB1i99QKQ5JUO1ZEF7dcjb6PAWxSSx_IRlA=s96-c)

Joseph Kunzler

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cma8s45dt016xad08g1goobcl)

> We use SSE extensively for Hello Interview and the amount of time we've spent dealing with networking edge cases is mind boggling.

I would read a developer blog about this if you write one. :)

Show more

1

Reply

S

SecondaryIvorySalamander436

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmadjb30901pmad08np7iuppu)

Correct me if I am wrong but one any mode of communication that needs to be long running needs additional infrastructure - so additional infrastructure is not necessarily a con of websockets alone. Ex: even SSE needs 'connection management servers' just like websockets.

Also, it is necessary to use consistent hashing to manage connection stickiness? Why not use a 'connection management' layer that scales based on number of active connections and connection allocation happens based on least connection?

Show more

0

Reply

H

HandsomeIvoryCrow799

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmb4smz8i03qnad08jzolsl8e)

It seems SSE currently not supported yet in major cloud providers such as AWS, Microsoft Azure, Google Cloud. In a product design interview, can we tell the interviewer that let's assume that major cloud providers do support SSE with their API Gateway component? If that's not possible, what's the chance that the interview would force us to discuss trade-offs between 2 below solutions?

-   Solution 1:Set up a custom API Gateway supporting SSE.
-   Solution 2: Use managed API Gateway supporting websocket.

I'm thinking perhaps solution 2 is more preferable if the interviewer forces us to choose. But please lemme know your second opinion. Thank you!

Show more

0

Reply

V

VoluntaryIndigoChickadee548

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbaug4sj00i7ad08p2jbt0cz)

I believe the "Consistent Hash Ring" image in the "Pushing via Consistent Hashes" section is incorrect. The depicted behaviour when node 8 is removed suggests that the remaining nodes are evenly redistributed across the ring, meaning every segment is affected. Have I misunderstood something?

Show more

1

Reply

![HottieAsian812](https://lh3.googleusercontent.com/a/ACg8ocKai4Y6bQ-snSpt6qK5zD7pHvwBjHMSL-EvCgEY6Z4zsFUHGw=s96-c)

HottieAsian812

[• 27 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdnbk9yf05t4ad08phhi319t)

I was also confused by this. Also it looks like the keys are being mapped going counter clockwise? Is there something I’m missing?

Show more

0

Reply

M

me.orca

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbcc0wur01rjad083t7gghpy)

Brilliant. Specially liked the Consistent Hashing and Pub/Sub parts!

Show more

0

Reply

B

bronze.gazer.6w

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbcfthpx01xbad07h7sgaoaz)

Amazing article! Wish I had found this earlier in my interview prep journey!

Show more

0

Reply

P

pointless.decommission

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbdnhtla01ctad08sav58t0m)

Pushing updates via pub/sub sounds clear enough with something like Redis as the pub/sub service. However, if we were to use Kafka, I don't think it would be remotely as easy as the article makes it sound. How could we ensure that endpoint servers consume messages for users connected to them if user connection assignment would be a non-deterministic process? Simply partitioning topics by user\_id would not solve the problem since: a) we would likely have much more users that topic partitions b) each topic partition would likely have messages from other many different users c) each server endpoint would have connected users that span many different topic partitions

Show more

0

Reply

S

SmoothTealNarwhal949

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbfl7m79001qad08dz58gsu0)

Great article. Summarizes multiple techniques succiently. I am a little confused by the two "hops". When using WebSockets when there are multiple servers maintaining connections, don't we use consistent hashing so that we avoid too many connections movement across the servers?

How is that different from the Pushing via consistent hashing?

Show more

0

Reply

O

OutdoorFuchsiaGerbil417

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbh0g47e0052ad08c65p50r0)

For pub/sub, how do you handle the HTTP connections on the client side? Would each client have to open a WebSocket connection? Because it would need to be a longrunning connection for the subscriber to be able to publish to all the clients, right?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbh2udpk0088ad070a2jobps)

Yes, you need an existing connection from client to server to be able to push messages down.

Show more

0

Reply

V

VoicelessBlackJackal625

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbplsh1w00b308adcgcdn8ja)

I'm new to these topics. Just to confirm my understanding

For Pub/Sub with Redis, the client will only get the updates when it is connected and misses all the updates when it is offline, right? What will happen if redis service is down and restarted? Will the states about the sub/pub be lost and endpoint servers need to re-sub the topics?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbs8zbjc003e08adfn2ftfej)

Yes and yes!

Redis pub/sub is fire-and-forget - if a client isn't connected, they miss messages. No message persistence.

When Redis restarts, all pub/sub state is lost and clients need to resubscribe.

Show more

0

Reply

A

altal

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbs6gg4u000908admys960hq)

Fantastic write-up. So many concepts that I was fuzzy about get clarified so concisely here!

Show more

1

Reply

W

WorthwhileJadeParrotfish147

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmbutxx5k06qv08ad9ecxiulx)

Why MQTT protocol is not discussed here? Isnt it benefical and better than HTTP when handling 10M+ connections, mainly for IoT devices?

Show more

0

Reply

R

RichMagentaBasilisk457

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmc2z7gvz006707adligj07dz)

Incredible content on this platform. Clear, concise, and super practical.

Show more

0

Reply

S

surajrider

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmc3pdedd047808ad487sj9ic)

> The most common objection from interviewers to polling is either that it's too slow or that it's not efficient. Be prepared to discuss why the polling frequency you've chosen is appropriate and sufficient for the problem. On the efficiency front, it's great to be able to discuss how you can reduce the overhead. One way to do this is to take advantage of HTTP keep-alive connections. Setting an HTTP keep-alive which is longer than the polling interval will mean that, in most cases, you'll only need to establish a TCP connection once which minimizes some of the setup and teardown overhead.

I found this inaccurate as it is the server which controls if connection can be kept open or not. It sends this information to the client in connection initiation as a hint. How does changing the connection timeout on client end would help here ?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 2 months ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmc3sp23p00if09adczfrm6tj)

Not sure what you mean, where are we talking about connection timeouts set client-side in the quote?

Show more

0

Reply

C

CheerfulTurquoiseGuppy693

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmccahy8y016had08yvdkuerf)

Having Quiz for this article would be great!

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmccajhwh016jad08dhw9u6tx)

Should have a quiz for everything in the coming days!

Show more

1

Reply

P

PhysicalSalmonManatee314

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmcdoxppb087xad08g7pnua05)

When can I expect more detailed content for other common patterns? In Premium content, I expected to see detailed atricles on all patterns mentioned in Common Patterns.

Show more

7

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmcz0yu6609n6ad0721052zt1)

Next week or earlier!

Show more

0

Reply

![Jinchi Zhou](https://lh3.googleusercontent.com/a/ACg8ocJlgfSIsqLZQci3oolk9otak9Gkg7Hrg0FZdg33CqKSAoPg9g=s96-c)

Jinchi Zhou

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmcgkhcq900qkad08f95qkcha)

Great doc!

Show more

0

Reply

![Sarthak Koherwal](https://lh3.googleusercontent.com/a/ACg8ocKpZp3eiAQTJ2LWQRH4TkEljg-vsm4usTpHiCI5Ed0Nnw3Cilfb=s96-c)

Sarthak Koherwal

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmchpd78e076oad08i9ddaari)

Question on server side pushing via pub / sub :- When can we expect a topic to have multiple subscribers? Since each topic should be unique to a user?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmchux7cf08rbad08kmsbfcdq)

Topics don't have to be users, they can be gameId or region or chatRoom or whatever.

Show more

0

Reply

![Sarthak Koherwal](https://lh3.googleusercontent.com/a/ACg8ocKpZp3eiAQTJ2LWQRH4TkEljg-vsm4usTpHiCI5Ed0Nnw3Cilfb=s96-c)

Sarthak Koherwal

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmchv8sq7086vad08e15ufckc)

So when a user comes online his server will have to subscribe multiple topics, instead of that when a message is delivered to a chat room, we can add it to the topic for each user in that chat room.

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmchv9pru08uvad08ury9wqis)

That's a design decision for you to make!

Show more

0

Reply

![Sarthak Koherwal](https://lh3.googleusercontent.com/a/ACg8ocKpZp3eiAQTJ2LWQRH4TkEljg-vsm4usTpHiCI5Ed0Nnw3Cilfb=s96-c)

Sarthak Koherwal

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmchvbhqt087cad08kexyi64h)

I see, let me think. Thanks !!

Show more

0

Reply

Y

YelpingGrayVole716

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmcqr9fol0037ad08kk3w5gjs)

> If we have 2 updates which occur within 10ms of each other, with long polling we’ll receive the first update 100ms after it occurred (100ms of network latency) but the second update may be up to 290ms after it occurred (90ms for the first response to finish returning, 100ms for the second request to be made, and another 100ms to get the response):

For some reason this simple paragraph took me a few minutes to fully grasp it. I ended up writing out the timeline below to clarify the sequence -- does this look accurate?

t=0ms: Server receives Update1 t=0ms: Server responds Update1 to Client

t=10ms: Server receives Update2

t=100ms: Client receives Update1 t=100ms: Client immediately requests Server again

t=200ms: Server responds Update2 to Client

t=300ms: Client receives Update2

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmcqriyj7005oad08crof31df)

You got it!

Show more

1

Reply

![Yogendra DR](https://lh3.googleusercontent.com/a/ACg8ocJ314t6bKSQligoizfDQ3c1HLci4hNDkCQEK1KT-upTc6x-6Cju=s96-c)

Yogendra DR

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmcywwcab08qpad08nyhacvn3)

Excellent document!! One suggestion is to have text inside the braces in italics so that it is easy for reading or when we are using 'read aloud' feature. Ex: The line that receives the updates is interrupted (_by the DB_) from the line that produces them

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmcz151k209otad077pszzze9)

Does your read aloud feature not put a different vocal emphasis on parentheticals? I'd normally do this but it makes it harder to read for the visual readers as parentheticals aren't de-facto italicized.

Show more

1

Reply

![Arjun Malhotra](https://lh3.googleusercontent.com/a/ACg8ocKp9g3WVSg7uKaiwQccmLIRqX8-8INmH47v34K4_r4n2jCbEQ=s96-c)

Arjun Malhotra

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmd2hxg2703lwad087j6vtuno)

"The effect of a L4 load balancer is as-if you randomly selected a backend server and assumed that TCP connections were established directly between the client and that server — this mental model is not far off."

I think that the L4 Load Balancer can use Round Robin or consistent Hashing (basedon the IP / port).

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmd2hzzkp03piad08xxnni8a1)

It can, yes, but the net result is always a single, persistent connection through the load balance to one backend host. (As compared to an L7 load balancer where you may have 2 HTTP requests on the same TCP connection hit 2 different backend hosts)

Show more

2

Reply

![Arjun Malhotra](https://lh3.googleusercontent.com/a/ACg8ocKp9g3WVSg7uKaiwQccmLIRqX8-8INmH47v34K4_r4n2jCbEQ=s96-c)

Arjun Malhotra

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmd2jgqgl046qad083m9btlct)

Right. Thank you for mentioning it, I had missed the point.

Show more

0

Reply

![Nidheesh](https://lh3.googleusercontent.com/a/ACg8ocKK16dJ7i5s5HqPrStvm0WB_0NrbHXUR8hVbUG2RYb9qrQ-65J5nQ=s96-c)

Nidheesh

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmd2m47yc059nad07cymeeh24)

Thank you for the deep dive. This is helpful. I see that SSE is getting deprecated to give way to Streamable HTTP considering the advancements related to MCP! It would be great if you could update the content to talk about Streamable HTTP as well. This will give a flavor of bi-directional realtime updates in the Agentic AI based applications

Show more

1

Reply

![Niranjhan Kantharaj](https://lh3.googleusercontent.com/a/ACg8ocISPQAOL90hPbJ5ilQZyBLjxhaLIFqJu_r3HTDm2rz8q-Wq=s96-c)

Niranjhan Kantharaj

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmd327p2v08read089ulfs65e)

Hey Stefan - great work, Thanks AI/Chatbots these days dont use SSE. They use "streamable http" MCP servers are becoming standard these days for tools usage and integration with AI agents/chat box, and they even depcrecate SSE in favor of "streamable http" https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/

Show more

0

Reply

![Ife Osuji](https://lh3.googleusercontent.com/a/ACg8ocJZW9xyKTaO3PsRPxH5eCmdYW6skIG1DPMAux10UAo5j2CQDCc0=s96-c)

Ife Osuji

[• 1 month ago• edited 5 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmd3bcw140bwgad085lzhwse0)

I wish we could get these in video format

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmeiqnyrp00kaad08jdc5j65d)

Your wish is my command.

Show more

0

Reply

B

BareCoffeeBasilisk390

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmd4eitdb0equad08bj9q24um)

Great stuff. Pedantic point in the SSE example, might make sense to include theresponse header: Transfer-Encoding: chunked,

to line up with the intro blurb.

Show more

0

Reply

O

OnlyAmberSalamander766

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmd6t11ae03njad08koycipy7)

Instead of having hashing/consistent-hashing, why cannot we just write connection mapping state to a fast caching service. Every time there is data to be sent to the user, update service just looks up the cache and finds out the right server and forwards the data to that server. It has been used at multiple companies and in many systems already.

Show more

0

Reply

![Rohit Bhattacharjee](https://lh3.googleusercontent.com/a/ACg8ocKLn7x7ATEJ1aR23j2UDEDs8EoCw834T1TzDwyPieIpkgXMD7A=s96-c)

Rohit Bhattacharjee

[• 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdaock1u0a69ad086t3nyli4)

Why not also include webhooks? For infrequent real time updates, webhooks can be a good option where an http connection doesnt need to be open at all until an update happens.

use cases like trip alerts or delivery notifications can benefit from webhooks.

Show more

0

Reply

S

SovietTanPanda828

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmddsukkg046qad095mf9dyyf)

can you elaborate more on what do you mean by minimizing state spread in architecture for Web Sockets? and any examples of this?

Show more

0

Reply

![sameer A](https://lh3.googleusercontent.com/a/ACg8ocIMZiadviONoTQ3nL2TfXxwutpfpCO_89_A_IHuZaK1Mi9jWg=s96-c)

sameer A

[• 28 days ago• edited 28 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdl55o7103aqad08el5972hl)

Hey, had a noob question. What is the best way to get the updates from a database to the update server(the 2nd hop that we're talking about in this article). Would the Update server just keep polling the DB at regular intervals to fetch new msgs? What if data in the DB changes too frequently like for example lot of new chat messages coming in?

Show more

0

Reply

![john desmond](https://lh3.googleusercontent.com/a/ACg8ocKIrxsIPMg_zE8C55aihUYIC6PNu8kwowJs1jApsuCiIpVreQ=s96-c)

john desmond

[• 27 days ago• edited 5 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdmajkwj04mgad08qvptudgy)

For Pub/Sub for a chat application, using one topic per user makes sense if you're using redis, but that would be too much overhead for kafka right?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 5 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmeiqnoj900k3ad08lfbkucow)

Depends on the application. Kafka has some disk overhead and O(2kb) per topic, Redis is O(50b).

Show more

0

Reply

![HottieAsian812](https://lh3.googleusercontent.com/a/ACg8ocKai4Y6bQ-snSpt6qK5zD7pHvwBjHMSL-EvCgEY6Z4zsFUHGw=s96-c)

HottieAsian812

[• 26 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdnf3gne0789ad08g7y8r4e8)

Am I missing something on the hash ring. It looks like the keys are mapped by going counter-clockwise instead of clockwise. And it seems like the key segment between \[n5, n1\] got remapped after n8 got removed. Shouldn't only the key segment between \[n7, n1\] get remmaped? Thanks.

Show more

0

Reply

R

RightEmeraldEarwig121

[• 25 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdpc98zg00f8ad09d3xz07t7)

Great write up

Show more

0

Reply

![Robert](https://lh3.googleusercontent.com/a/ACg8ocLzJOwmklmw_sU9vcF1R-hRXkUuvy4eEAaU4mc12lsZ0oK7Idw=s96-c)

Robert

[• 20 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdwh0ywl026aad07zml8q939)

Great article. Thanks!

Show more

0

Reply

![John Wang](https://lh3.googleusercontent.com/a/ACg8ocK8FGxHkN9q4dNfoZUQ6teUEMQGKcv4rkjY1p_2K9udBUFfbgc=s96-c)

John Wang

[• 20 days ago• edited 20 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdwijzwb000vad08o52r1jlf)

Hi, could you explain further why using a "least connections" strategy for the load balancer is helpful, especially in the context of minimizing the workload on WebSocket servers as they process messages? I thought WebSockets use persistent connections that can't be changed between clients and servers once established. So how exactly does the "least connections" strategy help in this case?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 20 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdwis873004pad08z46wo3sb)

"Least connections" helps when clients are _initially connecting_, not after. You're right that once a WebSocket connection is established, you can't move it between servers. But when new clients connect, you want to direct them to the server with the fewest existing connections to keep the load balanced. Otherwise you might end up with some servers handling way more connections than others, especially if connections are long-lived (which they typically are with WebSocket).

Show more

2

Reply

B

BottomTealSkunk679

[• 19 days ago• edited 19 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmdy44f8503wsad08t4xh7uon)

For consistent hashing and pub-sub, clients may unexpectedly disconnect, which might only last a few hundred milliseconds - but during that time they may miss a message or a few messages even. Is there a good pattern for handling these blind spots?

I assume that polling the DB works, but would cause a lot strain on the DB.

There's also storing the last message ID and checking that, but if there are a lot of messages, a client may have received the last message but missed one from a few milliseconds earlier due to race conditions. For example, messages "A" and "B" are going out, and due to latencies in the system, some clients may get "B" before "A". Say a client got message "B" and message "A" was about to get delivered as the client lost the connection. Their most recent message would be "B", but they are not aware "A" was missed. Are there good solutions to avoid this problem?

Show more

0

Reply

P

PreferredBronzeGoat654

[• 12 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cme8l7m8803vtad07glmpfdqj)

Small mismatch between the image and explanation, shouldn't "When the Update Server needs to send a message to User C, it can hash the user's id to figure out which server is responsible for them (Server 2) and sends the message there" but example image points to "User A"?

Show more

0

Reply

![Jonathan Livni](https://lh3.googleusercontent.com/a/ACg8ocJ4H03y4fBvctxUZgXc7eZh7Orm9YwTGM-YyS2qoccUpSDODZf2Xw=s96-c)

Jonathan Livni

[• 4 days ago• edited 4 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmeiuss7f02a0ad08h8c4z9b1)

Addressing the "Downside of Request/Response" section in the video and the "Request Lifecycle" section in the write up - both focus on the TCP connection overhead as motivation to avoid polling in some circumstance, however since HTTP/1 is obsolete, and HTTP/1.1 is supported but rarely used, modern implementations (including Google Docs which is given in the video as an example) maintain persistent connections. This means that it's actually the wrong motivation for reducing polling. Does that make sense?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 4 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmeiuvd1p02adad08ubftiqir)

Good question. Yes and no. I think (if not, my mistake!) I offered the caveat that the TCP connection isn't always torn down, as you mention. We still have the problem of the _request_ overhead which applies regardless of the underlying TCP connection.

Show more

1

Reply

AC

Ankit Chandra

[• 3 days ago• edited 3 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmekfh8900059ad08h77x3ttr)

What are CRDTs? in reference to be used in Google Docs?

Show more

0

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 days ago](https://www.hellointerview.com/learn/system-design/patterns/realtime-updates#comment-cmekfye3u00bxad08swxqbfrf)

[Conflict-free Replicated Data Types](https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type). See the [Google Docs](https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs) breakdown for a discussion

Show more

0

Reply

