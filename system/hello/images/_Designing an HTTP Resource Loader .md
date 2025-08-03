You asked:
----------

you are an interview for a senior staff position at meta and given this task:

Design HTTP Resource Loader

Design a system library's API that handles HTTP requests for client applications, such as a web browser or the Facebook app. For example, the Facebook app on an iPhone needs to send a request for a large video, which will be processed by our API.

  

Could you go through this interview and what possible solutions you might come up with

---

Gemini Replied:
---------------

Hello. I appreciate the opportunity to discuss this design challenge.

Here is my approach to designing an HTTP resource loader, suitable for a client application like the Facebook app.

High-Level API Design
---------------------

I'd design a library API that's **asynchronous**, **thread-safe**, and **protocol-agnostic** (within the HTTP family). It should be simple for developers to use while providing powerful underlying capabilities. The core API would center around a `ResourceLoader` class.

C++

    class ResourceLoader {
    public:
        // A callback for when a resource is partially or fully loaded
        using OnDataCallback = std::function<void(const std::vector<char>& chunk, size_t totalBytesReceived, bool isComplete)>;
        
        // A callback for when an error occurs
        using OnErrorCallback = std::function<void(const Error& error)>;
    
        // Load a resource with a given URL and callbacks
        void load(const URL& url, const OnDataCallback& onData, const OnErrorCallback& onError);
    
        // Cancel an ongoing request
        void cancel(const URL& url);
    };

This design separates the user-facing API from the internal implementation. The `load` method initiates a request and returns immediately, allowing the client to continue processing. Data chunks are delivered via the `onData` callback, and errors are handled by the `onError` callback. This asynchronous model is crucial for preventing UI freezes in a client application.

* * *

Core Components and Implementation Details
------------------------------------------

### 1\. `RequestScheduler`

The `RequestScheduler` is the brain of the system. It manages and prioritizes all incoming requests from the `ResourceLoader` API.

*   **Priority Queue:** Requests are not all equal. A user-initiated request for a profile picture should have a higher priority than a background ad. The scheduler would use a **priority queue** to process requests based on factors like:
    
    *   **User Interaction:** Is the resource currently visible on screen?
        
    *   **Resource Type:** Is it a critical image, or a large video?
        
    *   **Network Conditions:** Is the user on Wi-Fi or cellular data?
        
*   **Request Coalescing:** Multiple requests for the same URL, especially for popular resources like a shared image, can be combined into a single network request. The scheduler would check for pending requests and, if a match is found, add the new request's callbacks to the existing one.
    

### 2\. `NetworkTransportLayer`

This layer is responsible for the actual network communication. It abstracts away the low-level details of TCP/IP and HTTP.

*   **Multiplexing:** Modern protocols like **HTTP/2** and **HTTP/3** are essential here. HTTP/2's multiplexing over a single TCP connection is a key optimization for reducing latency. HTTP/3's use of **QUIC** over UDP is even better for reducing head-of-line blocking and connection establishment time, especially on mobile networks. The `NetworkTransportLayer` should be able to intelligently choose the best protocol based on server support and network conditions.
    
*   **Connection Pooling:** Reusing existing connections is far more efficient than establishing new ones. This layer would maintain a pool of active connections to various hosts.
    
*   **Byte-Range Requests:** For large resources, like videos, the loader would use **HTTP byte-range requests**. Instead of downloading the entire file at once, it would download small chunks sequentially. This allows for:
    
    *   **Progressive Loading:** Start playing the video before it's fully downloaded.
        
    *   **Resumable Downloads:** If a download is interrupted, we can resume from where we left off.
        

### 3\. `CacheManager`

Caching is a fundamental optimization. It prevents redundant network requests and speeds up resource loading.

*   **In-Memory Cache:** A small, fast cache for frequently accessed resources, like profile photos, that are currently in use. This provides near-instant access.
    
*   **Disk Cache:** A larger, persistent cache for resources that may be needed again later, even after the app is closed. This cache would be managed carefully to avoid excessive disk usage.
    
*   **Cache Invalidation:** The system must respect HTTP caching headers like `Cache-Control` and `ETag` to ensure that stale content isn't served.
    

### 4\. `TaskExecutor`

The `TaskExecutor` handles the asynchronous execution of tasks.

*   **Thread Pools:** The system would use dedicated thread pools for different tasks:
    
    *   **Networking Thread Pool:** A small number of threads dedicated to making network calls.
        
    *   **Processing Thread Pool:** A pool for tasks like decoding images, decompressing data, and managing cache operations.
        
    *   **Main Thread:** The UI thread, which should never be blocked. All callbacks to the UI should be dispatched to the main thread.
        

* * *

Example Walkthrough: Loading a Large Video
------------------------------------------

1.  The client calls `resourceLoader->load(videoUrl, onData, onError)`.
    
2.  The **`ResourceLoader`** creates a new `Request` object and passes it to the **`RequestScheduler`**.
    
3.  The **`RequestScheduler`** checks the cache. Assuming the video isn't cached, it places the request in the priority queue. Since it's a video, it might be given a medium priority.
    
4.  The **`NetworkTransportLayer`** picks up the request. It checks for a connection to the video's host. If one exists, it uses it; otherwise, it establishes a new one, preferably using HTTP/3. It sends an initial byte-range request for the first chunk of the video.
    
5.  As data chunks arrive, the **`NetworkTransportLayer`** passes them to a processing task in the **`TaskExecutor`**.
    
6.  The **`TaskExecutor`** sends the data chunk to the `onData` callback registered by the client. The client can now start playing the video.
    
7.  The **`CacheManager`** also receives the data chunks and writes them to the disk cache.
    
8.  This process of requesting and receiving chunks continues until the entire video is downloaded. If the user scrolls away, the client can call `cancel`, and the scheduler will stop the request.
    

This design provides a robust, scalable, and performant foundation for any client application that needs to load resources efficiently. It leverages modern network protocols, intelligent caching, and a multi-threaded asynchronous architecture to deliver a smooth user experience.

---

You asked:
----------

can you go over more about your design choices for the API, in particular, why certain choices were made, what pros and cons and alternatives you have considered, and also talk about typically overlooked principles in API design

---

Gemini Replied:
---------------

Sure, I'd be happy to elaborate on the design choices for the HTTP Resource Loader API.

### API Design Choices: Rationale and Alternatives

The API I proposed is built on a few core principles: **asynchrony**, **callback-based communication**, and **encapsulation**. Let's break down the reasoning behind these choices.

* * *

#### Asynchronous Design

The decision to make the `load` method **asynchronous** and non-blocking is paramount.

*   **Rationale:** In a client application like Facebook, the user interface (UI) must remain responsive. Blocking the UI thread for a network request, which could take seconds or even minutes, is a terrible user experience. An asynchronous design ensures that the app can continue to render frames, process user input, and perform other tasks while the network request is in flight.
    
*   **Pros:**
    
    *   **Responsive UI:** Prevents the application from freezing.
        
    *   **Efficient Resource Utilization:** Allows the system to manage multiple concurrent requests without dedicating a thread to each one. This is crucial for a mobile device with limited resources.
        
*   **Cons:**
    
    *   **Complexity:** Asynchronous programming can introduce complexity, particularly around managing state, handling race conditions, and debugging. Callback hell can be a real problem if not managed carefully.
        
*   **Alternatives:**
    
    *   **Blocking API:** A synchronous API where `load` would block until the resource is fully downloaded. This is a non-starter for a UI-centric application.
        
    *   **Futures/Promises:** Instead of callbacks, the `load` method could return a `Future` or `Promise` object. This is a very popular alternative in modern C++ (e.g., `std::future`) and other languages. It can make asynchronous code easier to read and compose, but it can also add a layer of indirection that some developers might find less intuitive. For a core system library, callbacks offer a more direct and performant approach.
        

* * *

#### Callback-Based Communication

The API uses `OnDataCallback` and `OnErrorCallback` to deliver data and communicate errors.

*   **Rationale:** Callbacks are a simple and direct way to handle asynchronous events. When a data chunk arrives or an error occurs, the system simply invokes the corresponding function provided by the client. This is a well-understood pattern in C and C++ programming.
    
*   **Pros:**
    
    *   **Simplicity:** The API is straightforward to use for developers familiar with this pattern.
        
    *   **Low Overhead:** There is minimal performance overhead associated with function pointers or `std::function` objects.
        
*   **Cons:**
    
    *   **"Callback Hell":** If a user needs to perform a series of dependent asynchronous operations, nesting callbacks can lead to deeply indented and difficult-to-read code. For a system library, however, the primary consumer of the API would likely be a higher-level object that manages this complexity, so it's less of a concern for the core API itself.
        
    *   **Lifetime Management:** It's the client's responsibility to ensure that the objects the callbacks operate on are still alive when the callback is invoked. If not handled correctly, this can lead to crashes (e.g., a "dangling pointer").
        
*   **Alternatives:**
    
    *   **Observer Pattern:** The `ResourceLoader` could be an observable subject that notifies registered observers of events. This is similar to callbacks but provides a more formal structure.
        
    *   **Event Loop/Message Queue:** The system could post messages to a central event loop, and the client would listen for messages related to their requests. This is a common pattern in frameworks like Android and iOS, but it would require a more intrusive design that might not be suitable for a standalone library.
        

* * *

#### Overlooked Principles in API Design

Beyond the core technical choices, here are some principles that are often overlooked but are critical for a great API.

1.  **Thread Safety and Concurrency:** A robust API for a multi-threaded application **must be thread-safe**. This means ensuring that multiple threads can call the API without corrupting internal state. The `ResourceLoader` API must handle multiple requests concurrently and safely. My design assumes the internal implementation handles synchronization, but this needs to be explicitly considered and tested.
    
2.  **Error Handling:** A good API makes it impossible to ignore errors. The `OnErrorCallback` is a deliberate choice to force the client to think about failure cases. A less robust design might return an error code that can be easily overlooked. The `Error` object passed to the callback should be rich with information, including the error code, a descriptive message, and potentially the HTTP status code.
    
3.  **Cancellation:** Providing a `cancel` method is crucial. Users don't always wait for a resource to load. A well-designed API allows for the graceful cancellation of in-flight requests. This frees up resources and prevents unnecessary network traffic.
    
4.  **Extensibility and `PIMPL` (Pointer to Implementation):** For a system library, the public API should be stable, even as the internal implementation changes. The **"PIMPL" idiom** is a design pattern that can help. The public `ResourceLoader` class would hold a private pointer to a concrete implementation class. This allows us to change the internals (e.g., switch from HTTP/2 to HTTP/3) without breaking the client's code.
    
5.  **Documentation and Examples:** A perfect API is useless if no one knows how to use it. A well-designed API includes clear, concise documentation for every method, class, and parameter, along with code examples illustrating common use cases.

---

You asked:
----------

Could you provide examples on how this api is used by a web browser or a facebook app?

---

Gemini Replied:
---------------

A web browser and the Facebook app both use an HTTP Resource Loader to manage a wide array of content, but they have different primary use cases and optimizations.

* * *

### Web Browser Usage

In a web browser, the Resource Loader is a core, foundational component. When you type a URL into the address bar, the browser's main thread initiates a chain of events handled by the Resource Loader.

1.  **HTML Document:** The browser requests the HTML file for the page. As it parses the HTML, it discovers other resources it needs to load, like CSS files, JavaScript files, and images. The Resource Loader then makes parallel requests for these resources.
    
2.  **Image Loading:** When the browser encounters an `<img>` tag, it calls the Resource Loader with the image URL. The loader handles the request, potentially from a local disk or memory cache. If the image is not cached, it fetches the data over the network and progressively sends chunks to the browser's rendering engine. This allows the image to be displayed even before it's fully downloaded.
    
3.  **JavaScript and CSS:** For `<script>` and `<link>` tags, the Resource Loader fetches the files. It prioritizes these requests because they are often "render-blocking," meaning the browser cannot paint the page until these files are processed.
    
4.  **Video and Audio:** When an HTML `<video>` or `<audio>` element is encountered, the Resource Loader is used to fetch the media data. It often uses **byte-range requests** to download the media in chunks, allowing for streaming playback. The loader will pre-fetch data a few seconds ahead of the current playback position to ensure a smooth experience.
    

The browser's Resource Loader is highly optimized for **parallelism** and **caching**, managing hundreds of resources per page to display the content as fast as possible.

* * *

### Facebook App Usage

The Facebook app uses a similar Resource Loader, but it's tailored to the specific needs of a mobile social media application. The focus is on providing a seamless, highly responsive experience, especially when scrolling through a feed with a continuous stream of new content.

1.  **Feed Pre-fetching:** As a user scrolls down the feed, the app's UI component will often notify the Resource Loader to start pre-fetching resources for posts that are about to come into view. For example, it might load the profile picture and the first image or video of the next post, even before the user sees it.
    
2.  **High-Priority Content:** Certain resources, like the profile picture of the person whose profile you're viewing, are given a very high priority. The app calls `ResourceLoader->load()` for this image, and the `RequestScheduler` ensures it is fetched and displayed immediately.
    
3.  **Large Video Streaming:** When a user taps to play a video, the app uses the Resource Loader to stream it in chunks. If the video is already partially cached from pre-fetching, the loader can start streaming from the local cache and seamlessly switch to the network for the remaining data. This is crucial for fast video playback and reducing data usage.
    
4.  **Cancellation:** If a user scrolls past a post before its media finishes loading, the app calls `ResourceLoader->cancel()` to stop the in-flight network request. This is a vital optimization for saving battery life and network bandwidth.
    

In both cases, the Resource Loader is a centralized, intelligent system that abstracts away the complexities of network requests, caching, and prioritization, allowing the client application to focus on rendering and user interaction.

<br> <br>

[Browser hacking: Let's build a caching resource loader](https://www.youtube.com/watch?v=AVsq0NqzUds) This video is relevant because it walks through the process of building a caching resource loader, which is a key component of a web browser's architecture.

[

![](https://www.gstatic.com/images/branding/productlogos/youtube/v9/192px.svg)

Browser hacking: Let's build a caching resource loader - YouTube

Andreas Kling · 2.4K views



](https://www.youtube.com/watch?v=AVsq0NqzUds)

---

You asked:
----------

can you go over http/2 multiplexing and http/3 quic

---

Gemini Replied:
---------------

HTTP/2 multiplexing and HTTP/3 QUIC are evolutions of the HTTP protocol designed to improve web performance, primarily by addressing the limitations of older protocols like HTTP/1.1. They both aim to solve the problem of **head-of-line (HOL) blocking** but do so in fundamentally different ways.

* * *

### HTTP/2 Multiplexing

HTTP/2 introduced **multiplexing** to allow multiple requests and responses to be sent concurrently over a single **TCP connection**.

*   **How it Works:** In HTTP/1.1, each request needed its own connection, or requests would be "queued" and sent one after the other. HTTP/2 solves this by converting requests and responses into a series of small, binary-encoded **frames**. These frames from different requests are then interleaved and sent over a single TCP stream. A header on each frame identifies which "stream" (request/response) it belongs to, allowing the receiving end to reassemble the data correctly. This eliminates the overhead of establishing multiple TCP connections and allows resources to be loaded in parallel.
    
*   **Head-of-Line Blocking (in HTTP/2):** While HTTP/2 solves the _HTTP-level_ HOL blocking (where a slow HTTP response blocks subsequent ones), it does not solve _TCP-level_ HOL blocking. Since HTTP/2 still runs on top of a single TCP connection, if a single packet is lost, all streams on that connection are blocked until the lost packet is retransmitted. This is a significant drawback on high-latency or lossy networks, like mobile connections.
    

* * *

### HTTP/3 and QUIC

HTTP/3 is the latest major revision of the protocol, and its core innovation is its use of **QUIC** (**Q**uick **U**DP **I**nternet **C**onnections) as its transport layer instead of TCP.

*   **How it Works:** QUIC runs over **UDP**, a simpler, connectionless protocol. However, QUIC adds the reliability, security, and multiplexing features that were previously handled by TCP and HTTP/2, but in a much more efficient way.
    
    *   **Independent Streams:** QUIC's multiplexing is native and operates at the transport layer. Each request/response stream is independent, so if a packet is lost for one stream, only that stream is affected. Other streams on the same connection can continue to send and receive data without being blocked. This is the key to solving the **TCP-level HOL blocking** that HTTP/2 couldn't.
        
    *   **Faster Handshake:** QUIC combines the transport handshake (like TCP's three-way handshake) and the cryptographic handshake (like TLS) into a single step. For a new connection, this reduces the number of round trips needed to establish a secure connection from two or three to just one. For a subsequent connection to a server, it can even achieve a **zero-round-trip (0-RTT)** time by using a pre-shared key, allowing data to be sent immediately.
        
    *   **Connection Migration:** QUIC connections are identified by a unique connection ID, not by the client's IP address and port. This allows a client (like a smartphone) to seamlessly switch between networks (e.g., from Wi-Fi to cellular data) without interrupting the connection, a major benefit for mobile applications.

---

You asked:
----------

what is the TLS handshake

---

Gemini Replied:
---------------

The TLS (Transport Layer Security) handshake is the process that initiates a secure communication session between a client (like a web browser) and a server. This cryptographic protocol ensures that the connection is secure by authenticating the server's identity, establishing the cryptographic algorithms to be used, and creating a shared secret key for the session. It's a foundational part of how HTTPS works.

* * *

### Key Goals of the TLS Handshake

The primary goals of the handshake are:

*   **Authentication:** The client verifies that the server it's communicating with is who it claims to be, preventing "man-in-the-middle" attacks.
    
*   **Key Exchange:** The client and server securely exchange information to create a **session key**. This key will be used for symmetric encryption, which is much faster than asymmetric encryption and is used to protect all subsequent data transferred during the session.
    
*   **Cipher Suite Negotiation:** Both parties agree on a **cipher suite**, a set of cryptographic algorithms that will be used for the session. This includes algorithms for key exchange, encryption, and message authentication.
    

* * *

### The Handshake Process (TLS 1.2)

While TLS 1.3 has a simpler process, the TLS 1.2 handshake is a good way to understand the core concepts. The process involves a series of messages exchanged between the client and server.

1.  **Client Hello:** The client initiates the handshake by sending a "Client Hello" message. This message includes the TLS versions it supports, a list of cipher suites it's willing to use, and a random string of bytes called the **client random**.
    
2.  **Server Hello:** The server responds with a "Server Hello" message, which contains the following:
    
    *   The agreed-upon TLS version and cipher suite selected from the client's list.
        
    *   Its own random string of bytes, the **server random**.
        
    *   The server's **digital certificate**, which contains its public key and is used to verify its identity.
        
3.  **Authentication:** The client receives the server's certificate and verifies its authenticity. It checks if the certificate is valid, has not expired, and was issued by a trusted **Certificate Authority (CA)**.
    
4.  **Key Exchange:** The client and server generate a shared secret, known as the **pre-master secret**.
    
    *   The client generates this secret and encrypts it using the server's public key from the certificate.
        
    *   The server then uses its private key to decrypt the message and retrieve the pre-master secret. Now, both the client and server have the three key pieces of information: the client random, server random, and pre-master secret.
        
5.  **Session Key Generation:** Using the three secrets, both the client and server independently generate a **master secret** and a unique **session key**.
    
6.  **Handshake Completion:** The client sends a "Change Cipher Spec" message to the server, indicating that it will now use the new session key for encryption. It then sends a "Finished" message, which is encrypted with the new key. The server does the same, and if both parties can successfully decrypt the other's "Finished" message, the handshake is complete and a secure connection is established.
    

This video on the TLS Handshake Explained by Computerphile provides a clear overview of the process.

[TLS Handshake Explained - Computerphile](https://www.youtube.com/watch?v=86cQJ0MMses)

[

![](https://www.gstatic.com/images/branding/productlogos/youtube/v9/192px.svg)

TLS Handshake Explained - Computerphile - YouTube

Computerphile · 604K views



](https://www.youtube.com/watch?v=86cQJ0MMses)

Sources

---

