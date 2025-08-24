# Design a Payment System

Multi-step Processes

[![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.66fdc8bf.png&w=96&q=75&dpl=e097d75362416d314ca97da7e72db8953ccb9c4d)

Evan King

Ex-Meta Staff Engineer

](https://www.linkedin.com/in/evan-king-40072280/)

hard

Updated Jul 10, 2025

* * *

###### Try This Problem Yourself

Practice with guided hints and real-time feedback

Start Practice

0:00

Play

Mute

0%

0:00

/

1:03:59

Premium Content

Closed-Captions On

Chapters

Settings

AirPlay

Google Cast

Enter PiP

Enter Fullscreen

## Understanding the Problem

**📸 What is [Stripe](https://www.stripe.com/)?** Payment processing systems like Stripe allow business (referred to throughout this breakdown as merchants) to accept payment from customers, without having to build their own payment processing infrastructure. Customer input their payment details on the merchant's website, and the merchant sends the payment details to Stripe. Stripe then processes the payment and returns the result to the merchant.

### [Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#1-functional-requirements)

**Core Requirements**

1.  Merchants should be able to initiate payment requests (charge a customer for a specific amount).
    
2.  Users should be able to pay for products with credit/debit cards.
    
3.  Merchants should be able to view status updates for payments (e.g., pending, success, failed).
    

**Below the line (out of scope):**

-   Customers should be able to save payment methods for future use.
    
-   Merchants should be able to issue full or partial refunds.
    
-   Merchants should be able to view transaction history and reports.
    
-   Support for alternative payment methods (e.g., bank transfers, digital wallets).
    
-   Handling recurring payments (subscriptions).
    
-   Payouts to merchants.
    

### [Non-Functional Requirements](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#2-non-functional-requirements)

Before defining your non-functional requirements in an interview, it's wise to inquire about the scale of the system as this will have a meaningful impact on your design. In this case, we'll be looking at a system handling about 10,000 transactions per second (TPS) at peak load.

**Core Requirements**

1.  The system should be highly secure
    
2.  The system should guarantee durability and auditability with no transaction data ever being lost, even in case of failures.
    
3.  The system should guarantee transaction safety and financial integrity despite the inherently asynchronous nature of external payment networks
    
4.  The system should be scalable to handle high transaction volume (10,000+ TPS) and potentially bursty traffic patterns (e.g., holiday sales).
    

**Below the line (out of scope):**

-   The system should adhere to various financial regulations globally (depending on supported regions).
    
-   The system should be extensible to easily add new payment methods or features later.
    

Here's how it might look on your whiteboard:

Payment System Requirements

## The Set Up

### [Defining the Core Entities](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#core-entities-2-minutes)

Let's start by identifying the core entities we'll need. I prefer to begin with a high-level overview before diving into specifics as this helps establish the foundation we'll build upon. That said, if you're someone who finds value in outlining the complete schema upfront with all columns and relationships defined, that's perfectly fine too! There's no single "right" approach in an interview, just do what works best for you. The key is to have a clear understanding of the main building blocks we'll need.

To satisfy our key functional requirements, we'll need the following entities:

1.  **Merchant:** This entity will store information about the businesses using our payment platform, including their identity details, bank account information, and API keys.
    
2.  **PaymentIntent:** This represents the merchant's intention to collect a specific amount from a customer and tracks the overall payment lifecycle from creation to completion. It owns the state machine from created → authorized → captured / canceled / refunded, and enforces idempotency for retries.
    
3.  **Transaction:** This represents a polymorphic money-movement record linked back to one PaymentIntent. Types include Charge (funds in), Refund (funds out), Dispute (potential reversal), and Payout (merchant withdrawal). Each row carries amount, currency, status, timestamps, and references to both the intent and the merchant. For simplicity, we'll only be focused on Charges as everything else is out of scope.
    

It's important to clarify the distinction between PaymentIntent and Transaction at this point as this can easily cause confusion. The relationship between them is one-to-many: a single PaymentIntent can have multiple Transactions associated with it. For example:

-   If a payment attempt fails due to insufficient funds, the merchant might retry with the same PaymentIntent ID but a different Transaction will be created.
    
-   For partial payments, multiple Transactions might be linked to the same PaymentIntent.
    
-   For refunds, a new Transaction with a negative amount would be created and linked to the original PaymentIntent.
    

From the merchant's perspective, they simply create a PaymentIntent and our system handles all the transaction complexities internally, providing a simplified view of the overall payment status.

In the actual interview, this can be as simple as a short list like this. Just make sure you talk through the entities with your interviewer to ensure you are on the same page.

A **Transaction** is a bit of a simplification to ensure we can focus on the important parts of the design with our limited time. In production you’d likely break this out into discrete types—Charge, Refund, Dispute, Payout, and the double-entry LedgerEntry rows that actually move balances. For interview purposes, though, collapsing them under a single polymorphic Transaction keeps the mental model tight: one PaymentIntent can spawn many money-movement events, each stamped with an amount, currency, status, and timestamps. That level of detail is enough to reason about idempotency, auditability, and failure handling without drowning in implementation minutiae.

Payment Entities

### [API or System Interface](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#api-or-system-interface-5-minutes)

The API is the main way merchants will interact with our payment system. Defining it early helps us structure the rest of our design. We'll start simple and, as always, we can add more detail as we go. I'll just create one endpoint for each of our core requirements.

First, merchants need to initiate PaymentIntent requests. This will happen when a customer reaches the checkout page of a merchant's website. We'll use a POST request to create the PaymentIntent.

`POST /payment-intents -> paymentIntentId {   "amountInCents": 2499,   "currency": "usd",   "description": "Order #1234", }`

Next, the system needs to securely accept and process payments. Since we're focusing on credit/debit cards initially, we'll need an endpoint for that:

`POST /payment-intents/{paymentIntentId}/transactions {   "type": "charge",   "card": {     "number": "4242424242424242",     "exp_month": 12,     "exp_year": 2025,     "cvc": "123"   } }`

In a real implementation, we'd never pass raw card details directly to our backend like this. We'd use a secure tokenization process to protect sensitive data. We'll get into the details of how we handle this securely when we get further into our design. In your interview, I would just callout that you understand this data will need to be encrypted.

Finally, merchants need to check the status of payments. This can be done with a simple GET request like so:

`GET /payment-intents/{paymentIntentId} -> PaymentIntent`

The response would include the payment's current status (e.g., "pending", "succeeded", "failed") and any relevant details like error messages for failed payments.

For a more real-time approach (and one that actually mirrors the industry standard), we could also provide webhooks that notify merchants when payment statuses change. In this way, the merchant would provide us with a callback URL that we would POST to when the payment status changes, allowing them to get real-time updates on the status of their payments.

`POST {merchant_webhook_url} {   "type": "payment.succeeded",   "data": {     "paymentId": "pay_123",     "amountInCents": 2499,     "currency": "usd",     "status": "succeeded"   } }`

Designing a webhook callback system is often a standalone question in system design interviews. Designing a payment system, complete with webhooks, would be a lot to complete in the time allotted. Have a discussing with your interviewer early on so you're on the same page about what you're building.

## [High-Level Design](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#high-level-design-10-15-minutes)

For our high-level design, we're simply going to work one-by-one through our functional requirements.

### 1) Merchants should be able to initiate payment requests

When a merchant wants to charge a customer, they need to initiate a payment request. In our system, this is accomplished through the creation of a PaymentIntent. As we stated in our Core Entities, a PaymentIntent represents the merchant's intention to collect a specific amount from a customer and tracks the lifecycle of the payment from initiation to completion.

Let's start by laying out the core components needed for this functionality:

Payment Intent Flow

1.  **API Gateway**: This serves as the entry point for all merchant requests. It handles authentication, rate limiting, and routes requests to the appropriate microservices.
    
2.  **Payment Service**: This microservice is responsible for creating and managing PaymentIntents. It interfaces with our database to store payment information.
    
3.  **Database**: A central database that stores all system data including PaymentIntents records (with their current status, amount, currency, and associated metadata) and merchant information (API keys, business details, and configuration preferences).
    

Here's the flow when a merchant initiates a payment:

1.  The merchant makes an API call to our system by sending a POST request to /payment-intents with details like amount, currency, and description.
    
2.  Once authenticated, the request is routed to the PaymentIntent Service (more on this later).
    
3.  The PaymentIntent Service creates a new PaymentIntent record with an initial status of "created" and stores it in the Database.
    
4.  The system generates a unique identifier for this PaymentIntent and returns it to the merchant in the API response.
    

This PaymentIntent ID is crucial as it will be used in subsequent steps of the payment flow. The merchant will typically embed this ID in their checkout page or pass it to their client-side code where it will be used when collecting the customer's payment details.

At this stage, no actual charging has occurred. We've simply recorded the merchant's intention to collect a payment and created a reference that will be used to track this payment throughout its lifecycle. The PaymentIntent exists in a "created" state, awaiting the next step where payment details will be provided and processing will begin.

### 2) Users should be able to pay for products with credit/debit cards.

Now that we have a PaymentIntent created, the next step is to securely collect payment details from the customer and process the payment. It's important to understand that payment processing is inherently asynchronous - the initial authorization response from the payment network is just the first step in a longer process that can take minutes or even days to fully complete. This is because payment networks need time to handle things like fraud checks, chargeback requests, etc.

###### Pattern: Multi-step Processes

Payment processing is a perfect example of the multi-step processes pattern. A single payment goes through multiple stages: authorization, capture, settlement, and potentially refunds or disputes. Each step can fail independently and may require retries or compensation. This requires careful state management and orchestration to ensure the payment workflow completes successfully while handling failures gracefully.

[Learn This Pattern](https://www.hellointerview.com/learn/system-design/patterns/multi-step-processes)

Our system needs to handle this asynchronous nature by maintaining the state of each payment and transaction, and keeping merchants informed of status changes throughout the entire lifecycle.

Let's expand our architecture to handle this critical part of the payment flow:

Process Payment Flow

1.  **Transaction Service**: A dedicated microservice responsible for receiving card details from the merchant server, managing transaction records throughout the payment lifecycle, and interfacing directly with external payment networks like Visa, Mastercard, and banking systems.
    
2.  **External Payment Network**: Shown as a dotted line in our diagram to demonstrate that this is external to our actual system, though a crucial part of the payment flow. These are the payment networks (Visa, Mastercard, etc.) and banking systems that actually authorize and process the financial transactions.
    

Here's how the flow works when a customer enters their payment details:

1.  The customer enters their credit card information into a payment form on the merchant's website.
    
2.  The merchant collects this data and sends it to our Transaction Service along with the original PaymentIntent ID.
    
3.  The Transaction Service creates a transaction record with status "pending" in our system.
    
4.  The Transaction Service directly handles the payment network interaction: a. Connects to the appropriate payment network and sends the authorization request b. Receives the initial response (approval/decline) c. Updates the transaction record with the initial status d. Continues to listen for callbacks from the payment network over the secure private connection e. When additional status changes occur (settlement, chargeback, etc.), receives callbacks and updates records accordingly
    
5.  The Transaction Service updates the PaymentIntent status as the transaction progresses through its lifecycle.
    

While it would be highly unusual for any of these details to be covered in anything but the most specialized interviews, you may be wondering how connections to payment networks work.

In short, payment networks operate on private, highly secure networks that are completely separate from the public internet. To connect with these networks, payment processors must establish dedicated connections through specialized hardware security modules (HSMs) and secure data centers that meet stringent [Payment Card Industry Data Security Standard (PCI DSS)](https://en.wikipedia.org/wiki/Payment_Card_Industry_Data_Security_Standard) requirements. These private networks use proprietary protocols with specific message formats (such as ISO 8583) and require formal certification processes to gain access. Unlike typical REST APIs, these connections often involve binary protocols, leased lines, and VPN tunnels with mutual TLS authentication.

This simplified architecture keeps the Transaction Service as the single point of responsibility for both managing our internal transaction records and interfacing with external payment networks. While this combines multiple concerns in one service, it reduces complexity and eliminates unnecessary network hops while still maintaining security through proper PCI compliance within the service.

### 3) The system should provide status updates for payments

After a payment is initiated and processed, merchants need a reliable way to determine its current status. This information is very important for business operations! Merchants need to know when a payment succeeds to trigger fulfillment actions like shipping physical products, granting access to digital content, or confirming reservations. Likewise, they need to know when payments fail so they can notify customers or attempt alternative payment methods.

Let's see how our existing architecture supports this functionality:

Payment Status Flow

Since we already have a PaymentIntent Service that manages PaymentIntents, we can leverage this same service to provide status updates to merchants. There's no need to create a separate service just for checking statuses.

Here's how the flow works when a merchant checks a payment's status:

1.  The merchant makes a GET request to /payment-intents/{paymentIntentId} to retrieve the current status of a specific PaymentIntent.
    
2.  The API Gateway validates the merchant's authenticity and routes the request to the PaymentIntent Service.
    
3.  The PaymentIntent Service queries the database for the current state of the PaymentIntent, including its status, any error messages (if failed), and related transaction details.
    
4.  The service returns this information to the merchant in a structured response format.
    

This simple polling mechanism allows merchants to programmatically check payment statuses and integrate the results into their business workflows.

The PaymentIntent can have various statuses throughout its lifecycle, such as:

-   created: Initial state after the merchant creates the PaymentIntent
    
-   processing: PaymentIntent details received and being processed
    
-   succeeded: PaymentIntent successfully processed
    
-   failed: Payment processing failed (with reason)
    

While this polling approach works well for many use cases, it's not ideal for real-time updates or high-frequency status checks. In a deep dive later, we'll explore how webhooks can be implemented to provide push-based notifications that eliminate the need for polling and reduce latency between payment completion and fulfillment actions.

Now that we've covered all three functional requirements, we have a basic payment processing system that can create payments, process payments securely, and provide status updates to merchants.

## [Potential Deep Dives](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery#deep-dives-10-minutes)

At this point, we have a basic system that satisfies the core functional requirements of our payment processing system. Merchants can initiate payments by creating a PaymentIntent, customers can pay with credit/debit cards, and merchants can view payment status updates. However, our current design has significant limitations, particularly around transaction safety, durability, and scaling to handle high volumes. Let's look back at our non-functional requirements and explore how we can improve our system to handle 10,000+ TPS with strong consistency and guaranteed durability.

### 1) The system should be highly secure

Let's start with security. For a payment processing system, there are two main things we care about when it comes to guaranteeing the security of the system.

1.  Is the person/merchant making the payment request who they say they are?
    
2.  Are we protecting customer personal information so that it can't be stolen or compromised?
    

Starting with #1, we need to validate that merchants connecting to our system are who they claim to be. After all, we're giving them the ability to charge people money, we better make sure they're legit! Most payment systems solve this with API keys, but there are different approaches with varying levels of security. Here are some options.

### 

Good Solution: Basic API Key Authentication

###### Approach

We can use standard, static API keys as the primary authentication mechanism for merchants. When a merchant onboards to our payment platform, we generate a unique API key (typically a random string like pk\_live\_51NzQRtGswQnXYZ8o) and store it in our database associated with the merchant's account.

For each API request, merchants include this key in the request headers, typically as Authorization: Bearer {api\_key} or a custom header like X-API-Key: {api\_key}. When our API Gateway receives a request, it extracts the API key, looks it up in the database, and identifies the corresponding merchant. If the key is valid, the request is authenticated and processed.

API Key

###### Challenges

Since API keys are static credentials passed in every request, they're vulnerable to interception through network sniffing, especially if transmitted over insecure connections. Once intercepted, an attacker can replay these credentials indefinitely, as there's no mechanism to detect or prevent replay attacks.

Additionally, and even more likely, these keys often end up hardcoded in client applications or configuration files, increasing the risk of exposure through code repositories or system backups.

For a financial system processing sensitive payment data, these vulnerabilities represent unacceptable risks that could lead to fraudulent transactions, data breaches, and significant financial and reputational damage.

### 

Great Solution: Enhanced API Key Management with Request Signing

###### Approach

To improve on the good solution, we can implement request signing which ensures that API requests are authentic, unmodified, and cannot be replayed by attackers. We can accomplish this with a combination of public API keys like before (to identify the merchant) and private secret keys (used to generate time bound signatures).

During merchant onboarding, we provide two keys: a public API key for identification and a private secret key stored securely on the merchant's server (never in client-side code). These keys are used for authenticating the merchant's server with our payment system, which is separate from how we handle customer card data (covered in the next section).

For each API request, the merchant's server generates a digital signature by hashing the request details (method, endpoint, parameters, body) along with a timestamp and a unique nonce using their secret key. This signature proves the request's authenticity and prevents tampering. This way, even if replayed, we'd know that the timestamp was outside our acceptable window or that the nonce was already used, allowing us to reject the request.

`// Example request with signature {   "method": "POST",   "path": "/payment-intents/{paymentIntentId}/transactions",   "body": {     // body here   },   "headers": {     "Authorization": "Bearer pk_live_51NzQRtGswQnXYZ8o", // API Key     "X-Request-Timestamp": "2023-10-15T14:22:31Z", // Timestamp     "X-Request-Nonce": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", // Nonce     "X-Signature": "sha256=7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069" // Hash of body   } }`

When our API Gateway receives a request, it:

1.  Retrieves the merchant's secret key based on the provided API key
    
2.  Recreates the HMAC signature using the same algorithm (SHA-256), secret key, and request data
    
3.  Compares the calculated signature with the one provided in the request headers
    
4.  Validates that the timestamp is within an acceptable time window (typically 5-15 minutes)
    
5.  Ensures the nonce hasn't been used before within the valid time window by checking the cache/DB
    

This approach uses HMAC (Hash-based Message Authentication Code) with a shared secret key, ensuring both authenticity (the request came from the merchant with the valid secret) and integrity (the request data hasn't been tampered with).

Now let's look at #2 - protecting sensitive customer data throughout the payment process. Allowing a bad actor to get a hold of someone else's credit card information can lead to fraud, identity theft, and a total loss of customer trust. Plus, there are strict regulations like PCI DSS that mandate how payment data must be handled. Let's look at the different approaches to tackling this challenge.

### 

Bad Solution: Server-Side Payment Data Collection

###### Approach

Merchants collect and process card data on their own servers before sending it to our payment system. The customer enters card details on the merchant website, these card details are forwarded to the merchants server over HTTPS, where they encrypt and transmit the data to our payment service. The merchant's server then receives authorization responses and transaction confirmations from our system.

###### Challenges

A key security consideration is that sensitive card data should never touch the merchant's servers. This is not only a best practice but also a requirement for PCI DSS.

Every merchant server becomes a potential attack target. If any merchant has vulnerabilities, customer card data is at risk. Merchants also bear enormous liability; a single breach could result in devastating financial penalties and reputation damage.

### 

Good Solution: Client-Side iFrame Isolation

###### Approach

Our big learning from the previous approach is that private data (like card numbers) should never hit the merchant's server. By handling this data ourselves, we protect merchants from compliance requirements and liability risks. Credit card information is transmitted directly from the client to our servers over HTTPS, bypassing the merchant's systems entirely.

To accomplish this, we provide a JavaScript SDK that loads iframes directly on the merchants client. Our SDK creates a secure frame on the merchant's webpage that loads directly from our domain. When customers enter card information, it goes straight to our servers without touching the merchant's systems.

The iframe creates a security boundary—the browser's same-origin policy prevents the merchant's code from accessing data inside the frame.

iframe

You can learn more about iframes and how they work [here](https://en.wikipedia.org/wiki/HTML_element#Frames).

###### Challenges

While this approach prevents card data from reaching merchant servers, it still has security limitations. The iframe's security depends on browser security policies, which could be compromised by sophisticated attacks. Additionally, the card data is only protected by HTTPS during transmission, meaning it could be exposed if the iframe is compromised. The iframe approach also limits customization options and can cause usability issues on mobile browsers.

### 

Great Solution: iFrame + Encryption

###### Approach

The most secure approach builds upon the iframe solution while addressing its limitations through multiple layers of security. Our payment system generates and manages encryption keypairs, and our JavaScript SDK utilizes our public key for encryption.

When a customer enters their card details in our iframe, our JavaScript SDK immediately encrypts the data using this public key before it even leaves their device. This means the card data never exists in an unencrypted form outside the customer's browser. Even if the merchant's site is compromised, the attacker would only get encrypted data that they cannot decrypt.

The encrypted data is then transmitted over HTTPS to our servers, where we use the corresponding private key (securely stored in Hardware Security Modules) to decrypt it. While HTTPS provides transport security, it's now protecting already-encrypted data, adding an extra layer of protection.

This multi-layered approach means that even if one security mechanism fails (e.g., if the iframe is compromised), the other layers continue to protect the card data. The data is encrypted before it leaves the customer's device, protected during transmission, and securely processed on our servers.

iframe with Encryption

### 2) The system should guarantee durability and auditability with no transaction data ever being lost, even in case of failures.

For a payment system, the worst thing you can do is lose transaction data. It would be both a financial and legal disaster. Every transaction represents real money moving between accounts, and regulations like PCI-DSS, SOX compliance, and financial auditing standards require us to maintain complete, immutable records of every payment attempt, success, and failure.

We need to track not just what the current state is, but the entire sequence of events that led to that state. When a customer disputes a charge six months later, we must be able to prove exactly what happened: when the payment was initiated, what amount was authorized, when it was captured, and whether any refunds were processed. A single missing record could mean inability to defend against chargebacks, failed compliance audits, or worse—being unable to determine the true state of customer accounts.

This durability requirement intersects with several other system needs. As we'll explore later, webhook delivery requires knowing which events have been sent to which merchants. Reconciliation with payment networks demands comparing our records against theirs. Fraud detection needs to analyze patterns across transaction history. All of these capabilities depend on having a durable, queryable audit trail as their foundation.

Let's examine different approaches to achieving this durability and auditability:

### 

Bad Solution: Single Database with Current State Only

###### Approach

The simplest (and worst!) approach is to store payment data in a traditional relational database, updating records in place as payment states change. When a payment moves from "pending" to "captured", we simply update the status field:

`UPDATE payment_intents SET status = 'captured',     captured_at = NOW(),     updated_at = NOW() WHERE payment_id = 'pay_123';`

This gives us the current state of every payment, which seems sufficient for basic operations. The database provides ACID guarantees, so we know updates are atomic and consistent. We can query current payment statuses quickly and build simple dashboards showing successful vs. failed payments.

For durability, we rely on database backups—perhaps daily snapshots plus transaction logs that let us restore to any point in time if disaster strikes.

###### Challenges

The massive problem is that we've thrown away history. When the update above executes, we lose the fact that this payment was in "pending" state, when that transition happened, and who initiated it. If a customer claims they were incorrectly charged, we cannot prove the sequence of events that led to the charge.

More critically, this approach is vulnerable to application bugs. A faulty deployment that incorrectly updates payment statuses would silently corrupt our data. By the time we notice, the original state is gone forever. Even with database backups, we might not know which records were affected or what their correct values should be.

From a compliance perspective, this approach fails most financial audit requirements. Auditors need to see not just current balances but the complete transaction history. They need proof that records haven't been tampered with. A mutable database where history can be overwritten provides neither capability.

Lastly, when payment disputes arise, customer support must manually piece together what happened from server logs, if they still exist. Debugging production issues becomes archeology rather than investigation.

### 

Good Solution: Separate Audit Tables

###### Approach

We can solve for most of the problem above with a more sophisticated approach that maintains separate audit tables alongside the main transaction tables. Every state change triggers an insert into an append-only audit log:

`BEGIN TRANSACTION;   -- Update main table   UPDATE payments  SET status = 'captured',        updated_at = NOW()   WHERE payment_id = 'pay_123';      -- Insert audit record   INSERT INTO payment_audit_log (     payment_id,      change_type,     old_status,     new_status,     changed_by,     changed_at,     metadata   ) VALUES (     'pay_123',     'status_change',     'authorized',     'captured',     'payment_service',     NOW(),     '{"amount": 2500, "auth_code": "ABC123"}'   ); COMMIT;`

This preserves history while keeping the main tables optimized for queries. The audit tables are append-only—we never UPDATE or DELETE audit records, providing some immutability guarantees. For additional safety, we might restrict database permissions so audit tables are insert-only even at the database level.

We can now answer compliance questions by querying the audit log. Customer disputes can be investigated by examining the complete state transition history. If the main tables become corrupted, we could theoretically reconstruct the current state by replaying all audit records.

###### Challenges

While this is much better, it still has significant limitations. The main problem is that application code must remember to perform both operations—updating the main table AND inserting the audit record. If a developer forgets the audit insert, or a bug skips it, we silently lose our audit trail with no way to recover the missing history.

Additionally, this approach forces your high-volume operational database to also store years of audit history. The same database handling 10,000 TPS of real-time payment processing must also manage ever-growing audit tables that will eventually contain billions of historical records. This coupling makes it impossible to optimize each use case independently—audit queries for compliance slow down operational queries, and operational schema changes become constrained by audit requirements.

### 

Great Solution: Database + Change Data Capture + Event Stream

###### Approach

A more robust approach, used by payment processors like Stripe, separates operational and audit concerns while guaranteeing consistency between them. We still use a traditional database for merchant-facing operations, but audit durability comes from an immutable event stream populated via Change Data Capture (CDC).

Here's how it works:

1.  **Operational Database**: Handles merchant API requests with optimized tables for current state queries. No audit tables needed—just pure operational data models.
    
2.  **Change Data Capture**: Monitors the database's write-ahead log (WAL) or oplog, capturing every committed change as an event. This happens at the database level, not application level, guaranteeing we never miss changes.
    
3.  **Immutable Event Stream**: CDC publishes all changes to Kafka, creating an append-only log of every state transition. Events are keyed by payment\_intent\_id and include the full before/after state.
    
4.  **Specialized Consumers**: Different services consume the event stream for their specific needs without impacting the operational database.
    

`# Example Kafka events from CDC {   "op": "insert",   "source": "payment_intents_db",   "table": "payment_intents",    "ts_ms": 1681234567890,   "before": null,   "after": {     "payment_intent_id": "pi_123",     "merchant_id": "merch_456",     "amount": 2500,     "status": "created"   } } {   "op": "update",   "source": "payment_intents_db",   "table": "payment_intents",   "ts_ms": 1681234568901,   "before": {     "payment_intent_id": "pi_123",     "status": "created"   },   "after": {     "payment_intent_id": "pi_123",      "status": "authorized"   } }`

The nice part about this architecture is that different consumers can materialize different views:

-   **Audit Service**: Maintains a complete, immutable history optimized for compliance queries
    
-   **Analytics**: Builds denormalized views for business intelligence
    
-   **Reconciliation**: Correlates our events with external payment network events (covered in detail later)
    
-   **Webhook Delivery**: Tracks which state changes need merchant notification (also covered later)
    

Granted, most of the above are out of scope for our requirements, but it's a useful callout that we recognize that the system will grow to require these things and this will make it easier.

CDC Architecture

For durability, Kafka provides configurable replication (typically 3x) across multiple brokers and availability zones. Events are retained for a configurable period (often 7-30 days) on disk, with older events archived to object storage for permanent retention. All events are automatically flushed to S3 for long-term auditability, ensuring we can reconstruct any payment's complete history even years later for compliance audits or dispute resolution. This gives us both hot data for operational use and cold storage for compliance.

This architecture provides the best of both worlds: merchants get sub-10ms API responses from an optimized operational database, while every change is automatically captured in an immutable event stream without impacting API latency. Since CDC operates at the database level, there's no reliance on application code remembering to write audit records—if a change commits to the database, it will appear in the event stream.

The separation of concerns allows independent scaling: your operational database handles 10k TPS focused purely on current state, while the event stream and its consumers handle audit, analytics, and reconciliation without competing for resources. This also satisfies compliance requirements with cryptographically verifiable immutability and provides the foundation for advanced capabilities like reconciliation and webhook delivery.

By implementing this hybrid approach, we achieve true durability and auditability without sacrificing the performance our merchants expect. The immutable event stream becomes the foundation for not just audit compliance, but for the advanced capabilities—reconciliation, analytics, and real-time notifications—that distinguish a professional payment system from a simple transaction processor.

A savvy interviewer might ask: "Isn't CDC a single point of failure? What happens if your CDC system fails and you miss critical payment events?"

This is a great question! CDC is technically a single point of failure - if it stops working, events stop flowing to Kafka even though database writes continue. Companies like Stripe handle this by running multiple independent CDC instances reading from the same database, each writing to different Kafka clusters. They also implement monitoring that alerts within seconds if CDC lag increases, and maintain recovery procedures to replay missed events from database logs if needed. For critical payment events, they might also implement application-level fallbacks that write directly to Kafka if CDC hasn't confirmed the event within a certain timeframe.

### 3) The system should guarantee transaction safety and financial integrity despite the inherently asynchronous nature of external payment networks

Payment networks operate in a fundamentally different way than our internal systems. When we send a charge request to Visa, Mastercard, or a bank, we're crossing into systems we don't control. These networks process millions of transactions across global infrastructure, with their own retry mechanisms, queue delays, and batch processing windows. A payment we consider "timed out" might still be winding its way through authorization systems, while another might have succeeded instantly but lost its response packet on the way back to us.

This asynchronous reality creates serious risks. The most dangerous is double-charging. A customer clicks "pay" for their $50 purchase, we timeout waiting for the bank's response, the merchant retries, and suddenly the customer sees two $50 charges on their statement. Even if we eventually refund one, we've damaged trust and created support headaches. The opposite risk is equally damaging: a payment succeeds at the bank but we never know it, so we tell the merchant the payment failed. The merchant never ships the product, the customer's card gets charged anyway, and now we have an angry customer who paid for goods they'll never receive.

The challenge isn't eliminating this asynchronous behavior, which is impossible. Instead, we need to build systems that acknowledge uncertainty and handle it gracefully. Fortunately, the CDC and event stream infrastructure we established for durability provides exactly the foundation we need to track payment attempts, handle timeouts intelligently, and maintain consistency even when networks misbehave.

Let's explore different approaches to handling this inherent uncertainty:

### 

Bad Solution: Assume Timeouts Mean Failure

###### Approach

A bad option is to just treat network communication like internal service calls. We set a reasonable timeout (typically 30-60 seconds) for payment network requests. If we don't receive a response within that window, we assume the payment failed and update our database accordingly.

This straightforward approach lets us give merchants quick feedback. After 30 seconds, they know to either fulfill the order or ask the customer to try again. Our database maintains a clean state with every payment marked as either succeeded or failed, never stuck in limbo.

###### Challenges

This approach is a disaster waiting to happen, and for obvious reasons. Network timeouts tell us nothing about what actually happened with the payment. Consider this scenario that happens daily in production systems:

1.  We send a $200 charge request to the customer's bank
    
2.  The bank approves and debits the customer's account
    
3.  The response packet gets delayed or lost in network congestion
    
4.  Our 30-second timeout triggers, we mark the payment as "failed"
    
5.  The merchant displays "Payment failed, please try again"
    
6.  The customer retries, creating a second $200 charge
    
7.  The customer now has $400 in charges for a $200 purchase
    

Yikes!

### 

Good Solution: Pending States with Manual Reconciliation

###### Approach

A better approach acknowledges that timeouts create uncertainty, not failure. Instead of guessing, we track unclear outcomes explicitly by expanding our payment status to include a "pending\_verification" state specifically for timeouts. We also create a separate table to track every attempt we make to charge a payment network, recording what we sent, when we sent it, and what reference ID the network assigned.

When a timeout occurs, we mark the payment as "pending\_verification" rather than failed. A background service then periodically queries the payment network using our recorded reference ID to determine what actually happened with the payment. This verification process runs independently of the merchant-facing API, gradually resolving uncertain states without blocking new payment processing.

Importantly, to prevent duplicate charges, we implement idempotency using a unique database constraint on merchant ID and idempotency key. When merchants retry with the same idempotency key, they get the existing payment record instead of creating a duplicate charge.

###### Challenges

The manual reconciliation burden is significant. Our financial team must review pending payments daily, check them against bank files, and update statuses manually. During busy periods like Black Friday, thousands of payments might be stuck in pending states, overwhelming our reconciliation processes.

More fundamentally, this approach still treats reconciliation as an exception case rather than a core system behavior. We're bolting verification onto a system designed for synchronous responses, creating operational complexity without addressing the root cause. The database bears the full load of both operational queries and reconciliation lookups, leading to performance degradation during peak times when we can least afford it.

### 

Great Solution: Event-Driven Safety with Reconciliation

###### Approach

The most robust approach leverages the CDC and event stream infrastructure we established for durability, treating asynchronous payment networks as first-class citizens in our architecture.

The key insight is to track our intentions before acting on them. Here's how the complete flow works:

1.  **Record the attempt**: Before calling any payment network, we write an attempt record to our database with the network name, reference ID, and what we're trying to accomplish. This triggers a CDC event capturing our intention.
    
2.  **Call the payment network**: We send the actual charge request with our configured timeout.
    
3.  **Handle the response** (branching based on outcome):
    
    -   **Success**: Update the attempt status to "succeeded" in the database, which triggers another CDC event with the successful outcome
        
    -   **Timeout**: Update the attempt status to "timeout" in the database, triggering a CDC event that the reconciliation service will process
        
    -   **Explicit failure**: Update the attempt status to "failed" with the failure reason
        
    
4.  **Automated reconciliation**: A dedicated reconciliation service consumes timeout events and proactively queries the payment network using our recorded reference ID to determine what actually happened.
    

This creates a complete audit trail where every step is captured as an immutable event, whether the call succeeds, fails, or times out.

The most powerful aspect is how we handle reconciliation. Payment networks provide two ways to verify transaction status: direct API queries for real-time verification and batch reconciliation files. These reconciliation files are comprehensive records of all processed transactions during a specific period, delivered on regular schedules (daily or hourly). They follow strict formatting specifications and serve as the definitive record of what actually happened in the payment network.

A dedicated reconciliation service correlates these external events with our internal attempts, updating payment statuses based on authoritative network data rather than guessing from timeouts. The reconciliation service continuously consumes events, proactively queries networks when timeouts occur, and systematically processes settlement files when they arrive.

Reconciliation

This architecture acknowledges that payment networks are asynchronous partners, not synchronous services. By tracking every attempt and its outcome through events, we build a system that handles uncertainty gracefully rather than fighting it.

In production payment systems like Stripe, transaction processing actually uses a two-phase event model:

1.  **Transaction Created Event**: Emitted when the transaction service begins processing, before the database write is complete. If this event fails to emit, the transaction enters a locked/failed state and can be retried.
    
2.  **Transaction Completed Event**: Emitted after the database write has successfully completed. If this event fails to emit, the transaction also enters a locked/failed state where further updates are blocked until the completion event is successfully emitted.
    

This two-phase approach allows the system to compare the "created" event data with the actual database state during retries. If the database write already occurred, the system only needs to re-emit the completion event rather than retry the entire transaction. This pattern provides stronger guarantees about data consistency between the transaction service and downstream consumers.

Most missing "completed" events in production are actually due to external system timeouts before the database write occurred, rather than event emission failures after successful writes.

Show More

The key to handling asynchronous payment networks is accepting that uncertainty is inevitable and building systems designed for eventual consistency. By combining a traditional database for synchronous merchant needs with an event stream for asynchronous network reality, we achieve both performance and correctness. This pattern, proven at companies like Stripe processing billions in payments, shows that the best distributed systems don't fight the nature of external dependencies — they embrace and design for them.

### 4) The system should be scalable to handle high transaction volume (10,000+ TPS)

We've come a long way! We have a payment processing system that meets just about all of our functional and non-functional requirements. Now we just need to discuss how we would handle scale. I like to save scale for my last deep dive because then I have a clear understanding of the full system I need to scale. It's likely that you already talked about scale (as we did a bit) during other parts of the interview, but we'll do our best to tie it all together here.

#### Servers

Let's start by acknowledging the basics. Each of our services is stateless and will scale horizontally with load balancers in front of them distributing the load. This is worth mentioning in the interview, but there is no need to draw it out; your interviewer knows what you mean, and this is largely taken for granted nowadays.

#### Kafka

Next, let's look deeper at our event log, Kafka. We are expecting ~10k tps. While [Kafka can support millions of messages per second at the cluster level](https://www.hellointerview.com/learn/system-design/deep-dives/numbers-to-know), it's important to note that each individual partition typically handles around 5,000-10,000 messages per second under normal production conditions. To comfortably support our 10k TPS requirement, we'd need multiple partitions—likely 3-5—to ensure both throughput and fault tolerance.

When partitioning, we need to guarantee ordering for transactions per PaymentIntent. This suggests partitioning our Kafka topics by payment\_intent\_id, which ensures all events for a given PaymentIntent are processed in order while allowing parallel processing across different PaymentIntents. This partitioning strategy balances our throughput needs with our ordering requirements—all state transitions for a single PaymentIntent (created → authorized → captured) will be processed sequentially, while different PaymentIntents can be processed in parallel across partitions.

For redundancy, we'd configure Kafka with a replication factor of 3, meaning each partition has three copies distributed across different brokers. This provides fault tolerance if a broker fails. We'd also set up consumer groups for each of our services, allowing us to scale consumers horizontally while maintaining exactly-once processing semantics.

Learn more about how to scale Kafka in our [Kafka deep dive here](https://www.hellointerview.com/learn/system-design/deep-dives/kafka).

#### Database

Our database should be approximately the same order of magnitude as our event log with regards to the number of transactions per second. With 10k TPS for our event stream, we can expect around 10k writes per second to our database as well. This is right on the edge of what a well-optimized PostgreSQL instance can handle this load, especially when combined with read replicas and proper indexing. To better handle the scale, we can shard our database by merchant\_id.

With regards to data growth, we can estimate that each row is ~500 bytes. This means we are storing 10,000 \* 500 bytes = 5mb of data a second, 5mb \* 100,000 (rounded seconds in a day) = 500gb of data a day, and 500gb \* 365 = ~180tb of data a year. This is significant storage growth that will require careful planning for data retention and archiving strategies.

For transactions older than a certain period (e.g., 3-6 months), we can move them to cold storage like Amazon S3 or Google Cloud Storage. This archived data would still be accessible for compliance and audit purposes but wouldn't impact our operational database performance. We'd implement a scheduled job to identify, export, and purge old records according to our retention policy.

For read scaling, we can implement read replicas of our database. Most payment queries are read operations (checking payment status, generating reports), so having multiple read replicas will distribute this load. We can also implement a caching layer using Redis or Memcached for frequently accessed data like recent payment statuses.

## Bonus Deep Dives

### 1) How can we expand the design to support Webhooks?

As mentioned earlier, a robust webhook system is a substantial system design topic in its own right - one that could easily fill an entire interview (and often does!). In the context of this payment processor discussion, we'll outline the high-level approach to webhooks rather than diving into all implementation details.

While our polling-based status endpoint works well for basic scenarios, merchants often need real-time updates about payment status changes to trigger business processes like order fulfillment or access provisioning. Webhooks solve this by allowing our system to proactively notify merchants about events as they occur.

Merchants provide us with two additional bits of information:

1.  Callback Url: This is the URL that we will POST updates to when we have them.
    
2.  Subscribed Events: This is a list of events that they want to subscribe to. We will notify them, at the callback url, when any of these events occur.
    

You hear real-time updates and you may think websockets or SSE! But no, these are often conflated concepts, but it's important to understand that webhooks represent server-to-server communication, not server-to-client (like websockets or SSE). The payment system's server sends notifications directly to the merchant's server via HTTP requests to a predefined endpoint. This is fundamentally different from client-facing real-time updates (like WebSockets or Server-Sent Events), which would deliver updates from a server to a browser or mobile app. Webhooks are designed for system-to-system communication and typically require the merchant to operate a publicly accessible endpoint to receive these notifications.

Here's how webhooks would work at a high level in our payment processing system:

Webhook System

1.  **Database Changes**: Our Transaction and PaymentIntent services update the operational database as payments progress through their lifecycle (created → authorized → captured, etc.).
    
2.  **CDC Events**: Change Data Capture automatically captures these database changes and publishes them to our Kafka event stream. These CDC events include payment status changes, transaction completions, and other state transitions.
    
3.  **Webhook Service**: We introduce a new Webhook Service that consumes from the same Kafka event stream as our other specialized consumers ie, Reconciliation. When the service receives a CDC event, it checks if the associated merchant has configured a webhook endpoint for that event type. If configured, it prepares the webhook payload with relevant event details, signs the payload with a shared secret to enable verification, and attempts delivery to the merchant's endpoint.
    
4.  **Delivery Management**: For each webhook, the delivery attempt is recorded with its status. If delivery fails, the Webhook Service implements a retry strategy with exponential backoff (e.g., retry after 5s, 25s, 125s, etc., up to a reasonable maximum interval like 1 hour).
    
5.  **Merchant Implementation**: On the merchant side, they would need to configure a publicly accessible HTTPS endpoint to receive webhooks. They must verify the signature of incoming webhooks using the shared secret to ensure authenticity. After verification, they would process the webhook payload and update their systems accordingly with the new information. Finally, they should return a 2xx HTTP status code to acknowledge receipt of the webhook, preventing unnecessary retries.
    

Simple example of a webhook payload:

`{   "id": "evt_1JklMnOpQrStUv",   "type": "payment.succeeded",   "created": 1633031234,   "data": {     "object": {       "id": "pay_1AbCdEfGhIjKlM",       "amountInCents": 2499,       "currency": "usd",       "status": "succeeded",       "created": 1633031200     }   } }`

In production systems, webhook delivery reliability is critical. Although we're not diving into all the details here, a complete webhook system would include features like idempotency keys, detailed delivery monitoring, webhook logs for debugging, and a dashboard for merchants to view and replay webhooks.

If this were a dedicated webhook system design interview, we would explore more intricate challenges such as exactly-once delivery semantics, handling webhook queue backlogs during outages, webhook payload versioning, and adaptive rate limiting to prevent overwhelming merchant systems. However, for our payment processor design, this high-level overview captures the essential functionality that would complement our core payment flows.

## [What is Expected at Each Level?](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

You may be thinking, "how much of that is actually required from me in an interview?" Let's break it down.

### Mid-level

For a mid-level candidate discussing a payment system, I'd primarily focus on your ability to establish the core functionality. You should be able to design a straightforward payment flow that handles the basic requirements of initiating payment requests, processing cards, and returning status updates.

You should demonstrate understanding of the need for security in payment systems and identify that sensitive card data shouldn't touch merchant servers. However, I wouldn't expect you to deeply understand all the nuances of payment security or complex tokenization approaches. When discussing consistency, a reasonable solution like assuming timeouts mean failure (even if not ideal) would be acceptable as long as you can explain the approach clearly. I'd be looking more at your problem-solving process than expecting a perfect solution to complex payment reconciliation challenges.

I'd anticipate that you might need guidance on deeper technical challenges, and that's perfectly fine. The interviewer will likely direct the conversation through the latter portions of the discussion.

### Senior

For a senior engineer, I'd expect you to move quickly through the basic functionality and drive the conversation toward the critical non-functional aspects of a payment system. You should proactively identify security and consistency as core challenges.

On the security front, you should propose a solution like the iframe isolation approach and understand its benefits in protecting sensitive card data. You should be able to articulate why this approach is superior to having merchants collect card data directly. For payment consistency, you should recognize that timeouts don't necessarily mean failure and propose a more sophisticated approach like idempotent transactions with pending resolution. You should be able to identify potential race conditions and explain how your design prevents double-charging.

For scaling discussions, I'd expect you to speak with confidence about how you'd implement horizontal scaling for services and databases, though you might need some prompting to explore more advanced topics like Kafka partitioning strategies or event sourcing.

### Staff+

As a staff+ candidate, I'd expect you to demonstrate deep expertise in payment systems design while also showing exceptional product and architectural thinking. Rather than rushing to complex solutions, you should first evaluate whether they're necessary. For instance, you might point out that for many payment scenarios, event sourcing with reconciliation is the right approach not just for technical reasons but because it aligns with how financial systems fundamentally work - acknowledging the inherently asynchronous nature of payment networks. For security, you should be able to explain multi-layered approaches like tokenization with client-side encryption, showing an understanding of the defense-in-depth strategy necessary for handling sensitive financial data.

You should proactively identify and address the most challenging edge cases in payment processing. For example, you might discuss how to handle payment network outages through fallback processing paths or how to design resilient reconciliation processes that guarantee eventual consistency with external payment networks.

###### Test Your Knowledge

Take a quick 15 question quiz to test what you've learned.

Start Quiz

Mark as read

Comment

Anonymous

Posting as Frankie Liu

​

Sort By

Old

Sort By

![Mikhail Tcirulnikov](https://lh3.googleusercontent.com/a/ACg8ocIHZb1TviUF35EKCd3obutg-kbaO5oX-EE7xneCWnsgK3fkoIyC=s96-c)

Mikhail Tcirulnikov

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaou1vcm0097ad07i8eghv19)

Thank you for event sourcing approach description here. It is brilliant application for it. How long we want to store events in Kafka?

Show more

6

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmapi8zf300oqad08ktql3g6a)

Great question, you typically want to retain events in Kafka for ~30 days max. The event log isn't meant for long-term storage, you'd archive the events to cold storage (S3, GCS) for audit/compliance, and maintain derived state in your databases.

Show more

18

Reply

W

walnatara2

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcz12c5b09alad08y4izqgar)

i think you need have regular snapshot/checkpointing also, so we can reply from derived state

Show more

0

Reply

![Илья Чубаров](https://lh3.googleusercontent.com/a/ACg8ocKMKBARZ4IvyH-htJ7nExn4tg9PKE5BeF7qm4qjhTjdB9AVRfo=s96-c)

Илья Чубаров

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaozlfyg00eaad08n6w13j7l)

Thanks for the great article! I'd like to ask about challenges "Idempotent transactions". It's not mentioned about logs and monitoring part. In order to answer any questions about what happened - enough to have logs (as a example - elk stack) - does it resolve this problem? Of course kafka as event sourcing mechanism resolve other problems.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmapia6h400ovad08d1jvxtq9)

Logs alone (even with ELK) won't solve the core problem because they're append-only records without guaranteed ordering or consistency guarantees. The issue with idempotency isn't just about debugging, it's about ensuring we can definitively prove the exact state of a payment at any point in time, handle retries correctly, and maintain consistency with external payment networks.

The event sourcing approach gives us ordered, immutable events that serve as the source of truth, which logs can't provide. Logs are great for debugging and monitoring, but they're not a replacement for proper transaction state management.

Show more

1

Reply

![Siddharth Chaudhary](https://lh3.googleusercontent.com/a/ACg8ocLTXu6bjB0zGAGXhVnkep8SRflBiq6KV-4cpcEZdkfm4JbrgPxq=s96-c)

Siddharth Chaudhary

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmapof31c0001ad08ahupg0fh)

I feel like designing a payment system requires so much domain knowledge! I think it's unfair to ask this question in a system design interview because the concepts you learned from designing other products/services are not applicable here. What do you think @Evan?

Show more

12

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmapogly60005ad0867tb69dc)

Completely agree. I was a bit surprised to see it requested as much as it was. It's not a question I would ask without providing a ton of background knowledge to the candidate upfront.

Show more

10

Reply

![Siddharth Chaudhary](https://lh3.googleusercontent.com/a/ACg8ocLTXu6bjB0zGAGXhVnkep8SRflBiq6KV-4cpcEZdkfm4JbrgPxq=s96-c)

Siddharth Chaudhary

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmapokda10005ad08kpg3ibde)

Yeah, that's because some companies ask this question to raise the difficulty level of their interviews especially in India. And everyone was asking for it because it's hard to find a good source to study this from. Thanks so much for taking the time to break this problem down. Learnt some new things, very helpful!!

Show more

7

Reply

![Phuong Le](https://lh3.googleusercontent.com/a/ACg8ocLXKeElj7ST0dmZk-sWBzBLYPhE589l3kPkoRBOinmX8__P_fFOBA=s96-c)

Phuong Le

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmau6y0ya01wqad08iqqlmb70)

I guess this question is usually asked by fintech companies e.g. checkout.com because it is closed to their internal systems, which helps them discuss it more comfortably, and also it can tell if a candidate have the relevant domain knowledge, which is a great bonus for the candidate.

Show more

2

Reply

Y

YelpingAmaranthBaboon850

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6hibst00bead09sbjp0jaf)

I was asked almost this question when I interviewed at Stripe last week.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6hj94g00dxad08tko33mse)

How'd it go? 🤞🏼

Show more

0

Reply

Y

YelpingAmaranthBaboon850

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6hq15q00fwad08gga2u8md)

Poorly. I hadn't seen this guide; just got a notification about it today. I was underprepared and the interviewer got really agitated about my API design immediately and wouldn't let me iterate on it or proceed. It was a weird experience.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6hr6st00ggad08dhsj6bx4)

Dang, super sorry to hear that 😔 Sounds like a frustrating experience

Show more

0

Reply

P

PleasantAquamarineHoverfly685

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmappxm320025ad08siavlxlw)

Evan - Waiting for the video explanation of this problem. Love your breakdowns and analysis. Hope the videos are in the pipeline!

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmappy48o001iad08bvr7c6am)

Yup! Working on adding videos to all premium content :)

Show more

20

Reply

V

VictoriousPeachGrasshopper634

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmar619wh001ead08v1cuudz5)

Cant wait for the video version!

Show more

2

Reply

N

nmonga88

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd0n4l44018zad098olvx0u5)

Great Blog, @evan when are you planning to have the video walkthroughs of the premium content? I feel Videos are much better for me to understand.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd0n5lkf016yad08nf8u06is)

in a couple days :)

Show more

4

Reply

N

nmonga88

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd0naivm015uad08q1qpwg7m)

Thanks Evan, also it will be awesome if you have a video on the recommendation model as this is coming a lot in the interviews recently. Something like Food recommendation service, Tiktok recommendations, spotify recommendations. As always, really appreciate your content.

Show more

3

Reply

![Vivek Tiwary](https://lh3.googleusercontent.com/a/ACg8ocJjflqocshnvEvuex4Xh43BNKPTUvLbEueHmmDNtlq6T8MuFNB-pw=s96-c)

Vivek Tiwary

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbs5i6zl001w08adqb617bl1)

Badly needed video for this one, takine hours to go through the written solution & taking notes.

Show more

1

Reply

A

Abhi

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmch8enc904qxad07hupcydn6)

+1 for video version of this, especially having an extensive deep dive on Security aspect.

Show more

0

Reply

M

ManyAzureImpala710

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaq4sbff0013ad08ikrc9e8d)

Quick suggestion: maybe change the service names to payment-> Sale/Order and Transactions-Payments. That's what we call at PayPal.

Show more

1

Reply

M

MatureOliveHalibut263

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaq5yyvk001xad08nwntbhya)

Hey Evan, thanks for your content. I got asked the dropbox question during a mock interview and was asked lots of questions that was not covered in your design (not saying it should). So I have been using ChatGPT to generate extra questions based on your solutions, but its not as good.

I'd strongly recommend adding 5-10 questions at the end where users can brainstorm themselves. You don't have to provide answers. For example, one question I was asked on dropbox question was: what if there are millions of small files and you need to upload them all. Apparently dropbox faced this problem in production and had to add a feature to compress and bulk upload small files to save on time.

Thanks again, love ur content!

Show more

7

Reply

F

FlyingTomatoHeron745

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaq617du003dad0842q26qu6)

What follow-up questions were you asked?

Show more

0

Reply

K

Karthik555

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmc8ij12702j9ad088db96vpe)

I think you can explain the concept of chunking its and provide and track it from the backend, customized for Dropbox.

Show more

0

Reply

F

FlyingTomatoHeron745

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaq63o1o003jad08t05h8lig)

Thanks for adding email notifications for new questions being added! would appreciate the same when you release premium videos as well :)

Show more

0

Reply

U

UltimateBronzeLemur440

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaqarc2s005aad08ioq5o5dh)

The description of the client side encryption approach is incorrect and misses the point of PCI. Even encrypted card numbers are considered pci data, so passing an encrypted card number to your server would bring that entire server, api gateway, network, and all connected services into PCI, which is not ideal. The point of the iframe is to load a form that comes from a pci compliant domain that is different from the parent origin.

Obviously only people who have worked on payment systems would know this in detail, so that brings me to my question: what is the point of this site or even these interviews? Is it for people to memorize the answers to commonly asked questions (cheat), or is it to show problem solving skills. I’m genuinely curious (and only being a little combative).

Show more

4

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaqumej500apad083txc43wb)

The goal definitely isn't to memorize perfect answers (you won't know every domain and perfect answers don't even exist), but to show you can reason about tradeoffs and constraints while being open to guidance when discussing unfamiliar territory. A payment system interview isn't testing if you know PCI compliance details, it's testing if you recognize that handling sensitive financial data requires special consideration and can propose reasonable approaches to address that. Or, given input from your interviewer, can you reason about the tradeoffs of multiple solutions. Like LeetCode, system design is about patterns. These articles aim to give candidates exposure to patterns they can apply to novel problems, not "correct" answers to memorize.

Show more

21

Reply

Q

QuietPinkOpossum868

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaqcgc9y0005ad08ijuyvc3v)

When sending webhook back to merchant, is it encrypted with the merchant's public key? A bit confused about the security part.

Is the high level idea that: for merchant authentication, they encrypt with their private key that they store securely, and we verify their identity with their public key. When sending webhook back, we encrypt with their public key, which they then decrypt with their private key.

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaqunbc700arad084c01r80r)

Webhooks aren't typically encrypted with public/private keys - that'd be overkill. Instead, we send them over HTTPS and include a signature (HMAC) created using a shared secret. The merchant verifies the signature to ensure the webhook came from us and wasn't tampered with.

Show more

3

Reply

Q

QuietPinkOpossum868

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaqy29fk00ddad08vhpgs6yu)

So we need to store both the public key and a symmetric key for the merchant?

Show more

0

Reply

![Volodymyr Trubachov](https://lh3.googleusercontent.com/a/ACg8ocJQHm2-YasV8OQcL2_nhTJDfThGnX5qIKZCYJ3uP45HMKtL4FmUvg=s96-c)

Volodymyr Trubachov

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbewhh0300csad08n6gjsxfx)

I would consider a design when webhooks are more like notifications of a status change for payment XYZ. Then the merchant's system uses the regular means to query for the payment status. It makes the security and webhook implementation much simpler.

Show more

0

Reply

![p Pig](https://lh3.googleusercontent.com/a/ACg8ocLwPtz170CgiCP_ETtnMO0fEXKmXy7x-a6LNdXIAkZUpCqMPQ=s96-c)

p Pig

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaqjiaee007dad08z1q79wwe)

great article！ I have one question, if the client initiates a payment request twice for the same payment due to network jitter and passes a different idempotency key, can our system only treat it as two payments?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaquock100avad08vdxzrmwv)

We need to generate the idempotency key on the client side by hashing the relevant payment details (amount, currency, order ID, etc). You can see that Stripe recommends this in their dev docs. Two requests for the same payment should produce the same hash, regardless of network issues. If the merchant's implementation isn't doing this and is generating random keys, then ya, our system would treat them as separate payments

Show more

5

Reply

![Omar Mihilmy](https://lh3.googleusercontent.com/a/ACg8ocKvgx5vPaqwzjwGib2tCK0DD1C2P79Ta3ZxOit12LsMf45LAT6v=s96-c)

Omar Mihilmy

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmc18ulhh0bmj08adtf7g7qld)

https://docs.stripe.com/api/idempotent\_requests

Show more

0

Reply

![Kartik Dutta](https://lh3.googleusercontent.com/a/ACg8ocKYwiZrWfQW4-orGP3B5id1__SEB1u-g4_g0uwcv4u6y57Wk7yk=s96-c)

Kartik Dutta

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaqo8gce0078ad08mzd24ci8)

Could we flip the order of event sourcing and the database? Let all the writes/updates be written to the database, then using change data capture, we will get a time-consistent log of what event took place at what time, and since it's a single database instance, the data timestamps will also be consistent?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaquphvw00azad08mlasijt5)

Nope, I wouldnt. Flipping the order would defeat several key benefits of event sourcing. The database becomes a single point of failure for writes, and you lose the ability to rebuild state from a complete history of events. CDC is also typically async, so you can miss events if the DB crashes before CDC picks them up. Event sourcing gives you a guaranteed, ordered log of all state changes that you can use for audit, replay, or building new projections. CDC is more suited for data replication or ETL pipelines where eventual consistency is acceptable.

Show more

1

Reply

Q

QuietPinkOpossum868

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmarkemi7007dad088lkq7ce8)

What if you wrote to the new data database as well as the "change" / history to the database in one transaction? Then send an event to basically tell other parts of the system about the change (e.g. analytics, webhooks)

Show more

1

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd7m5wp404xfad08pdoef4b4)

but we are doing event source with CDC only? , why we are saying saying event source vs CDC isnt we doing event source using CDC?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd7m9nu504wiad081a1ams9g)

The article was updated on the 11th. Comments before then are referring to an only write up.

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd7md18604ztad08d66gtexy)

ohk , but still here we are not using event sourcing? if not why?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd7meirz04xkad08sv4x93nz)

Its possible, but its makes idempotency challenging. this is best of both worlds. Immediately write to DB and have an event stream for any async processing

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd7nc7k805bmad08tjq98q1w)

so what I understood , now we are not using event sourcing ,earlier in this design we were using

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd7m7gji04xtad08gvwnjflb)

hey , I couldnt understand your question , can you expplain please?

Show more

0

Reply

![Le Hoang](https://lh3.googleusercontent.com/a/ACg8ocJ8oIYpVQuQwdBfzoFjZsEfS3-nN0QFPL4H6xFGHX9aNz4s4A=s96-c)

Le Hoang

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmar31aj800oiad07bhwvy92j)

Thanks a lot for high quality content, I've been waiting for Payment System for so long

Show more

1

Reply

![Julian Boilen](https://lh3.googleusercontent.com/a/ACg8ocI0zpG-CrL3RdGwBV_oS6fPtbsPLTZsgz0HJLPyQIo6Htv6aSHL=s96-c)

Julian Boilen

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmar90f5h005qad09rluiwfar)

Fun fact: Stripe uses Mongo, and not the newer version with transactions. There's an ad-hoc locks system built on top. You really can make any database work for anything haha.

Love the event-sourcing idea. Stripe doesn't do this but I built an event sourced billing system once, but we weren't willing to give up consistency so ended up putting the events in postgres in the same transaction as the projections.

Show more

4

Reply

![Stefan Mai](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstefan-headshot.05026b70.png&w=96&q=75)

Stefan Mai

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmar914wk005cad088s7k6xpk)

TIL! Thanks for sharing.

Show more

0

Reply

A

Abhi

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmch90slp04i9ad08r2qzoesp)

Great share, even Amazon Payments uses DynamoDB now.

Show more

0

Reply

![Kartik Dutta](https://lh3.googleusercontent.com/a/ACg8ocKYwiZrWfQW4-orGP3B5id1__SEB1u-g4_g0uwcv4u6y57Wk7yk=s96-c)

Kartik Dutta

[• 3 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmar9w33s0080ad080m5fof9c)

In the event sourcing architecture, with the payment and transaction service no longer accessing the database (as per the diagram), how would we ensure we are not producing duplicate messages to the Kafka topics? As far as I understand, Kafka transaction semantics only apply when reading and writing from Kafka topics.

Will the services continue to access the database for idempotency key-related checks to ensure that no duplicate events are produced, in case of merchant-side payment retries?

Show more

0

Reply

H

HushedCyanWalrus115

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbha2vlt00q6ad08f8x5wcrd)

Also, a bit confused on this

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbha432a00kkad08hjgi2l66)

Still needs to read the DB, just missing an arrow :)

Show more

1

Reply

H

HushedCyanWalrus115

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbhbsd4m00thad084laq4ixp)

Thanks for the quick response! 🐐

Show more

0

Reply

![Peter George](https://lh3.googleusercontent.com/a/ACg8ocKfLC6ZX2RPBZyCs5DJJ_B2U2vfuQsHOSmc5PBxg-IvqEV_bw=s96-c)

Peter George

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmasadelv00v0ad081ip2b2hm)

Great article. One wish - it would be nice if the final diagram showed where kafka, webhooks, and the other deep dive stuff would slot in on the diagram. For those interested in further study, Byte Byte Go has two similar/relevant designs: Payment System (the merchant server in this article) and Digital Wallet (like, PayPal - covers the event sourcing stuff too)

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmasldras00ccad07nxt6txhg)

The final diagram has Kafka, webhooks (high level), etc. What do you feel is missing?

Show more

0

Reply

![Peter George](https://lh3.googleusercontent.com/a/ACg8ocKfLC6ZX2RPBZyCs5DJJ_B2U2vfuQsHOSmc5PBxg-IvqEV_bw=s96-c)

Peter George

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmasulagj006pad08pb7r5uuf)

Oh wow, I must have been looking at the wrong one - can't believe I missed all that

Show more

0

Reply

A

abrar.a.hussain

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmasf72r50019ad087gxhi74f)

> When partitioning, we need to guarantee ordering for transactions per merchant and per payment. This suggests partitioning our Kafka topics by merchant\_id, which ensures all events for a given merchant are processed in order while allowing parallel processing across different merchants.

Wouldn't you use payment\_id here instead? Main reason you'd consider a partition key over round-robin is that you need strict ordering, and that mainly applies to a payment\_id not a merchant (different purchases from the same merchant aren't coupled). Thinking this just gives you lopsided partitions.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaslc34l00c5ad07n2xz0773)

Yah fair enough, I agree that ordering matters on the payment not the merchant. If you ran into hot partitions with merchants then this would be totally reasonable (could also start here).

Show more

1

Reply

A

Abhi

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmch9708t04klad08b6fxv81m)

+1, to resolve hot partitions (compositeKey, etc) on merchantId might also force us to redistribute the messages in the partitions losing the initial ordering.

Show more

0

Reply

X

XerothermicBlueGibbon404

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmasodtoq00kuad08mmodzd15)

Great article, thanks for writing this!

I have 2 questions:

1.  Are we storing the card details in our db? If yes, can you mention the card fields in the schema please? Feel like it’s important piece of information.
    
2.  It’s not clear to me if the money is going in our bank account or in the merchant’s bank account. Could you please clarify on this? And also specify how this would reflect in the schema.
    

Show more

1

Reply

A

Abhi

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmch9c2vq04lvad0874vmv5ok)

For #1, it seems we're storing it in our DB by decrypting using the private keys, but not sure from the write up - Explicitly callout of this will help. For #2, I think money is directly moved by networks and not stored with us.

Show more

0

Reply

G

ghanekar.omkar

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmastrsyf003sad08cjr6pmb2)

I'm not sure I understood the math (1K writes/sec) under the Database section of 3rd deep dive. https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#database

> Our database, given that it is storing projections of data, should be up to an order of magnitude smaller than our event log with regards to the number of transactions per second. 1k write per second in the context of modern databases is not a lot, we can handle that with a single well optimized postgres instance.

Our initial non fn requirement states 10K TPS which I assume to mean 10K unique payment transactions and not 10K overall interim states. If that assumption is right, it'd mean 10K DB entries/second and many more messages on the queue due to each transaction going through up to 4-5 interim states. Is my assumption right or am I misinterpreting something?

Show more

8

Reply

A

Abhi

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmch99qqp04l7ad08x3iwot8r)

I have the same question!

Show more

0

Reply

![Yuchen ZHANG](https://lh3.googleusercontent.com/a/ACg8ocJkfQyTCP6qedqfmsl0b3SXRPpUTA0B0EaVUHJJi4qv6y0K7w=s96-c)

Yuchen ZHANG

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaswpgk1009gad080zdya9ik)

Would it be okay if we don't have Kafka in the design, but just have a "event table" representing event for payments and transactions? We can still query this table for all events and sort it by timestamp.

Show more

0

Reply

![yotam oron](https://lh3.googleusercontent.com/a/ACg8ocJA58f0291Lzs534Q9iVQ5362TszS68DVXmHJsn9S9go2Bp-QPYcA=s96-c)

yotam oron

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmatbqjnl00llad08k96wwlz9)

As usual - exceptionally well written.

One enhancement I was thinking about - wouldn't it be good to immediately send all the events coming into Kafka to S3, with something like Athena over it, so we can properly query a specific transaction (and potentially generate reports on a specific merchant or set of transactions)?

Keep on the good work and just take my money!!!

Show more

0

Reply

G

GenuineSalmonMouse668

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmatqf65v01b6ad08f0e5jkvt)

Can we expect a video anytime soon? Thanks for the write up as well!

Show more

0

Reply

![Jerry Lee](https://lh3.googleusercontent.com/a/ACg8ocIbEQKYtKTRVdMYDjQglbBaKfAONGWVkRVFBsM-p5kbIbjywg=s96-c)

Jerry Lee

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaty0sif01fyad08thqkhoo0)

Thanks for the great writ up, Evan. If we need to change this to a general addmission tickets/inventory management scenario. Should we use a redis cache to hold the resverve amount of products and apply this? If so we need to handle inconsistency between the redis cache and our database(maybe DB just dont need to know the resverved items at all )Does it matter where the hold reversed items information either in our DB or in our distributed lock cache

Show more

0

Reply

Q

QuietPinkOpossum868

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmauaz16q026lad08bm7u5mv4)

How can we return a payment ID to the client if we put the payment request onto the Kafka queue in the event sourcing architecture?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaufw0bh02fnad092wcb4k3e)

Good question. We don't need to wait for Kafka. The Payment Service can first write directly to the database to create a payment record and get an ID, then publish an event to Kafka with that ID. The event sourcing pattern doesn't require every state change to go through Kafka first, it just requires that we eventually record all changes as events. This is a common pattern called "dual writes" where we write to both the database and event stream. The diagram should have another arrow potentially but dont want to overwhelm.

Show more

3

Reply

Q

QuietPinkOpossum868

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaum3kom02q6ad08ko5e6oq6)

I thought of using Redis to generate keys in the beginning (similar to how they're generated in the tiny url problem). But to my understanding Redis is not strongly consistent when it falls over in high availability mode, and I think we would need that strong consistency here. Alternative was to use something strongly consistent and replicated just to generate ID (e.g. Spanner, CockroachDB).

I also thought of "writing to table first then send to Kafka," (maybe via CDC) but wasn't sure how to handle the case where the database write succeeds and Kafka event send fails. But I found this transactional outbox pattern that seems to solve the problem https://www.confluent.io/blog/dual-write-problem/. (Btw the architecture described in this article is more similar to the listen-to-yourself pattern in the Confluent article)

Show more

1

Reply

![Phuong Le](https://lh3.googleusercontent.com/a/ACg8ocLXKeElj7ST0dmZk-sWBzBLYPhE589l3kPkoRBOinmX8__P_fFOBA=s96-c)

Phuong Le

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmauyxvyq02zzad088o3fv5na)

@QuietPinkOpossum868 I think the key design here is more about Event-sourcing than Listen-to-yourself but they are not mutually exclusive anyway. You can use both in your system. And by the way, thanks for mentioning the Confluent series 🙏. It is a gold mine for system design learners.

https://www.youtube.com/watch?v=U3o9Br6JsY8&list=PLa7VYi0yPIH0IpUKXb3q7NSjpJGO9GGGZ

Show more

0

Reply

A

Abhi

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmch9ozrm04ozad08harc1x0e)

Thanks for sharing!

Show more

0

Reply

W

walnatara2

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcz1ei8609dfad08ozgsmq53)

for dual writes, what happen writing to kafka is failed(?) I have once a problem in my current company where write to kafka is failed, but database is successfull.

In my current company, we recover through logs (which might be not reliable) and also through merchant complains.

Show more

0

Reply

X

XerothermicBlueGibbon404

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmauladbm02thad08dymg4flp)

also, it's always possible to generate the id in payment service, publish it to kafka and return to client

Show more

0

Reply

![Liran Jiao](https://lh3.googleusercontent.com/a/ACg8ocL9mVCDc2tZg0cRx_ImcFTZz1ZlRoBMyOV3J4EVbUF3-ZBGBw=s96-c)

Liran Jiao

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmauly4cu02p0ad097lulj571)

> The overall flow works like this:
> 
> 1.  Our Payment and Transaction services act as producers, writing events to Kafka whenever state changes occur
> 2.  Kafka durably stores these events in topic partitions, making them immutable and ordered
> 3.  Dedicated consumer services read these events and update the corresponding database tables (Payment and Transaction tables)
> 4.  Our database tables become read-optimized projections of the event log, used for serving API queries

If the Payment and Transaction services act as producers to Kafka, then these two services **cannot be aware of whether the sending events are written into DB successfully, or not**.

Will this cause the system internal inconsistent?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmav9bfx5033uad08q5fgitys)

Nope, this is actually a common misconception about event sourcing. The producers don't need to know about DB writes because Kafka itself is our source of truth - the DB is just a read-optimized projection. If DB writes fail, our consumers can replay events from Kafka to rebuild the DB state. This is actually one of the key benefits of event sourcing, we maintain consistency through the event log, not the DB.

Show more

0

Reply

![Aleksey Klintsevich](https://lh3.googleusercontent.com/a/ACg8ocL0px7VPQXZNEr2_xkgRSVls_QCeW-wbucIiICgpKJklg=s96-c)

Aleksey Klintsevich

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmaxng12100ztad08f9l0we10)

How about DB write succeeds, but Kafka call fails?

You now have an inconsistent view between both systems

Show more

2

Reply

![Rahul Garg](https://lh3.googleusercontent.com/a/ACg8ocLg1LAwUvjgYc05Syhag0OzUatbBN9Bcxatf3DTThrFQU77k7I=s96-c)

Rahul Garg

[• 3 months ago• edited 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmavep6ne03iiad08tn9t3868)

Hi, just wanted to say can you pick new questions from most popular questions from real questions tab. and also can we have a post update for web crawler with limited resource constraints as well. even on web crawler page most popular comment is that. i think alot of people having issues with that.

Show more

0

Reply

![Lovedeep Singh](https://lh3.googleusercontent.com/a/ACg8ocLz6zxfhEAbASWcvmGuiyB3iNT3lBO3JcMzEdBITqbNoFddEt_V=s96-c)

Lovedeep Singh

[• 3 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmb29eodi007vad0809zj0v8h)

Can we get a video on this as well? That would be really helpful ...

Show more

0

Reply

O

OkCrimsonSkunk471

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmb2f79y80011ad084nvi0bqe)

Hi Evan, In the system design round that my friend failed, He drew each service owner their DB (not a DB like you). Such as: payment-service hold paymentDB and transaction-service owner transactionDB. And in case after successful transaction, transaction-service updates payment status, He wrote direct connection from transaction-service to paymentDB. And the interviewer was not satisfied with that. He said in microServices architecture, each service will own their DB to extend independence and all requests should be accessed by payment-service. What do you think about that? Thanks and best regards,

Show more

2

Reply

W

WeeklyMoccasinLlama156

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmb3wrh4u01rxad08nwagj0nh)

How did 10000 TPS get converted to 1K writes per second to DB in the NFR#3 (The system should be able to handle 10k+ TPS)

Show more

1

Reply

Z

ZanyOliveEmu952

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmb4b6jbz02yzad08qvo5buon)

Mocked some candidates, find it hard to get easy understanding with the ones who haven't worked in payment domain. Suggest not to ask as mock questions for the sake of time...

Show more

1

Reply

![Abhishek Singh](https://lh3.googleusercontent.com/a/ACg8ocIxrpJa-ZQ_ZMPUsQRfrOeExeamAhNIUloQb8M5HwTpqDpI-A=s96-c)

Abhishek Singh

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmb62x2rm006zad073qf9l8lm)

Can we please have a question breakdown for webhook delivery system as well?

Show more

0

Reply

![Oussama H](https://lh3.googleusercontent.com/a/ACg8ocJa3ZWTqotfg35U1Ga4aimQBVn7hoX0x4QD0vXTOgTbPv9L5Q=s96-c)

Oussama H

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmb6wbk8200vuad08jtqu2tip)

Is a video planned on this one ?

Show more

0

Reply

S

SubtleBronzeParakeet321

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmba41vdk0021ad08rhgxuxoa)

This is very helpful, one question through, on a good hardware setup, Kafka can support 10k without problem, no? is the partition here we are talking about is how many brokers we need?

Show more

0

Reply

![Nitin Sharma](https://lh3.googleusercontent.com/a/ACg8ocJOCGPyried71gZAPdk11e4pu2dx6FQ-YG2r-AFm6G7MhA_a2GG=s96-c)

Nitin Sharma

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbcchfy101t6ad08pwzis32l)

Why do we need to have separate Payment gateway Service to connect with external payment network? It increases the probability of payment failure in case connection between transaction service & payment gateway service fails. Additionally, it's not like Transaction Service & Payment Gateway service are needed for different type of requests, for which we need to scale independently. Any idea?

Show more

0

Reply

![ziwen zhao](https://lh3.googleusercontent.com/a/ACg8ocKzCpfwv8lb2SfUk5la1IVl18nlpre-uclfHXs_80vF8USLs9VDFA=s96-c)

ziwen zhao

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbd8bw7801aiad083e9shap7)

I think there is a missing info when making transactions, what about the merchant/recipient account? The payment gateway needs to know both the payer credit card info, and also where the money is going to. Am I miss understanding some concept here?

Show more

0

Reply

![Hariharasudan Sreedharen](https://lh3.googleusercontent.com/a/ACg8ocJgH_SDQ3m2YpHIMsUG1ODeUEDiLuUCJNbH25pruIV8qFYM6A=s96-c)

Hariharasudan Sreedharen

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbdnkirk01nyad08ybwa6v65)

Great article as always, and thanks to both Evan and Stefan for sharing such valuable content; really appreciate the depth you bring to these discussions!

I had a follow-up question that I was hoping to clarify with you. Having worked in fintech for a few years, I’ve seen different approaches to handling consistency, and the suggestions in the article seem to highlight a trade-off between two patterns:

The DB-first approach, where data is persisted first and then using the latest version of the data, an event is published (commonly seen in traditional setups).

The event-first or event-sourced approach, where the event is the source of truth and the database acts as a projection.

I’ve mostly come across the former, where the database update triggers the event generation. The event-sourced approach, while powerful, seems to require careful versioning and introduces more or less an eventual consistency (not as fast as the normal data persist at least with Kafka coming into picture).

Would love to hear your thoughts on the choice made here and whether I might be misunderstanding something. Looking forward to more such insightful content!

Show more

2

Reply

![Abhirup Acharya](https://lh3.googleusercontent.com/a/ACg8ocIvglVSFWD-aHnP4JA7VE5p8-y07RbbiVgCV4zZ6HjqSCRqaA=s96-c)

Abhirup Acharya

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbdroyxd01gzad081073n4yr)

This misses and an important requirement of idempotency how to make sure the same payment is not initiated twice?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbfjpxx100ajad08pgg8udtd)

We talk about idempotency in the "Event Sourcing with Reconciliation" section. Each payment gets a unique idempotency key, and the Transaction Service stores this key with the result. If the same key appears again, we return the stored result instead of reprocessing. This prevents duplicate charges even if the merchant retries the same payment request multiple times.

Show more

0

Reply

![Volodymyr Trubachov](https://lh3.googleusercontent.com/a/ACg8ocJQHm2-YasV8OQcL2_nhTJDfThGnX5qIKZCYJ3uP45HMKtL4FmUvg=s96-c)

Volodymyr Trubachov

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbevi1dn00c5ad08gnaalq9s)

"When our API Gateway receives a request, it" - this is not how the request validation works in PKI. The server uses the merchant's public key to extract the hash. Then the server hashes the request and compare with the extracted hash.

Show more

0

Reply

![Anshuman Acharya](https://lh3.googleusercontent.com/a/ACg8ocIJddOsSZ_MCqxoUhATIfxeOqLQTS1CvVwZi1zgfByYQwADq8wV=s96-c)

Anshuman Acharya

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbfi990r005pad0803u3r2kz)

Thank you for the breakdown. This was completely different from the other system designs. I got to learn a lot especially Iframes, I did not know about them. I agree there is so much domain knowledge in this but still its great to learn about all these things as well.

Show more

1

Reply

![Jeremy Shi](https://lh3.googleusercontent.com/a/ACg8ocJ2sb0qOH8kcQXxe0Cn0yJ_g4LR3JiCmQXrdpnSwcy39kYvlNlm=s96-c)

Jeremy Shi

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbg1fs9500kpad079txxjt38)

It might be worth mentioning double entry bookkeeping in reconciliation part. But it's very domain specific concept though.

Show more

0

Reply

![Zhenghao Guo](https://lh3.googleusercontent.com/a/ACg8ocLevKKvDFVeLNyPDtktWZajkqmdMz6Ealzk6JqEjeYRKKdedA=s96-c)

Zhenghao Guo

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbl4gc9l008jad08eo967yr4)

Is idempotent key in the payment table same as the idempotent key in the transaction table? if we are using merchant request id as idempotent key, we could use that to prevent it from duplicate charge. However, we aren't using the merchant request id while our transaction service interacting with payment network right? we are just using the unique transaction id as the new request id? Also, with event sourcing architecture, are we still following idempotent mechanism? How do they work together?

Show more

0

Reply

C

CausalJadeEmu655

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbmb3onv00f008ad66b2v5yn)

How does the payment gateway "Retrieves the merchant's secret key" ?

Show more

1

Reply

G

GayGraySparrow764

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbnzchny017m07addhs832ql)

np: typo "Have a discussing with your interviewer" , I think you meant "discussion"

Show more

0

Reply

R

RemoteBlueGuan642

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmboth4h0025808ad2cmhic1p)

how does event sourcing reduce database TPS requirement? Is it due to batching of writes? if 10k users/sec are creating transactions, how does event sourcing flow through to DB to reduce to 1k/sec WPS?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbpdrxc7005308adc0pjwv03)

It just makes it so workers can consume and write to the DB at their own pace. So we can buffer spikes without slowing down DB queries

Show more

0

Reply

R

ResultingPinkMite610

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbpd8964004o08adzq1x7d2h)

Does merchant provide callback url and subscribe events during onboarding?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbpdqn6c005008adru1sepie)

Yah, typically they have a merchant dashboard to update these sorts of configuratios

Show more

1

Reply

C

ChangingTurquoisePython240

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbpj5nmh00b108adxmhi45zb)

I am still debating whether it is right use of kafka here. How would we provide a payment Id when initial request is received? Also, if payment Id is generated before message is put on the topic, what happens when we receive subsequent calls with paymentId but events are not yet processed by kafka?

Also, How can we identify that a consumer has sent the same request twice?

Using Kafka in a payment flow adds asynchrony, which can be powerful for scalability but tricky for idempotency and ordering guarantees spl. when Kafka ensure ordering only within a partition.

Show more

2

Reply

Y

YelpingGrayVole716

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbq53siy00kb08adczzemeh8)

For the security deep dive, would it be helpful if I leverage threat model framework e.g. STRIDE? For example, the two examples mentioned are spoofing and information disclosure risks respectively as per the STRIDE model. Will it also help if I mention more examples for other types of risks, e.g., tampering, repduation, deny of service and elevation of privigileges, or would they be overkill? If not, any advice on picking the most relevant security risks to discuss in general?

Show more

0

Reply

![Vivek Tiwary](https://lh3.googleusercontent.com/a/ACg8ocJjflqocshnvEvuex4Xh43BNKPTUvLbEueHmmDNtlq6T8MuFNB-pw=s96-c)

Vivek Tiwary

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbwjk7f303fw08addks7p3a6)

In our flow, the user first creates a payment intent, which generates a paymentId. Later, on the checkout page, the transaction is initiated, and a card token is generated via the iframe. What’s the link between the paymentId and the token generated from the iframe? How are they associated during the final transaction?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbx7k3xs087v08adm3ggfvm2)

The paymentId is passed as a parameter when loading the iframe (usually via a data attribute or URL parameter). When the customer enters their card details in the iframe, our SDK includes this paymentId when making the tokenization request to our servers. This way, when we receive the tokenized card data, we already know which payment it's associated with and can link them in our database. So the flow is: merchant creates payment (gets paymentId) -> loads iframe with paymentId -> customer enters card -> SDK sends card + paymentId to our servers -> we create token and associate it with the payment -> return token to merchant for final transaction.

Show more

2

Reply

Z

zyang.bizutil

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbwvh05505tj08adxpminprl)

"_During merchant onboarding, we provide two keys: a public API key for identification and a private secret key stored securely on the merchant's server (never in client-side code)._"

The wording here sounds like the public and private keys are provided by Stripe. Shouldn't it be the merchant generates both public and private keys on their side, only shares the public key with us (Stripe) for authentication purpose?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbx7j9qd087p08adneiszara)

No, the platform (Stripe) generates and provides both the public API key and private secret key to the merchant. This is different from asymmetric encryption where merchants would generate their own keypair. The API key is used for identification, while the secret key is used by merchants to sign requests, proving they are the legitimate owner of the API key

Show more

0

Reply

Z

zyang.bizutil

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmbxb2kt008hg07adllvwej1f)

If I understand you correctly, this public API key and the private key belong to two different key pairs? When I was reading it I thought they were one keypair.

Show more

0

Reply

P

PrimaryAmaranthSparrow330

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmc00yglf03rv08adifqd30zf)

@Evan King- Kindly upload the Video version of this . That will be super helpful, please.

Show more

0

Reply

B

BareCoffeeBasilisk390

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmc1530o40b6e08adak4yzbdx)

The TPS for the projections database is going to less than the Messages per sec of kafka. Why is this? Consider the example messages in the kafka stream:

`2023-04-15T14:32:18.423Z [PAYMENT_CREATED] payment_id=pay_789 merchant_id=merch_123 amount=2500 currency=usd 2023-04-15T14:32:19.105Z [PAYMENT_AUTHORIZED] payment_id=pay_789 transaction_id=txn_456 processor=visa last4=4242 2023-04-15T14:32:19.872Z [PAYMENT_CAPTURED] payment_id=pay_789 transaction_id=txn_456 amount=2500 2023-04-15T14:32:20.341Z [PAYMENT_SUCCEEDED] payment_id=pay_789 transaction_id=txn_456 2023-04-15T14:35:45.129Z [PAYMENT_REFUND_REQUESTED] payment_id=pay_789 refund_id=ref_123 amount=1000 reason="customer_request" 2023-04-15T14:36:12.784Z [PAYMENT_REFUNDED] payment_id=pay_789 refund_id=ref_123 amount=1000 status="succeeded"`

Each one of them could be a transaction for the DB? Are you saying you would roll them up, someway? Thanks

Show more

1

Reply

U

UrbanAmethystPrimate100

[• 2 months ago• edited 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmc1nn80a00jh08add5dp7s2z)

Question about introducing kafka for the payment service. Since this now makes thePOST /payments -> paymentId API endpoint async, would we get a payment ID back immediately? Since this endpoint only creates the payment intent ID, would it make sense to just have this service connect to the DB directly and return the new payment ID immediately and only have the transaction service leverage kafka?

Show more

0

Reply

C

ChemicalOrangeBaboon761

[• 2 months ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmc2l3kss05r908ad0i9ns7pq)

It stands to reason that a seperate Payment Gateway is required in the system. However, could you please explain the main reasons for having separate Payment Service and Transaction Service instead of one monolithic service?

Show more

1

Reply

A

Abhi

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdeb1mtt017tad084eljlild)

@Evan now that the version 2 has removed Payment Gateway altogether, my question is why was it removed?

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmce3q9r30bapad08rvopkij4)

> allowing us to scale consumers horizontally while maintaining exactly-once processing semantics.

could you elaborate "how" exactly once? is that through kafka transactions? or is it because of the idempotence set up?

Show more

0

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmce3s9l80bbkad08hw0q7ks9)

my biggest question is - the first API for payment intent, if that comes into kafka async, how does the caller know what the payment id is supposed to be?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmce472uj0014ad08r75wvlpb)

The payment ID is generated synchronously before the event is published to Kafka. The API creates the payment record in the database first (getting an ID), then publishes the event to Kafka with that ID, and finally returns the ID to the caller. The event sourcing pattern doesn't mean everything is async - we still need synchronous operations for things that require immediate responses.

Show more

2

Reply

R

RepresentativeLimeHarrier694

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcf5nvo000n2ad08yqrzeken)

thanks for the response, Evan. the design does show that everything goes through kafka first, and i didnt read it in text (sorry if I missed it) however, if we are anyway going to write it to DB first as you said, why not just say CDC and call it a day. what am I missing?

Show more

0

Reply

A

Abhi

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmchc9jah058fad08jaaiho56)

@Evan why do we need the createPayment event sourced in Kafka if it's already written to the DB? Since Kafka would just waste consumer bandwidth processing these duplicate idempotent createPayment API calls without any change in the DB record. Are we only putting it in Kafka to retain the sequence of events for 30 days? If yes, should Kafka consumer should omit all createPayment events from writing again to DB?

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 8 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmeeaigen033vad08d4mzwcge)

Hi , where it is mentioned that even for paymentintent , we are writing into Kafka

Show more

0

Reply

M

MinimalSapphireCarp228

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcjd9n3b08jdad07tjggwva9)

Hi Evan, great post! one question on this flow - client request goes to merchant to the API gateway to backend service to an external payment gateway. Should the TCP connection between client and merchant remain open or is it wiser to use SSE or polling from client considering the usual latency of the payment system?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd4ru2cn020nad08d9p1z164)

No you would not leave it open, it's fully async the response and the merchant gets updated either via polling or (as is the case in reality) via webhook.

Show more

0

Reply

A

aritra90tnp

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcoa6za802muad08cfvxcwxd)

> During merchant onboarding, we provide two keys: a public API key for identification and a private secret key stored securely on the merchant's server (never in client-side code).

One more question on the why do we(the payment provider) want to generate the private secret key and provide it to the merchant ? What is the advantage of that over letting the merchant sign using their own private key and the payment provider validates its authenticity using the merchant's public key ? I see one disadvantage, even though it would be an extreme case - the private key being compromised when payment provider sends it to the merchant at the time of onboarding.

Show more

0

Reply

P

PleasantAquamarineHoverfly685

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcr19q32022rad08iawzrxp2)

we need a video version of this problem **ASAP** otherwise the **world's gonna end**!!! Like if you agree :)

Show more

3

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcr1agev01moad08av987c78)

haha oh shit cant have that

Show more

0

Reply

P

PleasantAquamarineHoverfly685

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcr1mjhp025uad081u0g60wa)

We need Evan **Superman** King to finish encoding the video at 10000x speed....

Show more

0

Reply

J

jaswanthsai917

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcuhf6l403n5ad08yyrjjpw9)

Hi Evan, why cant the token and the idempotent key be the same or are they same ? because the token that we generate would also be unqiue to payment and is also persisted by the merchants.

Show more

0

Reply

G

GothicCopperPanther728

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcv58yk409w9ad08fizdedwm)

Couldn't the eventually-consistent nature of event sourcing with a projections DB cause functional issues if high lag were encountered? For example, a merchant could send two identical requests with the same idempotency key in a short time frame. We write a Kafka event for the first one, but due to lag, the event isn’t present in the projections database by the time the second request is processed, causing us to send duplicate requests to our payment gateway.

By the way, autocorrect doesn’t work in this text box on iOS for some reason.

Edit: Also, can you elaborate on the reasoning behind the DB’s TPS being one magnitude smaller than our Kafka cluster? At least for the types of events described here, it seems that every one would require a database write. Thanks so much.

Show more

1

Reply

C

ChristianTanMeadowlark637

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmcwrpgiv02zcad09ocxwy3z1)

I have a question about what happens when we send a request for a payment to our credit card gateway and due to a network failure or timeout, we don't know the status of the payment. You mentioned that would be bad practice to just assume that it failed, but if our CC gateway doesn't provide an API to query the status of the payment and only offers files for batch reconciliation hours later, what can we do? The merchant is likely needing an answer as to whether or not the payment is a success so they can put their order through or not. In this case, the answer is we don't know. If they retry with the idempotency key, again w/o an API from the gateway we have no choice to put this payment through again and later rely on batch reconciliation to refund the potential extra charge. Thoughts on what the best approach here is?

Show more

0

Reply

![Sid Khanna](https://lh3.googleusercontent.com/a/ACg8ocL0O_4eCgAsC4QmzlKqDVd9miKuiQx8-LaSM1xEHX7ObrBPqz_1=s96-c)

Sid Khanna

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd23i5j301bmad082kbsd2ar)

Why not use something like Apigee / APIM for validating the authorization of the merchant over storing private API Key in merchant server ?

Show more

0

Reply

E

EmotionalSalmonScorpion737

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd4pwo2b0bclad0831oxyiiy)

When events from Kafka are written to S3 for audit trail, how do we ensure that we don't have duplicate events written to S3?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd4ruth8020uad08753458bh)

You could use a combination of event ID and timestamp as the S3 key path (e.g., yyyy/mm/dd/hour/event\_id.json). Since S3 has strong consistency for PUTs, attempting to write the same event ID twice will either overwrite (fine, since it's the same data) or fail if we use S3's conditional write features.

Show more

1

Reply

C

ContinuousGrayBat761

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd5i9jff011had08guj2pqmh)

The last two images are not visible at all - zooming does not help. Could you fix them please?

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6h3bnu0077ad09qupl3117)

Fixing!

Show more

1

Reply

Y

YammeringOrangeParrotfish116

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6h1ksd007aad09guu1m3a0)

You briefly touch on the many entities (charge, refund, dispute, payout etc.) that represents different forms of a transaction which maps back to a payment intent. What if a payment intent successfully charge, but then at some future date its disputed, how are UUIDs and correlation IDs used? Would the original payment intent take form of the dispute transaction (mapping to the related entity)? Or is a new payment intent created, if so how is it mapped back to the initial transaction?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6h387e0073ad09souamcqf)

A dispute doesn't create a new PaymentIntent, it creates a new Transaction record (of type "dispute") that references the original PaymentIntent. The PaymentIntent remains the top-level entity that tracks the entire payment lifecycle, while different Transaction records (charge, dispute, refund) represent specific money movements related to that payment. Each Transaction gets its own UUID, but they all link back to the same PaymentIntent ID for correlation.

Show more

1

Reply

![Constantine Gerasimovich](https://lh3.googleusercontent.com/a/ACg8ocLAK4BenE3qvm_X199-aOqLPUDL6j5f5P8ZLNKo28XzW0zIiRU=s96-c)

Constantine Gerasimovich

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6szfhu03msad08su1izxca)

53:00 in the video, when we mark the status of the transaction to "Timeout" and offloaded processing the the Reconciliation service - how are we going to notify clients about the outcome? What will TransactionService return to the customer once ExternalPaymentSystem timed out?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6tyl3a043kad08l82c9p1s)

We return a "pending" status immediately to the client and provide them with the PaymentIntent ID. The client can then either poll the status endpoint or (better) receive a webhook when the Reconciliation service determines the final outcome. Payment transactions are always async, so we never return a sync success or failure immediately.

Show more

0

Reply

![Constantine Gerasimovich](https://lh3.googleusercontent.com/a/ACg8ocLAK4BenE3qvm_X199-aOqLPUDL6j5f5P8ZLNKo28XzW0zIiRU=s96-c)

Constantine Gerasimovich

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6t4yrv03p1ad08ep9am7qt)

RE CDC reliability - Debezium, being one of the most popular options for CDC with Postgres, always read from WAL and keep the current offset, so it knows where to continue reading in case of the crash or restart. Data might be lost if WAL was GCed by Postgres if CDC was unavailable for a long time, so it might make sense to have multiple connectors running to reduce the chances of losing events.

Show more

1

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd6tys7i043pad08mok7u2x6)

Nice! Good to know

Show more

1

Reply

![Colin Brown](https://lh3.googleusercontent.com/a/ACg8ocI5lHz892mgdacqBslhL-Qq_ocbgdVzTJyQwbybxkGg9uNrFA=s96-c)

Colin Brown

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdc65c9o01r6ad07nz3n206m)

Yes Debezium is great. For Postgres, you read off of the WAL and the database will store the offset, and WAL until all consumers read it. Same thing with MYSQL and the Binlog. From past experience, you can cap the size of the WAL to prevent the WAL from eating the database's resources or go unbounded if you want to YOLO it. If you have a cap, Postgres will break your connection and dump the WAL. New versions of Postgres 16+ allow you to CDC from a read replica as well. Another alternative to Debezium server that I like and implemented in production is using Flink with baked in CDC code and deploying these jobs.

Edit: if you are using a distributed manager like citus(postgres) or vitus(mysql), your set up is slightly different than what you'd naturally think. You read from the Citus and wait to get updates for all of the nodes instead of tapping into each one.

Show more

2

Reply

D

DistinctAmberHarrier825

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd779jg301zhad08k1oy3kv9)

Thank you Evan for simplifying this. Very logical. any suggestions on how to start with this baseline when asked about system design for digital wallet.

Show more

0

Reply

C

CautiousPurpleMacaw908

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd88xka602tsad08v62mzb1f)

Thanks for making a video on this! I have some questions about queues/streams:

-   The old written version (before the video) of this design had kafka before the database instead of after database-> CDC, would you mind comparing the 2 approaches?
-   From my understanding a queue between gateway and service isn't necessary because scaling the really lightweight service allows us to handle bursts just fine without a queue there?

Show more

2

Reply

![Amit K](https://lh3.googleusercontent.com/a/ACg8ocLWqC7IIa0dOro_z8cLQENraUWN1uJ9uVs8GwHcASu999CosOxU=s96-c)

Amit K

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmd96pvsj00t7ad083dt9xm27)

Excellent one. Insightful

Show more

0

Reply

![Rick Sanchez](https://lh3.googleusercontent.com/a/ACg8ocKuuBLuorMY90oxFsXim2wz42Im8wls4pJII4_br4rJvttKUtcd=s96-c)

Rick Sanchez

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmda5z7an05cdad08hwfsdhg7)

Can we introduce idempotency tokent in the design to handle retries?

Here's how it would fit in:

1.  A transaction is pushed into the stream with idempotency token
2.  A consumer reads it and makes an API call to the payments provider
3.  Happy path - status is updated to the db
4.  Sad path - consumer crashes before writing to the db. The message was not ACK'd so it would be retried and payment provider would return the status without double charging

I am glossing over several edge cases here but this is what the general idea is. Thoughts?

Show more

1

Reply

![Saurabh Mangalam](https://lh3.googleusercontent.com/a/ACg8ocKLfuhJnrWZjwdk3xzps2S_qjQ3pviKDGTDj8ikNeU9Q2h0dj4=s96-c)

Saurabh Mangalam

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdarhyxu0b6pad0837p55afc)

Thanks for the great breakdown. But just had some confusion around CDC as the single point of failure, how is that SPOF especially since you mentioned that the data capture will happen async. If the CDC is down, it won't bring the system down at all, it is just that it will stop consuming the WAL changes for whatever duration it is down. Once it comes back up, it will again start consuming the WAL changes and will send them wherever such as S3. The problem will only happen when CDC is down for a really long time, and the WAL has been cleaned up by Postgres. Am I missing something here?

Show more

0

Reply

P

pankti.majmudar

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdcd5evd04q4ad09h70gl1rw)

In the Scaling section, it is mentioned that the TPS handled by the database will be orders of magnitude smaller than the event stream. Can you please explain why it is the case? The event stream captures everything written to the database. So will both not have the same load?

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdcd7qvq04bjad07o5lfrr4z)

this is actualy residual from before the latest re-write of this article on the 10th. looks like i just missed a spot. will update! youre right theyre the same

Show more

1

Reply

P

pankti.majmudar

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmde1ozo506xsad085ic1l7cq)

Thanks a lot for clarifying

Show more

0

Reply

B

bhavsi

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmddsjhoo0449ad08v0aohupl)

Yay, the video is out too! Any chance you could upload this video on YouTube? Or is this premium only content? I am happy to watch it here, but it's always easier on YouTube :)

Show more

0

Reply

![Evan King](https://www.hellointerview.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fevan-headshot.36cce7dc.png&w=96&q=75)

Evan King

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmddskt1b03udad082j9ad9ca)

premium only :)

Show more

0

Reply

A

Abhi

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmde9uhvf00svad08y7b68ueg)

> We'd also set up consumer groups for each of our services, allowing us to scale consumers horizontally while maintaining exactly-once processing semantics.

Yes, but how are we ensuring exactly-once processing semantics in this design? I got grilled on this in a recent interview, I'd love to hear your take/deep dive on it.

Show more

0

Reply

S

StripedSalmonCheetah561

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdf5c0ws014wad087wq1eksm)

Thanks for a great explanation and cheers for chelsea winning the CWC!!!!

Show more

0

Reply

P

PrivateMaroonSwordfish354

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdffm1nd047aad08x27lvayp)

Using a ledger managed service might be a better idea than S3. E.g. Amazon offers QLDB which crytographically also signs each entry and ties them together essentially making mutability impossible without breaking the whole chain.

This ensures the auditability guarentee without foul play. Key for financial instruments.

https://aws.amazon.com/qldb/?refid=608c656c-dddb-4835-9aef-bbccd9e4f5eb

Show more

0

Reply

P

PrivateMaroonSwordfish354

[• 1 month ago• edited 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdfgak7204g1ad08egce0hc4)

Does the nonce serve as the idempotency key in general here? Idempotency in general would have been a good topic to cover!.

Show more

0

Reply

![Sourabh Upadhyay](https://lh3.googleusercontent.com/a/ACg8ocKpj06uaBaRfJhVsAJL98n9F7-IyL3NsYEkFdZuG1m_9wYa4Q=s96-c)

Sourabh Upadhyay

[• 1 month ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdg0sbtz04flad08s2je05si)

If one payment intent can have multiple transactions, should paymentIntentId be an idempotency key? It can be for the intent (like order) for transactions also we need an idempotency key.

Show more

1

Reply

O

OriginalWhiteTiglon757

[• 22 days ago• edited 22 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdtijbkb01fbad08pfyy1gay)

Thanks for the video breakdown, I found this really detailed and helpful.

I had a few questions on the failure scenarios:

1.  If the txn service blows up after firing off the call to the external service, how will the reconciliation worker be triggered? The data in CDC will be something like created instead of timedout.
    
2.  How does a reconciliation worker implement backoffs? You mentioned having separate topics for 1 min delay vs 5 min delay etc, but how is this delay actually implemented? I know SQS has a visibility timeout, but I'm wondering how you do something like this in Kafka.
    

Show more

0

Reply

S

SovietTanPanda828

[• 19 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmdxti8r600arad074ecx68dj)

API Gateway receives a request, it: Retrieves the merchant's secret key based on the provided API key Recreates the HMAC signature using the same algorithm (SHA-256), secret key, and request data Compares the calculated signature with the one provided in the request headers Validates that the timestamp is within an acceptable time window (typically 5-15 minutes) Ensures the nonce hasn't been used before within the valid time window by checking the cache/DB

The API gateway does all of that? I would have guessed that responsibility is delegated somewhere else.

Show more

0

Reply

![Aditya Rohan](https://lh3.googleusercontent.com/a/ACg8ocJe-7y5dWw2FJQBidbc24y_P9ud1cJDaHi_lXFhcG_xsg5Bwp_B=s96-c)

Aditya Rohan

[• 15 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cme2zpj1d088xad08rc2wgmgt)

Had query on this article. 🙂 Writing it down again. Instead of CDC, can services themselves act as producers and asynchronously write data to Kafka message bus ? Messages in different topics (named after different event) can then be consumed by downstream consumers, such as audit, ledger, and reconciliation pipeline, or for any other notification purpose, etc ?

Show more

0

Reply

P

PreciseAzureCuckoo716

[• 14 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cme4k0rvd08u5ad08tn4tbpjd)

@Evan, Could we get a video on webhooks please?

Show more

0

Reply

![Henry Chen](https://lh3.googleusercontent.com/a/ACg8ocJ_l-K3qJtLV5_6vt9fAw4uB4RVKrHKz49m3A2P0YHAnkW-Tw=s96-c)

Henry Chen

[• 13 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cme5z1h1h0lsaad082w4tnwle)

I was asked to design visa network in a system design interview of a general backend role and I was dead onsite..

Show more

0

Reply

![Shiksha Sharma](https://lh3.googleusercontent.com/a/ACg8ocIXFNZgiWWrmie5hyDCixenmQ1s5TIApjnvAx1vVLz3IC5xEQ=s96-c)

Shiksha Sharma

[• 8 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmeeajf1l0345ad0841bk4rzg)

so it is mentioned , separate attempt table , does this attempt table is audit table??

Show more

0

Reply

S

Sumant

[• 3 days ago](https://www.hellointerview.com/learn/system-design/problem-breakdowns/payment-system#comment-cmekz5gd300ciad085143rb9x)

Thank you for writing detailed walk through. In the deep dive for making the system secure you have generated private and public key, private key to be kept at the merchants end. However when it comes to verifying the request you are hinting at using HMAC which is for shared secret - using the same symmetric shared key that is with merchant and us. what is the use of the public key then?

Shouldn't it be signed with the private key that we generated for the merchant while we verify using the corresponding public key?

Show more

0

Reply
