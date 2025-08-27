##### Design Recommender System

by ExtraApricotSloth487• Senior• 21 days ago

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

[• 21 days ago](https://www.hellointerview.com/community/submissions/cmdz4fjki03mmad08cifxu8uu#comment-cmdz4fw6j0054ad3bsgrq5t2z)

Nice design! You've created a comprehensive recommender system architecture that demonstrates strong understanding of ML infrastructure patterns. Your separation of training and inference clusters with proper feature stores shows good architectural thinking, and the use of streaming for real-time feature updates is a solid choice. However, there are some critical issues around data flow clarity and operational complexity that need addressing.

1.  **Positives**
    
    -   Excellent separation of training and inference concerns with dedicated clusters, allowing independent scaling and optimization of each workload
    -   Smart use of hot/cold feature stores (Redis for frequently accessed user features, separate stores for items) to optimize for different access patterns
    -   Good implementation of A/B testing capability within the inference cluster, enabling safe model rollouts and experimentation
2.  **Potential Issues** **Unclear Data Flow for Training**
    
    -   The training cluster appears to read from multiple sources (OLAP DB, feature stores) but the data flow isn't clearly defined
    -   This ambiguity makes it impossible to understand how fresh interaction data reaches the training pipeline
    -   Without clear data lineage, you can't ensure models are trained on complete, consistent datasets
    
    **Overly Complex Component Naming**
    
    -   Components like "training cluster training pod validation Data Loader Gradient synchronization training pod" are confusingly named with unclear boundaries
    -   This makes it difficult to understand what each component actually does and how they interact
    -   The design becomes unmaintainable when component responsibilities aren't clearly delineated
    
    **Missing Model Deployment Pipeline**
    
    -   Models go from training cluster to Model Registry to inference cluster, but there's no deployment orchestration
    -   This creates risk of inconsistent model versions across inference pods during updates
    -   Without proper deployment coordination, you could serve different predictions to the same user
3.  **Follow-up Questions**
    
    -   How does interaction data flow from Kafka through S3/Athena to the training cluster for model updates?
    -   What triggers the cron job and what exactly does it do with the Items DB and feature store?
    -   How do you handle cold start users who have no features in the hot feature store?
    -   What's the strategy for model rollback if a new version performs poorly in production?
    -   How do you ensure feature consistency between training time and inference time to avoid training-serving skew?

_Warning: This feedback was AI generated and may contain errors._

Show more

1

Reply