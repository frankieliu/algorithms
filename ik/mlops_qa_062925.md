Frankie Y Liu (You) 09:57 AM 
what is management again?
This question has been answered live
Frankie Y Liu (You) 10:00 AM  
in what way is it different from kubernetes?
IK Academic Support 10:03 AM 
Airflow and Kubernetes serve fundamentally different purposes. Airflow is workflow orchestration platform specifically designed for scheduling and monitoring data pipelines. Kubernetes is a container orchestation platform for automating deployment, scaling and management of containerized applications.
Parminder Singh 10:01 AM  
How does Airflow interact with data? via the Metadatabase?
IK Academic Support 10:06 AM 
Airflow orchestrates the tools that process data. Metadatabase in airflow stores airflow’s internal state including: 
1) DAGs defnitions and their run history.
2) Task instances and their states
3) Variables and connections
4) Scheduler information
5) User authentication data

So, this is separate from the actual business/processing data that the workflows manipulate.

Let me know if you further questions
Benjamin Smith 10:01 AM  
Are workers typically entire VMs or jsut processes or something in between?
IK Academic Support 10:09 AM 
Workers can be deployed in various ways depending on the infrastructure needs and preferences.

In simple deployments, workers run as processes on the same machine as the scheduler. It is suitable for development or small-scale production environments.
It uses LocalExecutro or SequentialExecutor.

In complex deployments, processing takes place accross multiple machines. Using the CeleryExecutor, workers run as processes distributed across multiple machines. Each worker is a separate process that pulls tasks from a message queue. It requires a message broker (Redis or RabbitMQ)
Frankie Y Liu (You) 10:03 AM  
Does the scheduler have a dag of the tasks, how are dependencies expressed, and branching expressed.
IK Academic Support 10:10 AM 
Yep, airflow scheduler maintains an internal representation of the DAG structure. The scheduler parses DAG files to build this representation. It tracks task dependencies and their current states. It determines which tasks are eligible to run based on dependency satisfaction. The actual graph structure is stored in the metadata database
Santosh Singh 10:04 AM  
does each node in the dag gets executed by a one worker at a time or does an entrire dag gets run by a worker?
IK Academic Support 10:12 AM 
In Airflow, execution happens at the individual task level, not at the DAG level. Each task in a DAG is executed by a single worker at a time. Different tasks from the same DAG can run on different workers simultaneously

Multiple tasks from the same DAG can run concurrently if
1) Their dependencies are satisfied
2) There are available workers
3) Parallelism settings allow it
Frankie Y Liu (You) 10:06 AM  
Can it branch depending on result from Task A
IK Academic Support 10:13 AM 
Yep, airflow can branch based on the result of a previous task. This is one of its powerful features for creating dynamic workflows.
Frankie Y Liu (You) 10:11 AM  
Is airflow blind to the type of executor?
IK Academic Support 10:16 AM 
Airflow uses a well-designed abstraction layer for executors. DAG definitions are independent of the executor being used. Task dependencies, scheduling, and monitoring work consistently across executors. 

However, Airflow isn't entirely blind to executors in all contexts. Some executors provide unique capabilities (e.g., KubernetesExecutor allows pod templates)
Frankie Y Liu (You) 10:14 AM  
Can you please define what is a workflow?  Do workflows interact with one another?
IK Academic Support 10:17 AM 
A workflow is a DAG of tasks with defined dependencides that execute in a specific order to accomplish a complete business process.

A workflow represents a complete logical unit of work with a clear beginning and end
Frankie Y Liu (You) 10:18 AM  
do the workflows interact with one another?
IK Academic Support 10:19 AM 
Yes, airflow DAGs cna interact with one another,though they are designed to be independent by default. 

There are some common interaction methods like TriggerDagRunOperator, ExternalTaskSensor etc. to establish workflow interaction.
Frankie Y Liu (You) 10:18 AM  
If not then why not just provide a different instance of airflow?
This question has been answered live
Frankie Y Liu (You) 10:20 AM  
if dags interact with one another, then isn't this just a larger dag or workflow?
IK Academic Support 10:25 AM 
Yep you are right. Theoretically they could be respresented as a single larger DAG. But there are some reasons why airflow supports separate interacting DAGS.

1) Organizational Boundaries: Different teams can own and maintain separate DAGs
2) Deployment Independence: DAGs can be updated separately without affecting others
3) Failure Isolation: Problems in one workflow don't necessarily impact others
4) Scheduling Flexibility: Different DAGs can have different schedules
5) Parser Efficiency: The DAG parser has to process smaller files
6) Memory Usage: The scheduler manages smaller graphs in memory
7) Testing: Smaller DAGs are easier to test
8) Versioning: Easier to track changes to smaller, focused workflows
9) Reusability: Common patterns can be templated and reused
Frankie Y Liu (You) 10:22 AM  
can you go over why would someone choose celery over k8s?
IK Academic Support 10:28 AM 
Celery requires components and has a simpler architercture. It just needs a message broker (Redis/RabbitMQ) and worker processes. No container orchestration is need in celery. Celery is better suited for traditional VM based infrastructure and for non-containerised workflows.
Frankie Y Liu (You) 10:22 AM  
how does the scheduler handle zombie tasks -- tasks which don't terminate (stuck).
IK Academic Support 10:30 AM 
Airflow has task level timeouts. When a task excedds its execution timeout it is markes as failed and timeout error is recorded.  It also has DAG level timeouts.
Frankie Y Liu (You) 10:25 AM  
how does airflow pass authentication / passwords / API keys and related data security.  For example can a less secure worker ever touch data that was generated by another more secure data task?
IK Academic Support 10:35 AM 
Airflow connections can be stored in the metadata database after encrytion. Airflow supports external secret management systems as well like AWS secrets manager, GCP secret manager etc. Alternatively environment variables can be used for secrets as well injected directly into task environments
Frankie Y Liu (You) 10:27 AM  
regarding dynamic dags (one team changes their dag), what happens to other dags -- does one invalidate the dependent tasks in other workflows and start from scratch -- this seems tricky because there might be upstream branching.
IK Academic Support 10:37 AM 
Yep, this becomes a tricky situation. When a team changes their DAG, The scheduler periodically re-parses all DAG files (default: every 30 seconds). The new DAG structure replaces the old one in the metadata database.

The currently running DAGs continue running under the old DAG structure. Unscheduled tasks will follow the new DAG structure. Having said that, a single DAG run can end up with some tasks from the old version and some from the new
Frankie Y Liu (You) 10:29 AM  
can airflow handle a continuously running task that ingests data from other tasks and spits out some of its own output used by other tasks?   This may look like a cycle, but it is a common use pattern.
IK Academic Support 10:40 AM 
The robust approach is usually to have the continuous process run outside of Airflow entirely (as a dedicated service) and use Airflow to coordinate data flows into and out of it, using datasets or sensors to manage the dependencies
Frankie Y Liu (You) 10:32 AM  
Following up on zombie tasks, if scheduler marks it as terminated, what happens when the zombie task wakes up, how does airflow keep things idempotent?
IK Academic Support 10:43 AM 
So When a task is marked as failed due to being a zombie, Its task instance ID (combination of DAG ID, task ID, and execution date) is marked as "failed" in the metadata database. Any subsequent operation from that same task instance is considered invalid. Database constraints prevent duplicate task instance records.

For celery exectuor, task revocation is used to cancel zombie tasks. The Celery worker will terminate the task process. If a task somehow continues after revocation, its results are ignored by the broker.

For kubernetes executor,  Kubernetes pods are deleted when identified as zombies. Resource cleanup is handled by Kubernetes.
Frankie Y Liu (You) 10:36 AM  
How does airflow handle testing / debugging, how does one build in a test driven manner.
IK Academic Support 10:47 AM 
Airlfow support serveral levels of testing. We can use unit testing for the DAG structure to test if it properly constructed. There are unit testing operators to test individual task logic in isolation.  We can also test execution flow using dag_run mechanism
Frankie Y Liu (You) 10:49 AM  
Regarding debugging:  is it possible to put "breakpoints" into the dag, inspect the outputs and states and let it continue.
IK Academic Support 10:52 AM 
Nope airflow dosen’t have built in support for breakpoints. This is primarily due to tasks are executed async and distributed accross different workers.  Also, the scheduler is designed for throughput, not interactive debugging. 

However,you can implement pseudo breakpoint approaches like explict pause task in the DAG or having a conditional execution path.
Frankie Y Liu (You) 10:51 AM  
Can tasks that are sequentially related "send data" to each other, or must the data be dumped somewhere for the subsequent task to pick up.
IK Academic Support 10:56 AM 
There are several mechanisms for passing data between sequential tasks. Xcoms or cross communications is the built in mechanism designed for task to task communicaiton. Xcoms limitations:

1) Size limits (typically <48KB, depends on backend)
2) Stored in Airflow's metadata database
3) Not suitable for large datasets
4) Intended for control flow metadata, not bulk data transfer

In production environments, the external storage approach is generally preferred for robustness and scalability, with XComs used primarily for control flow and metadata.
Frankie Y Liu (You) 11:00 AM  
Is the bottom part happening in the scheduler?
This question has been answered live
Frankie Y Liu (You) 11:01 AM  
worker is changing the state?
IK Academic Support 11:04 AM 
Workers can change the task state. Workers write task state changes directly to the metadata database. 
States progress through: QUEUED → RUNNING → SUCCESS/FAILED/SKIPPED
Frankie Y Liu (You) 11:01 AM  
I thought the white boxes are task states?
IK Academic Support 11:05 AM 
Can you on the elaborate on the question. Did get your question
Frankie Y Liu (You) 11:01 AM  
I don't think the worker should determine task state
IK Academic Support 11:15 AM 
I hope this question got answered as part of the other questions.
Frankie Y Liu (You) 11:06 AM  
I don't understand, I thought executors are abstracted from airflow, so how can workers understand about airflow.
IK Academic Support 11:10 AM 
yep you are correct. 

So the executor component determines how tasks are executed (interface to execution environment). And the worker actually executes the tasks. 

The abstraction leakage occurs because:
1) Workers need direct access to the Airflow metadata database
2) Workers need to understand Airflow's internal task model
3) Workers need to implement Airflow-specific logic for state management
Frankie Y Liu (You) 11:07 AM  
For example if you choose k8 as the executor, what is the worker abstraction in this case?  It seems it should not know about airflow.
Frankie Y Liu (You) 11:14 AM 
Thanks question was answered above.
IK Academic Support 11:14 AM 
Yep this highlights a fundamentail architectural inconsistency in Airflow’s design. 

In K8s executor it particuallarly gets more interesting. 

What ideally should happen in clean architecture:
1) The executor would translate Airflow tasks into Kubernetes pods
2) The pods would run independently, unaware they're part of Airflow
3) Results would be reported back via an abstracted interface
4) The executor would translate these results back into Airflow concepts

But what actually happens:

Each pod contains, a full Airflow installation, access to the Airflow metadata database and Airflow's internal libraries and dependencies.

Scheduler determines task is ready
   ↓
KubernetesExecutor creates pod spec
   ↓
Pod launches with Airflow code inside
   ↓
Pod's entrypoint connects to Airflow DB
   ↓
Pod updates task state to RUNNING
   ↓
Pod executes actual task code
   ↓
Pod updates task state to SUCCESS/FAILED
   ↓
Pod terminates
IK Academic Support 11:22 AM 
Yep there are some workflow orchestration systems that have beend designed with cleaner architectural separation. 

1) Temporal: Clear separation between the control plane and worker processes. Event-sourced architecture with immutable history. Workers don't need database access.

2) Prefect 2.0: Lighter-weight execution environments. API-first design enables better integrations


3) Argo Workflows: Kubernetes-native workflow engine. 
Execution happens in isolated containers. Control plane (controller) is separate from execution

