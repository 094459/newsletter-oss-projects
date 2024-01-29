# AWS open source newsletter projects

This repo contains a list of projects featured in the AWS open source newsletter.Includes projects from #178 to #183 of the newsletter (retrospective projects will be added over time)


# By technology/use case

### AI & ML

**amazon-bedrock-serverless-prompt-chaining**

[amazon-bedrock-serverless-prompt-chaining](https://aws-oss.beachgeek.co.uk/3jy) this repository provides examples of using AWS Step Functions and Amazon Bedrock to build complex, serverless, and highly scalable generative AI applications with prompt chaining. (#184)

**amazon-sagemaker-pipeline-deploy-manage-100x-models-python-cdk**

[amazon-sagemaker-pipeline-deploy-manage-100x-models-python-cdk](https://aws-oss.beachgeek.co.uk/3lu) This GitHub repository showcases the implementation of a comprehensive end-to-end MLOps pipeline using Amazon SageMaker pipelines to deploy and manage 100x machine learning models. The pipeline covers data pre-processing, model training/re-training, hyper-parameter tuning, data quality check, model quality check, model registry, and model deployment. Automation of the MLOps pipeline is achieved through Continuous Integration and Continuous Deployment (CI/CD). Machine learning model for this sample code is SageMaker built-in XGBoost algorithm.(#186)

**awesome-codewhisperer**

[awesome-codewhisperer](https://aws-oss.beachgeek.co.uk/3cw) this repo from Christian Bonzelet is a great collection of resources for those of you who are experimenting with Generative AI coding assistants such as Amazon CodeWhisperer. This resource should keep you busy, and help you master Amazon CodeWhisperer in no time.  (#177)

**aws-inference-benchmark**
[aws-inference-benchmark ](https://aws-oss.beachgeek.co.uk/3cy)this project from Rustem Feyzkhanov contains code for running deep learning inference benchmarks on different AWS instances and service types. Check out his post, [Making LLMs Scalable: Cloud Inference with AWS Fargate and Copilot](https://aws-oss.beachgeek.co.uk/3d0) where Rustem shows you in more details how you can use this repo. (#177)


**bedrock-genai-workshop**

[bedrock-genai-workshop](https://aws-oss.beachgeek.co.uk/3lt) if you are looking to get hands on with generative AI, then check out this workshop that is aimed at developers and solution builders, introduces how to leverage foundation models (FMs) through Amazon Bedrock. Amazon Bedrock is a fully managed service that provides access to FMs from third-party providers and Amazon; available via an API. With Bedrock, you can choose from a variety of models to find the one that’s best suited for your use case. Within this series of labs, you'll explore some of the most common usage patterns, and Labs include: 1/ Text Generation, 2/ Text Summarization, 3/ Questions Answering, 4/ Chatbot , and 5/ Agent (#186)

**bedrock-multi-tenant-saas**

[bedrock-multi-tenant-saas](https://aws-oss.beachgeek.co.uk/3jx) In this repository, we show you how to build an internal SaaS service to access foundation models with Amazon Bedrock in a multi-tenant architecture. An internal software as a service (SaaS) for foundation models can address governance requirements while providing a simple and consistent interface for the end users. (#184)

**gen-ai-on-eks**

[gen-ai-on-eks](https://aws-oss.beachgeek.co.uk/3d2) this repository aims to showcase how to finetune a FM model in Amazon EKS cluster using, JupyterHub to provision notebooks and craft both serving and training scripts, RayOperator to manage Ray Clusters and Karpenter to manage Node Scaling. (#177)

**makit-llm-lambda**

[makit-llm-lambda ](https://aws-oss.beachgeek.co.uk/3cx)this repo from Martyn Kilbryde is an example of how you can run a Large Language Model (LLM) inside an AWS Lambda Function.  Whilst the code will help you deploy to AWS Lambda, it can be ran locally inside Docker for testing as well. The function contains the full LLM model and the code to use the model, allowing basic text generation from a HTTP call into it. (#177)

**partysmith**

[partysmith](https://aws-oss.beachgeek.co.uk/3l4) is an awesome project from AWS Community Builder Stephen Sennett, that provides an unofficial way to transform your AWS PartyRock apps into deployable full-stack SvelteKit applications. Users can enter the URL of a publicly published PartyRock app, select the desired settings, and PartySmith will forge an application into a ZIP archive which will be downloaded to your machine, and ready for use. How cool is that! (Very in case you were wondering). Find out more by reading the supporting blog post, [PartySmith - Bring PartyRock apps to your place](https://aws-oss.beachgeek.co.uk/3l5). (#185)

**rag-with-amazon-bedrock-and-pgvector**

[rag-with-amazon-bedrock-and-pgvector](https://aws-oss.beachgeek.co.uk/3lv) is an opinionated sample repo on how to configure and deploy RAG (Retrieval Augmented Retrieval) application. It uses a number of components including Amazon Bedrock for foundational models, Amazon RDS PostgreSQL with pgvector, LangChain, Streamlit, and a number of AWS services to bring it all together.(#186)

**RefChecker**

[RefChecker](https://aws-oss.beachgeek.co.uk/3l3) For all their remarkable abilities, large language models (LLMs) have an Achilles heel, which is their tendency to hallucinate, or make assertions that sound plausible but are factually inaccurate. RefChecker provides automatic checking pipeline and benchmark dataset for detecting fine-grained hallucinations generated by Large Language Models. Check out the supporting post for this tool, [New tool, dataset help detect hallucinations in large language models](https://aws-oss.beachgeek.co.uk/3l7) (#185)

**s3-connector-for-pytorch**

[s3-connector-for-pytorch](https://aws-oss.beachgeek.co.uk/3gw) the Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. Using the S3 Connector for PyTorch automatically optimises performance when downloading training data from and writing checkpoints to Amazon S3, eliminating the need to write your own code to list S3 buckets and manage concurrent requests. Amazon S3 Connector for PyTorch provides implementations of PyTorch's dataset primitives that you can use to load training data from Amazon S3. It supports both map-style datasets for random data access patterns and iterable-style datasets for streaming sequential data access patterns. The S3 Connector for PyTorch also includes a checkpointing interface to save and load checkpoints directly to Amazon S3, without first saving to local storage. (#181)

### Application integration and middleware

**aws-apn-connector**

[aws-apn-connector](https://aws-oss.beachgeek.co.uk/3l1) this project from the folks at Nearform provides a way of interacting with the AWS APN (AWS Partner Network) programatically, as this does not provide an API. If you are looking to automate your interactions with the AWS APN, you should check this project out.(#185)

**aws-cdk-python-for-amazon-mwaa**

[aws-cdk-python-for-amazon-mwaa](https://aws-oss.beachgeek.co.uk/3lq) this repo provides python code and uses AWS CDK to help you automate the deployment and configuration of Managed Workflows for Apache Airflow (MWAA). I have shared my own repos to help you do this, but you can never have enough of a good thing, so check out this repo and see if it is useful.(#186)

**dynamomq**

[dynamomq](https://aws-oss.beachgeek.co.uk/3j1) is a project from Yuichi Watanabe that provides a message queuing library that leverages the features of DynamoDB to achieve high scalability, reliability, and cost efficiency. Notably, its ability to dynamically edit message order and attributes enables flexible adaptation to application requirements. Compared to existing solutions, DynamoMQ offers ease of management for developers while providing the reliability of fully managed services like Amazon SQS. It also encompasses key functionalities expected from a message queue, such as concurrent processing with multiple goroutines, Dead Letter Queues, and ensuring FIFO (First In, First Out) order. (#183)

### Compute - Containers, EC2, Serverless

**aws-signer-oci-artifacts**

[aws-signer-oci-artifacts](https://aws-oss.beachgeek.co.uk/3km) this project is used to demonstrate how OCI artefacts can be signed and verified in a development pipeline. Zhuo-Wei Lee, Alontay Ellis, and Rajarshi Das have put together a blog post to help you get started, so if this project interests you, make sure you dive into [Signing and Validating OCI Artifacts with AWS Signer](https://aws-oss.beachgeek.co.uk/3kn).(#185)

**eks-shared-subnets**

[eks-shared-subnets](https://aws-oss.beachgeek.co.uk/3k2) this sample code demonstrates how a central networking team in an enterprise can create and share the VPC Subnets from their own AWS Account with other Workload Specific accounts. So that, Application teams can deploy and manage their own EKS clusters and related resources in those Subnets. (#184)

### Data, Big Data and Databases

**amazon-bedrock-synthetic-manufacturing-data-generator**

[amazon-bedrock-synthetic-manufacturing-data-generator](https://aws-oss.beachgeek.co.uk/3ln) is a industry aligned synthetic data generation solution. Manufacturing processes generate large amounts of sensor data that can be used for analytics and machine learning models. However, this data may contain sensitive or proprietary information that cannot be shared openly. Synthetic data allows the distribution of realistic example datasets that preserve the statistical properties and relationships in the real data, without exposing confidential information. This enables more open research and benchmarking on representative data. Additionally, synthetic data can augment real datasets to provide more training examples for machine learning algorithms to generalize better. Data augmentation with synthetic manufacturing data can help improve model accuracy and robustness. Overall, synthetic data enables sharing, research, and expanded applications of AI in manufacturing while protecting data privacy and security.(#186)

**aws-data-solutions-framework**

[aws-data-solutions-framework](https://github.com/awslabs/aws-data-solutions-framework) is a framework for implementation and delivery of data solutions with built-in AWS best practices. AWS Data Solutions Framework (DSF) is an abstraction atop AWS services based on AWS Cloud Development Kit (CDK) L3 constructs, packaged as a library. You can leverage AWS DSF to implement your data platform in weeks rather than in months. AWS DSF is available in TypeScript and Python. Use the framework to build your data solutions instead of building cloud infrastructure from scratch. Compose data solutions using integrated building blocks via Infrastructure as Code (IaC), that allow you to benefit from smart defaults and built-in AWS best practices. You can also customize or extend according to your requirements. Check out the dedicated documentation page, complete with examples to get you started. (#178)

**glide-for-redis**

[glide-for-redis](https://aws-oss.beachgeek.co.uk/3l2) or General Language Independent Driver for the Enterprise (GLIDE) for Redis (mayeb GLIDER would have been cooler :-) is a new open source client for Redis that works with any Redis distribution that adheres to the Redis Serialization Protocol (RESP) specification. The client is optimised for security, performance, minimal downtime, and observability, and comes pre-configured with best practices learned from over a decade of operating Redis-compatible services used by hundreds of thousands of customers. (#185)

**sso-sync-to-amazon-rds**

[sso-sync-to-amazon-rds](https://aws-oss.beachgeek.co.uk/3e2) This project sets up AWS Lambda functions, Amazon EventBridge rule, Amazon VPC Endpoint for AWS IAM Identity Center (successor to AWS Single Sign-On), the related Security Groups and permissions necessary to automatically provision database users to the Amazon Relational Database Service (Amazon RDS) cluster using AWS Cloud Development Kit (AWS CDK). When a new user is created in IAM Identity Center and the user belongs to the group specified in a IAM_IDC_GROUP_NAME variable, EventBridge rule will trigger the Lambda function. The Lambda function will create a new user in a specified Amazon RDS cluster. The user will then be able to login to the database using their SSO username and IAM credentials. Adding a user to the configured group will trigger the Lambda function as well. (#178)

[tokenizing-db-data-tool](https://aws-oss.beachgeek.co.uk/3lp) provides a handy solution to help you address challenges around masking and keeping safe private or sensitive data (PII), as you need to move data from production to non production systems for activities like testing, debugging, load and integration tests, and more. (#186)

### Developer Tools & DevOps

**aws-iatk**

[aws-iatk](https://aws-oss.beachgeek.co.uk/3fc) AWS Integrated Application Test Kit (IATK), a new open-source test library that makes it easier for developers to create tests for cloud applications with increased speed and accuracy. With AWS IATK, developers can quickly write tests that exercise their code and its AWS integrations against an environment in the cloud, making it easier to catch mistakes early in the development process. IATK includes utilities to generate test events, validate event delivery and structure in Amazon EventBridge Event Bus, and assertions to validate call flow using AWS X-Ray traces. The [AWS IATK](https://aws-oss.beachgeek.co.uk/3g0) is available for Python3.8+. To help you get started, check out the supporting blog post from Dan Fox and Brian Krygsman, [Introducing the AWS Integrated Application Test Kit (IATK)](https://aws-oss.beachgeek.co.uk/3fz). (#180)

**aws-cdk-stack-builder-tool**

[aws-cdk-stack-builder-tool](https://aws-oss.beachgeek.co.uk/3g3) or AWS CDK Builder, is a browser-based tool designed to streamline bootstrapping of Infrastructure as Code (IaC) projects using the AWS Cloud Development Kit (CDK). Equipped with a dynamic visual designer and instant TypeScript code generation capabilities, the CDK Builder simplifies the construction and deployment of CDK projects. It stands as a resource for all CDK users, providing a platform to explore a broad array of CDK constructs. Very cool indeed, and you can deploy on AWS Cloud9, so that this project on my weekend to do list. (#180)

**aws-external-package-security**

[aws-external-package-security](https://aws-oss.beachgeek.co.uk/3g2) provides code to setup a solution that demonstrates how you can deploy AWS Code Services (e.g., AWS CodePipeline, AWS CodeBuild, Amazon CodeGuru Security, AWS CodeArtifact) to orchestrate secure access to external package repositories from an Amazon SageMaker data science environment configured with multi-layer security. The solution can also be expanded upon to account for general developer workflows, where developers use external package dependencies. (#180)

**aws-signer-oci-artifacts**

[aws-signer-oci-artifacts](https://aws-oss.beachgeek.co.uk/3km) this project is used to demonstrate how OCI artefacts can be signed and verified in a development pipeline. Zhuo-Wei Lee, Alontay Ellis, and Rajarshi Das have put together a blog post to help you get started, so if this project interests you, make sure you dive into [Signing and Validating OCI Artifacts with AWS Signer](https://aws-oss.beachgeek.co.uk/3kn).(#185)

**cdk-notifier**

[cdk-notifier](https://aws-oss.beachgeek.co.uk/3it) is a lightweight CLI tool to parse a CDK log file and post changes to pull request requests. Can be used to get more confidence on approving pull requests because reviewer will be aware of changes done to your environments. I am not sure whether this is an old tool, but I have only just found out about it thanks to the blog post from AWS Community Builder, Johannes Konings. He put together [Use cdk-notifier to compare changes in pull requests](https://aws-oss.beachgeek.co.uk/3iu) that explains in more details how this works and walks you through using it. (#183)

**cfn-pipeline**

[cfn-pipeline](https://aws-oss.beachgeek.co.uk/3kv) is a repo from Wolfgang Unger that contains an AWS Codepipeline that will allow automated Cloudformation deployments from within AWS Codepipeline. To help you get started, Wolfgang has put together a detailed blog post that includes videos. Go check it out, [Pipeline for automatic CloudFormation Deployments](https://aws-oss.beachgeek.co.uk/3kw) (#185)

**cloudwatch-macros**

[cloudwatch-macros](https://aws-oss.beachgeek.co.uk/3gs) is the latest open source creation from AWS Hero Efi Merdler-Kravitz, focused on improving the CloudFormation and AWS SAM developer experience. This project features a collection of (basic at the moment) CloudFormation macros, written in Rust, offering seamless deployment through SAM. Check out [Efi's post on LinkedIn](https://aws-oss.beachgeek.co.uk/3gt) for more details and additional useful resources. (#181)

**codecatalyst-blueprints**

[codecatalyst-blueprints](https://aws-oss.beachgeek.co.uk/3kr) This repository contains common blueprint components, the base blueprint constructs and several public blueprints. Blueprints are code generators used to create and maintain projects in Amazon CodeCatalyst. (#185)

**cumuli-aws-console-chat**

[cumuli-aws-console-chat](https://aws-oss.beachgeek.co.uk/3j2)  is an open source Chrome extension that provides similar functionality to Amazon Q. The Cumuli Chrome extension adds a side panel with LLM chat to all AWS pages, and lets you add screenshots of the console to your queries to get context-aware responses. It's similar to Amazon Q but uses GPT-4 Turbo with vision. Check out the repo that includes a demo video of it in action.(#183)

**eks-saas-gitops**

[eks-saas-gitops](https://aws-oss.beachgeek.co.uk/3k1) This repository offers a sample pattern to manage multi-tenancy in a Kubernetes cluster using GitOps with Flux. The provided CloudFormation template automates the deployment of necessary AWS resources and sets up an environment ready for GitOps practices. (#184)

**dynamomq**

[dynamomq](https://aws-oss.beachgeek.co.uk/3j1) is a project from Yuichi Watanabe that provides a message queuing library that leverages the features of DynamoDB to achieve high scalability, reliability, and cost efficiency. Notably, its ability to dynamically edit message order and attributes enables flexible adaptation to application requirements. Compared to existing solutions, DynamoMQ offers ease of management for developers while providing the reliability of fully managed services like Amazon SQS. It also encompasses key functionalities expected from a message queue, such as concurrent processing with multiple goroutines, Dead Letter Queues, and ensuring FIFO (First In, First Out) order. (#183)

**localstack-aws-cdk-example**

[localstack-aws-cdk-example](https://aws-oss.beachgeek.co.uk/3dw) This repo aims to showcase the usage of [Localstack](https://aws-oss.beachgeek.co.uk/3dx) and AWS CDK to address specific integration challenges regarding local development where the end target is the AWS platform. If you are unfamiliar with Localstack, it is an open source, fully functional local AWS cloud stack that allows you to develop and test your cloud and Serverless apps offline. (#178)

**pagemosaic-website-starter**

[pagemosaic-website-starter](https://aws-oss.beachgeek.co.uk/3gp) is an open source tool from Alex Pust that helps you to host static websites on AWS, using AWS CDK under the covers from the looks of things. To deploy your website, simply transfer your website files to the /platform/web-app directory. Following this, execute the command pnpm deploy-platform to initiate the deployment process. Nice use of You Tube videos in the README to help you get started. (#181)

**aws-pdk**

[aws-pdk](https://aws-oss.beachgeek.co.uk/3jb) the AWS Project Development Kit (AWS PDK) is an open-source tool to help bootstrap and maintain cloud projects. It provides building blocks for common patterns together with development tools to manage and build your projects. The AWS PDK lets you define your projects programatically via the expressive power of type safe constructs available in one of 3 languages (typescript, python or java). Under the covers, AWS PDK is built on top of Projen. The AWS Bites Podcast provides an overview of the AWS Project Development Kit (PDK), and the hosts discuss what PDK is, how it can help generate boilerplate code and infrastructure, keep configuration consistent across projects, and some pros and cons of using a tool like this versus doing it manually. (#184)

**rust-s3-cdn**

[rust-s3-cdn](https://aws-oss.beachgeek.co.uk/3j4) provides a Least Recently Used (LRU) cached proxy for AWS S3 written in Rust. I actually had to look up [LRU](https://helpful.knobs-dials.com/index.php/Cache_and_proxy_notes#Least_Recently_Used_(LRU)) as this was a new term for me. The repo provides a couple of reasons why you might want to use this tool, as well as helping you be aware of caveats too.(#183)

**s3-fast-list**

[s3-fast-list](https://aws-oss.beachgeek.co.uk/3k6) is a rust based tool that concurrently list Amazon S3 bucket with ListObjectsV2 API. Check out the README for use cases as to when s3-fast-list is going to help you out (against existing options you have) (#184)

**tokenizing-db-data-tool**

[tokenizing-db-data-tool](https://aws-oss.beachgeek.co.uk/3lp) provides a handy solution to help you address challenges around masking and keeping safe private or sensitive data (PII), as you need to move data from production to non production systems for activities like testing, debugging, load and integration tests, and more. (#186)

**vscode-on-ec2-for-prototyping**

[vscode-on-ec2-for-prototyping](https://aws-oss.beachgeek.co.uk/3lo) This repository introduces how to access and use VSCode hosted on EC2 from a browser. The connection is made via Session Manager, so IAM permissions are used for authentication. The access destination will be localhost. Please note that this repository does not introduce connecting from your local VSCode to an EC2 instance via Remote SSH. (#186)

### Governance & Risk

**appfabric-data-analytics**

[appfabric-data-analytics](https://aws-oss.beachgeek.co.uk/3k3) this project enables you to maintain logs from various SaaS applications and provides the ability to search and display log data. This solution leverages AWS AppFabric to create a data repository that you can query with Amazon Athena. While customers can obtain normalized and enriched SaaS audit log data (OCSF) from AppFabric, many prefer not only to forward these logs to a security tool. Some have the requirement to preserve logs for post-incident analysis, while others need to utilize the logs for tracking SaaS subscription and license usage. Additionally, some customers aim to analyze user activity to discover patterns. This project establishes a data pipeline that empowers customers to construct dashboards on top of it. (#184)

**aws-control-tower-controls-cdk**

[aws-control-tower-controls-cdk](https://aws-oss.beachgeek.co.uk/3e3) This pattern describes how to use AWS CloudFormation and AWS Cloud Development Kit (AWS CDK) to implement and administer preventive, detective, and proactive AWS Control Tower controls as infrastructure as code (IaC). A control (also known as a guardrail) is a high-level rule that provides ongoing governance for your overall AWS Control Tower environment. For example, you can use controls to require logging for your AWS accounts and then configure automatic notifications if specific security-related events occur. Check out the REAMDE for more details on what you can do with this. (#178)

**aws-external-package-security**

[aws-external-package-security](https://aws-oss.beachgeek.co.uk/3g2) provides code to setup a solution that demonstrates how you can deploy AWS Code Services (e.g., AWS CodePipeline, AWS CodeBuild, Amazon CodeGuru Security, AWS CodeArtifact) to orchestrate secure access to external package repositories from an Amazon SageMaker data science environment configured with multi-layer security. The solution can also be expanded upon to account for general developer workflows, where developers use external package dependencies. (#180)

**aws-organizations-tag-inventory**

[aws-organizations-tag-inventory](https://aws-oss.beachgeek.co.uk/3jz)  This project provides a solution to AWS customers for reporting on what tags exists, the resources they are applied to, and what resources don't have tags across their entire AWS organization. The solution is designed to be deployed in an AWS Organization with multiple accounts. Detailed information and deployment guidelines are in the README, including some sample dashboards so you can see what you can expect.(#184)

**aws-waf-for-event-analysis-dashboard**

[aws-waf-for-event-analysis-dashboard](https://aws-oss.beachgeek.co.uk/3ls) finding the information you need during security incidents is what this project aims to help with. During major online events like live broadcasts, security teams need a fast and clear understanding of attack patterns and behaviours to distinguish between normal and malicious traffic flows. The solution outlined here allows filtering flow logs by "Client IP", "URI", "Header name", and "Header value" to analyse these fields and pinpoint values specifically associated with attack traffic versus normal traffic. For example, the dashboard can identify the top header values that are atypical for normal usage. The security team can then create an AWS WAF rule to block requests containing these header values, stopping the attack. This project demonstrates using AWS Glue crawlers to categorise and structure WAF flow log data and Amazon Athena for querying. Amazon Quicksight is then employed to visualise query results in a dashboard. Once deployed, the dashboard provides traffic visualisation similar to the example graphs shown in Images folder in under project , empowering security teams with insight into attacks and defence.(#186)

**awskillswitch**

[awskillswitch](https://aws-oss.beachgeek.co.uk/3gr) is an open sourced tool from Jeffrey Lyon that is worth checking out. AWS Kill Switch is a Lambda function (and proof of concept client) that an organisation can implement in a dedicated "Security" account to give their security engineers the ability to delete IAM roles or apply a highly restrictive service control policy (SCP) on any account in their organisation. Make sure you check out the README for full details, but this looks like it might be one of those tools that are useful to have in the back pocket in times of need. (#181)

**observability-solution-kit**

[observability-solution-kit](https://aws-oss.beachgeek.co.uk/3d3) this repository is the Ollyv sdk. The Ollyv sdk offers a simple way to configure its behaviour through *.properties files, which are environment-specific. Currently code is provide from NodeLambda ✨ · NodeExpress 👟 · JavaSpring 🦚 (#177)

### Java, Kotlin, Scala, OpenJDK

**powertools-lambda-kotlin**

[powertools-lambda-kotlin](https://aws-oss.beachgeek.co.uk/3dv) This project demonstrates the Lambda for Powertools Kotlin module deployed using Serverless Application Model with Gradle running the build. This example is configured for Java 8 only; in order to use a newer version, check out the Gradle configuration guide in the main project README. You can also use sam init to create a new Gradle-powered Powertools application - choose to use the AWS Quick Start Templates, and then Hello World Example with Powertools for AWS Lambda, Java 17 runtime, and finally gradle. (#178)

### Networking

**eks-shared-subnets**

[eks-shared-subnets](https://aws-oss.beachgeek.co.uk/3k2) this sample code demonstrates how a central networking team in an enterprise can create and share the VPC Subnets from their own AWS Account with other Workload Specific accounts. So that, Application teams can deploy and manage their own EKS clusters and related resources in those Subnets. (#184)

**rust-s3-cdn**

[rust-s3-cdn](https://aws-oss.beachgeek.co.uk/3j4) provides a Least Recently Used (LRU) cached proxy for AWS S3 written in Rust. I actually had to look up [LRU](https://helpful.knobs-dials.com/index.php/Cache_and_proxy_notes#Least_Recently_Used_(LRU)) as this was a new term for me. The repo provides a couple of reasons why you might want to use this tool, as well as helping you be aware of caveats too.

**trading-latency-benchmark**

[trading-latency-benchmark](https://aws-oss.beachgeek.co.uk/3d1) This repository contains a network latency test stack that consists of Java based trading client and Ansible playbooks to coordinate distributed tests. Java based trading client is designed to send limit and cancel orders, allowing you to measure round-trip times of the network communication. (#177)

### Security

**avp-toy-store-sample**

[avp-toy-store-sample](https://aws-oss.beachgeek.co.uk/3jw) is a great sample project if you want to explore Cedar, and how this fits in with Amazon Verified Permissions. This sample web application demonstrates authentication and policy-based authorization for different user types to an imaginary toy store. The toy store takes orders online and then send them to customers through multiple warehouses. This application is used by warehouses to help sending orders to customers. The application uses Amazon Cognito for authentication and uses Amazon Verified Permissions for policy-based authorization. Additionally, the application uses API-Gateway as the front-door to the application, and Lambda to process requests. (#184)

**aws-waf-for-event-analysis-dashboard**

[aws-waf-for-event-analysis-dashboard](https://aws-oss.beachgeek.co.uk/3ls) finding the information you need during security incidents is what this project aims to help with. During major online events like live broadcasts, security teams need a fast and clear understanding of attack patterns and behaviours to distinguish between normal and malicious traffic flows. The solution outlined here allows filtering flow logs by "Client IP", "URI", "Header name", and "Header value" to analyse these fields and pinpoint values specifically associated with attack traffic versus normal traffic. For example, the dashboard can identify the top header values that are atypical for normal usage. The security team can then create an AWS WAF rule to block requests containing these header values, stopping the attack. This project demonstrates using AWS Glue crawlers to categorise and structure WAF flow log data and Amazon Athena for querying. Amazon Quicksight is then employed to visualise query results in a dashboard. Once deployed, the dashboard provides traffic visualisation similar to the example graphs shown in Images folder in under project , empowering security teams with insight into attacks and defence.(#186)

**awskillswitch**

[awskillswitch](https://aws-oss.beachgeek.co.uk/3gr) is an open sourced tool from Jeffrey Lyon that is worth checking out. AWS Kill Switch is a Lambda function (and proof of concept client) that an organisation can implement in a dedicated "Security" account to give their security engineers the ability to delete IAM roles or apply a highly restrictive service control policy (SCP) on any account in their organisation. Make sure you check out the README for full details, but this looks like it might be one of those tools that are useful to have in the back pocket in times of need. (#181)

**s3-presignedurl-staticips-endpoint-with-cdk**

[s3-presignedurl-staticips-endpoint-with-cdk](https://aws-oss.beachgeek.co.uk/3k5) this solution simplifies access to Amazon S3 by creating secure, custom presigned URLs for object downloads through a single endpoint with a unique domain and static IPs. The use case involves users following an IP and domain Allowlist firewall policy, limiting API access to specific domains and IPs. The architecture employs key AWS services, including AWS Global Accelerator, Amazon API Gateway, AWS Lambda, Application Load Balancer(ALB), VPC Endpoint, and Amazon S3. This design centralizes the API for generating presigned URLs and the S3 endpoint under a single domain, linked to an AWS Global Accelerator with two static IPs. Consequently, users can effortlessly request presigned URLs and download S3 objects through a unified domain endpoint with static IPs. This architecture is especially beneficial for customers with strict policies or compliance requirements, such as those in the public, medical, and finance sectors. (#184)

**sso-sync-to-amazon-rds**

[sso-sync-to-amazon-rds](https://aws-oss.beachgeek.co.uk/3e2) This project sets up AWS Lambda functions, Amazon EventBridge rule, Amazon VPC Endpoint for AWS IAM Identity Center (successor to AWS Single Sign-On), the related Security Groups and permissions necessary to automatically provision database users to the Amazon Relational Database Service (Amazon RDS) cluster using AWS Cloud Development Kit (AWS CDK). When a new user is created in IAM Identity Center and the user belongs to the group specified in a IAM_IDC_GROUP_NAME variable, EventBridge rule will trigger the Lambda function. The Lambda function will create a new user in a specified Amazon RDS cluster. The user will then be able to login to the database using their SSO username and IAM credentials. Adding a user to the configured group will trigger the Lambda function as well. (#178)

### Storage

**amazon-s3-glacier-archive-data-delete**

[amazon-s3-glacier-archive-data-delete](https://aws-oss.beachgeek.co.uk/3lr) Amazon S3 Glacier Archive (data) Delete solution provides an automated workflow to delete ALL of your data in an S3 Glacier Vault. This solution only applies to Amazon S3 Glacier Vault Archives. Within S3 Glacier, data is stored as an Archive within a Vault. This solution does not apply to objects in Glacier Deep Archive, Glacier Flexible Retrieval, and Glacier Instant Retrieval stored in an Amazon S3 Bucket. Good README with clear guidance and overview of how this works.(#186)

**s3-fast-list**

[s3-fast-list](https://aws-oss.beachgeek.co.uk/3k6) is a rust based tool that concurrently list Amazon S3 bucket with ListObjectsV2 API. Check out the README for use cases as to when s3-fast-list is going to help you out (against existing options you have) (#184)

**s3-presignedurl-staticips-endpoint-with-cdk**

[s3-presignedurl-staticips-endpoint-with-cdk](https://aws-oss.beachgeek.co.uk/3k5) this solution simplifies access to Amazon S3 by creating secure, custom presigned URLs for object downloads through a single endpoint with a unique domain and static IPs. The use case involves users following an IP and domain Allowlist firewall policy, limiting API access to specific domains and IPs. The architecture employs key AWS services, including AWS Global Accelerator, Amazon API Gateway, AWS Lambda, Application Load Balancer(ALB), VPC Endpoint, and Amazon S3. This design centralizes the API for generating presigned URLs and the S3 endpoint under a single domain, linked to an AWS Global Accelerator with two static IPs. Consequently, users can effortlessly request presigned URLs and download S3 objects through a unified domain endpoint with static IPs. This architecture is especially beneficial for customers with strict policies or compliance requirements, such as those in the public, medical, and finance sectors. (#184)


# AWS Services

**appfabric-data-analytics**

[appfabric-data-analytics](https://aws-oss.beachgeek.co.uk/3k3) this project enables you to maintain logs from various SaaS applications and provides the ability to search and display log data. This solution leverages AWS AppFabric to create a data repository that you can query with Amazon Athena. While customers can obtain normalized and enriched SaaS audit log data (OCSF) from AppFabric, many prefer not only to forward these logs to a security tool. Some have the requirement to preserve logs for post-incident analysis, while others need to utilize the logs for tracking SaaS subscription and license usage. Additionally, some customers aim to analyze user activity to discover patterns. This project establishes a data pipeline that empowers customers to construct dashboards on top of it. (#184)

**amazon-chime-sdk-voice-voice-translator**

[amazon-chime-sdk-voice-voice-translator](https://aws-oss.beachgeek.co.uk/3k4) this project leverages the Amazon Chime SDK to create a voice to voice live translator. It facilitates real time translation in voice calls enabling seamless communication between participants speaking different languages. The system integrates various AWS services, including Amazon Chime SDK, Amazon Kinesis Video Streams (KVS), Amazon Transcribe, Amazon Translate, Amazon Polly, etc. to achieve efficient and accurate translation. (#184)

**aws-cdk-python-for-amazon-mwaa**

[aws-cdk-python-for-amazon-mwaa](https://aws-oss.beachgeek.co.uk/3lq) this repo provides python code and uses AWS CDK to help you automate the deployment and configuration of Managed Workflows for Apache Airflow (MWAA). I have shared my own repos to help you do this, but you can never have enough of a good thing, so check out this repo and see if it is useful.(#186)

**aws-control-tower-controls-cdk**

[aws-control-tower-controls-cdk](https://aws-oss.beachgeek.co.uk/3e3) This pattern describes how to use AWS CloudFormation and AWS Cloud Development Kit (AWS CDK) to implement and administer preventive, detective, and proactive AWS Control Tower controls as infrastructure as code (IaC). A control (also known as a guardrail) is a high-level rule that provides ongoing governance for your overall AWS Control Tower environment. For example, you can use controls to require logging for your AWS accounts and then configure automatic notifications if specific security-related events occur. Check out the REAMDE for more details on what you can do with this. (#178)

**aws-cdk-stack-builder-tool**

[aws-cdk-stack-builder-tool](https://aws-oss.beachgeek.co.uk/3g3) or AWS CDK Builder, is a browser-based tool designed to streamline bootstrapping of Infrastructure as Code (IaC) projects using the AWS Cloud Development Kit (CDK). Equipped with a dynamic visual designer and instant TypeScript code generation capabilities, the CDK Builder simplifies the construction and deployment of CDK projects. It stands as a resource for all CDK users, providing a platform to explore a broad array of CDK constructs. Very cool indeed, and you can deploy on AWS Cloud9, so that this project on my weekend to do list. (#180)

**aws-external-package-security**

[aws-external-package-security](https://aws-oss.beachgeek.co.uk/3g2) provides code to setup a solution that demonstrates how you can deploy AWS Code Services (e.g., AWS CodePipeline, AWS CodeBuild, Amazon CodeGuru Security, AWS CodeArtifact) to orchestrate secure access to external package repositories from an Amazon SageMaker data science environment configured with multi-layer security. The solution can also be expanded upon to account for general developer workflows, where developers use external package dependencies. (#180)

**aws-inference-benchmark**
[aws-inference-benchmark ](https://aws-oss.beachgeek.co.uk/3cy)this project from Rustem Feyzkhanov contains code for running deep learning inference benchmarks on different AWS instances and service types. Check out his post, [Making LLMs Scalable: Cloud Inference with AWS Fargate and Copilot](https://aws-oss.beachgeek.co.uk/3d0) where Rustem shows you in more details how you can use this repo. (#177)

**aws-organizations-tag-inventory**

[aws-organizations-tag-inventory](https://aws-oss.beachgeek.co.uk/3jz)  This project provides a solution to AWS customers for reporting on what tags exists, the resources they are applied to, and what resources don't have tags across their entire AWS organization. The solution is designed to be deployed in an AWS Organization with multiple accounts. Detailed information and deployment guidelines are in the README, including some sample dashboards so you can see what you can expect.(#184)

**cdk-notifier**

[cdk-notifier](https://aws-oss.beachgeek.co.uk/3it) is a lightweight CLI tool to parse a CDK log file and post changes to pull request requests. Can be used to get more confidence on approving pull requests because reviewer will be aware of changes done to your environments. I am not sure whether this is an old tool, but I have only just found out about it thanks to the blog post from AWS Community Builder, Johannes Konings. He put together [Use cdk-notifier to compare changes in pull requests](https://aws-oss.beachgeek.co.uk/3iu) that explains in more details how this works and walks you through using it. (#183)

**gen-ai-on-eks**

[gen-ai-on-eks](https://aws-oss.beachgeek.co.uk/3d2) this repository aims to showcase how to finetune a FM model in Amazon EKS cluster using, JupyterHub to provision notebooks and craft both serving and training scripts, RayOperator to manage Ray Clusters and Karpenter to manage Node Scaling. (#177)

**makit-llm-lambda**

[makit-llm-lambda ](https://aws-oss.beachgeek.co.uk/3cx)this repo from Martyn Kilbryde is an example of how you can run a Large Language Model (LLM) inside an AWS Lambda Function.  Whilst the code will help you deploy to AWS Lambda, it can be ran locally inside Docker for testing as well. The function contains the full LLM model and the code to use the model, allowing basic text generation from a HTTP call into it. (#177)

**observability-solution-kit**

[observability-solution-kit](https://aws-oss.beachgeek.co.uk/3d3) this repository is the Ollyv sdk. The Ollyv sdk offers a simple way to configure its behaviour through *.properties files, which are environment-specific. Currently code is provide from NodeLambda ✨ · NodeExpress 👟 · JavaSpring 🦚 (#177)

**resource-explorer-with-organizations**

[resource-explorer-with-organizations](https://aws-oss.beachgeek.co.uk/3dy) you may have a use cases where you are eager to find lingering resources, or resources that were not at their optimal settings. By utilising Resource Explorer and Step Functions, you can gather all the necessary information from these accounts, and use them to create a report to gain a wider understanding of the state of your AWS accounts. As of this release, the limitation of Resource Explorer is that it is done on a per account basis. However, the README provides details of a workaround to deploy this to all your accounts in our AWS Organization using StackSets. The use case shown in the repo shows you how you can find resources in an multiple AWS accounts over multiple regions, and generating an Excel Document displaying the Account it belongs to, Name, Resource Type, and ARN of the resource. The repo provides details of how you can deploy this tool, so make sure you check that out too. (#178)

**s3-fast-list**

[s3-fast-list](https://aws-oss.beachgeek.co.uk/3k6) is a rust based tool that concurrently list Amazon S3 bucket with ListObjectsV2 API. Check out the README for use cases as to when s3-fast-list is going to help you out (against existing options you have) (#184)

**s3-presignedurl-staticips-endpoint-with-cdk**

[s3-presignedurl-staticips-endpoint-with-cdk](https://aws-oss.beachgeek.co.uk/3k5) this solution simplifies access to Amazon S3 by creating secure, custom presigned URLs for object downloads through a single endpoint with a unique domain and static IPs. The use case involves users following an IP and domain Allowlist firewall policy, limiting API access to specific domains and IPs. The architecture employs key AWS services, including AWS Global Accelerator, Amazon API Gateway, AWS Lambda, Application Load Balancer(ALB), VPC Endpoint, and Amazon S3. This design centralizes the API for generating presigned URLs and the S3 endpoint under a single domain, linked to an AWS Global Accelerator with two static IPs. Consequently, users can effortlessly request presigned URLs and download S3 objects through a unified domain endpoint with static IPs. This architecture is especially beneficial for customers with strict policies or compliance requirements, such as those in the public, medical, and finance sectors. (#184)

**terraform-aws-ecr-watch**

[terraform-aws-ecr-watch](https://aws-oss.beachgeek.co.uk/3f2) is a project out of the folks from Porsche, when they are not busy designing super fast cars, their engineers are busy creating useful open source tools for folks to use. This project is a Terraform module to configure an AWS ECR Usage Dashboard based on AWS CloudWatch log insight queries with data fetched from AWS CloudTrail. (#180)

# Open Source projects on AWS

**cumuli-aws-console-chat**

[cumuli-aws-console-chat](https://aws-oss.beachgeek.co.uk/3j2)  is an open source Chrome extension that provides similar functionality to Amazon Q. The Cumuli Chrome extension adds a side panel with LLM chat to all AWS pages, and lets you add screenshots of the console to your queries to get context-aware responses. It's similar to Amazon Q but uses GPT-4 Turbo with vision. Check out the repo that includes a demo video of it in action.(#183)

**weaviate-on-eks**

[weaviate-on-eks](https://aws-oss.beachgeek.co.uk/3d4) this repository includes sample code that can be used to deploy and configure an instance of the [Weaviate](https://aws-oss.beachgeek.co.uk/3d5) distributed vector database on EKS. (#177)

**ragna**

[ragna](https://aws-oss.beachgeek.co.uk/3f3) this is a repo I put together to show you how you can add Amazon Bedrock models from Anthropic and Meta within the Ragna tool. I blogged last week about this [#179](https://dev.to/aws/unboxing-ragna-getting-hands-on-and-making-it-to-work-with-amazon-bedrock-7k3) but I have put together this repo that shows the actual code as I had received quite a few DMs, and as a bonus, I have also added the recently announced Llama2 13B model from Meta. To help with this, a new blog post, [Adding Amazon Bedrock Llama2 as an assistant in Ragna](https://dev.to/aws/adding-amazon-bedrock-llama2-as-an-assistant-in-ragna-pdl) will help you get this all up and running. There is also lots of useful info in the project README. (#180)

# Demos and Samples

**aws-agentic-document-assistant**

[aws-agentic-document-assistant](https://aws-oss.beachgeek.co.uk/3gu) The Agentic Documents Assistant is an LLM assistant that provides users with easy access to information and insights stored across their business documents, through natural conversations and question answering. It supports answering factual questions by retrieving information directly from documents using semantic search with the popular RAG design pattern. Additionally, it answers analytical questions by translating user questions into SQL queries and running them against a database of entities extracted from the documents with a batch process. It is also able to answer complex multi-step questions by combining different tools and data sources using an LLM agent design pattern.(#181)

**aws-clean-rooms-lab**

[aws-clean-rooms-lab ](https://aws-oss.beachgeek.co.uk/3j5)is a workshop from AWS Security Hero Richard Fan that  will walk you through the setup of AWS Clean Rooms so you can try its different features. Richard wrote about this repo in his write up [Start building my AWS Clean Rooms lab](https://aws-oss.beachgeek.co.uk/3j6), which you should read to help you get started. This is a work in progress, but there is still a lot of stuff to get stuck into so worth checking out if AWS Clean Rooms is something that you are exploring. (#183)

**big-data-summarization-using-griptape-bedrock-redshift**

[big-data-summarization-using-griptape-bedrock-redshift](https://aws-oss.beachgeek.co.uk/3gv) I have looked at Griptape in other blog posts, so it was nice to see this repo that provides sample code and instructions for a Big data summarisation example using this popular open-source library, together with Amazon Bedrock and Amazon Redshift. In this sample,  TitanXL LLM is used to summarise but Anthropic's Claude v2 is also used to drive the application. This application sample demonstrates how data can be pulled from Amazon Redshift and then passed to the summarisation model. The driving model is isolated from the actual data and uses the tools provided to it to orchestrate the application. (#181)

**gen-ai-on-eks**

[gen-ai-on-eks](https://aws-oss.beachgeek.co.uk/3d2) this repository aims to showcase how to finetune a FM model in Amazon EKS cluster using, JupyterHub to provision notebooks and craft both serving and training scripts, RayOperator to manage Ray Clusters and Karpenter to manage Node Scaling. (#177)

**get-the-news-rss-atom-feed-summary**

[get-the-news-rss-atom-feed-summary](https://aws-oss.beachgeek.co.uk/3l6) is a fantastic demo that demonstrates some of the capabilities that using large language models can help you introduce into your applications. The demo code provides a summary of the most recent news from an RSS or Atom feed using Amazon Bedrock. (#185)


**golang-url-shortener**

[golang-url-shortener](https://aws-oss.beachgeek.co.uk/3j3) is a project that you can build from Furkan Gulsen that deploys a URL shortener service, built with Go and Hexagonal Architecture, leverages a serverless approach for efficient scalability and performance. It uses a variety of AWS services to provide a robust, maintainable, and highly available URL shortening service. Are URL Shortners the new todo app? Not sure but I really like the quality of the documentation of this project, and like I did last year with implementing a serverless web analytics solution, I am going to check this project out and see if it would be a good replacement for the tool I currently use, YOURLS. Check out the discussion on reddit [here](https://www.reddit.com/r/aws/comments/18nnfix/url_shortener_hexagonal_serverless_architecture/).(#183)

**makit-llm-lambda**

[makit-llm-lambda ](https://aws-oss.beachgeek.co.uk/3cx)this repo from Martyn Kilbryde is an example of how you can run a Large Language Model (LLM) inside an AWS Lambda Function.  Whilst the code will help you deploy to AWS Lambda, it can be ran locally inside Docker for testing as well. The function contains the full LLM model and the code to use the model, allowing basic text generation from a HTTP call into it. (#177)

**multi-tenant-chatbot-using-rag-with-amazon-bedrock**

[multi-tenant-chatbot-using-rag-with-amazon-bedrock](https://aws-oss.beachgeek.co.uk/3c5) provides a solution for building a multi-tenant chatbot with Retrieval Augmented Generation (RAG). RAG is a common pattern where a general-purpose language model is queried with a user question along with additional contextual information extracted from private documents. To help you understand and deploy the code, check out the supporting blog post from Farooq Ashraf, Jared Dean, and Ravi Yadav, [Build a multi-tenant chatbot with RAG using Amazon Bedrock and Amazon EKS](https://aws-oss.beachgeek.co.uk/3c6) (#177)

**quarkus-bedrock-demo**

[quarkus-bedrock-demo](https://aws-oss.beachgeek.co.uk/3cv) This is a sample project from my colleague Denis Traub, based on [work from Vini](https://aws-oss.beachgeek.co.uk/3b2)  , that demonstrates how to access Amazon Bedrock from a Quarkus application deployed on AWS Lambda. (#177)

**reinvent-session-concierge**

[reinvent-session-concierge](https://aws-oss.beachgeek.co.uk/3gq) is potentially a very useful tool for those of you heading out to re:Invent, and wanting to make sure that you make the most of your time there by attending the sessions of most interest to you. This project uses Amazon Bedrock Titan text embeddings stored in a PostgreSQL database to enable generative AI queries across the re:Invent session data. It combines both semantic search and traditional queries. I am going to be trying it out later today to help me plan my online viewing. (#181)

**serverless-rss-filtered-feed-gen**

[serverless-rss-filtered-feed-gen](https://aws-oss.beachgeek.co.uk/3dz) This is a configurable serverless solution that generates filtered rss feeds and makes them public accessible. Defined RSS sources are read at a given interval and new filtered feeds are generated and stored. The architecture uses a minimum number of AWS services to keep it easy to maintain and cost-effective. (#178)

**youtube-video-summarizer-with-bedrock**

[youtube-video-summarizer-with-bedrock](https://aws-oss.beachgeek.co.uk/3j7) is a example project from Zied Ben Tahar that uses large language models to create a YouTube video summariser, allowing you to sift through You Tube videos and get an high level summary of them, allowing you to make better decisions as to whether you want to spend more time watching the video.  Zied has also put together a supporting blog post, [AI powered video summariser with Amazon Bedrock](https://aws-oss.beachgeek.co.uk/3j8) that provides everything you need to get this project up and running for yourself. (#183)

# Industry use cases

**garnet-framework**

[garnet-framework](https://aws-oss.beachgeek.co.uk/3e1) Garnet is an open-source framework for building scalable, reliable and interoperable platforms leveraging open standards, FIWARE open source technology and AWS Cloud services. It supports the development and integration of smart and efficient solutions across multiple domains such as Smart Cities, Regions and Campuses, Energy and Utilities, Agriculture, Smart Building, Automotive and Manufacturing. The repo provides code and links to the dedicated documentation site to help you get started. (#178)

**geo-location-api**

[geo-location-api](https://aws-oss.beachgeek.co.uk/3k0) is a project for the .NET developers out there, that provides a NET Web API for retrieving geolocations. The  geolocation data is provided by MaxMind GeoLite2. (#184)

**res**

[res](https://aws-oss.beachgeek.co.uk/3f0) Research and Engineering Studio on AWS (RES) is an open source, easy-to-use web-based portal for administrators to create and manage secure cloud-based research and engineering environments. Using RES, scientists and engineers can visualise data and run interactive applications without the need for cloud expertise. With just a few clicks, scientists and engineers can create and connect to Windows and Linux virtual desktops that come with pre-installed applications, shared data, and collaboration tools they need. With RES, administrators can define permissions, set budgets, and monitor resource utilisation through a single web interface. RES virtual desktops are powered by Amazon EC2 instances and NICE DCV. RES is available at no additional charge. You pay only for the AWS resources needed to run your applications. (#180)









