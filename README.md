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

**bedrust**

[bedrust](https://aws-oss.beachgeek.co.uk/3n1) is a demo repo from my colleague Darko Mesaros that shows you how you can use Amazon Bedrock in your Rust code, and allows you to currently choose between Claude V2, Llama2 70B, and Cohere Command.(#188)

**bedrock-vscode-playground**

[bedrock-vscode-playground](https://aws-oss.beachgeek.co.uk/3nb) is a Visual Studio Code (VS Code) extension which allows developers to easily explore and experiment with large language models (LLMs) available in Amazon Bedrock. Check out the README for details of what you can do with it and how you can configure it to work with your specific setup.(#188)

**ecs-gpu-scaling**

[ecs-gpu-scaling](https://aws-oss.beachgeek.co.uk/3mh) This repository is intended for engineers looking to horizontally scale GPU-based Machine Learning (ML) workloads on Amazon ECS. By default, GPU utilisation metrics are not part of the predefined metrics available with Application Autoscaling. As such, you implement auto scaling based on custom metrics.  For NVIDIA-based GPUs, you use DCGM-Exporter in your container to expose GPU metrics. You can then use metrics such as DCGM_FI_DEV_GPU_UTIL and DCGM_FI_DEV_GPU_TEMP to determine your auto scaling behaviour. The README provides links to all the additional resources you need to get this up and running.(#187)

**foundation-model-benchmarking-tool**

[foundation-model-benchmarking-tool](https://aws-oss.beachgeek.co.uk/3mj) is a Foundation model (FM) benchmarking tool. Run any model on Amazon SageMaker and benchmark for performance across instance type and serving stack options. A key challenge with FMs is the ability to benchmark their performance in terms of inference latency, throughput and cost so as to determine which model running with what combination of the hardware and serving stack provides the best price-performance combination for a given workload.(#187)

**gen-ai-on-eks**

[gen-ai-on-eks](https://aws-oss.beachgeek.co.uk/3d2) this repository aims to showcase how to finetune a FM model in Amazon EKS cluster using, JupyterHub to provision notebooks and craft both serving and training scripts, RayOperator to manage Ray Clusters and Karpenter to manage Node Scaling. (#177)

**makit-llm-lambda**

[makit-llm-lambda ](https://aws-oss.beachgeek.co.uk/3cx)this repo from Martyn Kilbryde is an example of how you can run a Large Language Model (LLM) inside an AWS Lambda Function.  Whilst the code will help you deploy to AWS Lambda, it can be ran locally inside Docker for testing as well. The function contains the full LLM model and the code to use the model, allowing basic text generation from a HTTP call into it. (#177)

**partysmith**

[partysmith](https://aws-oss.beachgeek.co.uk/3l4) is an awesome project from AWS Community Builder Stephen Sennett, that provides an unofficial way to transform your AWS PartyRock apps into deployable full-stack SvelteKit applications. Users can enter the URL of a publicly published PartyRock app, select the desired settings, and PartySmith will forge an application into a ZIP archive which will be downloaded to your machine, and ready for use. How cool is that! (Very in case you were wondering). Find out more by reading the supporting blog post, [PartySmith - Bring PartyRock apps to your place](https://aws-oss.beachgeek.co.uk/3l5). (#185)

**promptus**

[promptus](https://aws-oss.beachgeek.co.uk/3mu) Prompt engineering is key for generating high-quality AI content. But crafting effective prompts can be time-consuming and difficult. That's why I built Promptus. Promptus allows you to easily create, iterate, and organise prompts for generative AI models. With Promptus, you can:

* Quickly build prompts with an intuitive interface
* Automatically version and compare prompt iterations to optimise quality
* Organize prompts into projects and share with teammates
* See a history of your prompt and easily go back to any previous prompt execution

(#188)

**rag-with-amazon-bedrock-and-pgvector**

[rag-with-amazon-bedrock-and-pgvector](https://aws-oss.beachgeek.co.uk/3lv) is an opinionated sample repo on how to configure and deploy RAG (Retrieval Augmented Retrieval) application. It uses a number of components including Amazon Bedrock for foundational models, Amazon RDS PostgreSQL with pgvector, LangChain, Streamlit, and a number of AWS services to bring it all together.(#186)

**RefChecker**

[RefChecker](https://aws-oss.beachgeek.co.uk/3l3) For all their remarkable abilities, large language models (LLMs) have an Achilles heel, which is their tendency to hallucinate, or make assertions that sound plausible but are factually inaccurate. RefChecker provides automatic checking pipeline and benchmark dataset for detecting fine-grained hallucinations generated by Large Language Models. Check out the supporting post for this tool, [New tool, dataset help detect hallucinations in large language models](https://aws-oss.beachgeek.co.uk/3l7) (#185)

**s3-connector-for-pytorch**

[s3-connector-for-pytorch](https://aws-oss.beachgeek.co.uk/3gw) the Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. Using the S3 Connector for PyTorch automatically optimises performance when downloading training data from and writing checkpoints to Amazon S3, eliminating the need to write your own code to list S3 buckets and manage concurrent requests. Amazon S3 Connector for PyTorch provides implementations of PyTorch's dataset primitives that you can use to load training data from Amazon S3. It supports both map-style datasets for random data access patterns and iterable-style datasets for streaming sequential data access patterns. The S3 Connector for PyTorch also includes a checkpointing interface to save and load checkpoints directly to Amazon S3, without first saving to local storage. (#181)

### Application integration and middleware

**active-active-cache**

[active-active-cache](https://aws-oss.beachgeek.co.uk/3o5) is a repo that helps you build a solution that implements an active-active cache across 2 AWS regions, using ElastiCache for Redis. This solution is automated with CDK and SAM.(#189)

**aws-apn-connector**

[aws-apn-connector](https://aws-oss.beachgeek.co.uk/3l1) this project from the folks at Nearform provides a way of interacting with the AWS APN (AWS Partner Network) programatically, as this does not provide an API. If you are looking to automate your interactions with the AWS APN, you should check this project out.(#185)

**aws-cdk-python-for-amazon-mwaa**

[aws-cdk-python-for-amazon-mwaa](https://aws-oss.beachgeek.co.uk/3lq) this repo provides python code and uses AWS CDK to help you automate the deployment and configuration of Managed Workflows for Apache Airflow (MWAA). I have shared my own repos to help you do this, but you can never have enough of a good thing, so check out this repo and see if it is useful.(#186)

**dynamomq**

[dynamomq](https://aws-oss.beachgeek.co.uk/3j1) is a project from Yuichi Watanabe that provides a message queuing library that leverages the features of DynamoDB to achieve high scalability, reliability, and cost efficiency. Notably, its ability to dynamically edit message order and attributes enables flexible adaptation to application requirements. Compared to existing solutions, DynamoMQ offers ease of management for developers while providing the reliability of fully managed services like Amazon SQS. It also encompasses key functionalities expected from a message queue, such as concurrent processing with multiple goroutines, Dead Letter Queues, and ensuring FIFO (First In, First Out) order. (#183)

### Compute - Containers, EC2, Serverless

**aws-cdk-imagebuilder-sample**

[aws-cdk-imagebuilder-sample](https://aws-oss.beachgeek.co.uk/3o2) this repo uses AWS CDK (TypeScript) that demonstrates how to create a fully functional ImageBuilder pipeline that builds an Amazon Linux 2023 container image, installing git, docker and nodejs, all the way to pushing the resulting image to an ECR repository.(#189)

**aws-signer-oci-artifacts**

[aws-signer-oci-artifacts](https://aws-oss.beachgeek.co.uk/3km) this project is used to demonstrate how OCI artefacts can be signed and verified in a development pipeline. Zhuo-Wei Lee, Alontay Ellis, and Rajarshi Das have put together a blog post to help you get started, so if this project interests you, make sure you dive into [Signing and Validating OCI Artifacts with AWS Signer](https://aws-oss.beachgeek.co.uk/3kn).(#185)

**ecs-gpu-scaling**

[ecs-gpu-scaling](https://aws-oss.beachgeek.co.uk/3mh) This repository is intended for engineers looking to horizontally scale GPU-based Machine Learning (ML) workloads on Amazon ECS. By default, GPU utilisation metrics are not part of the predefined metrics available with Application Autoscaling. As such, you implement auto scaling based on custom metrics.  For NVIDIA-based GPUs, you use DCGM-Exporter in your container to expose GPU metrics. You can then use metrics such as DCGM_FI_DEV_GPU_UTIL and DCGM_FI_DEV_GPU_TEMP to determine your auto scaling behaviour. The README provides links to all the additional resources you need to get this up and running.(#187)

**eks-shared-subnets**

[eks-shared-subnets](https://aws-oss.beachgeek.co.uk/3k2) this sample code demonstrates how a central networking team in an enterprise can create and share the VPC Subnets from their own AWS Account with other Workload Specific accounts. So that, Application teams can deploy and manage their own EKS clusters and related resources in those Subnets. (#184)

**fargate-on-demand**

[fargate-on-demand](https://aws-oss.beachgeek.co.uk/3mv) this repo provides the code that shows you how you can cost optimise your container applications and then control scale down (or up) as needed. Yoanna Krasteva has put together a blog post that provides you with why and how you can configure this in the post, [Cost saving with AWS Fargate On-Demand](https://aws-oss.beachgeek.co.uk/3mw).(#188)

**llrt**

[llrt](https://aws-oss.beachgeek.co.uk/3mm) - Low Latency Runtime (LLRT) is a lightweight JavaScript runtime designed to address the growing demand for fast and efficient Serverless applications. LLRT offers up to over 10x faster startup and up to 2x overall lower cost compared to other JavaScript runtimes running on AWS Lambda. It's is built in Rust, utilising QuickJS as JavaScript engine, ensuring efficient memory usage and swift startup. (#188)

### Data, Big Data and Databases

**active-active-cache**

[active-active-cache](https://aws-oss.beachgeek.co.uk/3o5) is a repo that helps you build a solution that implements an active-active cache across 2 AWS regions, using ElastiCache for Redis. This solution is automated with CDK and SAM.(#189)

**amazon-bedrock-synthetic-manufacturing-data-generator**

[amazon-bedrock-synthetic-manufacturing-data-generator](https://aws-oss.beachgeek.co.uk/3ln) is a industry aligned synthetic data generation solution. Manufacturing processes generate large amounts of sensor data that can be used for analytics and machine learning models. However, this data may contain sensitive or proprietary information that cannot be shared openly. Synthetic data allows the distribution of realistic example datasets that preserve the statistical properties and relationships in the real data, without exposing confidential information. This enables more open research and benchmarking on representative data. Additionally, synthetic data can augment real datasets to provide more training examples for machine learning algorithms to generalize better. Data augmentation with synthetic manufacturing data can help improve model accuracy and robustness. Overall, synthetic data enables sharing, research, and expanded applications of AI in manufacturing while protecting data privacy and security.(#186)

**automated-data-validation-framework**

[automated-data-validation-framework](https://aws-oss.beachgeek.co.uk/3mi) When you are undertaking data migration projects, a significant time is spent in doing the data validation and lot of manual efforts being spent. This repo provides a framework developed that helps to simplifying this problem by automating full data validation with some simple config files, and running the framework on EMR. It will create summary and detail data validation report in S3 and show up on Athena tables. You will need to do some initial work to setup this framework and create config files which has table names to compare. (#187)

**aws-data-solutions-framework**

[aws-data-solutions-framework](https://github.com/awslabs/aws-data-solutions-framework) is a framework for implementation and delivery of data solutions with built-in AWS best practices. AWS Data Solutions Framework (DSF) is an abstraction atop AWS services based on AWS Cloud Development Kit (CDK) L3 constructs, packaged as a library. You can leverage AWS DSF to implement your data platform in weeks rather than in months. AWS DSF is available in TypeScript and Python. Use the framework to build your data solutions instead of building cloud infrastructure from scratch. Compose data solutions using integrated building blocks via Infrastructure as Code (IaC), that allow you to benefit from smart defaults and built-in AWS best practices. You can also customize or extend according to your requirements. Check out the dedicated documentation page, complete with examples to get you started. (#178)

**glide-for-redis**

[glide-for-redis](https://aws-oss.beachgeek.co.uk/3l2) or General Language Independent Driver for the Enterprise (GLIDE) for Redis (mayeb GLIDER would have been cooler :-) is a new open source client for Redis that works with any Redis distribution that adheres to the Redis Serialization Protocol (RESP) specification. The client is optimised for security, performance, minimal downtime, and observability, and comes pre-configured with best practices learned from over a decade of operating Redis-compatible services used by hundreds of thousands of customers. (#185)

**pinecone-db-construct**

[pinecone-db-construct](https://aws-oss.beachgeek.co.uk/3mn) The Pinecone DB Construct for AWS CDK is a JSII-constructed library that simplifies the creation and management of Pinecone indexes in your AWS infrastructure. It allows you to define, configure, and orchestrate your vector database resources alongside your AWS resources within your CDK application. The maintainer has shared some of the noteworthy features, which include:

* Handles CRUDs for both Pod and Serverless Spec indexes
* Deploy multiple indexes at the same time with isolated state management
* Adheres to AWS-defined removal policies (DESTROY, SNAPSHOT, etc.)
* Creates stack-scoped index names, to avoid name collisions

(#188)

**prometheus-rds-exporter**

[prometheus-rds-exporter](https://aws-oss.beachgeek.co.uk/3mx) is a project from Vincent Mercier that provides a Prometheus exporter for AWS RDS. Check out the README, it is very detailed and well put together. It provides a lot of information on how they built this, examples of configurations as well as detailed configuration options. (#188)

**rds-instances-locator**

[rds-instances-locator](https://aws-oss.beachgeek.co.uk/3o0) There are times when you want to know exactly in which physical AZ your RDS instances are running. For those instances, you can use this script to help you. Details on the various scenarios it has been designed for as well as example commands make this something you can easily get started with.(#189)

**remote-debugging-with-emr**

[remote-debugging-with-emr](https://aws-oss.beachgeek.co.uk/3ne) is a Python-based CDK stack that deploys an EC2 bastion host and EMR Serverless and EMR on EKS environments configured for remote debugging. (#188)

**sso-sync-to-amazon-rds**

[sso-sync-to-amazon-rds](https://aws-oss.beachgeek.co.uk/3e2) This project sets up AWS Lambda functions, Amazon EventBridge rule, Amazon VPC Endpoint for AWS IAM Identity Center (successor to AWS Single Sign-On), the related Security Groups and permissions necessary to automatically provision database users to the Amazon Relational Database Service (Amazon RDS) cluster using AWS Cloud Development Kit (AWS CDK). When a new user is created in IAM Identity Center and the user belongs to the group specified in a IAM_IDC_GROUP_NAME variable, EventBridge rule will trigger the Lambda function. The Lambda function will create a new user in a specified Amazon RDS cluster. The user will then be able to login to the database using their SSO username and IAM credentials. Adding a user to the configured group will trigger the Lambda function as well. (#178)

[tokenizing-db-data-tool](https://aws-oss.beachgeek.co.uk/3lp) provides a handy solution to help you address challenges around masking and keeping safe private or sensitive data (PII), as you need to move data from production to non production systems for activities like testing, debugging, load and integration tests, and more. (#186)

### Developer Tools & DevOps

**aws-iatk**

[aws-iatk](https://aws-oss.beachgeek.co.uk/3fc) AWS Integrated Application Test Kit (IATK), a new open-source test library that makes it easier for developers to create tests for cloud applications with increased speed and accuracy. With AWS IATK, developers can quickly write tests that exercise their code and its AWS integrations against an environment in the cloud, making it easier to catch mistakes early in the development process. IATK includes utilities to generate test events, validate event delivery and structure in Amazon EventBridge Event Bus, and assertions to validate call flow using AWS X-Ray traces. The [AWS IATK](https://aws-oss.beachgeek.co.uk/3g0) is available for Python3.8+. To help you get started, check out the supporting blog post from Dan Fox and Brian Krygsman, [Introducing the AWS Integrated Application Test Kit (IATK)](https://aws-oss.beachgeek.co.uk/3fz). (#180)

**aws-cdk-imagebuilder-sample**

[aws-cdk-imagebuilder-sample](https://aws-oss.beachgeek.co.uk/3o2) this repo uses AWS CDK (TypeScript) that demonstrates how to create a fully functional ImageBuilder pipeline that builds an Amazon Linux 2023 container image, installing git, docker and nodejs, all the way to pushing the resulting image to an ECR repository.(#189)

**aws-cdk-stack-builder-tool**

[aws-cdk-stack-builder-tool](https://aws-oss.beachgeek.co.uk/3g3) or AWS CDK Builder, is a browser-based tool designed to streamline bootstrapping of Infrastructure as Code (IaC) projects using the AWS Cloud Development Kit (CDK). Equipped with a dynamic visual designer and instant TypeScript code generation capabilities, the CDK Builder simplifies the construction and deployment of CDK projects. It stands as a resource for all CDK users, providing a platform to explore a broad array of CDK constructs. Very cool indeed, and you can deploy on AWS Cloud9, so that this project on my weekend to do list. (#180)

**aws-external-package-security**

[aws-external-package-security](https://aws-oss.beachgeek.co.uk/3g2) provides code to setup a solution that demonstrates how you can deploy AWS Code Services (e.g., AWS CodePipeline, AWS CodeBuild, Amazon CodeGuru Security, AWS CodeArtifact) to orchestrate secure access to external package repositories from an Amazon SageMaker data science environment configured with multi-layer security. The solution can also be expanded upon to account for general developer workflows, where developers use external package dependencies. (#180)

**aws-pdk**

[aws-pdk](https://aws-oss.beachgeek.co.uk/3jb) the AWS Project Development Kit (AWS PDK) is an open-source tool to help bootstrap and maintain cloud projects. It provides building blocks for common patterns together with development tools to manage and build your projects. The AWS PDK lets you define your projects programatically via the expressive power of type safe constructs available in one of 3 languages (typescript, python or java). Under the covers, AWS PDK is built on top of Projen. The AWS Bites Podcast provides an overview of the AWS Project Development Kit (PDK), and the hosts discuss what PDK is, how it can help generate boilerplate code and infrastructure, keep configuration consistent across projects, and some pros and cons of using a tool like this versus doing it manually. (#184)

**aws-signer-oci-artifacts**

[aws-signer-oci-artifacts](https://aws-oss.beachgeek.co.uk/3km) this project is used to demonstrate how OCI artefacts can be signed and verified in a development pipeline. Zhuo-Wei Lee, Alontay Ellis, and Rajarshi Das have put together a blog post to help you get started, so if this project interests you, make sure you dive into [Signing and Validating OCI Artifacts with AWS Signer](https://aws-oss.beachgeek.co.uk/3kn).(#185)

**bedrust**

[bedrust](https://aws-oss.beachgeek.co.uk/3n1) is a demo repo from my colleague Darko Mesaros that shows you how you can use Amazon Bedrock in your Rust code, and allows you to currently choose between Claude V2, Llama2 70B, and Cohere Command.(#188)

**bedrock-vscode-playground**

[bedrock-vscode-playground](https://aws-oss.beachgeek.co.uk/3nb) is a Visual Studio Code (VS Code) extension which allows developers to easily explore and experiment with large language models (LLMs) available in Amazon Bedrock. Check out the README for details of what you can do with it and how you can configure it to work with your specific setup.(#188)

**cdk-notifier**

[cdk-notifier](https://aws-oss.beachgeek.co.uk/3it) is a lightweight CLI tool to parse a CDK log file and post changes to pull request requests. Can be used to get more confidence on approving pull requests because reviewer will be aware of changes done to your environments. I am not sure whether this is an old tool, but I have only just found out about it thanks to the blog post from AWS Community Builder, Johannes Konings. He put together [Use cdk-notifier to compare changes in pull requests](https://aws-oss.beachgeek.co.uk/3iu) that explains in more details how this works and walks you through using it. (#183)

**cloudcatalog**

[cloudcatalog](https://aws-oss.beachgeek.co.uk/3mf) colleague David Boyne has put together another project, that is a fork of one his earlier projects ([EventCatalog](https://dev.to/aws/aws-open-source-news-and-updates-96-ig8)) that provides a similar capability, but this time helping you to document your AWS architecture. Check out the README for more details, including an example architecture that was documented. (#187)

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

**llrt**

[llrt](https://aws-oss.beachgeek.co.uk/3mm) - Low Latency Runtime (LLRT) is a lightweight JavaScript runtime designed to address the growing demand for fast and efficient Serverless applications. LLRT offers up to over 10x faster startup and up to 2x overall lower cost compared to other JavaScript runtimes running on AWS Lambda. It's is built in Rust, utilising QuickJS as JavaScript engine, ensuring efficient memory usage and swift startup. (#188)

**localstack-aws-cdk-example**

[localstack-aws-cdk-example](https://aws-oss.beachgeek.co.uk/3dw) This repo aims to showcase the usage of [Localstack](https://aws-oss.beachgeek.co.uk/3dx) and AWS CDK to address specific integration challenges regarding local development where the end target is the AWS platform. If you are unfamiliar with Localstack, it is an open source, fully functional local AWS cloud stack that allows you to develop and test your cloud and Serverless apps offline. (#178)

**pagemosaic-website-starter**

[pagemosaic-website-starter](https://aws-oss.beachgeek.co.uk/3gp) is an open source tool from Alex Pust that helps you to host static websites on AWS, using AWS CDK under the covers from the looks of things. To deploy your website, simply transfer your website files to the /platform/web-app directory. Following this, execute the command pnpm deploy-platform to initiate the deployment process. Nice use of You Tube videos in the README to help you get started. (#181)

**promptus**

[promptus](https://aws-oss.beachgeek.co.uk/3mu) Prompt engineering is key for generating high-quality AI content. But crafting effective prompts can be time-consuming and difficult. That's why I built Promptus. Promptus allows you to easily create, iterate, and organise prompts for generative AI models. With Promptus, you can:

* Quickly build prompts with an intuitive interface
* Automatically version and compare prompt iterations to optimise quality
* Organize prompts into projects and share with teammates
* See a history of your prompt and easily go back to any previous prompt execution

(#188)


**rust-s3-cdn**

[rust-s3-cdn](https://aws-oss.beachgeek.co.uk/3j4) provides a Least Recently Used (LRU) cached proxy for AWS S3 written in Rust. I actually had to look up [LRU](https://helpful.knobs-dials.com/index.php/Cache_and_proxy_notes#Least_Recently_Used_(LRU)) as this was a new term for me. The repo provides a couple of reasons why you might want to use this tool, as well as helping you be aware of caveats too.(#183)

**s3-fast-list**

[s3-fast-list](https://aws-oss.beachgeek.co.uk/3k6) is a rust based tool that concurrently list Amazon S3 bucket with ListObjectsV2 API. Check out the README for use cases as to when s3-fast-list is going to help you out (against existing options you have) (#184)

**stree**

[stree](https://aws-oss.beachgeek.co.uk/3o1) this project from Takafumi Miyanaga is a CLI tool designed to visualize the directory tree structure of an S3 bucket.
By inputting an S3 bucket/prefix and utilizing various flags to customize your request, you can obtain a colorized or non-colorized directory tree right in your terminal. Whether it's for verifying the file structure, sharing the structure with your team, or any other purpose, stree offers an easy and convenient way to explore your S3 buckets. (#189)

**tokenizing-db-data-tool**

[tokenizing-db-data-tool](https://aws-oss.beachgeek.co.uk/3lp) provides a handy solution to help you address challenges around masking and keeping safe private or sensitive data (PII), as you need to move data from production to non production systems for activities like testing, debugging, load and integration tests, and more. (#186)

**vscode-on-ec2-for-prototyping**

[vscode-on-ec2-for-prototyping](https://aws-oss.beachgeek.co.uk/3lo) This repository introduces how to access and use VSCode hosted on EC2 from a browser. The connection is made via Session Manager, so IAM permissions are used for authentication. The access destination will be localhost. Please note that this repository does not introduce connecting from your local VSCode to an EC2 instance via Remote SSH. (#186)

### Governance & Risk

**appfabric-data-analytics**

[appfabric-data-analytics](https://aws-oss.beachgeek.co.uk/3k3) this project enables you to maintain logs from various SaaS applications and provides the ability to search and display log data. This solution leverages AWS AppFabric to create a data repository that you can query with Amazon Athena. While customers can obtain normalized and enriched SaaS audit log data (OCSF) from AppFabric, many prefer not only to forward these logs to a security tool. Some have the requirement to preserve logs for post-incident analysis, while others need to utilize the logs for tracking SaaS subscription and license usage. Additionally, some customers aim to analyze user activity to discover patterns. This project establishes a data pipeline that empowers customers to construct dashboards on top of it. (#184)

**aws-account-tag-association-imported-portfolios**

[aws-account-tag-association-imported-portfolios](https://aws-oss.beachgeek.co.uk/3o4) This repo provides a solution that is designed to automate associating account level tags to shared and local portfolios in the AWS environment which in turn inherits the tags to launched resources. AWS ServiceCatalog TagOption feature is used for this association.(#189)

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

**cloudcatalog**

[cloudcatalog](https://aws-oss.beachgeek.co.uk/3mf) colleague David Boyne has put together another project, that is a fork of one his earlier projects ([EventCatalog](https://dev.to/aws/aws-open-source-news-and-updates-96-ig8)) that provides a similar capability, but this time helping you to document your AWS architecture. Check out the README for more details, including an example architecture that was documented. (#187)

**s3-restore-and-copy-progress-monitoring**

[s3-restore-and-copy-progress-monitoring](https://aws-oss.beachgeek.co.uk/3o3) this is a very comprehensive and polished repo that provides an example of how you can restore data that you have stored in S3, providing you a single visualised dashboard to monitor the restore and copy progress within a defined scope.(#189)


### Java, Kotlin, Scala, OpenJDK

**java-on-aws**

[java-on-aws](https://aws-oss.beachgeek.co.uk/3lx) is a fantastic resource for all Java developers who want to dive deeper on how to deploy their Java applications on AWS. Taking a sample application, the workshop looks at how you can containerise it, and then deploy it across a number of different compute environments - from serverless to Kubernetes.(#187) 

**powertools-lambda-kotlin**

[powertools-lambda-kotlin](https://aws-oss.beachgeek.co.uk/3dv) This project demonstrates the Lambda for Powertools Kotlin module deployed using Serverless Application Model with Gradle running the build. This example is configured for Java 8 only; in order to use a newer version, check out the Gradle configuration guide in the main project README. You can also use sam init to create a new Gradle-powered Powertools application - choose to use the AWS Quick Start Templates, and then Hello World Example with Powertools for AWS Lambda, Java 17 runtime, and finally gradle. (#178)

**serverless-java-container**

[serverless-java-container](https://aws-oss.beachgeek.co.uk/3my) this repo provides a Java wrapper to run Spring, Spring Boot, Jersey, and other apps inside AWS Lambda. Serverless Java Container natively supports API Gateway's proxy integration models for requests and responses, you can create and inject custom models for methods that use custom mappings. Check out the supporting blog post from Dennis Kieselhorst, [Re-platforming Java applications using the updated AWS Serverless Java Container](https://aws-oss.beachgeek.co.uk/3mz). (#188)

### Networking

**eks-shared-subnets**

[eks-shared-subnets](https://aws-oss.beachgeek.co.uk/3k2) this sample code demonstrates how a central networking team in an enterprise can create and share the VPC Subnets from their own AWS Account with other Workload Specific accounts. So that, Application teams can deploy and manage their own EKS clusters and related resources in those Subnets. (#184)

**route53-hostedzone-migrator**

[route53-hostedzone-migrator](https://aws-oss.beachgeek.co.uk/3nc) is a handy script will help you to automate the migration of an AWS Route 53 hosted zone from an AWS account to another one. It will follow all the needed steps published in the official AWS Route 53 documentation regarding the migration of a hosted zone.(#188)

**rust-s3-cdn**

[rust-s3-cdn](https://aws-oss.beachgeek.co.uk/3j4) provides a Least Recently Used (LRU) cached proxy for AWS S3 written in Rust. I actually had to look up [LRU](https://helpful.knobs-dials.com/index.php/Cache_and_proxy_notes#Least_Recently_Used_(LRU)) as this was a new term for me. The repo provides a couple of reasons why you might want to use this tool, as well as helping you be aware of caveats too.

**trading-latency-benchmark**

[trading-latency-benchmark](https://aws-oss.beachgeek.co.uk/3d1) This repository contains a network latency test stack that consists of Java based trading client and Ansible playbooks to coordinate distributed tests. Java based trading client is designed to send limit and cancel orders, allowing you to measure round-trip times of the network communication. (#177)

### Observability

**observability-solution-kit**

[observability-solution-kit](https://aws-oss.beachgeek.co.uk/3d3) this repository is the Ollyv sdk. The Ollyv sdk offers a simple way to configure its behaviour through *.properties files, which are environment-specific. Currently code is provide from NodeLambda ✨ · NodeExpress 👟 · JavaSpring 🦚 (#177)

**prometheus-rds-exporter**

[prometheus-rds-exporter](https://aws-oss.beachgeek.co.uk/3mx) is a project from Vincent Mercier that provides a Prometheus exporter for AWS RDS. Check out the README, it is very detailed and well put together. It provides a lot of information on how they built this, examples of configurations as well as detailed configuration options. (#188)

**s3-restore-and-copy-progress-monitoring**

[s3-restore-and-copy-progress-monitoring](https://aws-oss.beachgeek.co.uk/3o3) this is a very comprehensive and polished repo that provides an example of how you can restore data that you have stored in S3, providing you a single visualised dashboard to monitor the restore and copy progress within a defined scope.(#189)

### Security

**avp-toy-store-sample**

[avp-toy-store-sample](https://aws-oss.beachgeek.co.uk/3jw) is a great sample project if you want to explore Cedar, and how this fits in with Amazon Verified Permissions. This sample web application demonstrates authentication and policy-based authorization for different user types to an imaginary toy store. The toy store takes orders online and then send them to customers through multiple warehouses. This application is used by warehouses to help sending orders to customers. The application uses Amazon Cognito for authentication and uses Amazon Verified Permissions for policy-based authorization. Additionally, the application uses API-Gateway as the front-door to the application, and Lambda to process requests. (#184)

**aws-waf-for-event-analysis-dashboard**

[aws-waf-for-event-analysis-dashboard](https://aws-oss.beachgeek.co.uk/3ls) finding the information you need during security incidents is what this project aims to help with. During major online events like live broadcasts, security teams need a fast and clear understanding of attack patterns and behaviours to distinguish between normal and malicious traffic flows. The solution outlined here allows filtering flow logs by "Client IP", "URI", "Header name", and "Header value" to analyse these fields and pinpoint values specifically associated with attack traffic versus normal traffic. For example, the dashboard can identify the top header values that are atypical for normal usage. The security team can then create an AWS WAF rule to block requests containing these header values, stopping the attack. This project demonstrates using AWS Glue crawlers to categorise and structure WAF flow log data and Amazon Athena for querying. Amazon Quicksight is then employed to visualise query results in a dashboard. Once deployed, the dashboard provides traffic visualisation similar to the example graphs shown in Images folder in under project , empowering security teams with insight into attacks and defence.(#186)

**awskillswitch**

[awskillswitch](https://aws-oss.beachgeek.co.uk/3gr) is an open sourced tool from Jeffrey Lyon that is worth checking out. AWS Kill Switch is a Lambda function (and proof of concept client) that an organisation can implement in a dedicated "Security" account to give their security engineers the ability to delete IAM roles or apply a highly restrictive service control policy (SCP) on any account in their organisation. Make sure you check out the README for full details, but this looks like it might be one of those tools that are useful to have in the back pocket in times of need. (#181)

**cedar-antlr-grammar**

[cedar-antlr-grammar](https://aws-oss.beachgeek.co.uk/3n0) - ANTLR (ANother Tool for Language Recognition) is a powerful parser generator for reading, processing, executing, or translating structured text or binary files. It's widely used to build languages, tools, and frameworks. From a grammar, ANTLR generates a parser that can build and walk parse trees. AWS Hero Ian Mckay has created one for Cedar. (#188)

**s3-presignedurl-staticips-endpoint-with-cdk**

[s3-presignedurl-staticips-endpoint-with-cdk](https://aws-oss.beachgeek.co.uk/3k5) this solution simplifies access to Amazon S3 by creating secure, custom presigned URLs for object downloads through a single endpoint with a unique domain and static IPs. The use case involves users following an IP and domain Allowlist firewall policy, limiting API access to specific domains and IPs. The architecture employs key AWS services, including AWS Global Accelerator, Amazon API Gateway, AWS Lambda, Application Load Balancer(ALB), VPC Endpoint, and Amazon S3. This design centralizes the API for generating presigned URLs and the S3 endpoint under a single domain, linked to an AWS Global Accelerator with two static IPs. Consequently, users can effortlessly request presigned URLs and download S3 objects through a unified domain endpoint with static IPs. This architecture is especially beneficial for customers with strict policies or compliance requirements, such as those in the public, medical, and finance sectors. (#184)

**sso-sync-to-amazon-rds**

[sso-sync-to-amazon-rds](https://aws-oss.beachgeek.co.uk/3e2) This project sets up AWS Lambda functions, Amazon EventBridge rule, Amazon VPC Endpoint for AWS IAM Identity Center (successor to AWS Single Sign-On), the related Security Groups and permissions necessary to automatically provision database users to the Amazon Relational Database Service (Amazon RDS) cluster using AWS Cloud Development Kit (AWS CDK). When a new user is created in IAM Identity Center and the user belongs to the group specified in a IAM_IDC_GROUP_NAME variable, EventBridge rule will trigger the Lambda function. The Lambda function will create a new user in a specified Amazon RDS cluster. The user will then be able to login to the database using their SSO username and IAM credentials. Adding a user to the configured group will trigger the Lambda function as well. (#178)

### Storage

**amazon-s3-glacier-archive-data-delete**

[amazon-s3-glacier-archive-data-delete](https://aws-oss.beachgeek.co.uk/3lr) Amazon S3 Glacier Archive (data) Delete solution provides an automated workflow to delete ALL of your data in an S3 Glacier Vault. This solution only applies to Amazon S3 Glacier Vault Archives. Within S3 Glacier, data is stored as an Archive within a Vault. This solution does not apply to objects in Glacier Deep Archive, Glacier Flexible Retrieval, and Glacier Instant Retrieval stored in an Amazon S3 Bucket. Good README with clear guidance and overview of how this works.(#186)

**ebs-bootstrap**

[ebs-bootstrap](https://aws-oss.beachgeek.co.uk/3mg) is a very handy tool from Lasith Koswatta Gamage that solves a very specific problem. Lasith reached out to explain more about the "itch" that needed to be scratched. ebs-bootstrap is a tool that provides a safe and as-code approach for managing block devices on AWS EC2. If you need precise and consistent control over your EBS volumes when attaching them to your EC2 Nitro based instances, you need to check out this project. The README provides some additional example configurations, and there is a blog post in the works which I will share once it has been published. (#187)

**s3-fast-list**

[s3-fast-list](https://aws-oss.beachgeek.co.uk/3k6) is a rust based tool that concurrently list Amazon S3 bucket with ListObjectsV2 API. Check out the README for use cases as to when s3-fast-list is going to help you out (against existing options you have) (#184)

**s3-presignedurl-staticips-endpoint-with-cdk**

[s3-presignedurl-staticips-endpoint-with-cdk](https://aws-oss.beachgeek.co.uk/3k5) this solution simplifies access to Amazon S3 by creating secure, custom presigned URLs for object downloads through a single endpoint with a unique domain and static IPs. The use case involves users following an IP and domain Allowlist firewall policy, limiting API access to specific domains and IPs. The architecture employs key AWS services, including AWS Global Accelerator, Amazon API Gateway, AWS Lambda, Application Load Balancer(ALB), VPC Endpoint, and Amazon S3. This design centralizes the API for generating presigned URLs and the S3 endpoint under a single domain, linked to an AWS Global Accelerator with two static IPs. Consequently, users can effortlessly request presigned URLs and download S3 objects through a unified domain endpoint with static IPs. This architecture is especially beneficial for customers with strict policies or compliance requirements, such as those in the public, medical, and finance sectors. (#184)

**s3-restore-and-copy-progress-monitoring**

[s3-restore-and-copy-progress-monitoring](https://aws-oss.beachgeek.co.uk/3o3) this is a very comprehensive and polished repo that provides an example of how you can restore data that you have stored in S3, providing you a single visualised dashboard to monitor the restore and copy progress within a defined scope.(#189)

**s3-small-object-compaction**

[s3-small-object-compaction](https://aws-oss.beachgeek.co.uk/3mk) This solution deploys a serverless application to combine ("compact") small objects stored in a given Amazon S3 prefix into a single larger file. Larger files enable cost effective use of S3 storage tiers that have a minimum billable object size (e.g. 128 KB). It can also improve performance when querying data directly with Amazon Athena. The sample code is written using the AWS Cloud Development Kit in Python.(#187)

**stree**

[stree](https://aws-oss.beachgeek.co.uk/3o1) this project from Takafumi Miyanaga is a CLI tool designed to visualize the directory tree structure of an S3 bucket.
By inputting an S3 bucket/prefix and utilizing various flags to customize your request, you can obtain a colorized or non-colorized directory tree right in your terminal. Whether it's for verifying the file structure, sharing the structure with your team, or any other purpose, stree offers an easy and convenient way to explore your S3 buckets. (#189)

# AWS Services

**appfabric-data-analytics**

[appfabric-data-analytics](https://aws-oss.beachgeek.co.uk/3k3) this project enables you to maintain logs from various SaaS applications and provides the ability to search and display log data. This solution leverages AWS AppFabric to create a data repository that you can query with Amazon Athena. While customers can obtain normalized and enriched SaaS audit log data (OCSF) from AppFabric, many prefer not only to forward these logs to a security tool. Some have the requirement to preserve logs for post-incident analysis, while others need to utilize the logs for tracking SaaS subscription and license usage. Additionally, some customers aim to analyze user activity to discover patterns. This project establishes a data pipeline that empowers customers to construct dashboards on top of it. (#184)

**amazon-chime-sdk-voice-voice-translator**

[amazon-chime-sdk-voice-voice-translator](https://aws-oss.beachgeek.co.uk/3k4) this project leverages the Amazon Chime SDK to create a voice to voice live translator. It facilitates real time translation in voice calls enabling seamless communication between participants speaking different languages. The system integrates various AWS services, including Amazon Chime SDK, Amazon Kinesis Video Streams (KVS), Amazon Transcribe, Amazon Translate, Amazon Polly, etc. to achieve efficient and accurate translation. (#184)

**automated-meeting-scribe-and-summarizer**

[automated-meeting-scribe-and-summarizer](https://aws-oss.beachgeek.co.uk/3nz) Using this application's website, you can invite an AI-assisted scribe bot to your upcoming Amazon Chime meeting(s) to get a follow-up email with the attendee list, chat history, attachments, and transcript, as well as a summary and action items. You don't even need to be present in a meeting for your invited scribe bot to join. Each scribe bot is linked to your email for identification. The scribe bot also redacts sensitive personally identifiable information (PII) by default, with the option to redact additional PII. (#189)

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

**ebs-bootstrap**

[ebs-bootstrap](https://aws-oss.beachgeek.co.uk/3mg) is a very handy tool from Lasith Koswatta Gamage that solves a very specific problem. Lasith reached out to explain more about the "itch" that needed to be scratched. ebs-bootstrap is a tool that provides a safe and as-code approach for managing block devices on AWS EC2. If you need precise and consistent control over your EBS volumes when attaching them to your EC2 Nitro based instances, you need to check out this project. The README provides some additional example configurations, and there is a blog post in the works which I will share once it has been published. (#187)

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









