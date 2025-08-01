# AWS open source newsletter projects

This repo contains a list of projects featured in the AWS open source newsletter. The (#) identifies which edition of the newsletter you can find the original mention.

If you want your projects featured, please get in touch with ricsue at amazon dot com.

# By technology/use case

### AI & ML

**abc**

[abc](https://aws-oss.beachgeek.co.uk/47t) is an AI bash tool that integrates with Amazon Bedrock as a foundation model provider. This tool will generate shell command(s) from natural language description using which ever LLM (generative AI) you configure, and place on the next shell prompt for editing and execution. Plenty of details in the README on how to configure and run this, so give it a go if you are looking for something that integrates with Amazon Bedrock.(#206)

**agent-dev-toolkit**

[agent-dev-toolkit](https://aws-oss.beachgeek.co.uk/4gx) or Agent Development Toolkit (ADT),  provides a single command-line interface that lets developers build, test, and iterate on Strands agents with ease during local development. The CLI unifies agent execution, observability, UI, and local containerisation into one cohesive developer experience, enabling developers to iterate and experiment faster with immediate feedback loops and streamlined workflows. Check out the detailed README file for more info on how to get started with this command line tool.(#212)

**agent-evaluation**

[agent-evaluation](https://aws-oss.beachgeek.co.uk/3vw)  is a generative AI-powered framework for testing virtual agents. Agent Evaluation implements an LLM agent (evaluator) that will orchestrate conversations with your own agent (target) and evaluate the responses during the conversation. The repo has links to detailed docs that provide example configurations and a reference guide to get you started. (#197)

**ai-on-eks**

[ai-on-eks](https://aws-oss.beachgeek.co.uk/4bw) is your go to repo if you are interested in getting AI up and running on Amazon EKS. It provides a gateway to help you scale AI and ML workloads on Amazon EKS. It provides a rich collection of Terraform Blueprints featuring current good practices for deploying robust solutions with advanced logging and observability. You can explore practical patterns for running AI/ML workloads on EKS, leveraging the power of the Ray ecosystem for distributed computing. Utilise advanced serving solutions like NVIDIA Triton Server, vLLM for efficient and scalable model inference, and TensorRT-LLM for optimising deep learning models. (#209)


**alarm-context-tool**

[alarm-context-tool](https://aws-oss.beachgeek.co.uk/40g) enhances AWS CloudWatch Alarms by providing additional context to aid in troubleshooting and analysis. By leveraging AWS services such as Lambda, CloudWatch, X-Ray, and Amazon Bedrock, this solution aggregates and analyses metrics, logs, and traces to generate meaningful insights. Using generative AI capabilities from Amazon Bedrock, it summarises findings, identifies potential root causes, and offers relevant documentation links to help operators resolve issues more efficiently. The implementation is designed for easy deployment and integration into existing observability pipelines, significantly reducing response times and improving root cause analysis. (#201)

**amazon-bedrock-client-for-mac**

[amazon-bedrock-client-for-mac](https://aws-oss.beachgeek.co.uk/3um) this repo provides the code for the Amazon Bedrock Client for Mac is a macOS demo application built with SwiftUI. It serves as a client interface for AWS Bedrock, allowing users to interact with AWS Bedrock models.(#196)

**amazon-bedrock-serverless-prompt-chaining**

[amazon-bedrock-serverless-prompt-chaining](https://aws-oss.beachgeek.co.uk/3jy) this repository provides examples of using AWS Step Functions and Amazon Bedrock to build complex, serverless, and highly scalable generative AI applications with prompt chaining. (#184)

**amazon-q-developer-cli-webui**

[amazon-q-developer-cli-webui](https://aws-oss.beachgeek.co.uk/4f5) is a very nice project from **AWS Community Builder Gabriel Koo** that allows you to run Amazon Q CLI via a web ui. He put together a post that explains the itch he was trying to scratch, and I have to say that this is on my weekend list of projects to try out. Go read the post, [Why I Built a Web UI for Amazon Q Developer CLI (And How I Vibe-Coded It)](https://dev.to/aws-builders/why-i-built-a-web-ui-for-amazon-q-developer-cli-and-how-i-vibe-coded-it-54d6).(#211)

**amazon-sagemaker-pipeline-deploy-manage-100x-models-python-cdk**

[amazon-sagemaker-pipeline-deploy-manage-100x-models-python-cdk](https://aws-oss.beachgeek.co.uk/3lu) This GitHub repository showcases the implementation of a comprehensive end-to-end MLOps pipeline using Amazon SageMaker pipelines to deploy and manage 100x machine learning models. The pipeline covers data pre-processing, model training/re-training, hyper-parameter tuning, data quality check, model quality check, model registry, and model deployment. Automation of the MLOps pipeline is achieved through Continuous Integration and Continuous Deployment (CI/CD). Machine learning model for this sample code is SageMaker built-in XGBoost algorithm.(#186)

**amazon-q-vibes-memory-banking**

[amazon-q-vibes-memory-banking](https://aws-oss.beachgeek.co.uk/4fd) is from **AWS Serverless Community Builder Nicola Cremaschini** who shares his approach to using AI Coding Assistants like Amazon Q Developer, to provide more consistent outcomes. The Q-Vibes framework helps maintain context across AI assistant sessions through 5 lightweight files, enabling quick prototype development without losing momentum between sessions. (#211)

**awesome-codewhisperer**

[awesome-codewhisperer](https://aws-oss.beachgeek.co.uk/3cw) this repo from Christian Bonzelet is a great collection of resources for those of you who are experimenting with Generative AI coding assistants such as Amazon CodeWhisperer. This resource should keep you busy, and help you master Amazon CodeWhisperer in no time.  (#177)

**aws-advanced-nodejs-wrapper**

[aws-advanced-nodejs-wrapper](https://aws-oss.beachgeek.co.uk/465) is complementary to an existing NodeJS driver and aims to extend the functionality of the driver to enable applications to take full advantage of the features of clustered databases such as Amazon Aurora.  In other words, the AWS Advanced NodeJS Wrapper does not connect directly to any database, but enables support of AWS and Aurora functionalities on top of an underlying NodeJS driver of the user's choice. Hosting a database cluster in the cloud via Aurora is able to provide users with sets of features and configurations to obtain maximum performance and availability, such as database failover. However, at the moment, most existing drivers do not currently support those functionalities or are not able to entirely take advantage of it. The main idea behind the AWS Advanced NodeJS Wrapper is to add a software layer on top of an existing NodeJS driver that would enable all the enhancements brought by Aurora, without requiring users to change their workflow with their databases and existing NodeJS drivers. (#205)

**aws-chatbot-fargate-python**

[aws-chatbot-fargate-python](https://aws-oss.beachgeek.co.uk/43f) is a new repo from AWS Hero Ran Isenberg that deploys a Streamlit Chatbot in an AWS Fargate-based ESC cluster web application using AWS CDK (Cloud Development Kit). The infrastructure includes an ECS cluster, Fargate service, Application Load Balancer, VPC, and WAF and includes security best practices with CDK-nag verification. The chatbot is based on an implementation by Streamlit and the initial prompt is that the chatbot is me, Ran the builder, a serverless hero and attempts to answer as me. The Chatbot uses custom domain (you can remove it or change it to your own domain) and assume an OpenAI token exists in the account in the form of a secrets manager secret for making API calls to OpenAI.(#203)

**aws-cost-explorer-mcp-server**

[aws-cost-explorer-mcp-server](https://aws-oss.beachgeek.co.uk/4ac) builds on the current wave of excitement around Model Context Protocol (MCP) that provides a powerful open-source tool that brings natural language querying to your AWS spending data. Ask questions like "What was my EC2 spend yesterday?" or "Break down my Bedrock usage by model" and get detailed answers through Claude. It helps track cloud costs across services, identifies expensive resources, and provides deep insights into your Amazon Bedrock model usage. (#208)

**aws-genai-rfpassistant**

[aws-genai-rfpassistant](https://aws-oss.beachgeek.co.uk/43a) this repository contains the code and infrastructure as code for a Generative AI-powered Request for Proposal (RFP) Assistant leveraging Amazon Bedrock and AWS Cloud Development Kit (CDK). This could be very hand if responding to RFP's is something that you do and you want to look at ways of optimising your approach. The documentation in the repo is very comprehensive. I have not tried this one out for myself, but I have been involved in both writing and reviewing RFPs in the past, so understand the pain that led to the creation of this project.(#203)

**aws-inference-benchmark**
[aws-inference-benchmark ](https://aws-oss.beachgeek.co.uk/3cy)this project from Rustem Feyzkhanov contains code for running deep learning inference benchmarks on different AWS instances and service types. Check out his post, [Making LLMs Scalable: Cloud Inference with AWS Fargate and Copilot](https://aws-oss.beachgeek.co.uk/3d0) where Rustem shows you in more details how you can use this repo. (#177)

**aws-mcp**

[aws-mcp](https://aws-oss.beachgeek.co.uk/4ag) is a project from Rafal Wilinski that enables AI assistants like Claude to interact with your AWS environment through MCP. This allows for natural language querying and management of your AWS resources during conversations. 

![example prompt using aws-mcp to find out about costs](https://github.com/RafalWilinski/aws-mcp/blob/main/images/aws-mcp-demo.png?raw=true) (#208)


**aws-piday2024**

[aws-piday2024 ](https://aws-oss.beachgeek.co.uk/3r3)my colleague Suman Debnath has put together this AWS Pi Day 2024 repository, where you can explore various applications and examples using Amazon Bedrock, fine-tuning, and Retrieval-Augmented Generation (RAG). (#193)

**aws-serverless-mcp-server**

[aws-serverless-mcp-server](https://aws-oss.beachgeek.co.uk/4df) enhances the serverless development experience by providing AI coding assistants with comprehensive knowledge of serverless patterns, best practices, and AWS services. Using AWS Serverless Application Model Command Line Interface (AWS SAM CLI) integration, you can handle events and deploy infrastructure while implementing proven architectural patterns. This integration streamlines function lifecycles, service integrations, and operational requirements throughout your application development process. The server also provides contextual guidance for infrastructure as code decisions, AWS Lambda specific best practices, and event schemas for AWS Lambda event source mappings. (#210)

**bedrock-access-gateway**

[bedrock-access-gateway](https://aws-oss.beachgeek.co.uk/3us) provides an OpenAI-compatible RESTful APIs for Amazon Bedrock. Amazon Bedrock offers a wide range of foundation models (such as Claude 3 Opus/Sonnet/Haiku, Llama 2/3, Mistral/Mixtral, etc.) and a broad set of capabilities for you to build generative AI applications. Check the Amazon Bedrock landing page for additional information. Sometimes, you might have applications developed using OpenAI APIs or SDKs, and you want to experiment with Amazon Bedrock without modifying your codebase. Or you may simply wish to evaluate the capabilities of these foundation models in tools like AutoGen etc. Well, this repository allows you to access Amazon Bedrock models seamlessly through OpenAI APIs and SDKs, enabling you to test these models without code changes.(#196)

**bedrock-agentcore-sdk-python**

[bedrock-agentcore-sdk-python](https://aws-oss.beachgeek.co.uk/4gt) is the public preview of the Bedrock AgentCore SDK. Amazon Bedrock AgentCore, a comprehensive set of enterprise-grade services that help developers quickly and securely deploy and operate AI agents at scale using any framework and model, hosted on Amazon Bedrock or elsewhere. Check out my colleague Danilo's blog post to find out more, [Introducing Amazon Bedrock AgentCore: Securely deploy and operate AI agents at any scale (preview)](https://aws-oss.beachgeek.co.uk/4gu) (#212)

**bedrock-agentcore-starter-toolkit**

[bedrock-agentcore-starter-toolkit](https://aws-oss.beachgeek.co.uk/4h3) provides a CLI toolkit for deploying AI agents to Amazon Bedrock AgentCore. Zero infrastructure management with built-in gateway and memory integrations. This is currently in public preview, so subject to breaking changes - so bear that in mind when you try this out. (#212)



**bedrock-embed-web**

[bedrock-embed-web](https://aws-oss.beachgeek.co.uk/43i) is a new project from my colleague Olivier Leplus that makes it easier than ever to embed Amazon Bedrock foundation models within a chat interface in your web applications. (#203)

**bedrock-engineer**

[bedrock-engineer](https://aws-oss.beachgeek.co.uk/497) looks like an amazing project and one I am trying to find some time to play with. bedrock-engineer is Autonomous software development agent apps using Amazon Bedrock, capable of customise to create/edit files, execute commands, search the web, use knowledge base, use multi-agents, generative images and more. The project README has a short video that goes over some of the functionality and its pretty neat. (#207)

**bedrock-genai-workshop**

[bedrock-genai-workshop](https://aws-oss.beachgeek.co.uk/3lt) if you are looking to get hands on with generative AI, then check out this workshop that is aimed at developers and solution builders, introduces how to leverage foundation models (FMs) through Amazon Bedrock. Amazon Bedrock is a fully managed service that provides access to FMs from third-party providers and Amazon; available via an API. With Bedrock, you can choose from a variety of models to find the one that’s best suited for your use case. Within this series of labs, you'll explore some of the most common usage patterns, and Labs include: 1/ Text Generation, 2/ Text Summarization, 3/ Questions Answering, 4/ Chatbot , and 5/ Agent (#186)

**bedrock-litellm**

[bedrock-litellm](https://aws-oss.beachgeek.co.uk/43m) is an awesome project that provides a way of proxying requests in the OpenAI format, so that they will work with Amazon Bedrock. OpenAI is often one of the default options for integrating various generative AI tools and libraries, and now you have a way of being able to point those to use foundational models managed by Amazon Bedrock. It uses [litellm](https://www.litellm.ai/) to do this, and is deployed in a Kubernetes cluster.(#203)

**bedrock-multi-tenant-saas**

[bedrock-multi-tenant-saas](https://aws-oss.beachgeek.co.uk/3jx) In this repository, we show you how to build an internal SaaS service to access foundation models with Amazon Bedrock in a multi-tenant architecture. An internal software as a service (SaaS) for foundation models can address governance requirements while providing a simple and consistent interface for the end users. (#184)

**bedrock-vscode-playground**

[bedrock-vscode-playground](https://aws-oss.beachgeek.co.uk/3nb) is a Visual Studio Code (VS Code) extension which allows developers to easily explore and experiment with large language models (LLMs) available in Amazon Bedrock. Check out the README for details of what you can do with it and how you can configure it to work with your specific setup.(#188)

**bedrust**

[bedrust](https://aws-oss.beachgeek.co.uk/3n1) is a demo repo from my colleague Darko Mesaros that shows you how you can use Amazon Bedrock in your Rust code, and allows you to currently choose between Claude V2, Llama2 70B, and Cohere Command.(#188)

**build-an-agentic-llm-assistant**

[build-an-agentic-llm-assistant](https://aws-oss.beachgeek.co.uk/43e) this repo provides code that you can follow along as part of the "Build an agentic LLM assistant on AWS" workshop. This hands-on workshop, aimed at developers and solution builders, trains you on how to build a real-life serverless LLM application using foundation models (FMs) through Amazon Bedrock and advanced design patterns such as: Reason and Act (ReAct) Agent, text-to-SQL, and Retrieval Augemented Generation (RAG). It complements the Amazon Bedrock Workshop by helping you transition from practicing standalone design patterns in notebooks to building an end-to-end llm serverless application. Check out the README for additional links to the workshop text, as well as more details on how this repo works.(#203)

**building-reactjs-gen-ai-apps-with-amazon-bedrock-javascript-sdk**

[building-reactjs-gen-ai-apps-with-amazon-bedrock-javascript-sdk](https://aws-oss.beachgeek.co.uk/3op) provides a sample application that integrates the power of generative AI with a call to the Amazon Bedrock API from a web application such SPA built with JavaScript and react framework. The sample application uses  Amazon Cognito credentials and IAM Roles to invoke Amazon Bedrock API in a react-based application with JavaScript and the CloudScape design system. You will deploy all the resources and host the app using AWS Amplify. Nice detailed README, so what are you waiting for, go check this out. (#190)

**cfn-bedrock-notify**

[cfn-bedrock-notify](https://aws-oss.beachgeek.co.uk/3sc) is a new tool from my colleague Du'an Lightfoot that is a very creative an interesting way to incorporate large language models to help you troubleshoot failed Cloudformation deployments. How many times have you had a deployment fail, only to reach out to your preferred search tool to help solve the problem. Well with this project deployed, any issues are sent via SNS to Amazon Bedrock using Anthropic Claude v3, and (hopefully) the answer to your problems are returned via the email of the SNS subscriber. (#194)

**chat-cli**

[chat-cli](https://aws-oss.beachgeek.co.uk/47o) is a terminal based program that lets you interact with LLMs available on Amazon Bedrock. You can install and run via source or for ease, Homebrew. The README provides you with more details including how to configure the application. If you want an easy, text base way to interact with the foundational models you run via Amazon Bedrock, check this out.(#206)

**chorus**

[chorus](https://aws-oss.beachgeek.co.uk/4aq) is a new easy-to-use framework for building scalable solutions with LLM-driven multi-agent collaboration. Chorus allows you to develop and test solutions using multi-agent collaboration with zero or minimal coding. Chorus provides a multi-agent playground for easy visualisation and testing. Finally, Chorus helps you deploy your solution to Amazon Bedrock Agents with a single command (coming soon).(#208)

**chronos-forecasting**

[chronos-forecasting](https://aws-oss.beachgeek.co.uk/3rh) is a family of pre-trained time series forecasting models based on language model architectures. A time series is transformed into a sequence of tokens via scaling and quantisation, and a language model is trained on these tokens using the cross-entropy loss. Once trained, probabilistic forecasts are obtained by sampling multiple future trajectories given the historical context. Chronos models have been trained on a large corpus of publicly available time series data, as well as synthetic data generated using Gaussian processes. (#194)

**cloudysetup**

[cloudysetup](https://aws-oss.beachgeek.co.uk/40e) is a CLI tool designed to streamline AWS resource management using AWS Cloud Control API. It leverages Amazon Bedrock fully managed service with Anthropic - Claude V2 Gen AI model to create, read, update, list, and delete AWS resources by generating configurations compatible with AWS Cloud Control API.(#201)

**content-based-item-recommender**

[content-based-item-recommender](https://aws-oss.beachgeek.co.uk/44r) provides some example code the helps you deploy a content-based recommender system. It is called "content-based" as it bases the recommendation based on the matching between the input's content and the items' content from your database. This uses prompt to large-language models (LLM) and vector search to perform the recommendation. (#204)

**deploy-crewai-agents-terraform**

[deploy-crewai-agents-terraform](https://aws-oss.beachgeek.co.uk/4d6) is a project designed to help you perform security audits and generate reports for your AWS infrastructure using a multi-agent AI system, leveraging the powerful and flexible framework provided by CrewAI. The AWS Security Auditor Crew architecture combines CrewAI's multi-agent framework with AWS services to provide comprehensive security auditing capabilities. The system can be deployed locally or to AWS using Terraform, with Amazon Bedrock powering the AI agents. (#210)

**distill-cli**

[distill-cli](https://aws-oss.beachgeek.co.uk/3yz) is a new project from Amazon CTO Dr Werner Vogels, which uses Amazon Transcribe and Amazon Bedrock to create summaries of your audio recordings (e.g., meetings, podcasts, etc.) directly from the command line. Distill CLI takes a dependency on Amazon Transcribe, and as such, supports the following media formats: AMR, FLAC, M4A, MP3, MP4, Ogg, WebM, WAV. It is great to feature this latest project, with the previous one being featured in [#197](https://community.aws/content/2gPNtsdSfQRIpmbUrNyPrjUg54D/aws-open-source-newsletter-197). To go with this repo, there is a post too, [Introducing Distill CLI: An efficient, Rust-powered tool for media summarization](https://aws-oss.beachgeek.co.uk/3yy) where Werner shares his experience building this tool in Rust, and provides some closing thoughts too. (#200)

**draw-an-app**

[draw-an-app](https://aws-oss.beachgeek.co.uk/49c) is a neat demo application that uses AWS Bedrock's Claude 3 Sonnet model to convert hand-drawn UI sketches into working web applications. It supports two modes of operation: 1/ Real-time webcam capture for immediate sketching and conversion, or 2/ Image upload through a user-friendly Streamlit interface. I have not tried this one yet, but if you do, let me know how you get on.(#207)

**ecs-gpu-scaling**

[ecs-gpu-scaling](https://aws-oss.beachgeek.co.uk/3mh) This repository is intended for engineers looking to horizontally scale GPU-based Machine Learning (ML) workloads on Amazon ECS. By default, GPU utilisation metrics are not part of the predefined metrics available with Application Autoscaling. As such, you implement auto scaling based on custom metrics.  For NVIDIA-based GPUs, you use DCGM-Exporter in your container to expose GPU metrics. You can then use metrics such as DCGM_FI_DEV_GPU_UTIL and DCGM_FI_DEV_GPU_TEMP to determine your auto scaling behaviour. The README provides links to all the additional resources you need to get this up and running.(#187)

**ecs-mcp-server**

[ecs-mcp-server](https://aws-oss.beachgeek.co.uk/4db) containerises and deploys applications to Amazon ECS within minutes by configuring all relevant AWS resources, including load balancers, networking, auto-scaling, monitoring, Amazon ECS task definitions, and services. Using natural language instructions, you can manage cluster operations, implement auto-scaling strategies, and use real-time troubleshooting capabilities to identify and resolve deployment issues quickly. (#210)

**eks-mcp-server**

[eks-mcp-server](https://aws-oss.beachgeek.co.uk/4dd) provides similar capabilities to the **ecs-mcp-server** project, providing AI assistants with up-to-date, contextual information about your specific EKS environment. It offers access to the latest EKS features, knowledge base, and cluster state information. This gives AI code assistants more accurate, tailored guidance throughout the application lifecycle, from initial setup to production deployment. (#210)

**evaluating-large-language-models-using-llm-as-a-judge**

[evaluating-large-language-models-using-llm-as-a-judge](https://aws-oss.beachgeek.co.uk/3vz) This lab addresses this challenge by providing a practical solution for evaluating LLMs using LLM-as-a-Judge with Amazon Bedrock. This is relevant for developers and researchers working on evaluating LLM based applications. In the notebook you are guided using MT-Bench questions to generate test answers and evaluate them with a single-answer grading using the Bedrock API, Python and Langchain.

Evaluating large language models (LLM) is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences. To address this, strong LLMs are used as judges to evaluate these models on more open-ended questions. The agreement between LLM judges and human preferences has been verified by introducing two benchmarks: Multi Turn (MT)-bench, a multi-turn question set, and Chatbot Arena, a crowdsourced battle platform. The results reveal that strong LLM judges can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans This makes LLM-as-a-judge a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain. (#197)

**fmbench-orchestrator**

[fmbench-orchestrator](https://aws-oss.beachgeek.co.uk/467) this repo is a tool designed to automate the deployment and management of FMBench for benchmarking on Amazon EC2, Amazon SageMaker and Amazon Bedrock. In case of benchmarking on EC2, we could benchmark on multiple instances simultaneously, and these instances can be of different instance types (so you could run g6e, p4de and a trn1 instances via the same config file), in different AWS regions and also test multiple FMBench config files. This orchestrator automates the creation of Security Groups, Key Pairs, EC2 instances, runs FMBench for a specific config, retrieves the results, and shuts down the instances after completion. Thus it simplifies the benchmarking process (no more manual creation of SageMaker Notebooks, EC2 instances and cleanup, downloading results folder) and ensures a streamlined and scalable workflow. Very detailed README that provides much more details on how this works. (#205)

**fm-leaderboarder**

[fm-leaderboarder](https://aws-oss.beachgeek.co.uk/3sf) provides resources to help you benchmark against the growing number of different models to help you find the best LLM for your own business use case based on your own tasks, prompts, and data. (#194)

**foundation-model-benchmarking-tool**

[foundation-model-benchmarking-tool](https://aws-oss.beachgeek.co.uk/3mj) is a Foundation model (FM) benchmarking tool. Run any model on Amazon SageMaker and benchmark for performance across instance type and serving stack options. A key challenge with FMs is the ability to benchmark their performance in terms of inference latency, throughput and cost so as to determine which model running with what combination of the hardware and serving stack provides the best price-performance combination for a given workload.(#187)

**gh-relfind**

[gh-relfind](https://aws-oss.beachgeek.co.uk/3tf) is a project from AWS Community Builder Colin Duggan, that provides a simple Go CLI tool that uses Claude to parse package changes from GitHub repos release history. Significant change information is often contained in the release text field (body tag in the JSON response from ListReleases). Adding a semantic search makes it easier to extract and summarise actual change details. The project was built to get a better understanding of how to integrate with Claude through AWS Bedrock. (#195)

**gen-ai-on-eks**

[gen-ai-on-eks](https://aws-oss.beachgeek.co.uk/3d2) this repository aims to showcase how to finetune a FM model in Amazon EKS cluster using, JupyterHub to provision notebooks and craft both serving and training scripts, RayOperator to manage Ray Clusters and Karpenter to manage Node Scaling. (#177)

**graphrag-toolkit**

[graphrag-toolkit](https://aws-oss.beachgeek.co.uk/464) is a Python toolkit for building GraphRAG applications. It provides a framework for automating the construction of a graph from unstructured data, and composing question-answering strategies that query this graph when answering user questions. The toolkit uses low-level LlamaIndex components – data connectors, metadata extractors, and transforms – to implement much of the graph construction process. By default, the toolkit uses Amazon Neptune Analytics or Amazon Neptune Database for its graph store, and Neptune Analytics or Amazon OpenSearch Serverless for its vector store, but it also provides extensibility points for adding alternative graph stores and vector stores. The default backend for LLMs and embedding models is Amazon Bedrock; but, as with the stores, the toolkit can be configured for other LLM and embedding model backends using LlamaIndex abstractions. (#205)

**guidance-for-natural-language-queries-of-relational-databases-on-aws**

[guidance-for-natural-language-queries-of-relational-databases-on-aws](https://aws-oss.beachgeek.co.uk/337) this AWS Solution contains a demonstration of Generative AI, specifically, the use of Natural Language Query (NLQ) to ask questions of an Amazon RDS for PostgreSQL database. This solution offers three architectural options for Foundation Models: 1. Amazon SageMaker JumpStart, 2. Amazon Bedrock, and 3. OpenAI API. The demonstration's web-based application, running on Amazon ECS on AWS Fargate, uses a combination of LangChain, Streamlit, Chroma, and HuggingFace SentenceTransformers. The application accepts natural language questions from end-users and returns natural language answers, along with the associated SQL query and Pandas DataFrame-compatible result set.(#190)

**langgraph-aws-deployment**

[langgraph-aws-deployment](https://aws-oss.beachgeek.co.uk/4gj) this repo from **Ali mz** provides a script that simplifies deploying langgraph on yuor AWS environments. If you just want to kick the tires on an agent without managing EC2 or writing Terraform, this gets you from git clone to a public HTTPS endpoint in ~10 min. It’s opinionated (Fargate, ALB, Parameter Store) but easy to tweak. (#212)

**llm-colosseum**

[llm-colosseum](https://github.com/aws-banjo/llm-colosseum) is another repo that takes a more creative look at benchmarking your LLM's, this time using a classic video arcade fighting game.(#194)

**load-test-llm-with-locust**

[load-test-llm-with-locust](https://aws-oss.beachgeek.co.uk/3qg) provides an example of how to perform load testing on the LLM API to evaluate your production requirements. The code is developed within a SageMaker Notebook and utilises the command line interface to conduct load testing on both the SageMaker and Bedrock LLM API. If you are not familiar with Locust, it is an open source load testing tool, and is a popular framework for load testing HTTP and other protocols. Its developer friendly approach lets you to define your tests in regular Python code. Locust tests can be run from command line or using its web-based UI. Throughput, response times and errors can be viewed in real time and/or exported for later analysis.(#192)

**Log-Analyzer-with-MCP**

[Log-Analyzer-with-MCP](https://aws-oss.beachgeek.co.uk/4bu) - this repo provides a Model Context Protocol (MCP) server that provides AI assistants access to AWS CloudWatch Logs for analysis, searching, and correlation. The README provides everything you need to get started, as well as providing links that dive deeper into how this works. (#209)


**llm-test-mate**

[llm-test-mate](https://aws-oss.beachgeek.co.uk/47v) is a project from my colleague Danilo Poccia that is a simple testing framework to evaluate and validate LLM-generated content using string similarity, semantic. Check out the README for a list of features currently supported, as well as examples of how to use it. The README really is a thing of beauty, and I wish more projects provided as clear and detailed info as this project.(#206)

**makit-llm-lambda**

[makit-llm-lambda ](https://aws-oss.beachgeek.co.uk/3cx)this repo from Martyn Kilbryde is an example of how you can run a Large Language Model (LLM) inside an AWS Lambda Function.  Whilst the code will help you deploy to AWS Lambda, it can be ran locally inside Docker for testing as well. The function contains the full LLM model and the code to use the model, allowing basic text generation from a HTTP call into it. (#177)

**mlspace**

[mlspace](https://aws-oss.beachgeek.co.uk/3r8) provides code that will help you deploy [MLSpace](https://aws-oss.beachgeek.co.uk/3r9) into your AWS account. [MLSpace](https://aws-oss.beachgeek.co.uk/3r9) is an open source no-hassle tool for data science, machine learning and deep learning, and has pre-made environments for pytorch, tensorflow and everything else you might need. (#193)

**mcp**

[mcp](https://aws-oss.beachgeek.co.uk/4bi) is a comprehensive repo that provides a suite of specialised Model Context Protocol (MCP) servers that help you get the most out of AWS, wherever you use MCP. This list is growing, so check out the list (currently contains MCP servers that will help you with AWS documentation, AWS Cost Analysis, using AWS CDK, and more) - (#209)


**mcp-lambda-layer**

[mcp-lambda-layer](https://aws-oss.beachgeek.co.uk/4ab) is project that uses a Node.js package that provides MCP (Model Context Protocol) server infrastructure for AWS Lambda functions with SSE support. The repo provides example code for tools and prompts which should get you started and simplify how you might provide these to your MCP clients. Check out the README for other important notes.(#208)

**mcp-server-aws**

[mcp-server-aws](https://aws-oss.beachgeek.co.uk/47r) is a project from Rishi Kavikondala that provides a  [Model Context Protocol](https://aws-oss.beachgeek.co.uk/47s) server implementation for AWS operations that currently supports S3 and DynamoDB services. All operations are automatically logged and can be accessed through the audit://aws-operations resource endpoint.(#206)

**MCP2Lambda**

[MCP2Lambda](https://aws-oss.beachgeek.co.uk/498) is a project from my good friend Danilo Poccia and is a really great example of how Model Control Protocol (MCP) provides Large Language Model (LLM) with additional capabilities and flexibility. In this demo sample, an MCP server acts as a bridge between MCP clients and AWS Lambda functions, allowing generative AI models to access and run Lambda functions as tools. This is useful, for example, to access private resources such as internal applications and databases without the need to provide public network access. This approach allows the model to use other AWS services, private networks, and the public internet.(#207)

**middy-mcp**

[middy-mcp](https://aws-oss.beachgeek.co.uk/4bk) provides [Middy](https://github.com/middyjs/middy) [middleware](https://middy.js.org/)  (a Node.js middleware engine for AWS Lambda that helps you organise your Lambda code, remove code duplication, and focus on business logic) for Model Context Protocol (MCP) server integration with AWS Lambda functions. It provides a convenient way to handle MCP requests and responses within your Lambda functions using the Middy middleware framework. It supports requests sent to AWS Lambda from API Gateway (both REST API / v1 and HTTP API / v2) using the Proxy integration, as well as requests sent form an ALB. Check out the README for some code examples of how you might use this. (#209)

**mkdocs-mcp**

[mkdocs-mcp](https://aws-oss.beachgeek.co.uk/4d2) is another project from **AWS Hero Michael Walmsley** that provides search functionality for any MkDocs powered site. This server relies on the existing MkDocs search implementation using the Lunr.Js search engine. (#210)

**multi-agent-orchestrator**

[multi-agent-orchestrator](https://aws-oss.beachgeek.co.uk/41q) is a new open source project that provides a flexible and powerful framework for managing multiple AI agents and handling complex conversations. It intelligently routes queries and maintains context across interactions. The system offers pre-built components for quick deployment, while also allowing easy integration of custom agents and conversation messages storage solutions. This adaptability makes it suitable for a wide range of applications, from simple chatbots to sophisticated AI systems, accommodating diverse requirements and scaling efficiently. (#202)

**multi-table-benchmark**

[multi-table-benchmark](https://aws-oss.beachgeek.co.uk/3vv) this repo is the DBInfer Benchmark (DBB), a set of benchmarks for measuring machine learning solutions over data stored as multiple tables. The repo provides detailed instructions on the different steps needed to set up your testing, as well as a Jupyter notebook tutorial to walk you through the key concepts.(#197)

**news-clustering-and-summarization**

[news-clustering-and-summarization](https://aws-oss.beachgeek.co.uk/44l) this repository contains code for a near real-time news clustering and summarisation solution using AWS services like Lambda, Step Functions, Kinesis, and Bedrock. It demonstrates how to efficiently process, embed, cluster, and summarise large volumes of news articles to provide timely insights for financial services and other industries. This solution aims to launch a news Event feature that clusters related news stories into summaries, providing customers with near real-time updates on unfolding events. This augmented news consumption experience will enable users to easily follow evolving stories while maximising relevance and reducing the firehose of information for articles covering the same event. By tailoring news clusters around key events, this application can improve customer satisfaction and engagement. Detailed docs will help you get up and running in no time. (#204)

**nova-prompt-optimizer**

[nova-prompt-optimizer](https://aws-oss.beachgeek.co.uk/4h4) is a Python SDK for optimising prompts for Nova. Looks very comprehensive, so worth checking out the README. Again, public preview folks so bear this in mind.(#212)

**observation-extractor**

[observation-extractor](https://aws-oss.beachgeek.co.uk/4at) is a tool for collecting observations from data. Observations are useful bits of data related to questions that you define that is extracted from the data you pass in. Use Observation Extractor to process pdf (and maybe someday other files) into formats like csv (and later parquet) to turn unstructured documents into structured observations that you can query and use directly or through your application. When you output to a format like csv or parquet, observations are the row level records.

Observation Extractor takes an unstructured data file as input (like a pdf) and outputs a list of Observation objects. Each observation includes standard fields that are extracted from the document together with metadata like the document name and page number.

You can populate observations into a datastore and make them available to your human and AI users. They can be queried based on metadata like date and the specific questions they relate too. You can define question sets that represent thought process of a subject-matter-expert coming up to speed on this case to start mapping a document into useful observations.(#208)


**partysmith**

[partysmith](https://aws-oss.beachgeek.co.uk/3l4) is an awesome project from AWS Community Builder Stephen Sennett, that provides an unofficial way to transform your AWS PartyRock apps into deployable full-stack SvelteKit applications. Users can enter the URL of a publicly published PartyRock app, select the desired settings, and PartySmith will forge an application into a ZIP archive which will be downloaded to your machine, and ready for use. How cool is that! (Very in case you were wondering). Find out more by reading the supporting blog post, [PartySmith - Bring PartyRock apps to your place](https://aws-oss.beachgeek.co.uk/3l5). (#185)

**powertools-mcp**

[powertools-mcp](https://aws-oss.beachgeek.co.uk/4cz) is a very nice project from **AWS Hero  Michael Walmsley** that provides search functionality for AWS Lambda Powertools documentation across multiple runtimes. This project implements an MCP server that enables Large Language Models (LLMs) to search through AWS Lambda Powertools documentation. It uses lunr.js for efficient local search capabilities and provides results that can be summarised and presented to users.  Good documentation, with examples on how to get started with MCP clients like Claude Desktop (but should work with Amazon Q CLI too) (#210)

**project-lakechain**

[project-lakechain](https://aws-oss.beachgeek.co.uk/401) is an experimental framework based on the AWS Cloud Development Kit (CDK) that makes it easy to express and deploy scalable document processing pipelines on AWS using infrastructure-as-code. It emphasis is on modularity of pipelines, and provides 40+ ready to use components for prototyping complex document pipelines that can scale out of the box to millions of documents. This project has been designed to help AWS customers build and scale different types of document processing pipelines, ranging a wide array of use-cases including metadata extraction, document conversion, NLP analysis, text summarisation, translations, audio transcriptions, computer vision, Retrieval Augmented Generation pipelines, and much more! It is in Alpha stage at the moment, so if you catch any oddities, be sure to flag an issue.(#201)

**promptus**

[promptus](https://aws-oss.beachgeek.co.uk/3mu) Prompt engineering is key for generating high-quality AI content. But crafting effective prompts can be time-consuming and difficult. That's why I built Promptus. Promptus allows you to easily create, iterate, and organise prompts for generative AI models. With Promptus, you can:

* Quickly build prompts with an intuitive interface
* Automatically version and compare prompt iterations to optimise quality
* Organize prompts into projects and share with teammates
* See a history of your prompt and easily go back to any previous prompt execution

(#188)

**python-bedrock-converse-generate-docs**

[python-bedrock-converse-generate-docs](https://aws-oss.beachgeek.co.uk/409) is a project from AWS Community Builder Alan Blockley that generates documentation for a given source code file using the Anthropic Bedrock Runtime API. The generated documentation is formatted in Markdown and stored in the specified output directory. Alan also put a blog together, [It’s not a chat bot: Writing Documentation](https://aws-oss.beachgeek.co.uk/40a), that shows you how it works and how to get started. The other cool thing about this project is that it is using the [Converse API](https://aws-oss.beachgeek.co.uk/40b) which you should check out if you have not already seen/used it. (#201)

**rag-with-amazon-bedrock-and-pgvector**

[rag-with-amazon-bedrock-and-pgvector](https://aws-oss.beachgeek.co.uk/3lv) is an opinionated sample repo on how to configure and deploy RAG (Retrieval Augmented Retrieval) application. It uses a number of components including Amazon Bedrock for foundational models, Amazon RDS PostgreSQL with pgvector, LangChain, Streamlit, and a number of AWS services to bring it all together.(#186)

**real-time-social-media-analytics-with-generative-ai**

[real-time-social-media-analytics-with-generative-ai](https://aws-oss.beachgeek.co.uk/3x5) this repo helps you to build and deploy an AWS Architecture that is able to combine streaming data with GenAI using Amazon Managed Service for Apache Flink and Amazon Bedrock. (198)

**rhubarb**

[rhubarb](https://aws-oss.beachgeek.co.uk/3vt) is a light-weight Python framework that makes it easy to build document understanding applications using Multi-modal Large Language Models (LLMs) and Embedding models. Rhubarb is created from the ground up to work with Amazon Bedrock and Anthropic Claude V3 Multi-modal Language Models, and Amazon Titan Multi-modal Embedding model. Rhubarb can perform multiple document processing and understanding tasks. Fundamentally, Rhubarb uses Multi-modal language models and multi-modal embedding models available via Amazon Bedrock to perform document extraction, summarisation, Entity detection, Q&A and more. Rhubarb comes with built-in system prompts that makes it easy to use it for a number of different document understanding use-cases. You can customise Rhubarb by passing in your own system and user prompts. It supports deterministic JSON schema based output generation which makes it easy to integrate into downstream applications. Looks super interesting, on my to do list and will report back my findings.(#197)

**RefChecker**

[RefChecker](https://aws-oss.beachgeek.co.uk/3l3) For all their remarkable abilities, large language models (LLMs) have an Achilles heel, which is their tendency to hallucinate, or make assertions that sound plausible but are factually inaccurate. RefChecker provides automatic checking pipeline and benchmark dataset for detecting fine-grained hallucinations generated by Large Language Models. Check out the supporting post for this tool, [New tool, dataset help detect hallucinations in large language models](https://aws-oss.beachgeek.co.uk/3l7) (#185)

**rockhead-extensions**

[rockhead-extensions ](https://aws-oss.beachgeek.co.uk/3r5)another repo from a colleague, this time it is .NET aficionado Francois Bouteruche, who has put together this repo that provides code (as well as a nuget package) to make your .NET developer life easier when you invoke foundation models on Amazon Bedrock. More specifically, Francois has created a set of extension methods for the AWS SDK for .NET Bedrock Runtime client. It provides you strongly typed parameters and responses to make your developer life easier. (#193)

**s3-connector-for-pytorch**

[s3-connector-for-pytorch](https://aws-oss.beachgeek.co.uk/3gw) the Amazon S3 Connector for PyTorch delivers high throughput for PyTorch training jobs that access or store data in Amazon S3. Using the S3 Connector for PyTorch automatically optimises performance when downloading training data from and writing checkpoints to Amazon S3, eliminating the need to write your own code to list S3 buckets and manage concurrent requests. Amazon S3 Connector for PyTorch provides implementations of PyTorch's dataset primitives that you can use to load training data from Amazon S3. It supports both map-style datasets for random data access patterns and iterable-style datasets for streaming sequential data access patterns. The S3 Connector for PyTorch also includes a checkpointing interface to save and load checkpoints directly to Amazon S3, without first saving to local storage. (#181)

**safeaws-cli**

[safeaws-cli](https://aws-oss.beachgeek.co.uk/3te) is a project from AWS Community Builder Gabriel Koo that provides an AWS CLI wrapper that helps you avoid common mistakes and pitfalls with Amazon Bedrock's Large Language Models, checking your command arguments against the command docs. safeaws-cli empowers you to execute AWS commands confidently by leveraging Amazon Bedrock's AI language models to analyze CLI commands, fetch help text, and identify potential issues or concerns before execution. By providing a safety net that mitigates pitfalls, safeaws-cli allows users to explore AWS securely, fostering a more informed approach to working with the CLI.(#195)

**sagemaker-mlflow**

[sagemaker-mlflow](https://aws-oss.beachgeek.co.uk/3ya)  This plugin generates Signature V4 headers in each outgoing request to the Amazon SageMaker with MLflow capability, determines the URL of capability to connect to tracking servers, and registers models to the SageMaker Model Registry. It generates a token with the SigV4 Algorithm that the service will use to conduct Authentication and Authorization using AWS IAM.(#200)

**sample-convert-codebase-to-graphrag**

[sample-convert-codebase-to-graphrag](https://aws-oss.beachgeek.co.uk/4ak) is a demo to show how to leverages AI to analyze, index, and query code repositories. It creates a searchable graph representation of code structures, enabling developers to explore and understand complex codebases efficiently. This project combines several AWS services, including Lambda, Neptune, OpenSearch, and Bedrock, to process code repositories, generate metadata, and provide powerful search capabilities. The system is designed to handle large-scale code analysis tasks and offer semantic code search functionality. It uses CDK to simplify how to deploy this in your own AWS environments. (#208)

**sample-devgenius-aws-solution-builder**

[sample-devgenius-aws-solution-builder](https://aws-oss.beachgeek.co.uk/4bx) is a super interesting project that you can deploy and provides an AI-powered application that transforms project ideas into complete, ready-to-deploy AWS solutions. It leverages Amazon Bedrock and Claude AI models to provide architecture diagrams, cost estimates, infrastructure as code, and comprehensive technical documentation. In "Conversational Solution Architecture Building" mode,  DevGenius enables customers to design solution architectures in a conversational manner. Users can create architecture diagrams (in draw.io format) and refine them interactively. Once the design is finalised, they can generate end-to-end code automation using CDK or CloudFormation templates, and deploy it in their AWS account with a single click. Additionally, customers can receive cost estimates for running the architecture in production, along with detailed documentation for the solution. In the "Build Solution Architecture from Whiteboard Drawings" mode, those of you who already have their architecture in image form (e.g., whiteboard drawings), DevGenius allows you to upload the image. Once uploaded, DevGenius analyses the architecture and provides a detailed explanation. You can then refine the design conversationally and, once finalised, generate end-to-end code automation using CDK or CloudFormation. Cost estimates and comprehensive documentation are also available.(#209)

**sample-ollama-server**

[sample-ollama-server](https://aws-oss.beachgeek.co.uk/4bv) is a project that folks who are interested in or already using ollama need to check out. This repo provides a AWS CloudFormation template to provision NVIDIA GPU EC2 instances with Ollama and Open WebUI, and include access to Amazon Bedrock foundation models (FMs). The solution can be deployed as a website for LLM interaction through Open WebUI, or as application development environment with Amazon DCV server.(#209)

**serverless-genai-food-analyzer-app**

[serverless-genai-food-analyzer-app](https://aws-oss.beachgeek.co.uk/3x6) provides code for a personalised GenAI nutritional web application for your shopping and cooking recipes built with serverless architecture and generative AI capabilities. It was first created as the winner of the AWS Hackathon France 2024 and then introduced as a booth exhibit at the AWS Summit Paris 2024. You use your cell phone to scan a bar code of a product to get the explanations of the ingredients and nutritional information of a grocery product personalised with your allergies and diet. You can also take a picture of food products and discover three personalised recipes based on their food preferences. The app is designed to have minimal code, be extensible, scalable, and cost-efficient. It uses Lazy Loading to reduce cost and ensure the best user experience. (#198)

**smart-assistant-agent**

[smart-assistant-agent](https://aws-oss.beachgeek.co.uk/3rd) is a project from AWS Community Builder Darya Petrashka that provides a solution to building an AWS Bedrock agent acting as a Telegram chat assistant. Check out the README for example videos of what this can do, as well as very detailed deployment instructions. (#193)

**strands-ts**

[strands-ts](https://aws-oss.beachgeek.co.uk/4ed) is an experimental SDK for Strands for TypeScript developers from AWS Serverless Community Builder **Ryan Cormack**.  It is an AI generated migration of the Python SDK to TypeScript, using the same architecture and design principles. It is not a direct translation but rather a reimagining of the SDK in TypeScript, leveraging its features and idioms.(#211)

**streamlit-bedrock-claude-sample**

[streamlit-bedrock-claude-sample](https://aws-oss.beachgeek.co.uk/437) - I have featured Gary Stafford's open source projects and blog posts regularly in this newsletter. Gary has built a number of simple Streamlit applications to make it easy access the latest models and features of Amazon Web Services (AWS) Amazon Bedrock as part of several talks, workshops, and demonstrations he has done.  As part these, he has put together a simple Streamlit application that uses the Amazon Bedrock boto3 Python SDK to call the latest Anthropic Claude 3 family of multimodal foundation models. The application accepts a system and user prompt and generates a text-based response. The Streamlit app can be easily modified to incorporate new Bedrock features or as a starting point for your own applications. (#203)

**video-understanding-solution**

[video-understanding-solution](https://aws-oss.beachgeek.co.uk/3sj) This is a deployable solution to help save your time in understanding videos without having to watch every video. You can upload videos and this solution can generate AI-powered summary and entities extraction for each video. It also supports Q&A about the video like "What is funny about the video?", "How does Jeff Bezos look like there?", and "What shirt did he wear?". You can also search for videos using semantic search e.g. "Amazon's culture and history". This solution extracts information from visual scenes, audio, visible texts, and detected celebrities or faces in the video. It leverages an LLM which can understand visual and describe the video frames. You can upload videos to your Amazon Simple Storage Service (S3) bucket bucket by using AWS console, CLI, SDK, or other means (e.g. via AWS Transfer Family). This solution will automatically trigger processes including call to Amazon Transcribe for voice transcription, call to Amazon Rekognition to extract the objects visible, and call to Amazon Bedrock with Claude 3 model to extract scenes and visually visible text. The LLM used can perform VQA (visual question answering) from images (video frames), which is used to extract the scene and text. This combined information is used to generate the summary and entities extraction as powered by generative AI with Amazon Bedrock. The UI chatbot also uses Amazon Bedrock for the Q&A chatbot. The summaries, entities, and combined extracted information are stored in S3 bucket, available to be used for further custom analytics. (#194)

**whats-new-summary-notifier**

[whats-new-summary-notifier](https://aws-oss.beachgeek.co.uk/3x4) is a demo repo that lets you build a generative AI application that summarises the content of AWS What's New and other web articles in multiple languages, and delivers the summary to Slack or Microsoft Teams. (#198)

**whisperx-on-aws-lambda**

[whisperx-on-aws-lambda](https://aws-oss.beachgeek.co.uk/49a) is a project from Vincent Claes that shows you how you can run [WhisperX](https://aws-oss.beachgeek.co.uk/49b) (one of the most versatile and feature-rich Whisper variation that provides fast automatic speech recognition) on AWS Lambda - WhisperX goes serverless! (#207)


**ziya**

[ziya](https://aws-oss.beachgeek.co.uk/44k) is a code assist tool for Amazon Bedrock models that can read your entire codebase and answer questions. The tool currently operates in Read only mode, but doing more that this is on the road map.(#204)

### Application integration and middleware

**active-active-cache**

[active-active-cache](https://aws-oss.beachgeek.co.uk/3o5) is a repo that helps you build a solution that implements an active-active cache across 2 AWS regions, using ElastiCache for Redis. This solution is automated with CDK and SAM.(#189)

**apigw-multi-region-failover**

[apigw-multi-region-failover](https://aws-oss.beachgeek.co.uk/3rc) provides demo code that demonstrates an Amazon API Gateway multi-region active-passive public API that proxies two independent multi-region active-passive service APIs. The primary and secondary regions can be configured independently for the external API and each service. This allows you to fail over the external API and each service independently as needed for disaster recovery. (#193)

**aws-apn-connector**

[aws-apn-connector](https://aws-oss.beachgeek.co.uk/3l1) this project from the folks at Nearform provides a way of interacting with the AWS APN (AWS Partner Network) programatically, as this does not provide an API. If you are looking to automate your interactions with the AWS APN, you should check this project out.(#185)

**aws-cdk-python-for-amazon-mwaa**

[aws-cdk-python-for-amazon-mwaa](https://aws-oss.beachgeek.co.uk/3lq) this repo provides python code and uses AWS CDK to help you automate the deployment and configuration of Managed Workflows for Apache Airflow (MWAA). I have shared my own repos to help you do this, but you can never have enough of a good thing, so check out this repo and see if it is useful.(#186)

**domino**

[domino](https://aws-oss.beachgeek.co.uk/3uj) is a new open source workflow management platform that provides a very nice GUI and drag and drop experience for creating workflows. Now regular readers of this newsletter will know I am a big fan of the Node Red open source project, and I got very strong Node Red vibes about the GUI, which is a good thing. Under the covers, we have another favourite project of mine, Apache Airflow. (#196)

**dynamomq**

[dynamomq](https://aws-oss.beachgeek.co.uk/3j1) is a project from Yuichi Watanabe that provides a message queuing library that leverages the features of DynamoDB to achieve high scalability, reliability, and cost efficiency. Notably, its ability to dynamically edit message order and attributes enables flexible adaptation to application requirements. Compared to existing solutions, DynamoMQ offers ease of management for developers while providing the reliability of fully managed services like Amazon SQS. It also encompasses key functionalities expected from a message queue, such as concurrent processing with multiple goroutines, Dead Letter Queues, and ensuring FIFO (First In, First Out) order. (#183)

**language-server-runtimes**

[language-server-runtimes](https://aws-oss.beachgeek.co.uk/3qe) is a JSON-RPC based protocol for interactions between servers and clients (typically embedded in development tools). The README covers details around specification support and features supported, that will help you tailor this to your needs.(#192)

### Compute - Containers, EC2, Serverless

**ai-on-eks**

[ai-on-eks](https://aws-oss.beachgeek.co.uk/4bw) is your go to repo if you are interested in getting AI up and running on Amazon EKS. It provides a gateway to help you scale AI and ML workloads on Amazon EKS. It provides a rich collection of Terraform Blueprints featuring current good practices for deploying robust solutions with advanced logging and observability. You can explore practical patterns for running AI/ML workloads on EKS, leveraging the power of the Ray ecosystem for distributed computing. Utilise advanced serving solutions like NVIDIA Triton Server, vLLM for efficient and scalable model inference, and TensorRT-LLM for optimising deep learning models. (#209)

**amazon-eks-running-webassembly**

[amazon-eks-running-webassembly](https://aws-oss.beachgeek.co.uk/3th) This repository contains code for building custom Amazon EKS AMIs using HashiCorp Packer. The AMIs include necessary binaries and configurations to enable you to run WebAssembly workloads in an EKS cluster and are based on Amazon Linux 2023. The runtimes used in the AMIs are Spin and WasmEdge. The respective containerd-shims are used for both runtimes. Deploying the cluster is done using Hashicorp Terraform. After the cluster is created, RuntimeClasses and example workloads are deployed to the cluster. If you are exploring Wasm, then this is for you. (#195)

**amazon-mwaa-docker-images**

[amazon-mwaa-docker-images](https://aws-oss.beachgeek.co.uk/3zu) this repo was new to me, so making sure that everyone knows that this repo contains the standard container images used for the Managed Worksflows for Apache Airflow (#201)

**amplify-hosting-astro**

[amplify-hosting-astro](https://aws-oss.beachgeek.co.uk/3r7) is a repo from AWS Amplify's Matt Auerbach that provides a walk through on how to build a simple blog using Astro's starter blog template, and deploy it using AWS Amplify Hosting. (#193)

**apigw-multi-region-failover**

[apigw-multi-region-failover](https://aws-oss.beachgeek.co.uk/3rc) provides demo code that demonstrates an Amazon API Gateway multi-region active-passive public API that proxies two independent multi-region active-passive service APIs. The primary and secondary regions can be configured independently for the external API and each service. This allows you to fail over the external API and each service independently as needed for disaster recovery. (#193)

**aws-cdk-imagebuilder-sample**

[aws-cdk-imagebuilder-sample](https://aws-oss.beachgeek.co.uk/3o2) this repo uses AWS CDK (TypeScript) that demonstrates how to create a fully functional ImageBuilder pipeline that builds an Amazon Linux 2023 container image, installing git, docker and nodejs, all the way to pushing the resulting image to an ECR repository.(#189)

**aws-nitro-enclaves-eif-build-action**

[aws-nitro-enclaves-eif-build-action](https://aws-oss.beachgeek.co.uk/3pj) is a new project from AWS Hero Richard Fan that uses a number of tools to help you build a reproducible AWS Nitro Enclaves EIF (Enclave Image File). This GitHub Action use kaniko and Amazon Linux container with nitro-cli, and provides examples of how you can use other tools such as sigstore to sign artefacts as well. (#191)

**aws-signer-oci-artifacts**

[aws-signer-oci-artifacts](https://aws-oss.beachgeek.co.uk/3km) this project is used to demonstrate how OCI artefacts can be signed and verified in a development pipeline. Zhuo-Wei Lee, Alontay Ellis, and Rajarshi Das have put together a blog post to help you get started, so if this project interests you, make sure you dive into [Signing and Validating OCI Artifacts with AWS Signer](https://aws-oss.beachgeek.co.uk/3kn).(#185)

**beta9**

[beta9](https://aws-oss.beachgeek.co.uk/3xu) is a self-hosted serverless framework that you can run in your AWS account. Think of AWS Lambda, but with GPUs and a Python-first developer experience. You can run workloads that instantly scale up to thousands of GPU containers running in parallel. The instances scale down automatically after each workload. You can also do things like deploy web endpoints, run task queues, and mount storage volumes for accessing large datasets. If you already have an EKS cluster, you can install Beta9 with a Helm chart. We think this would be a great way to save money on EC2 GPU resources while also getting a magical Python-first developer experience. If you have feedback or feature ideas, the maintainers would like to hear them. 

**cedar-access-control-for-k8s**

[cedar-access-control-for-k8s](https://aws-oss.beachgeek.co.uk/47k) is a very very cool project from Micah Hausler, that extends Cedar to the Kubernetes control plane and allows you to implement fine grain policies in Cedar that allow you to have much greater control and flexibility of authorisation within your Kubernetes environments. If you are using Kubernetes, then reviewing this project is a must. Check out the video in the Videos section at the end for more info, where Micah walks you through how this works in more detail. (#206)

**codecatalyst-runner-cli**

[codecatalyst-runner-cli](https://aws-oss.beachgeek.co.uk/3up) This repository contains a command line tool that will allow you to run Amazon CodeCatalyst workflows locally. The README provides the instructions for quickly installing and getting started, so if  you have been using Amazon CodeCatalyst and looking for this, look no more.(#196)

**containers-cost-allocation-dashboard**

[containers-cost-allocation-dashboard](https://aws-oss.beachgeek.co.uk/3x9) provides everything you need to create a QuickSight dashboard for containers cost allocation based on data from Kubecost. The dashboard provides visibility into EKS in-cluster cost and usage in a multi-cluster environment, using data from a self-hosted Kubecost pod. The README contains additional links to resources to help you understand how this works, dependencies, and how to deploy and configure this project.(#198)

**container-resiliency**

[container-resiliency](https://aws-oss.beachgeek.co.uk/3vs) the primary goal of this repository is to provide a comprehensive guide and patterns for organisations to design, deploy, and operate highly resilient and fault-tolerant containerised applications on AWS. These patterns aims to provide the knowledge and practical guidance necessary to mitigate risks, minimise downtime, and ensure the continuous availability and resilience of containerised applications on AWS, ultimately enhancing their overall operational efficiency and customer experience.(#197)

**e1s**

[e1s](https://aws-oss.beachgeek.co.uk/3w1)  is a terminal application from [Xing Yahao](https://twitter.com/keidarcy) to easily browse and manage AWS ECS resources, supports both Fargate and EC2 ECS launch types. Inspired by k9s. e1s uses the default aws-cli configuration. It does not store or send your access and secret key anywhere. The access and secret key are used only to securely connect to AWS API via AWS SDK. e1s is available on Linux, macOS and Windows platforms. (#197)

**ec2RuntimeMonitor**

[ec2RuntimeMonitor](https://aws-oss.beachgeek.co.uk/3ra) EC2 runtime monitor is a serverless solution to get a notification when an EC2 instance is running for a time exceeding a user defined threshold. The README covers use cases why you might find this useful, but principally cost optimisation as well as reducing your carbon footprint are two key reasons why this might be a useful tool to keep your toolkit. (#193)

**ecs-gpu-scaling**

[ecs-gpu-scaling](https://aws-oss.beachgeek.co.uk/3mh) This repository is intended for engineers looking to horizontally scale GPU-based Machine Learning (ML) workloads on Amazon ECS. By default, GPU utilisation metrics are not part of the predefined metrics available with Application Autoscaling. As such, you implement auto scaling based on custom metrics.  For NVIDIA-based GPUs, you use DCGM-Exporter in your container to expose GPU metrics. You can then use metrics such as DCGM_FI_DEV_GPU_UTIL and DCGM_FI_DEV_GPU_TEMP to determine your auto scaling behaviour. The README provides links to all the additional resources you need to get this up and running.(#187)

**ecs-will-it-fit**

[ecs-will-it-fit](https://aws-oss.beachgeek.co.uk/47h) (or wily for short) is a new project from my good friend Ivica Kolenkaš, that provides a very useful tool for those of you using Amazon ECS. It is a CLI tool that helps you answer the question: "Will this ECS service fit on my ECS cluster backed by EC2 instances?". It does so by mimicking1 the selection process that the ECS scheduler performs while selecting suitable container instances for your service. Nice detailed README provides everything you need to know about how to use this cli. This is in alpha stage, so if you give it a go, make sure to pass on your feedback to Ivica. (#206)

**eks-shared-subnets**

[eks-shared-subnets](https://aws-oss.beachgeek.co.uk/3k2) this sample code demonstrates how a central networking team in an enterprise can create and share the VPC Subnets from their own AWS Account with other Workload Specific accounts. So that, Application teams can deploy and manage their own EKS clusters and related resources in those Subnets. (#184)

**fargate-on-demand**

[fargate-on-demand](https://aws-oss.beachgeek.co.uk/3mv) this repo provides the code that shows you how you can cost optimise your container applications and then control scale down (or up) as needed. Yoanna Krasteva has put together a blog post that provides you with why and how you can configure this in the post, [Cost saving with AWS Fargate On-Demand](https://aws-oss.beachgeek.co.uk/3mw).(#188)

**finch-mcp-server**

[finch-mcp-server](https://aws-oss.beachgeek.co.uk/4ep) is the source code for the Finch MCP Server,  that enables generative AI models to build and push container images through finch cli leveraged MCP tools. It currently can build container images using Finch, push those images onto container registries (like Amazon ECR) creating them if needed, and manage all your Finch configuration files. The json configuration files that you will need for your MCP Clients [can be found here](https://awslabs.github.io/mcp/servers/finch-mcp-server/), together with additional info on dependencies and how this works. (#211)

**how-high-is-my-salary-enclave-app**

[how-high-is-my-salary-enclave-app](https://aws-oss.beachgeek.co.uk/3y3) is a rather cool project from AWS Hero Richard Fan that provides a simple app showcases how to protect software supply chain security using GitHub Actions, SLSA, and AWS Nitro Enclaves. (#199)

**kro**

[kro](https://aws-oss.beachgeek.co.uk/468) This project aims to simplify the creation and management of complex custom resources for Kubernetes. Kube Resource Orchestrator (kro) provides a powerful abstraction layer that allows you to define complex multi-resource constructs as reusable components in your applications and systems. You define these using kro's fundamental custom resource, ResourceGroup. This resource serves as a blueprint for creating and managing collections of underlying Kubernetes resources. With kro, you define custom resources as your fundamental building blocks for Kubernetes. These building blocks can include other Kubernetes resources, either native or custom, and can specify the dependencies between them. This lets you define complex custom resources, and include default configurations for their use. The kro controller will determine the dependencies between resources, establish the correct order of operations to create and configure them, and then dynamically create and manage all of the underlying resources for you. kro is Kubernetes native and integrates seamlessly with existing tools to preserve familiar processes and interfaces. (#205)

**llrt**

[llrt](https://aws-oss.beachgeek.co.uk/3mm) - Low Latency Runtime (LLRT) is a lightweight JavaScript runtime designed to address the growing demand for fast and efficient Serverless applications. LLRT offers up to over 10x faster startup and up to 2x overall lower cost compared to other JavaScript runtimes running on AWS Lambda. It's is built in Rust, utilising QuickJS as JavaScript engine, ensuring efficient memory usage and swift startup. (#188)

**serverless-lambda-cron-cdk**

[serverless-lambda-cron-cdk](https://aws-oss.beachgeek.co.uk/3uf) This repository provides a starter kit for setting up cron jobs using AWS Lambda. It includes the necessary AWS Cloud Development Kit (CDK) deployment code, a CI/CD pipeline, as well as the source code for the Lambda function. The kit is designed to be easily configurable and deployable, allowing for quick setup and iteration. It's ideal for developers looking to automate tasks on a schedule using AWS Lambda. (#196)

### Data, Big Data and Databases

**active-active-cache**

[active-active-cache](https://aws-oss.beachgeek.co.uk/3o5) is a repo that helps you build a solution that implements an active-active cache across 2 AWS regions, using ElastiCache for Redis. This solution is automated with CDK and SAM.(#189)

**activerecord-dsql-adapter**

[activerecord-dsql-adapter](https://aws-oss.beachgeek.co.uk/47u) is a new project from Samuel Cochran
 that provides the beginnings of an Active Record connection adapter for Amazon's AWS Aurora DSQL database.(#206)

**ai-driven-sql-generation**

[ai-driven-sql-generation](https://aws-oss.beachgeek.co.uk/3w0) this sample code from AWS Community Builder Hardik Singh Behl uses Amazon Bedrock with Spring AI to convert natural language queries to SQL queries, using Anthropic's Claude 3 Haiku model.(#197)

**amazon-bedrock-synthetic-manufacturing-data-generator**

[amazon-bedrock-synthetic-manufacturing-data-generator](https://aws-oss.beachgeek.co.uk/3ln) is a industry aligned synthetic data generation solution. Manufacturing processes generate large amounts of sensor data that can be used for analytics and machine learning models. However, this data may contain sensitive or proprietary information that cannot be shared openly. Synthetic data allows the distribution of realistic example datasets that preserve the statistical properties and relationships in the real data, without exposing confidential information. This enables more open research and benchmarking on representative data. Additionally, synthetic data can augment real datasets to provide more training examples for machine learning algorithms to generalize better. Data augmentation with synthetic manufacturing data can help improve model accuracy and robustness. Overall, synthetic data enables sharing, research, and expanded applications of AI in manufacturing while protecting data privacy and security.(#186)

**amazon-datazone-mcp-server**

[amazon-datazone-mcp-server](https://aws-oss.beachgeek.co.uk/4gz) repo provides a high-performance Model Context Protocol (MCP) server that provides seamless integration with Amazon DataZone services. This server enables AI assistants and applications to interact with Amazon DataZone APIs through a standardised interface.(#212)

**amazon-kinesis-video-streams-dcep**

[amazon-kinesis-video-streams-dcep](https://aws-oss.beachgeek.co.uk/4am) is the repo for the Data Channel Establishment Protocol (DCEP) library.  DCEP is a simple protocol for establishing symmetric data channels between WebRTC peers. It uses a two-way handshake and allows sending of user data without waiting for the handshake to complete. The peer that initiates opening a data channel selects a stream identifier for which the corresponding incoming and outgoing streams are unused and sends a DATA_CHANNEL_OPEN message on the outgoing stream. The peer responds with a DATA_CHANNEL_ACK message on its corresponding outgoing stream. Then the data channel is open. DCEP messages are sent on the same stream as the user messages belonging to the data channel. The demultiplexing is based on the SCTP Payload Protocol Identifier (PPID), since DCEP uses a specific PPID.(#208)

**amazon-mwaa-docker-images**

[amazon-mwaa-docker-images](https://aws-oss.beachgeek.co.uk/3zu) this repo was new to me, so making sure that everyone knows that this repo contains the standard container images used for the Managed Worksflows for Apache Airflow (#201)

**amazon-sqs-python-extended-client-lib**

[amazon-sqs-python-extended-client-lib](https://aws-oss.beachgeek.co.uk/3ti) this repo (Amazon SQS Extended Client) allows clients to manage Amazon SQS message payloads that exceed the 256 KB message size limit, up to a size of 2 GB. In the event of publishing such large messages, the client accomplishes this feat by storing the actual payload in a S3 bucket and by storing the reference of the stored object in the SQS queue. Similarly, the extended-client is also used for retrieving and dereferencing these references of message objects stored in S3.  Check out the docs for more details on how this works and some sample code to get you going. (#195)

**analytics-accelerator-s3**

[analytics-accelerator-s3](https://aws-oss.beachgeek.co.uk/463) the Analytics Accelerator Library for Amazon S3 is an open source library that accelerates data access from client applications to Amazon S3. With this tool you can 1/ lower processing times and compute costs for data analytics workloads, 2/ implement S3 best practices for performance, 3/utilise optimisations specific to Apache Parquet files, such as pre-fetching metadata located in the footer of the object and predictive column pre-fetching, and 4/improve the price performance for your data analytics applications, such as workloads based on Apache Spark.Project is currently in Alpha, so bear that in mind. More examples and details in the README. (#205)

**apache-xtable-on-aws-samples**

[apache-xtable-on-aws-samples](https://aws-oss.beachgeek.co.uk/3ze) provides sample code to build an Apache Airflow Operator that uses Apache XTable to make a single physical dataset readable in different formats by translating its metadata and avoiding reprocessing of actual data files. The repo will help you build and compile your custom jar file, which you can then use within your Airflow DAG. Check out the supporting blog post from Matthias Rudolph and Stephen Said, [Run Apache XTable on Amazon MWAA to translate open table formats](https://aws-oss.beachgeek.co.uk/3zf).(#201)

**automated-datastore-discovery-with-aws-glue**

[automated-datastore-discovery-with-aws-glue](https://aws-oss.beachgeek.co.uk/3x7) This sample shows you how to automate the discovery of various types of data sources in your AWS estate. Examples include - S3 Buckets, RDS databases, or DynamoDB tables. All the information is curated using AWS Glue - specifically in its Data Catalog. It also attempts to detect potential PII fields in the data sources via the Sensitive Data Detection transform in AWS Glue. This framework is useful to get a sense of all data sources in an organisation's AWS estate - from a compliance standpoint. An example of that could be GDPR Article 30. Check out the README for detailed architecture diagrams and a break down of each component as to how it works. (#198)

**automated-data-validation-framework**

[automated-data-validation-framework](https://aws-oss.beachgeek.co.uk/3mi) When you are undertaking data migration projects, a significant time is spent in doing the data validation and lot of manual efforts being spent. This repo provides a framework developed that helps to simplifying this problem by automating full data validation with some simple config files, and running the framework on EMR. It will create summary and detail data validation report in S3 and show up on Athena tables. You will need to do some initial work to setup this framework and create config files which has table names to compare. (#187)

**aws-aurora-db-vertical-autoscaler**

[aws-aurora-db-vertical-autoscaler](https://aws-oss.beachgeek.co.uk/3uu) is a project that I heard about from Dmitry Shurupov (thanks for reaching out!) that helps you implement vertical autoscaling for Aurora for Postgres using Lambda functions. Oleg Mironov put together a blog post to go into more details, including a nice detailed flow diagram of how this code works.(#196)

**aws-az-mapper**

[aws-az-mapper](https://aws-oss.beachgeek.co.uk/3om) is a new tool from Jeremy Barnes that maps an AWS Account and it's regions physical availability zones to their logical availability zone. This project is new to me (although was released a while ago) and what got my interest was this blog post, [Tool - AWS Availability Zone Mapper](https://aws-oss.beachgeek.co.uk/3on) where Jeremy walks you through how you can use this tool, to help with our cost optimisation strategies. (#190)

**aws-config-rule-rds-logging-enabled-remediation**

[aws-config-rule-rds-logging-enabled-remediation](https://aws-oss.beachgeek.co.uk/43l) provides code that will help you checks if respective logs of Amazon Relational Database Service (Amazon RDS) are enabled using AWS Config rule 'rds-logging-enabled'. The rule is NON_COMPLIANT if any log types are not enabled. AWS Systems Manager Automation document used as remediation action to enable the logs export to CloudWatch for the RDS instances that are marked NON_COMPLIANT.(#203)

**aws-data-solutions-framework**

[aws-data-solutions-framework](https://github.com/awslabs/aws-data-solutions-framework) is a framework for implementation and delivery of data solutions with built-in AWS best practices. AWS Data Solutions Framework (DSF) is an abstraction atop AWS services based on AWS Cloud Development Kit (CDK) L3 constructs, packaged as a library. You can leverage AWS DSF to implement your data platform in weeks rather than in months. AWS DSF is available in TypeScript and Python. Use the framework to build your data solutions instead of building cloud infrastructure from scratch. Compose data solutions using integrated building blocks via Infrastructure as Code (IaC), that allow you to benefit from smart defaults and built-in AWS best practices. You can also customize or extend according to your requirements. Check out the dedicated documentation page, complete with examples to get you started. (#178)

**aws-emr-advisor**

[aws-emr-advisor](https://aws-oss.beachgeek.co.uk/3ut) started as fork of Qubole SparkLens, this tool can be used to analyse Spark Event Logs to generate insights and costs recommendations using different deployment options for Amazon EMR. The tool generates an HTML report that can be stored locally or on Amazon S3 bucket for a quick review.(#196)

**aws-advanced-python-wrapper**

[aws-advanced-python-wrapper](https://aws-oss.beachgeek.co.uk/3xc) is complementary to and extends the functionality of an existing Python database driver to help an application take advantage of the features of clustered databases on AWS. It wraps the open-source Psycopg and the MySQL Connector/Python drivers and supports Python versions 3.8 or newer. You can install the aws-advanced-python-wrapper package using the pip command along with either the psycpg or mysql-connector-python open-source packages. The wrapper driver relies on monitoring database cluster status and being aware of the cluster topology to determine the new writer. This approach reduces switchover and failover times from tens of seconds to single digit seconds compared to the open-source drivers. Check the README for more details and example code on how to use this. (#199)

**build-neptune-graphapp-cdk**

[build-neptune-graphapp-cdk](https://aws-oss.beachgeek.co.uk/3z9) this repo provides a quick example on how to build a graph application with Amazon Neptune and AWS Amplify. (#200)

**cloudwatch-to-opensearch**

[cloudwatch-to-opensearch](https://aws-oss.beachgeek.co.uk/44v) provides sample code that shows you how to ingest Amazon CloudWatch logs into Amazon OpenSearch Serverless. While CloudWatch Logs excels at collecting and storing log data, OpenSearch Serverless provides more powerful search, analytics, and visualisation capabilities on that log data. This project implements a serverless pipeline to get the best of both services - using CloudWatch Logs for log aggregation, and OpenSearch Serverless for log analysis.(#204)

**config-rds-ca-expiry**

[config-rds-ca-expiry](https://aws-oss.beachgeek.co.uk/3z7) provides sample code to create a custom AWS Config rule to detect expiring CA certificates. Everyone loves TLS certs, but we all hate it when we realise that stuff has broken because they expired. It can happen to anyone, so check this out and make sure you are proactively managing your certs on your Amazon RDS instances, and how this is different to the out of the box notifications you already get with Amazon RDS. (#200)

**cruise-control-for-msk**

[cruise-control-for-msk](https://aws-oss.beachgeek.co.uk/3uq) is a repo that provides AWS CloudFormation templates that simplifies the deployment and management of Cruise Control and Prometheus for monitoring and rebalancing Amazon MSK clusters. Amazon MSK is a fully managed service that makes it easy to build and run applications that use Apache Kafka to process streaming data. With this new CloudFormation template, you can now integrate Cruise Control and Prometheus to gain deeper insights into your Amazon MSK cluster's performance and optimise resource utilisation. By automating the deployment and configuration of Cruise Control and Prometheus, you can improve operational efficiency, reduce the time and effort required for manual setup and maintenance, and allow you to focus on higher-value tasks. Check out the README for more details.(#196)

**da-top-monitoring**

[da-top-monitoring](https://aws-oss.beachgeek.co.uk/3sg)  ADTop Monitoring is lightweight application to perform real-time monitoring for AWS Data Analytics Resources. Based on same simplicity concept of Unix top utility, provide quick and fast view of performance, just all in one screen. (#194)

**db-top-monitoring**

[db-top-monitoring](https://aws-oss.beachgeek.co.uk/3ph)  is lightweight application to perform realtime monitoring for AWS Database Resources. Based on same simplicity concept of Unix top utility, provide quick and fast view of database performance, just all in one screen.  The README is very details and comprehensive, so if you are doing any sort of work with databases, and need to understand the performance characteristics, this is a project you should explore. (#191)

**dynamodb-parallel-scan**

[dynamodb-parallel-scan](https://aws-oss.beachgeek.co.uk/47l) is a new node library from shelfio that looks to speed up scanning your DynamoDB tables, through parallelism. Vlad Holubiev has put together a blog post, [How to Scan a 23 GB DynamoDB Table in One Minute](https://aws-oss.beachgeek.co.uk/47m) where he shares more details on how this library works, how to get started, as well as some benchmarks he has done.

**glide-for-redis**

[glide-for-redis](https://aws-oss.beachgeek.co.uk/3l2) or General Language Independent Driver for the Enterprise (GLIDE) for Redis (mayeb GLIDER would have been cooler :-) is a new open source client for Redis that works with any Redis distribution that adheres to the Redis Serialization Protocol (RESP) specification. The client is optimised for security, performance, minimal downtime, and observability, and comes pre-configured with best practices learned from over a decade of operating Redis-compatible services used by hundreds of thousands of customers. (#185)

**guidance-for-natural-language-queries-of-relational-databases-on-aws**

[guidance-for-natural-language-queries-of-relational-databases-on-aws](https://aws-oss.beachgeek.co.uk/337) this AWS Solution contains a demonstration of Generative AI, specifically, the use of Natural Language Query (NLQ) to ask questions of an Amazon RDS for PostgreSQL database. This solution offers three architectural options for Foundation Models: 1. Amazon SageMaker JumpStart, 2. Amazon Bedrock, and 3. OpenAI API. The demonstration's web-based application, running on Amazon ECS on AWS Fargate, uses a combination of LangChain, Streamlit, Chroma, and HuggingFace SentenceTransformers. The application accepts natural language questions from end-users and returns natural language answers, along with the associated SQL query and Pandas DataFrame-compatible result set.(#190)

**kafka-client-metrics-to-cloudwatch-with-kip-714**

[kafka-client-metrics-to-cloudwatch-with-kip-714](https://aws-oss.beachgeek.co.uk/407) provides reference code from my colleague Ricardo Ferreria, that shows how to push metrics from your Apache Kafka clients to Amazon CloudWatch using the KIP-714: Client Metrics and Observability. To use this feature, you must use a Kafka cluster with the version 3.7.0 or higher. It also requires the Kraft mode enabled, which is the new mode to run Kafka brokers without requiring Zookeeper. (#201)

**opensearch-for-gophers**

[opensearch-for-gophers](https://aws-oss.beachgeek.co.uk/3un) This project contains an example that showcases different features from the official Go Client for OpenSearch that you can use as a reference about how to get started with OpenSearch in your Go apps. It is not intended to provide the full spectrum of what the client is capable of—but it certainly puts you on the right track. You can run this code with an OpenSearch instance running locally, to which you can leverage the Docker Compose code available in the project. Alternatively, you can also run this code with Amazon OpenSearch that can be easily created using the Terraform code also available in the project. Nice README that provides useful examples to get you going.(#196)

**pinecone-db-construct**

[pinecone-db-construct](https://aws-oss.beachgeek.co.uk/3mn) The Pinecone DB Construct for AWS CDK is a JSII-constructed library that simplifies the creation and management of Pinecone indexes in your AWS infrastructure. It allows you to define, configure, and orchestrate your vector database resources alongside your AWS resources within your CDK application. The maintainer has shared some of the noteworthy features, which include:

* Handles CRUDs for both Pod and Serverless Spec indexes
* Deploy multiple indexes at the same time with isolated state management
* Adheres to AWS-defined removal policies (DESTROY, SNAPSHOT, etc.)
* Creates stack-scoped index names, to avoid name collisions

(#188)

**prometheus-rds-exporter**

[prometheus-rds-exporter](https://aws-oss.beachgeek.co.uk/3mx) is a project from Vincent Mercier that provides a Prometheus exporter for AWS RDS. Check out the README, it is very detailed and well put together. It provides a lot of information on how they built this, examples of configurations as well as detailed configuration options. (#188)


**rds-extended-support-cost-estimator**

[rds-extended-support-cost-estimator](https://aws-oss.beachgeek.co.uk/3rb) provides scripts can be used to help estimate the cost of RDS Extended Support for RDS instances & clusters in your AWS account and organisation. In September 2023, we announced Amazon RDS Extended Support, which allows you to continue running your database on a major engine version past its RDS end of standard support date on Amazon Aurora or Amazon RDS at an additional cost. These scripts should be run from the payer account of your organisation to identify the RDS clusters in your organisation that will be impacted by the extended support and the estimated additional cost. Check the README for additional details as to which database engines it will scan and provide estimations for. (#193)

**rds-instances-locator**

[rds-instances-locator](https://aws-oss.beachgeek.co.uk/3o0) There are times when you want to know exactly in which physical AZ your RDS instances are running. For those instances, you can use this script to help you. Details on the various scenarios it has been designed for as well as example commands make this something you can easily get started with.(#189)

**remote-debugging-with-emr**

[remote-debugging-with-emr](https://aws-oss.beachgeek.co.uk/3ne) is a Python-based CDK stack that deploys an EC2 bastion host and EMR Serverless and EMR on EKS environments configured for remote debugging. (#188)

**salmon**

[salmon](https://aws-oss.beachgeek.co.uk/41o) is a new open source solution from Soname Solutions that provides an alerting solution for your AWS based data pipelines, which has been designed with ease of use in mind. The solution focus' on AWS services such as Glue, Step Functions, EMR, and others. Check out the repo for details on how it works, as well as some examples to get you going. (#202)

**sparklepop**

[sparklepop](https://aws-oss.beachgeek.co.uk/3z2) is a simple Python package from Daniel B designed to check the free disk space of an AWS RDS instance. It leverages AWS CloudWatch to retrieve the necessary metrics. This package is intended for users who need a straightforward way to monitor disk space without setting up complex alerts. (#200)

**sso-sync-to-amazon-rds**

[sso-sync-to-amazon-rds](https://aws-oss.beachgeek.co.uk/3e2) This project sets up AWS Lambda functions, Amazon EventBridge rule, Amazon VPC Endpoint for AWS IAM Identity Center (successor to AWS Single Sign-On), the related Security Groups and permissions necessary to automatically provision database users to the Amazon Relational Database Service (Amazon RDS) cluster using AWS Cloud Development Kit (AWS CDK). When a new user is created in IAM Identity Center and the user belongs to the group specified in a IAM_IDC_GROUP_NAME variable, EventBridge rule will trigger the Lambda function. The Lambda function will create a new user in a specified Amazon RDS cluster. The user will then be able to login to the database using their SSO username and IAM credentials. Adding a user to the configured group will trigger the Lambda function as well. (#178)

**terraform-provider-mongodb**

[terraform-provider-mongodb](https://aws-oss.beachgeek.co.uk/4gh) is new repo from **AWS Community Builder Anuj Tyagi** that provides a new custom Terraform/OpenTofu provider for working with DocumentDB (as well as MongoDB). Anuj has also put together a nice post that looks at the motivation for building this and how you can get started, so check out [Automate DocumentDB Operations using custom Terraform provider](https://community.aws/content/2vhsRbjHTMTMrkchpr0QdHySsdC/automate-documentdb-operations-using-custom-terraform-provider) if this sounds like something useful to you. (#212)

**tokenizing-db-data-tool**

[tokenizing-db-data-tool](https://aws-oss.beachgeek.co.uk/3lp) provides a handy solution to help you address challenges around masking and keeping safe private or sensitive data (PII), as you need to move data from production to non production systems for activities like testing, debugging, load and integration tests, and more. (#186)


**tsynamo**

[tsynamo](https://aws-oss.beachgeek.co.uk/3td) is a project from that Olli Warro that simplifies the DynamoDB API so that you don't have to write commands with raw expressions and hassle with the attribute names and values. Moreover, Tsynamo makes sure you use correct types in your DynamoDB expressions, and the queries are nicer to write with autocompletion. Olli was inspired by another project ([Kysely](https://aws-oss.beachgeek.co.uk/3tm)), and so built this project so that he could do similar using Amazon DynamoDB. (#195)

**user-behavior-insights**

[user-behavior-insights](https://aws-oss.beachgeek.co.uk/3yx) This repository contains the OpenSearch plugin for the User Behavior Insights (UBI) capability. This plugin facilitates persisting client-side events (e.g. item clicks, scroll depth) and OpenSearch queries for the purpose of analyzing the data to improve search relevance and user experience.(#200)

**valkey-python-demo**

[valkey-python-demo](https://aws-oss.beachgeek.co.uk/41k) provides some sample code that shows you how you can connect to a Valkey server using three different types of client. Existing Redis clients, the Valkey client, and the all new GLIDE client too. I put together a quick blog post on how I put this code together, so check it out - [Using Amazon Q Developer to update Valkey client code](https://aws-oss.beachgeek.co.uk/41m) (#202)

**valkey-finch**

[valkey-finch](https://aws-oss.beachgeek.co.uk/41n) is a quick recipe on how to run Valkey in a container using Finch. It did not work out of the box for me, and I had to figure out how to get it working. Now you can save yourself the trouble and check out this configuration. I also put a short blog on this, so check out [Getting started with Valkey and Finch](https://aws-oss.beachgeek.co.uk/40u) (#202)

### Developer Tools & DevOps

**.NET Aspire**

[aspire](https://aws-oss.beachgeek.co.uk/3x1) Provides extension methods and resources definition for a .NET Aspire AppHost to configure the AWS SDK for .NET and AWS application resources. If you are not familiar with Aspire,  it is an opinionated, cloud ready stack for building observable, production ready, distributed applications in .NET. You can now use this with AWS resources, so check out the repo and the documentation that provides code examples and more.(#198)

**abc**

[abc](https://aws-oss.beachgeek.co.uk/47t) is an AI bash tool that integrates with Amazon Bedrock as a foundation model provider. This tool will generate shell command(s) from natural language description using which ever LLM (generative AI) you configure, and place on the next shell prompt for editing and execution. Plenty of details in the README on how to configure and run this, so give it a go if you are looking for something that integrates with Amazon Bedrock.(#206)

**alarm-context-tool**

[alarm-context-tool](https://aws-oss.beachgeek.co.uk/40g) enhances AWS CloudWatch Alarms by providing additional context to aid in troubleshooting and analysis. By leveraging AWS services such as Lambda, CloudWatch, X-Ray, and Amazon Bedrock, this solution aggregates and analyses metrics, logs, and traces to generate meaningful insights. Using generative AI capabilities from Amazon Bedrock, it summarises findings, identifies potential root causes, and offers relevant documentation links to help operators resolve issues more efficiently. The implementation is designed for easy deployment and integration into existing observability pipelines, significantly reducing response times and improving root cause analysis. (#201)

**amazon-bedrock-client-for-mac**

[amazon-bedrock-client-for-mac](https://aws-oss.beachgeek.co.uk/3um) this repo provides the code for the Amazon Bedrock Client for Mac is a macOS demo application built with SwiftUI. It serves as a client interface for AWS Bedrock, allowing users to interact with AWS Bedrock models.(#196)

**amazon-q-developer-cli**

[amazon-q-developer-cli](https://aws-oss.beachgeek.co.uk/466) is the repo that houses most of the core code for the Amazon Q Developer desktop app and CLI, adding autocomplete and AI to your existing terminal on macOS & Linux. (#205)

**amazon-q-developer-cli-webui**

[amazon-q-developer-cli-webui](https://aws-oss.beachgeek.co.uk/4f5) is a very nice project from **AWS Community Builder Gabriel Koo** that allows you to run Amazon Q CLI via a web ui. He put together a post that explains the itch he was trying to scratch, and I have to say that this is on my weekend list of projects to try out. Go read the post, [Why I Built a Web UI for Amazon Q Developer CLI (And How I Vibe-Coded It)](https://dev.to/aws-builders/why-i-built-a-web-ui-for-amazon-q-developer-cli-and-how-i-vibe-coded-it-54d6).(#211)

**amazon-q-vibes-memory-banking**

[amazon-q-vibes-memory-banking](https://aws-oss.beachgeek.co.uk/4fd) is from **AWS Serverless Community Builder Nicola Cremaschini** who shares his approach to using AI Coding Assistants like Amazon Q Developer, to provide more consistent outcomes. The Q-Vibes framework helps maintain context across AI assistant sessions through 5 lightweight files, enabling quick prototype development without losing momentum between sessions. (#211)

**amazon-sqs-python-extended-client-lib**

[amazon-sqs-python-extended-client-lib](https://aws-oss.beachgeek.co.uk/3ti) this repo (Amazon SQS Extended Client) allows clients to manage Amazon SQS message payloads that exceed the 256 KB message size limit, up to a size of 2 GB. In the event of publishing such large messages, the client accomplishes this feat by storing the actual payload in a S3 bucket and by storing the reference of the stored object in the SQS queue. Similarly, the extended-client is also used for retrieving and dereferencing these references of message objects stored in S3.  Check out the docs for more details on how this works and some sample code to get you going. (#195)

**amplify-hosting-astro**

[amplify-hosting-astro](https://aws-oss.beachgeek.co.uk/3r7) is a repo from AWS Amplify's Matt Auerbach that provides a walk through on how to build a simple blog using Astro's starter blog template, and deploy it using AWS Amplify Hosting. (#193)

**arctic**

[arctic](https://aws-oss.beachgeek.co.uk/4bz) is an open-source multi platform tool developed by the Corretto Team to automate interactive UI tests. Arctic supports existing tests and is agnostic to how those tests are written. Arctic can be used to validate any type of UI test, as it does not require any special support from the application side. Arctic relies on the operating system to capture all the required events during recording time and then reproduce them during replay time. This allows Arctic to operate with older tests that were not written with automation in mind without the need to modify them. Arctic runs on Linux, macOS (aarch64 and x86_64), and Windows (x86_64). (#210)

**awsid**

[awsid](https://aws-oss.beachgeek.co.uk/41p) is the latest novel project from Aidan Steele, author of many curiosities in the past, and this time (using his own words) he has created a tool that is "an incredibly niche tool, that might be of interest to half a dozen people. It turns AWS unique IDs into ARNs. I used generative AI to generate the UI." Check it out and see if you are part of that niche! (also make sure you check out the README about what he can potentially see) (#202)

**aws-advanced-nodejs-wrapper**

[aws-advanced-nodejs-wrapper](https://aws-oss.beachgeek.co.uk/465) is complementary to an existing NodeJS driver and aims to extend the functionality of the driver to enable applications to take full advantage of the features of clustered databases such as Amazon Aurora.  In other words, the AWS Advanced NodeJS Wrapper does not connect directly to any database, but enables support of AWS and Aurora functionalities on top of an underlying NodeJS driver of the user's choice. Hosting a database cluster in the cloud via Aurora is able to provide users with sets of features and configurations to obtain maximum performance and availability, such as database failover. However, at the moment, most existing drivers do not currently support those functionalities or are not able to entirely take advantage of it. The main idea behind the AWS Advanced NodeJS Wrapper is to add a software layer on top of an existing NodeJS driver that would enable all the enhancements brought by Aurora, without requiring users to change their workflow with their databases and existing NodeJS drivers. (#205)

**aws-advanced-python-wrapper**

[aws-advanced-python-wrapper](https://aws-oss.beachgeek.co.uk/3xc) is complementary to and extends the functionality of an existing Python database driver to help an application take advantage of the features of clustered databases on AWS. It wraps the open-source Psycopg and the MySQL Connector/Python drivers and supports Python versions 3.8 or newer. You can install the aws-advanced-python-wrapper package using the pip command along with either the psycpg or mysql-connector-python open-source packages. The wrapper driver relies on monitoring database cluster status and being aware of the cluster topology to determine the new writer. This approach reduces switchover and failover times from tens of seconds to single digit seconds compared to the open-source drivers. Check the README for more details and example code on how to use this. (#199)

**aws-appsync-events-swift**

[aws-appsync-events-swift](https://aws-oss.beachgeek.co.uk/4h2) provides a Swift library to interact with AWS AppSync Events API in your Apple iOS, macOS and tvOS applications.(#212)

**aws-cdk-imagebuilder-sample**

[aws-cdk-imagebuilder-sample](https://aws-oss.beachgeek.co.uk/3o2) this repo uses AWS CDK (TypeScript) that demonstrates how to create a fully functional ImageBuilder pipeline that builds an Amazon Linux 2023 container image, installing git, docker and nodejs, all the way to pushing the resulting image to an ECR repository.(#189)

**aws-cdk-stack-builder-tool**

[aws-cdk-stack-builder-tool](https://aws-oss.beachgeek.co.uk/3g3) or AWS CDK Builder, is a browser-based tool designed to streamline bootstrapping of Infrastructure as Code (IaC) projects using the AWS Cloud Development Kit (CDK). Equipped with a dynamic visual designer and instant TypeScript code generation capabilities, the CDK Builder simplifies the construction and deployment of CDK projects. It stands as a resource for all CDK users, providing a platform to explore a broad array of CDK constructs. Very cool indeed, and you can deploy on AWS Cloud9, so that this project on my weekend to do list. (#180)

**aws-client-monitor**

[aws-client-monitor](https://aws-oss.beachgeek.co.uk/496) is a new tool from Roman Tsypuk designed to analyse applications interacting with AWS services. It's particularly useful during local development, troubleshooting, or profiling third-party applications. It allows you to do real-time monitoring, capturing and displaying AWS API invocations in real time, and is easy to get started with. It comes with a gorgeous dashboard, so this is on my todo list to try out. (#207)

**aws-cloudformation-starterkit**

[aws-cloudformation-starterkit](https://aws-oss.beachgeek.co.uk/434) is a new project from AWS Community Builder Danny Steenman that should accelerate AWS infrastructure deployment for CloudFormation users. It's designed for both beginners and seasoned pros, featuring quick CI/CD setup, multi-environment support, and automated security checks. Very nice repo, clear and detailed documentation, so make sure you check this project out.(#203)

**aws-external-package-security**

[aws-external-package-security](https://aws-oss.beachgeek.co.uk/3g2) provides code to setup a solution that demonstrates how you can deploy AWS Code Services (e.g., AWS CodePipeline, AWS CodeBuild, Amazon CodeGuru Security, AWS CodeArtifact) to orchestrate secure access to external package repositories from an Amazon SageMaker data science environment configured with multi-layer security. The solution can also be expanded upon to account for general developer workflows, where developers use external package dependencies. (#180)

**aws-lambda-mcp-cookbook**

[aws-lambda-mcp-cookbook](https://aws-oss.beachgeek.co.uk/4fc) is a repository from **AWS Hero Ran Isenberg** that provides a working, deployable, open source-based, serverless MCP server blueprint with an AWS Lambda function and AWS CDK Python code with all the best practices and a complete CI/CD pipeline.  Checkout the README for details of how this is put together, and how to get started. (#211)

**aws-lambda-java-profiler**

[aws-lambda-java-profiler](https://aws-oss.beachgeek.co.uk/4a1) is an experimental (currently) release that provides a new capability that helps builders gain deep performance insights into their Lambda functions. The Lambda Profiler enables you to analyse your serverless applications with precision and ease. What can it do I can hear you all saying. It's current capabilities include:

* Generate detailed flame graphs for every function invocation
* Leverage low overhead sampling async-profiler technology with no safepoint bias
* Automatically store profiling data in your designated Amazon S3 bucket

Getting started is simple - just attach the Lambda layer to your function. No code changes required. This new capability helps you optimise cost and performance, identify bottlenecks, and deliver better experiences to your customers. It's another example of how we're listening to our builders and delivering the tools they need to innovate faster. (#208)

**aws-lint-iam-policies**

[aws-lint-iam-policies](https://aws-oss.beachgeek.co.uk/3pe)  runs IAM policy linting checks against either a single AWS account or all accounts of an AWS Organization. Reports on policies that violate security best practices or contain errors. Supports both identity-based and resource-based policies. Optionally dumps all policies analysed. The actual linting is performed by the AWS IAM Access Analyzer policy validation feature, which is mostly known for showing recommendations when manually editing IAM policies on the AWS Console UI. The repo provides additional blog posts to help you get started, as well as more details on how this works with supporting resources (#191)

**aws-iatk**

[aws-iatk](https://aws-oss.beachgeek.co.uk/3fc) AWS Integrated Application Test Kit (IATK), a new open-source test library that makes it easier for developers to create tests for cloud applications with increased speed and accuracy. With AWS IATK, developers can quickly write tests that exercise their code and its AWS integrations against an environment in the cloud, making it easier to catch mistakes early in the development process. IATK includes utilities to generate test events, validate event delivery and structure in Amazon EventBridge Event Bus, and assertions to validate call flow using AWS X-Ray traces. The [AWS IATK](https://aws-oss.beachgeek.co.uk/3g0) is available for Python3.8+. To help you get started, check out the supporting blog post from Dan Fox and Brian Krygsman, [Introducing the AWS Integrated Application Test Kit (IATK)](https://aws-oss.beachgeek.co.uk/3fz). (#180)

**aws-pdk**

[aws-pdk](https://aws-oss.beachgeek.co.uk/3jb) the AWS Project Development Kit (AWS PDK) is an open-source tool to help bootstrap and maintain cloud projects. It provides building blocks for common patterns together with development tools to manage and build your projects. The AWS PDK lets you define your projects programatically via the expressive power of type safe constructs available in one of 3 languages (typescript, python or java). Under the covers, AWS PDK is built on top of Projen. The AWS Bites Podcast provides an overview of the AWS Project Development Kit (PDK), and the hosts discuss what PDK is, how it can help generate boilerplate code and infrastructure, keep configuration consistent across projects, and some pros and cons of using a tool like this versus doing it manually. (#184)

**aws-rotate-key**

[aws-rotate-key](https://aws-oss.beachgeek.co.uk/3sd) is a project from AWS Community Builder  Stefan Sundin, that helps you implement security good practices around periodically regenerating your API keys. This command line tool simplifies the rotation of those access keys as defined in your local ~/.aws/credentials file. Check out the README for plenty of helpful info and examples of how you might use this. (#194)

**aws-scps-for-sandbox-and-training-accounts**

[aws-scps-for-sandbox-and-training-accounts](https://aws-oss.beachgeek.co.uk/3pf) is a collection of example Service Control Policies (SCPs) that are useful for sandbox and training AWS accounts. The SCPs deny API calls that change baseline account settings (contacts, billing, tax settings, etc.), have long-term financial effects (purchases and reservations) or operate outside allow-listed AWS regions or services. (#191)

**aws-sdk-python**

[aws-sdk-python](https://aws-oss.beachgeek.co.uk/4da) is a repo that contains **experimental async clients for the AWS SDK for Python**. These new clients will allow you to interact with select AWS services that can best utilise Python's async functionality. Unlike Boto3, these clients are distributed per-service, leaving you the option to pick what fits your needs. Please note that this new project is in early development and will be seeing rapid iteration over the coming months. This may mean instability in both public interfaces and general behaviours. Until the project releases version 1.0.0, breaking changes may occur between minor versions of the SDK. We'd strongly advise strict pinning to a version of the SDK for any non-experimental use cases.(#210)

**awsesh**

[awsesh](https://aws-oss.beachgeek.co.uk/4ad) is a command line tool for managing your AWS SSO sessions for those of you who are using AWS IAM Identity Centre (IdC). Whilst the repo provides pre built binaries for Mac, Linux, and Windows, you can also build from source too. There is a brief README with videos showing how this works - it is pretty simple to get up and running. If you are not using a vanity name for your IdC, then just use the prefix you see on the IdC dashboard when configuring this tool.(#208)

**aws-sdk-python-signers**

[aws-sdk-python-signers](https://aws-oss.beachgeek.co.uk/3x3) AWS SDK Python Signers provides stand-alone signing functionality. This enables users to create standardised request signatures (currently only SigV4) and apply them to common HTTP utilities like AIOHTTP, Curl, Postman, Requests and urllib3. This project is currently in an Alpha phase of development. There likely will be breakages and redesigns between minor patch versions as we collect user feedback. We strongly recommend pinning to a minor version and reviewing the changelog carefully before upgrading. Check out the README for details on how to use the signing module. (#198)

**aws-secret-inject**

[aws-secret-inject](https://aws-oss.beachgeek.co.uk/3pg) this handy command line tool from Quincy Mitchell allows you to inject AWS Secrets or SSM Parameters into your configuration files (.env, or whatever you like to call your configuration files these days). The README contains examples of how you can use this. Very handy indeed. (#191)

**aws-signer-oci-artifacts**

[aws-signer-oci-artifacts](https://aws-oss.beachgeek.co.uk/3km) this project is used to demonstrate how OCI artefacts can be signed and verified in a development pipeline. Zhuo-Wei Lee, Alontay Ellis, and Rajarshi Das have put together a blog post to help you get started, so if this project interests you, make sure you dive into [Signing and Validating OCI Artifacts with AWS Signer](https://aws-oss.beachgeek.co.uk/3kn).(#185)

**awsviz**

[awsviz](https://aws-oss.beachgeek.co.uk/3z1) is a super nice little tool from Bour Mohamed Abdelhadi, that helps you quickly visualy your IAM policies. You can check out the hosted version of [awsviz](https://aws-oss.beachgeek.co.uk/3z3) and there are some sample policies to show you what you can expect. Check out the[ use cases doc](https://aws-oss.beachgeek.co.uk/3z4) to see why you might want to try this tool out. (#200)

**bedrock-embed-web**

[bedrock-embed-web](https://aws-oss.beachgeek.co.uk/43i) is a new project from my colleague Olivier Leplus that makes it easier than ever to embed Amazon Bedrock foundation models within a chat interface in your web applications. (#203)

**bedrock-engineer**

[bedrock-engineer](https://aws-oss.beachgeek.co.uk/497) looks like an amazing project and one I am trying to find some time to play with. bedrock-engineer is Autonomous software development agent apps using Amazon Bedrock, capable of customise to create/edit files, execute commands, search the web, use knowledge base, use multi-agents, generative images and more. The project README has a short video that goes over some of the functionality and its pretty neat. (#207)

**bedrock-vscode-playground**

[bedrock-vscode-playground](https://aws-oss.beachgeek.co.uk/3nb) is a Visual Studio Code (VS Code) extension which allows developers to easily explore and experiment with large language models (LLMs) available in Amazon Bedrock. Check out the README for details of what you can do with it and how you can configure it to work with your specific setup.(#188)

**bedrust**

[bedrust](https://aws-oss.beachgeek.co.uk/3n1) is a demo repo from my colleague Darko Mesaros that shows you how you can use Amazon Bedrock in your Rust code, and allows you to currently choose between Claude V2, Llama2 70B, and Cohere Command.(#188)

**beta9**

[beta9](https://aws-oss.beachgeek.co.uk/3xu) is a self-hosted serverless framework that you can run in your AWS account. Think of AWS Lambda, but with GPUs and a Python-first developer experience. You can run workloads that instantly scale up to thousands of GPU containers running in parallel. The instances scale down automatically after each workload. You can also do things like deploy web endpoints, run task queues, and mount storage volumes for accessing large datasets. If you already have an EKS cluster, you can install Beta9 with a Helm chart. We think this would be a great way to save money on EC2 GPU resources while also getting a magical Python-first developer experience. If you have feedback or feature ideas, the maintainers would like to hear them.  (#199)

**bluesky-pds-cdk**

[bluesky-pds-cdk](https://aws-oss.beachgeek.co.uk/47x) if you are looking to deploy a self-hosted a fully containerized, serverless Bluesky Personal Data Server (PDS) on AWS, then this is the repo for you. It provides an opinionated AWS CDK construct that makes deploying this on AWS a breeze. (#206)

**csr-builder-for-kms**

[csr-builder-for-kms](https://aws-oss.beachgeek.co.uk/40h) provides a Python library for creating and signing X.509 certificate signing requests (CSRs) with KMS Keys. (#201)

**cdk-diff-action**

[cdk-diff-action](https://aws-oss.beachgeek.co.uk/4f0) from **Cory Hall**, provides a GitHub Action to run "cdk diff" on your PRs to track infrastructure changes.(#211)

**cdk-express-pipeline**

[cdk-express-pipeline](https://aws-oss.beachgeek.co.uk/43g) is the latest project from AWS Hero Rehan van der Merwe (who's project I use to track usage metrics of this newsletter!) which provides an alternative to those of you who want to use some of the features of AWS CDK Pipelines, but perhaps need it less opinionated. CDK Express Pipelines is a library built on the AWS CDK, allowing you to define pipelines in a CDK-native method. It leverages the CDK CLI to compute and deploy the correct dependency graph between Waves, Stages, and Stacks using the .addDependency method, making it build-system agnostic and an alternative to AWS CDK Pipelines. Check out the clear documentation which will help you get started in no time.(#203)

**cdk-notifier**

[cdk-notifier](https://aws-oss.beachgeek.co.uk/3it) is a lightweight CLI tool to parse a CDK log file and post changes to pull request requests. Can be used to get more confidence on approving pull requests because reviewer will be aware of changes done to your environments. I am not sure whether this is an old tool, but I have only just found out about it thanks to the blog post from AWS Community Builder, Johannes Konings. He put together [Use cdk-notifier to compare changes in pull requests](https://aws-oss.beachgeek.co.uk/3iu) that explains in more details how this works and walks you through using it. (#183)

**cdk-sops-secrets**

[cdk-sops-secrets](https://aws-oss.beachgeek.co.uk/4ex) helps you create secret values in AWS with infrastructure-as-code easily by providing a CDK construct library that facilitate syncing SOPS-encrypted secrets to AWS Secrets Manager and SSM Parameter Store. It enables secure storage of secrets in Git repositories while allowing seamless synchronisation and usage within AWS. Even large sets of SSM Parameters can be created quickly from a single file. Detailed README with plenty of examples of how you can use this. Very nice.(#211)

**cdk-vscode-server**

[cdk-vscode-server](https://aws-oss.beachgeek.co.uk/47j) is a new CDK construct from Manuel Vogel that provides a speed way to provision VSCode servers on AWS. Check out [his LinedIn post here](https://www.linkedin.com/posts/manuel-vogel_aws-cdk-cdkconstruct-ugcPost-7285010738505523201-YbLw/?utm_source=share&utm_medium=member_ios) for more details, as well as the detailed README. I have done this in the past with CloudFormation (check out my [gist here](https://gist.github.com/094459/0dc0eefcffbbc2c843e11e96940c2011)) but will be switching over to this construct from now on. (#206)

**cedar-go**

[cedar-go](https://aws-oss.beachgeek.co.uk/3qf) provides the Go implementation of the Cedar policy language. Check out the README for a quick example of how to use Cedar within your Go applications, and am looking forward to seeing how Go developers start to incorporate this into their applications.(#192)

**cfn-bedrock-notify**

[cfn-bedrock-notify](https://aws-oss.beachgeek.co.uk/3sc) is a new tool from my colleague Du'an Lightfoot that is a very creative an interesting way to incorporate large language models to help you troubleshoot failed Cloudformation deployments. How many times have you had a deployment fail, only to reach out to your preferred search tool to help solve the problem. Well with this project deployed, any issues are sent via SNS to Amazon Bedrock using Anthropic Claude v3, and (hopefully) the answer to your problems are returned via the email of the SNS subscriber. (#194)

**cfn-changeset-viewer**

[cfn-changeset-viewer](https://aws-oss.beachgeek.co.uk/3xy) is a tool all developers who work with and use AWS CloudFormation will want to check out. cfn-changeset-viewer is a CLI that will view the changes calculated in a CloudFormation ChangeSet in a more human-friendly way, including rendering details from a nested change set. Diffs are displayed in a logical way, making it easy to see changes, additions and deletions. Checkout the doc for more details and an example. (#199)

**cfn-pipeline**

[cfn-pipeline](https://aws-oss.beachgeek.co.uk/3kv) is a repo from Wolfgang Unger that contains an AWS Codepipeline that will allow automated Cloudformation deployments from within AWS Codepipeline. To help you get started, Wolfgang has put together a detailed blog post that includes videos. Go check it out, [Pipeline for automatic CloudFormation Deployments](https://aws-oss.beachgeek.co.uk/3kw) (#185)

**chaos-machine**

[chaos-machine](https://aws-oss.beachgeek.co.uk/4ao) is a complete chaos engineering workflow that enables customers to run controlled chaos experiments and test hypotheses related to system behaviour. Chaos Machine uses metric and alarm data from both Amazon CloudWatch and Prometheus as inputs to evaluate system behaviour before, during, and after the experiment. The Chaos Machine provides a simple, consistent way to organise and execute chaos experiments, and is appropriate to use for both building and conducting ad-hoc experiments or integrating into more sophisticated automation pipelines. Chaos Machine uses the AWS Fault Injection Service (FIS) to run controlled experiments, and AWS Step Functions and AWS Lambda for orchestration and execution.(#208)

**chorus**

[chorus](https://aws-oss.beachgeek.co.uk/4aq) is a new easy-to-use framework for building scalable solutions with LLM-driven multi-agent collaboration. Chorus allows you to develop and test solutions using multi-agent collaboration with zero or minimal coding. Chorus provides a multi-agent playground for easy visualisation and testing. Finally, Chorus helps you deploy your solution to Amazon Bedrock Agents with a single command (coming soon).(#208)

**cloudcatalog**

[cloudcatalog](https://aws-oss.beachgeek.co.uk/3mf) colleague David Boyne has put together another project, that is a fork of one his earlier projects ([EventCatalog](https://dev.to/aws/aws-open-source-news-and-updates-96-ig8)) that provides a similar capability, but this time helping you to document your AWS architecture. Check out the README for more details, including an example architecture that was documented. (#187)

**cloudfront-hosting-toolkit**

[cloudfront-hosting-toolkit](https://aws-oss.beachgeek.co.uk/3xv) is a new an open source command line tool to help developers deploy fast and secure frontends in the cloud. This project offers the convenience of a managed frontend hosting service while retaining full control over the hosting and deployment infrastructure to make it your own. The CLI simplifies AWS platform interaction for deploying static websites. It walks you through configuring a new repository, executing the deployment process, and provides the domain name upon completion. By following these steps, you effortlessly link your GitHub repository and deploy the necessary infrastructure, simplifying the deployment process. This enables you to focus on developing website content without dealing with the intricacies of infrastructure management. A few of my colleagues have tried this out and they are loving it. You can also find out more by reading the blog post, [Introducing CloudFront Hosting Toolkit](https://aws-oss.beachgeek.co.uk/3xw) where Achraf Souk, Corneliu Croitoru, and Cristian Graziano help you get started with a hands on guide to this project. (#199)

**cloudwatch-macros**

[cloudwatch-macros](https://aws-oss.beachgeek.co.uk/3gs) is the latest open source creation from AWS Hero Efi Merdler-Kravitz, focused on improving the CloudFormation and AWS SAM developer experience. This project features a collection of (basic at the moment) CloudFormation macros, written in Rust, offering seamless deployment through SAM. Check out [Efi's post on LinkedIn](https://aws-oss.beachgeek.co.uk/3gt) for more details and additional useful resources. (#181)

**cloudysetup**

[cloudysetup](https://aws-oss.beachgeek.co.uk/40e) is a CLI tool designed to streamline AWS resource management using AWS Cloud Control API. It leverages Amazon Bedrock fully managed service with Anthropic - Claude V2 Gen AI model to create, read, update, list, and delete AWS resources by generating configurations compatible with AWS Cloud Control API.(#201)

**codecatalyst-blueprints**

[codecatalyst-blueprints](https://aws-oss.beachgeek.co.uk/3kr) This repository contains common blueprint components, the base blueprint constructs and several public blueprints. Blueprints are code generators used to create and maintain projects in Amazon CodeCatalyst. (#185)

**codecatalyst-runner-cli**

[codecatalyst-runner-cli](https://aws-oss.beachgeek.co.uk/3up) This repository contains a command line tool that will allow you to run Amazon CodeCatalyst workflows locally. The README provides the instructions for quickly installing and getting started, so if  you have been using Amazon CodeCatalyst and looking for this, look no more.(#196)

**cumuli-aws-console-chat**

[cumuli-aws-console-chat](https://aws-oss.beachgeek.co.uk/3j2)  is an open source Chrome extension that provides similar functionality to Amazon Q. The Cumuli Chrome extension adds a side panel with LLM chat to all AWS pages, and lets you add screenshots of the console to your queries to get context-aware responses. It's similar to Amazon Q but uses GPT-4 Turbo with vision. Check out the repo that includes a demo video of it in action.(#183)

**db-top-monitoring**

[db-top-monitoring](https://aws-oss.beachgeek.co.uk/3ph)  is lightweight application to perform realtime monitoring for AWS Database Resources. Based on same simplicity concept of Unix top utility, provide quick and fast view of database performance, just all in one screen.  The README is very details and comprehensive, so if you are doing any sort of work with databases, and need to understand the performance characteristics, this is a project you should explore. (#191)

**deploy-time-build**

[deploy-time-build](https://aws-oss.beachgeek.co.uk/4ey) is an AWS CDK L3 construct that allows you to run a build job for specific purposes. Currently this library supports the following use cases: 1/ Build web frontend static files, 2/Build a container image, and 3/Build Seekable OCI (SOCI) indices for container images. Nice README with plenty of example code on how you can use against these use cases.(#211)

**diagram-as-code**

[diagram-as-code](https://aws-oss.beachgeek.co.uk/3ql) is a command line interface (CLI) tool enables drawing infrastructure diagrams for Amazon Web Services through YAML code. It facilitates diagram-as-code without relying on image libraries. The CLI tool promotes code reuse, testing, integration, and automating the diagramming process. It allows managing diagrams with Git by writing human-readable YAML. The README provides an example diagram (and the source that this tool used to generate it). (#192)

**draw-an-app**

[draw-an-app](https://aws-oss.beachgeek.co.uk/49c) is a neat demo application that uses AWS Bedrock's Claude 3 Sonnet model to convert hand-drawn UI sketches into working web applications. It supports two modes of operation: 1/ Real-time webcam capture for immediate sketching and conversion, or 2/ Image upload through a user-friendly Streamlit interface. I have not tried this one yet, but if you do, let me know how you get on.(#207)

**dynamomq**

[dynamomq](https://aws-oss.beachgeek.co.uk/3j1) is a project from Yuichi Watanabe that provides a message queuing library that leverages the features of DynamoDB to achieve high scalability, reliability, and cost efficiency. Notably, its ability to dynamically edit message order and attributes enables flexible adaptation to application requirements. Compared to existing solutions, DynamoMQ offers ease of management for developers while providing the reliability of fully managed services like Amazon SQS. It also encompasses key functionalities expected from a message queue, such as concurrent processing with multiple goroutines, Dead Letter Queues, and ensuring FIFO (First In, First Out) order. (#183)

**e1s**

[e1s](https://aws-oss.beachgeek.co.uk/3w1)  is a terminal application from [Xing Yahao](https://twitter.com/keidarcy) to easily browse and manage AWS ECS resources, supports both Fargate and EC2 ECS launch types. Inspired by k9s. e1s uses the default aws-cli configuration. It does not store or send your access and secret key anywhere. The access and secret key are used only to securely connect to AWS API via AWS SDK. e1s is available on Linux, macOS and Windows platforms. (#197)

**eks-saas-gitops**

[eks-saas-gitops](https://aws-oss.beachgeek.co.uk/3k1) This repository offers a sample pattern to manage multi-tenancy in a Kubernetes cluster using GitOps with Flux. The provided CloudFormation template automates the deployment of necessary AWS resources and sets up an environment ready for GitOps practices. (#184)

**eslint-cdk-plugin**

[eslint-cdk-plugin](https://aws-oss.beachgeek.co.uk/45z) provides an ESLint plugin for AWS CDK. ESLint is a static code analysis tool for identifying problematic patterns found in JavaScript code. This provides rules for the AWS CDK to help you write readable, maintainable, and reusable code. There are extensive rules provided ([eslint-rules](https://aws-oss.beachgeek.co.uk/461)) to get you going. Check out the project webpage [here](https://aws-oss.beachgeek.co.uk/460). (#205)

**gh-folder-dl**

[gh-folder-dl](https://aws-oss.beachgeek.co.uk/47i) is a handy tool from my colleague Denis Traub, a Python library and CLI tool for downloading files from GitHub repository folders, with recursive support and smart caching. It tracks downloaded files in a SQLite database to avoid re-downloading unchanged files and maintains the original directory structure. Denis has provided helpful examples of how to use this in the README, so go check it out. (#206)

**gh-relfind**

[gh-relfind](https://aws-oss.beachgeek.co.uk/3tf) is a project from AWS Community Builder Colin Duggan, that provides a simple Go CLI tool that uses Claude to parse package changes from GitHub repos release history. Significant change information is often contained in the release text field (body tag in the JSON response from ListReleases). Adding a semantic search makes it easier to extract and summarise actual change details. The project was built to get a better understanding of how to integrate with Claude through AWS Bedrock. (#195)

**git-remote-s3**

[git-remote-s3](https://aws-oss.beachgeek.co.uk/44s) is a neat tool that provides you with the ability to use Amazon S3 as a [Git Large File Storage (LFS)](https://git-lfs.com/) remote provider. It provides an implementation of a git remote helper to use S3 as a serverless Git server. The README provides good examples of how to set this up and example git commands that allow you to use this setup. This is pretty neat, and something I am going to try out for myself in future projects. (#204)

**kiro-steering-docs**

[kiro-steering-docs](https://aws-oss.beachgeek.co.uk/4gi) is a new repo from **AWS Hero Matthew Bonig** that shares Kiro steering files that are helpful for providing more concise and accurate responses when using AI Coding Assistants, in this case Kiro but they might work in other tools too. (#212)

**lambda_helpers_metrics**

[lambda_helpers_metrics](https://aws-oss.beachgeek.co.uk/40c) is a metrics helper library for AWS Lambda functions that provides the way to put metrics to the CloudWatch using the Embedded Metric Format ([EMF](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format_Specification.html)). Check out the supporting post, [AWS Lambda Rust EMF metrics helper](https://aws-oss.beachgeek.co.uk/40d). (#201)

**language-server-runtimes**

[language-server-runtimes](https://aws-oss.beachgeek.co.uk/3qe) is a JSON-RPC based protocol for interactions between servers and clients (typically embedded in development tools). The README covers details around specification support and features supported, that will help you tailor this to your needs.(#192)

**llm-test-mate**

[llm-test-mate](https://aws-oss.beachgeek.co.uk/47v) is a project from my colleague Danilo Poccia that is a simple testing framework to evaluate and validate LLM-generated content using string similarity, semantic. Check out the README for a list of features currently supported, as well as examples of how to use it. The README really is a thing of beauty, and I wish more projects provided as clear and detailed info as this project.(#206)

**llrt**

[llrt](https://aws-oss.beachgeek.co.uk/3mm) - Low Latency Runtime (LLRT) is a lightweight JavaScript runtime designed to address the growing demand for fast and efficient Serverless applications. LLRT offers up to over 10x faster startup and up to 2x overall lower cost compared to other JavaScript runtimes running on AWS Lambda. It's is built in Rust, utilising QuickJS as JavaScript engine, ensuring efficient memory usage and swift startup. (#188)

**load-test-llm-with-locust**

[load-test-llm-with-locust](https://aws-oss.beachgeek.co.uk/3qg) provides an example of how to perform load testing on the LLM API to evaluate your production requirements. The code is developed within a SageMaker Notebook and utilises the command line interface to conduct load testing on both the SageMaker and Bedrock LLM API. If you are not familiar with Locust, it is an open source load testing tool, and is a popular framework for load testing HTTP and other protocols. Its developer friendly approach lets you to define your tests in regular Python code. Locust tests can be run from command line or using its web-based UI. Throughput, response times and errors can be viewed in real time and/or exported for later analysis.(#192)

**localstack-aws-cdk-example**

[localstack-aws-cdk-example](https://aws-oss.beachgeek.co.uk/3dw) This repo aims to showcase the usage of [Localstack](https://aws-oss.beachgeek.co.uk/3dx) and AWS CDK to address specific integration challenges regarding local development where the end target is the AWS platform. If you are unfamiliar with Localstack, it is an open source, fully functional local AWS cloud stack that allows you to develop and test your cloud and Serverless apps offline. (#178)

**Log-Analyzer-with-MCP**

[Log-Analyzer-with-MCP](https://aws-oss.beachgeek.co.uk/4bu) - this repo provides a Model Context Protocol (MCP) server that provides AI assistants access to AWS CloudWatch Logs for analysis, searching, and correlation. The README provides everything you need to get started, as well as providing links that dive deeper into how this works. (#209)

**mcp**

[mcp](https://aws-oss.beachgeek.co.uk/4bi) is a comprehensive repo that provides a suite of specialised Model Context Protocol (MCP) servers that help you get the most out of AWS, wherever you use MCP. This list is growing, so check out the list (currently contains MCP servers that will help you with AWS documentation, AWS Cost Analysis, using AWS CDK, and more). (#209)

**MCP2Lambda**

[MCP2Lambda](https://aws-oss.beachgeek.co.uk/498) is a project from my good friend Danilo Poccia and is a really great example of how Model Control Protocol (MCP) provides Large Language Model (LLM) with additional capabilities and flexibility. In this demo sample, an MCP server acts as a bridge between MCP clients and AWS Lambda functions, allowing generative AI models to access and run Lambda functions as tools. This is useful, for example, to access private resources such as internal applications and databases without the need to provide public network access. This approach allows the model to use other AWS services, private networks, and the public internet.(#207)

**middy-mcp**

[middy-mcp](https://aws-oss.beachgeek.co.uk/4bk) provides [Middy](https://github.com/middyjs/middy) [middleware](https://middy.js.org/)  (a Node.js middleware engine for AWS Lambda that helps you organise your Lambda code, remove code duplication, and focus on business logic) for Model Context Protocol (MCP) server integration with AWS Lambda functions. It provides a convenient way to handle MCP requests and responses within your Lambda functions using the Middy middleware framework. It supports requests sent to AWS Lambda from API Gateway (both REST API / v1 and HTTP API / v2) using the Proxy integration, as well as requests sent form an ALB. Check out the README for some code examples of how you might use this. (#209)

**mkdocs-mcp**

[mkdocs-mcp](https://aws-oss.beachgeek.co.uk/4d2) is another project from **AWS Hero Michael Walmsley** that provides search functionality for any MkDocs powered site. This server relies on the existing MkDocs search implementation using the Lunr.Js search engine. (#210)

**orbits**

[orbits](https://aws-oss.beachgeek.co.uk/4gm) is an open-source framework for orchestrating long-lived resources and workflows involving long-running processes. It lets you manage your infrastructure templates from a single, unified place using native TypeScript code. By combining the flexibility of Node.js with a powerful workflow engine, Orbits helps you build modern, adaptable infrastructure operations. It’s ideal for automating processes that self-adapt to their environment and for exposing workflows as reusable, reliable building blocks. It also provides a solid foundation for building crash-resilient Node.js applications. **Louis Dussarps** provides you with a real use case of how you can use this tool in the blog post, [Solving cross-account resources for AWS CDK](https://aws-oss.beachgeek.co.uk/4gn) (#212)

**outtasync**

[outtasync](https://aws-oss.beachgeek.co.uk/3y0) helps users quickly identify the CloudFormation stacks that have gone out of sync with the state represented by their counterpart stack files. This can occur when someone updates a stack but fails to commit the latest stack file to the codebase. Alternatively, it may happen when a stack is updated on one deployment environment but not on others. Great documentation with examples and a video that provides everything you need to know. (#199)

**pagemosaic-website-starter**

[pagemosaic-website-starter](https://aws-oss.beachgeek.co.uk/3gp) is an open source tool from Alex Pust that helps you to host static websites on AWS, using AWS CDK under the covers from the looks of things. To deploy your website, simply transfer your website files to the /platform/web-app directory. Following this, execute the command pnpm deploy-platform to initiate the deployment process. Nice use of You Tube videos in the README to help you get started. (#181)

**powertools-mcp**

[powertools-mcp](https://aws-oss.beachgeek.co.uk/4cz) is a very nice project from **AWS Hero  Michael Walmsley** that provides search functionality for AWS Lambda Powertools documentation across multiple runtimes. This project implements an MCP server that enables Large Language Models (LLMs) to search through AWS Lambda Powertools documentation. It uses lunr.js for efficient local search capabilities and provides results that can be summarised and presented to users.  Good documentation, with examples on how to get started with MCP clients like Claude Desktop (but should work with Amazon Q CLI too) (#210)

**pristup**

[pristup](https://aws-oss.beachgeek.co.uk/3se) is along similar lines to the previous project, except this project from my colleague Dark Mesaros, provides a way to generate temporary AWS Console sign-in URLs. The purpose of this is to enable your users that do not have AWS Console access, temporary access to it without the need for a username and password. As with all of Darko's projects, excellent documentation and examples abound in the README. (#194)

**projen-vitest**

[projen-vitest](https://aws-oss.beachgeek.co.uk/47n) is a new component for Projen from Niko Virtala that helps you switch from Jest to Vitest in your Projen managed projects. Depending on your configuration, it should be more or less a drop-in replacement. Check out and follow [Niko here](https://bsky.app/profile/nikovirtala.io/post/3lejn3leh4k2b) (#206)

**promptus**

[promptus](https://aws-oss.beachgeek.co.uk/3mu) Prompt engineering is key for generating high-quality AI content. But crafting effective prompts can be time-consuming and difficult. That's why I built Promptus. Promptus allows you to easily create, iterate, and organise prompts for generative AI models. With Promptus, you can:

* Quickly build prompts with an intuitive interface
* Automatically version and compare prompt iterations to optimise quality
* Organize prompts into projects and share with teammates
* See a history of your prompt and easily go back to any previous prompt execution

(#188)

**python-bedrock-converse-generate-docs**

[python-bedrock-converse-generate-docs](https://aws-oss.beachgeek.co.uk/409) is a project from AWS Community Builder Alan Blockley that generates documentation for a given source code file using the Anthropic Bedrock Runtime API. The generated documentation is formatted in Markdown and stored in the specified output directory. Alan also put a blog together, [It’s not a chat bot: Writing Documentation](https://aws-oss.beachgeek.co.uk/40a), that shows you how it works and how to get started. The other cool thing about this project is that it is using the [Converse API](https://aws-oss.beachgeek.co.uk/40b) which you should check out if you have not already seen/used it. (#201)

**repository-migration-helper**

[repository-migration-helper](https://aws-oss.beachgeek.co.uk/3vu) is a Python CLI Helper tool to migrate Amazon CodeCommit repositories across AWS Accounts and Regions. Migrating CodeCommit repositories between AWS Accounts is a simple but repetitive process that can be automated for large-scale migrations. In this artefact, we share a Python script that provides a user-friendly interface to automate the migration of repositories in bulk. Using profiles configured for the AWS CLI, this tool makes it easy to move hundreds CodeCommit repositories in one command. The tool can also be used to migrate repositories between regions in one account when using the same profile for source and destination. First the script fetches the full list of CodeCommit repositories in the source account. Then the user is asked to filter and/or validate the list of repositories to be migrated to the destination account. For each of the selected repositories, it clones the source repository locally (as a mirror including files and metadata). The script then creates the target repository on the destination account with matching name (with an optional custom prefix) and description. The local clone is then pushed to the destination remote and removed from the local disk.(#197)

**rhubarb**

[rhubarb](https://aws-oss.beachgeek.co.uk/3vt) is a light-weight Python framework that makes it easy to build document understanding applications using Multi-modal Large Language Models (LLMs) and Embedding models. Rhubarb is created from the ground up to work with Amazon Bedrock and Anthropic Claude V3 Multi-modal Language Models, and Amazon Titan Multi-modal Embedding model. Rhubarb can perform multiple document processing and understanding tasks. Fundamentally, Rhubarb uses Multi-modal language models and multi-modal embedding models available via Amazon Bedrock to perform document extraction, summarisation, Entity detection, Q&A and more. Rhubarb comes with built-in system prompts that makes it easy to use it for a number of different document understanding use-cases. You can customise Rhubarb by passing in your own system and user prompts. It supports deterministic JSON schema based output generation which makes it easy to integrate into downstream applications. Looks super interesting, on my to do list and will report back my findings.(#197)

**rockhead-extensions**

[rockhead-extensions ](https://aws-oss.beachgeek.co.uk/3r5)another repo from a colleague, this time it is .NET aficionado Francois Bouteruche, who has put together this repo that provides code (as well as a nuget package) to make your .NET developer life easier when you invoke foundation models on Amazon Bedrock. More specifically, Francois has created a set of extension methods for the AWS SDK for .NET Bedrock Runtime client. It provides you strongly typed parameters and responses to make your developer life easier. (#193)

**rust-s3-cdn**

[rust-s3-cdn](https://aws-oss.beachgeek.co.uk/3j4) provides a Least Recently Used (LRU) cached proxy for AWS S3 written in Rust. I actually had to look up [LRU](https://helpful.knobs-dials.com/index.php/Cache_and_proxy_notes#Least_Recently_Used_(LRU)) as this was a new term for me. The repo provides a couple of reasons why you might want to use this tool, as well as helping you be aware of caveats too.(#183)

**s3-fast-list**

[s3-fast-list](https://aws-oss.beachgeek.co.uk/3k6) is a rust based tool that concurrently list Amazon S3 bucket with ListObjectsV2 API. Check out the README for use cases as to when s3-fast-list is going to help you out (against existing options you have) (#184)

**safeaws-cli**

[safeaws-cli](https://aws-oss.beachgeek.co.uk/3te) is a project from AWS Community Builder Gabriel Koo that provides an AWS CLI wrapper that helps you avoid common mistakes and pitfalls with Amazon Bedrock's Large Language Models, checking your command arguments against the command docs. safeaws-cli empowers you to execute AWS commands confidently by leveraging Amazon Bedrock's AI language models to analyze CLI commands, fetch help text, and identify potential issues or concerns before execution. By providing a safety net that mitigates pitfalls, safeaws-cli allows users to explore AWS securely, fostering a more informed approach to working with the CLI.(#195)

**sample-convert-codebase-to-graphrag**

[sample-convert-codebase-to-graphrag](https://aws-oss.beachgeek.co.uk/4ak) is a demo to show how to leverages AI to analyze, index, and query code repositories. It creates a searchable graph representation of code structures, enabling developers to explore and understand complex codebases efficiently. This project combines several AWS services, including Lambda, Neptune, OpenSearch, and Bedrock, to process code repositories, generate metadata, and provide powerful search capabilities. The system is designed to handle large-scale code analysis tasks and offer semantic code search functionality. It uses CDK to simplify how to deploy this in your own AWS environments. (#208)

**sample-devgenius-aws-solution-builder**

[sample-devgenius-aws-solution-builder](https://aws-oss.beachgeek.co.uk/4bx) is a super interesting project that you can deploy and provides an AI-powered application that transforms project ideas into complete, ready-to-deploy AWS solutions. It leverages Amazon Bedrock and Claude AI models to provide architecture diagrams, cost estimates, infrastructure as code, and comprehensive technical documentation. In "Conversational Solution Architecture Building" mode,  DevGenius enables customers to design solution architectures in a conversational manner. Users can create architecture diagrams (in draw.io format) and refine them interactively. Once the design is finalised, they can generate end-to-end code automation using CDK or CloudFormation templates, and deploy it in their AWS account with a single click. Additionally, customers can receive cost estimates for running the architecture in production, along with detailed documentation for the solution. In the "Build Solution Architecture from Whiteboard Drawings" mode, those of you who already have their architecture in image form (e.g., whiteboard drawings), DevGenius allows you to upload the image. Once uploaded, DevGenius analyses the architecture and provides a detailed explanation. You can then refine the design conversationally and, once finalised, generate end-to-end code automation using CDK or CloudFormation. Cost estimates and comprehensive documentation are also available.(#209)

**sample-ollama-server**

[sample-ollama-server](https://aws-oss.beachgeek.co.uk/4bv) is a project that folks who are interested in or already using ollama need to check out. This repo provides a AWS CloudFormation template to provision NVIDIA GPU EC2 instances with Ollama and Open WebUI, and include access to Amazon Bedrock foundation models (FMs). The solution can be deployed as a website for LLM interaction through Open WebUI, or as application development environment with Amazon DCV server.(#209)

**sbt-aws**

[sbt-aws](https://aws-oss.beachgeek.co.uk/3x8) SaaS Builder Toolkit for AWS (SBT) is an open-source developer toolkit to implement SaaS best practices and increase developer velocity. It offers a high-level object-oriented abstraction to define SaaS resources on AWS imperatively using the power of modern programming languages. Using SBT’s library of infrastructure constructs, you can easily encapsulate SaaS best practices in your SaaS application, and share it without worrying about boilerplate logic. The README contains all the resources you need to get started with this project, so if you are doing anything in the SaaS space, check it out. (#198)

**service-screener-v2**

[service-screener-v2](https://aws-oss.beachgeek.co.uk/3ol) Service Screener is a tool for AWS customers to analyse their AWS accounts against best practices for architecture. It provides an easy-to-use report with recommendations across various areas like cost optimisation and security, highlighting quick fixes that are cost-effective and downtime-free. Service Screener checks environments against the Well-Architected framework and other standards, such as the Foundational Technical Review and Startup Security Baseline, offering a comprehensive, stylish report that's cost-free and easy to understand, often running within minutes. Check out the README for lots of examples and explainer videos. (#190)

**sigv4a-signing-examples**

[sigv4a-signing-examples](https://aws-oss.beachgeek.co.uk/1jl) provides a useful set of examples that show examples of sigv4 implementations. Josh Hart, maintainer of this repo, has put together a companion repo that has a set of sigv4 examples in different frameworks and using the sdk or from scratch. Josh is also looking for community contributions in other frameworks, so if you are working on a framework and want to combine efforts, drop Josh a note.(#195)

**sfn-profiler**

[sfn-profiler ](https://aws-oss.beachgeek.co.uk/4bn)is a nice project from James Sanders that provides a package of utilities for profiling AWS Step Function executions. This utility provides relevant performance metrics and information about particular Step Function executions and their child workflows (called 'contributor' workflows) in your local web browser. It displays information such as the largest contributors to the overall duration as well as a gantt chart representation of the workflow execution. Check out the README for more details, this really is an essential tool if you spend any time at all developing AWS Step Functions. (#209)

**sparklepop**

[sparklepop](https://aws-oss.beachgeek.co.uk/3z2) is a simple Python package from Daniel B designed to check the free disk space of an AWS RDS instance. It leverages AWS CloudWatch to retrieve the necessary metrics. This package is intended for users who need a straightforward way to monitor disk space without setting up complex alerts. (#200)

**strands-ts**

[strands-ts](https://aws-oss.beachgeek.co.uk/4ed) is an experimental SDK for Strands for TypeScript developers from AWS Serverless Community Builder **Ryan Cormack**.  It is an AI generated migration of the Python SDK to TypeScript, using the same architecture and design principles. It is not a direct translation but rather a reimagining of the SDK in TypeScript, leveraging its features and idioms.(#211)

**stree**

[stree](https://aws-oss.beachgeek.co.uk/3o1) this project from Takafumi Miyanaga is a CLI tool designed to visualize the directory tree structure of an S3 bucket.
By inputting an S3 bucket/prefix and utilizing various flags to customize your request, you can obtain a colorized or non-colorized directory tree right in your terminal. Whether it's for verifying the file structure, sharing the structure with your team, or any other purpose, stree offers an easy and convenient way to explore your S3 buckets. (#189)

**StsSamlDriver**

[StsSamlDriver](https://aws-oss.beachgeek.co.uk/49d) is A Python-based SAML authentication handler for AWS STS that allows you to get temporary credentials using SAML to the AWS CLI, or an application written using an AWS SDK without the need to screen scrape or emulate a browser.(#207)

**tokenizing-db-data-tool**

[tokenizing-db-data-tool](https://aws-oss.beachgeek.co.uk/3lp) provides a handy solution to help you address challenges around masking and keeping safe private or sensitive data (PII), as you need to move data from production to non production systems for activities like testing, debugging, load and integration tests, and more. (#186)

**tsynamo**

[tsynamo](https://aws-oss.beachgeek.co.uk/3td) is a project from that Olli Warro that simplifies the DynamoDB API so that you don't have to write commands with raw expressions and hassle with the attribute names and values. Moreover, Tsynamo makes sure you use correct types in your DynamoDB expressions, and the queries are nicer to write with autocompletion. Olli was inspired by another project ([Kysely](https://aws-oss.beachgeek.co.uk/3tm)), and so built this project so that he could do similar using Amazon DynamoDB. (#195)

**type-safe-cdk-env**

[type-safe-cdk-env](https://aws-oss.beachgeek.co.uk/4f4) is from **AWS Community Builder Dakota Lewallen** and provides a TypeScript library that provides type-safe environment configuration for AWS CDK stacks using Zod for schema validation (Helper function to parse JSON files into environment variables within CDK stacks). Check out the README for example code and more details.(#211)

**vscode-on-ec2-for-prototyping**

[vscode-on-ec2-for-prototyping](https://aws-oss.beachgeek.co.uk/3lo) This repository introduces how to access and use VSCode hosted on EC2 from a browser. The connection is made via Session Manager, so IAM permissions are used for authentication. The access destination will be localhost. Please note that this repository does not introduce connecting from your local VSCode to an EC2 instance via Remote SSH. (#186)

**wide-logger**

[wide-logger](https://aws-oss.beachgeek.co.uk/3pi) is a canonical wide logger that is built to gather key, value pairs and then flush them all to the console in a single log message. This does not replace your existing detailed debug logging, it is an addition. All logs emitted by the Wide Logger will be prefixed by WIDE so you can quickly and easily find them or use filtered subscriptions to record these in a single place for easy searching and correlation. (#191)

**vscode-iam-service-principal-snippets**

[vscode-iam-service-principal-snippets](https://aws-oss.beachgeek.co.uk/495) is the latest VSCode plugin from AWS Community Builder Danny Steenman ([his fifth](https://www.linkedin.com/posts/dannysteenman_today-marks-a-small-but-significant-milestone-activity-7292070581011660800-iwbT/)). This VS Code extension provides autocompletion of all AWS services that can be used as Service Principals in your IAM policies. (#207)

**zero-downtime-deployment-tofu**

[zero-downtime-deployment-tofu](https://aws-oss.beachgeek.co.uk/438) is a repo from AWS Community Build Jorge Tovar that contains code examples using OpenTofu that shows how you can achieve zero downtime using a number of different approaches. Check out the supporting blog post for more details, [Zero Downtime Deployment in AWS with Tofu/Terraform and SAM](https://aws-oss.beachgeek.co.uk/439). This is this weeks essential repo to check out, and a good opportunity to learn about and become familiar with the different techniques and options you have. (#203)

**ziya**

[ziya](https://aws-oss.beachgeek.co.uk/44k) is a code assist tool for Amazon Bedrock models that can read your entire codebase and answer questions. The tool currently operates in Read only mode, but doing more that this is on the road map.(#204)

### Governance & Risk

**aft-account-suspend-close-solution**

[aft-account-suspend-close-solution](https://aws-oss.beachgeek.co.uk/41h) provides a sample solution that leverages AWS Control Tower Account Factory Terraform (AFT) to streamline the account closure and suspension process. The solution aims to provide a reliable, efficient, and fast way to manage the decommissioning of AWS accounts from organisations.(#202)

**alarm-context-tool**

[alarm-context-tool](https://aws-oss.beachgeek.co.uk/40g) enhances AWS CloudWatch Alarms by providing additional context to aid in troubleshooting and analysis. By leveraging AWS services such as Lambda, CloudWatch, X-Ray, and Amazon Bedrock, this solution aggregates and analyses metrics, logs, and traces to generate meaningful insights. Using generative AI capabilities from Amazon Bedrock, it summarises findings, identifies potential root causes, and offers relevant documentation links to help operators resolve issues more efficiently. The implementation is designed for easy deployment and integration into existing observability pipelines, significantly reducing response times and improving root cause analysis. (#201)

**appfabric-data-analytics**

[appfabric-data-analytics](https://aws-oss.beachgeek.co.uk/3k3) this project enables you to maintain logs from various SaaS applications and provides the ability to search and display log data. This solution leverages AWS AppFabric to create a data repository that you can query with Amazon Athena. While customers can obtain normalized and enriched SaaS audit log data (OCSF) from AppFabric, many prefer not only to forward these logs to a security tool. Some have the requirement to preserve logs for post-incident analysis, while others need to utilize the logs for tracking SaaS subscription and license usage. Additionally, some customers aim to analyze user activity to discover patterns. This project establishes a data pipeline that empowers customers to construct dashboards on top of it. (#184)

**automated-datastore-discovery-with-aws-glue**

[automated-datastore-discovery-with-aws-glue](https://aws-oss.beachgeek.co.uk/3x7) This sample shows you how to automate the discovery of various types of data sources in your AWS estate. Examples include - S3 Buckets, RDS databases, or DynamoDB tables. All the information is curated using AWS Glue - specifically in its Data Catalog. It also attempts to detect potential PII fields in the data sources via the Sensitive Data Detection transform in AWS Glue. This framework is useful to get a sense of all data sources in an organisation's AWS estate - from a compliance standpoint. An example of that could be GDPR Article 30. Check out the README for detailed architecture diagrams and a break down of each component as to how it works. (#198)

**aws-account-tag-association-imported-portfolios**

[aws-account-tag-association-imported-portfolios](https://aws-oss.beachgeek.co.uk/3o4) This repo provides a solution that is designed to automate associating account level tags to shared and local portfolios in the AWS environment which in turn inherits the tags to launched resources. AWS ServiceCatalog TagOption feature is used for this association.(#189)

**aws-cloudformation-starterkit**

[aws-cloudformation-starterkit](https://aws-oss.beachgeek.co.uk/434) is a new project from AWS Community Builder Danny Steenman that should accelerate AWS infrastructure deployment for CloudFormation users. It's designed for both beginners and seasoned pros, featuring quick CI/CD setup, multi-environment support, and automated security checks. Very nice repo, clear and detailed documentation, so make sure you check this project out.(#203)

**aws-control-tower-controls-cdk**

[aws-control-tower-controls-cdk](https://aws-oss.beachgeek.co.uk/3e3) This pattern describes how to use AWS CloudFormation and AWS Cloud Development Kit (AWS CDK) to implement and administer preventive, detective, and proactive AWS Control Tower controls as infrastructure as code (IaC). A control (also known as a guardrail) is a high-level rule that provides ongoing governance for your overall AWS Control Tower environment. For example, you can use controls to require logging for your AWS accounts and then configure automatic notifications if specific security-related events occur. Check out the REAMDE for more details on what you can do with this. (#178)

**aws-config-rule-rds-logging-enabled-remediation**

[aws-config-rule-rds-logging-enabled-remediation](https://aws-oss.beachgeek.co.uk/43l) provides code that will help you checks if respective logs of Amazon Relational Database Service (Amazon RDS) are enabled using AWS Config rule 'rds-logging-enabled'. The rule is NON_COMPLIANT if any log types are not enabled. AWS Systems Manager Automation document used as remediation action to enable the logs export to CloudWatch for the RDS instances that are marked NON_COMPLIANT.(#203)

**aws-cost-explorer-mcp-server**

[aws-cost-explorer-mcp-server](https://aws-oss.beachgeek.co.uk/4ac) builds on the current wave of excitement around Model Context Protocol (MCP) that provides a powerful open-source tool that brings natural language querying to your AWS spending data. Ask questions like "What was my EC2 spend yesterday?" or "Break down my Bedrock usage by model" and get detailed answers through Claude. It helps track cloud costs across services, identifies expensive resources, and provides deep insights into your Amazon Bedrock model usage. (#208)


**aws-external-package-security**

[aws-external-package-security](https://aws-oss.beachgeek.co.uk/3g2) provides code to setup a solution that demonstrates how you can deploy AWS Code Services (e.g., AWS CodePipeline, AWS CodeBuild, Amazon CodeGuru Security, AWS CodeArtifact) to orchestrate secure access to external package repositories from an Amazon SageMaker data science environment configured with multi-layer security. The solution can also be expanded upon to account for general developer workflows, where developers use external package dependencies. (#180)

**aws-finops-dashboard**

[aws-finops-dashboard](https://aws-oss.beachgeek.co.uk/4d3) is a terminal-based AWS cost and resource dashboard from **Ravi Kiran Vallamkonda** built with Python and the Rich library. It provides multi-account cost summaries by time period, service, and cost allocation tags; budget limits vs. actuals; EC2 instance status; six‑month cost trend charts; and “FinOps audit” reports (e.g. untagged or idle resources). It can export data to CSV/JSON/PDF.(#210)

**aws-health-events-insight**

[aws-health-events-insight](https://aws-oss.beachgeek.co.uk/48z) this project provides a very nice solution to centralise, store and then analyse your AWS Health events. Health Events Intelligence Dashboards and Insights (HEIDI) Data Collection Framework enables you to collect data from different accounts, regions and organisations. Check out the README for more details  including how to deploy and configure this solution in your AWS environment.(#207)

**aws-organizations-tag-inventory**

[aws-organizations-tag-inventory](https://aws-oss.beachgeek.co.uk/3jz)  This project provides a solution to AWS customers for reporting on what tags exists, the resources they are applied to, and what resources don't have tags across their entire AWS organization. The solution is designed to be deployed in an AWS Organization with multiple accounts. Detailed information and deployment guidelines are in the README, including some sample dashboards so you can see what you can expect.(#184)

**aws-rotate-key**

[aws-rotate-key](https://aws-oss.beachgeek.co.uk/3sd) is a project from AWS Community Builder  Stefan Sundin, that helps you implement security good practices around periodically regenerating your API keys. This command line tool simplifies the rotation of those access keys as defined in your local ~/.aws/credentials file. Check out the README for plenty of helpful info and examples of how you might use this. (#194)

**aws-scps-for-sandbox-and-training-accounts**

[aws-scps-for-sandbox-and-training-accounts](https://aws-oss.beachgeek.co.uk/3pf) is a collection of example Service Control Policies (SCPs) that are useful for sandbox and training AWS accounts. The SCPs deny API calls that change baseline account settings (contacts, billing, tax settings, etc.), have long-term financial effects (purchases and reservations) or operate outside allow-listed AWS regions or services. (#191)

**aws-summarize-account-activity**

[aws-summarize-account-activity](https://aws-oss.beachgeek.co.uk/3pd) helps you to analyse CloudTrail data of a given AWS account and generates a summary of recently active IAM principals, API calls they made and regions that were used. The summary is written to a JSON output file and can optionally be visualised as PNG files. Michael has put together a couple of supporting blog posts for this project too. (#191)

**aws-waf-for-event-analysis-dashboard**

[aws-waf-for-event-analysis-dashboard](https://aws-oss.beachgeek.co.uk/3ls) finding the information you need during security incidents is what this project aims to help with. During major online events like live broadcasts, security teams need a fast and clear understanding of attack patterns and behaviours to distinguish between normal and malicious traffic flows. The solution outlined here allows filtering flow logs by "Client IP", "URI", "Header name", and "Header value" to analyse these fields and pinpoint values specifically associated with attack traffic versus normal traffic. For example, the dashboard can identify the top header values that are atypical for normal usage. The security team can then create an AWS WAF rule to block requests containing these header values, stopping the attack. This project demonstrates using AWS Glue crawlers to categorise and structure WAF flow log data and Amazon Athena for querying. Amazon Quicksight is then employed to visualise query results in a dashboard. Once deployed, the dashboard provides traffic visualisation similar to the example graphs shown in Images folder in under project , empowering security teams with insight into attacks and defence.(#186)

**awskillswitch**

[awskillswitch](https://aws-oss.beachgeek.co.uk/3gr) is an open sourced tool from Jeffrey Lyon that is worth checking out. AWS Kill Switch is a Lambda function (and proof of concept client) that an organisation can implement in a dedicated "Security" account to give their security engineers the ability to delete IAM roles or apply a highly restrictive service control policy (SCP) on any account in their organisation. Make sure you check out the README for full details, but this looks like it might be one of those tools that are useful to have in the back pocket in times of need. (#181)

**awsesh**

[awsesh](https://aws-oss.beachgeek.co.uk/4ad) is a command line tool for managing your AWS SSO sessions for those of you who are using AWS IAM Identity Centre (IdC). Whilst the repo provides pre built binaries for Mac, Linux, and Windows, you can also build from source too. There is a brief README with videos showing how this works - it is pretty simple to get up and running. If you are not using a vanity name for your IdC, then just use the prefix you see on the IdC dashboard when configuring this tool.(#208)

**chaos-machine**

[chaos-machine](https://aws-oss.beachgeek.co.uk/4ao) is a complete chaos engineering workflow that enables customers to run controlled chaos experiments and test hypotheses related to system behaviour. Chaos Machine uses metric and alarm data from both Amazon CloudWatch and Prometheus as inputs to evaluate system behaviour before, during, and after the experiment. The Chaos Machine provides a simple, consistent way to organise and execute chaos experiments, and is appropriate to use for both building and conducting ad-hoc experiments or integrating into more sophisticated automation pipelines. Chaos Machine uses the AWS Fault Injection Service (FIS) to run controlled experiments, and AWS Step Functions and AWS Lambda for orchestration and execution.(#208)

**cloudwatch-logs-analyzer**

[cloudwatch-logs-analyzer](https://aws-oss.beachgeek.co.uk/4f2) is an interesting tool from **AWS Community Builder Hafiz Syed Ashir Hassan** that uses Amazon Agent Strands to analyse CloudWatch logs, identify errors, and provide solutions based on either its own knowledge or a connected knowledge base. He has put together a blog post to help you get started, [Automating CloudWatch Log Analysis with Amazon Strands Agent: Meet the CloudWatch Analyzer](https://aws-oss.beachgeek.co.uk/4f3). (#211)

**cloudcatalog**

[cloudcatalog](https://aws-oss.beachgeek.co.uk/3mf) colleague David Boyne has put together another project, that is a fork of one his earlier projects ([EventCatalog](https://dev.to/aws/aws-open-source-news-and-updates-96-ig8)) that provides a similar capability, but this time helping you to document your AWS architecture. Check out the README for more details, including an example architecture that was documented. (#187)

**CloudGrappler**

[CloudGrappler](https://aws-oss.beachgeek.co.uk/3qb) is a purpose-built tool designed for effortless querying of high-fidelity and single-event detections related to well-known threat actors in AWS. Andi Ahmeti has put together a blog post, [Introducing CloudGrappler: A Powerful Open-Source Threat Detection Tool for Cloud Environments](https://aws-oss.beachgeek.co.uk/3qc), that provides an overview of how this works with examples.(#192)

**cloudysetup**

[cloudysetup](https://aws-oss.beachgeek.co.uk/40e) is a CLI tool designed to streamline AWS resource management using AWS Cloud Control API. It leverages Amazon Bedrock fully managed service with Anthropic - Claude V2 Gen AI model to create, read, update, list, and delete AWS resources by generating configurations compatible with AWS Cloud Control API.(#201)

**containers-cost-allocation-dashboard**

[containers-cost-allocation-dashboard](https://aws-oss.beachgeek.co.uk/3x9) provides everything you need to create a QuickSight dashboard for containers cost allocation based on data from Kubecost. The dashboard provides visibility into EKS in-cluster cost and usage in a multi-cluster environment, using data from a self-hosted Kubecost pod. The README contains additional links to resources to help you understand how this works, dependencies, and how to deploy and configure this project.(#198)

**create-and-delete-ngw**

[create-and-delete-ngw](https://aws-oss.beachgeek.co.uk/3x2) This project contains source code and supporting files for a serverless application that allocates an Elastic IP address, creates a NAT Gateway, and adds a route to the NAT Gateway in a VPC route table. The application also deletes the NAT Gateway and releases the Elastic IP address. The process to create and delete a NAT Gateway is orchestrated by an AWS Step Functions State Machine, triggered by an EventBridge Scheduler. The schedule can be defined by parameters during the SAM deployment process.(#198)

**diagram-as-code**

[diagram-as-code](https://aws-oss.beachgeek.co.uk/3ql) is a command line interface (CLI) tool enables drawing infrastructure diagrams for Amazon Web Services through YAML code. It facilitates diagram-as-code without relying on image libraries. The CLI tool promotes code reuse, testing, integration, and automating the diagramming process. It allows managing diagrams with Git by writing human-readable YAML. The README provides an example diagram (and the source that this tool used to generate it). (#192)

**duplicate-rule-detection-tool**

[duplicate-rule-detection-tool](https://aws-oss.beachgeek.co.uk/3oq) is a project to assess the current active AWS Config rules with potential duplicate scope in an AWS account. Our goal is to help customers can make informed decisions on how to streamline their AWS Config rules and reduce complexity. Plenty of examples and detailed breakdown of how this works in the README, so give it a look. (#190)

**ebsight**
                                                                 
[ebsight](https://aws-oss.beachgeek.co.uk/490) is a Python-based analysis tool developed by Stephen Jones in response to AWS's new EBS snapshot size reporting feature. This tool leverages the newly introduced FullSnapshotSizeInBytes field to provide comprehensive insights into EBS volume usage, performance metrics, and cost optimisation opportunities. After you have checked the repo out and gone through the README, why not read the supporting blog post, [Unleash the Power of EBSight for Optimal AWS Storage Management](https://aws-oss.beachgeek.co.uk/491) (#207)

**ec2RuntimeMonitor**

[ec2RuntimeMonitor](https://aws-oss.beachgeek.co.uk/3ra) EC2 runtime monitor is a serverless solution to get a notification when an EC2 instance is running for a time exceeding a user defined threshold. The README covers use cases why you might find this useful, but principally cost optimisation as well as reducing your carbon footprint are two key reasons why this might be a useful tool to keep your toolkit. (#193)

**ha-aws-cost**

[ha-aws-cost](https://aws-oss.beachgeek.co.uk/44i)  is a project from Diego Marques for folks who use [Home Assistant](https://www.home-assistant.io/) (a very popular open source home automation system), and provides custom component interacts with AWS to get the month to date cost and forecast cost and make it available on Home Assistant. It adds two new entities 1/Month to date cost: The current cost of your AWS account, and 2/Forecasted month costs: The forecasted cost based in your current consumption. Check out Diego's[ post on LinkedIn](https://www.linkedin.com/posts/diego7marques_aws-cost-my-first-home-assistant-integration-activity-7239741496013340672-cCSY/) that provides some more background to this project.(#204)

**innovation-sandbox-on-aws**

[innovation-sandbox-on-aws](https://aws-oss.beachgeek.co.uk/4g6) - The Innovation Sandbox on AWS solution allows cloud administrators to set up and recycle temporary sandbox environments by automating the implementation of security and governance policies, spend management mechanisms, and account recycling preferences through a web user interface (UI). Using the solution, customers can empower their teams to experiment, learn, and innovate with AWS services in production-isolated AWS accounts that are recycled after use.You can find out more, including a detailed implementation guide by checking out the homepage, [Innovation Sandbox on AWS](https://aws-oss.beachgeek.co.uk/4g7) (#212)

**integrate-control-tower-with-ipam**

[integrate-control-tower-with-ipam](https://aws-oss.beachgeek.co.uk/41f) This project implements a solution which integrates Amazon IP Address Management (IPAM) within AWS Control Tower through the use of Lifecycle Events. It presents the architecture view and shows how this solution extends your AWS Control Tower environment with Amazon IPAM to allow teams to access IPAM pools for their workload accounts.(#202)

**my-steampipe-arsenal**

[my-steampipe-arsenal](https://aws-oss.beachgeek.co.uk/436) Sometimes GitHub Gists or snippets are super useful, and Gabriel Soltz shares one such useful snippet in his repo of Steampipe queries that allows you to check for inactive Route53 alias'. Check out some of the other ones he shares too. (#203)

**networking-costs-calculator**

[networking-costs-calculator](https://aws-oss.beachgeek.co.uk/44t) provide a useful sample code for a networking costs calculator, helping to estimate the networking costs such as Data Transfer, Transit Gateway Attachments, NAT Gateways etc. The calculator has two main components: a serverless backend part, that uses the AWS Price List Query APIs to get the updated prices for the relevant networking services. These prices are cached in a DynamoDB table, and a ReactJS frontend web application, that is the user interface for estimating the costs for various networking services (hosted with S3 and CloudFront).(#204) 

**observation-extractor**

[observation-extractor](https://aws-oss.beachgeek.co.uk/4at) is a tool for collecting observations from data. Observations are useful bits of data related to questions that you define that is extracted from the data you pass in. Use Observation Extractor to process pdf (and maybe someday other files) into formats like csv (and later parquet) to turn unstructured documents into structured observations that you can query and use directly or through your application. When you output to a format like csv or parquet, observations are the row level records.

Observation Extractor takes an unstructured data file as input (like a pdf) and outputs a list of Observation objects. Each observation includes standard fields that are extracted from the document together with metadata like the document name and page number.

You can populate observations into a datastore and make them available to your human and AI users. They can be queried based on metadata like date and the specific questions they relate too. You can define question sets that represent thought process of a subject-matter-expert coming up to speed on this case to start mapping a document into useful observations.(#208)

**openops**

[openops](https://aws-oss.beachgeek.co.uk/4d4)  is a No-Code FinOps automation platform that helps organisations reduce cloud costs and streamline financial operations. It provides customisable workflows to automate key FinOps processes like allocation, unit economics, anomaly management, workload optimisation, safe de-provisioning and much, much more. It is not limited to AWS, and you can use it across your broader technical stack.(#210)

**orgs-prescriptive-guidance**

[orgs-prescriptive-guidance](https://aws-oss.beachgeek.co.uk/41g) This repository contains a collection of AWS CloudFormation templates to create up an AWS Organizations structure. So if you are looking to implement this, or are curious and want to dig into the code to find out more, check out this repo. (#202)

**powerpipe**

[powerpipe](https://aws-oss.beachgeek.co.uk/3qd) is dashboards and benchmarks as code. Use it to visualise any data source, and run compliance benchmarks and controls, for effective decision-making and ongoing compliance monitoring. As with all the Turbot open source projects, excellent documentation, and they have included a video that provides a demo of this at work. (#192)

**pristup**

[pristup](https://aws-oss.beachgeek.co.uk/3se) is along similar lines to the previous project, except this project from my colleague Dark Mesaros, provides a way to generate temporary AWS Console sign-in URLs. The purpose of this is to enable your users that do not have AWS Console access, temporary access to it without the need for a username and password. As with all of Darko's projects, excellent documentation and examples abound in the README. (#194)

**rds-extended-support-cost-estimator**

[rds-extended-support-cost-estimator](https://aws-oss.beachgeek.co.uk/3rb) provides scripts can be used to help estimate the cost of RDS Extended Support for RDS instances & clusters in your AWS account and organisation. In September 2023, we announced Amazon RDS Extended Support, which allows you to continue running your database on a major engine version past its RDS end of standard support date on Amazon Aurora or Amazon RDS at an additional cost. These scripts should be run from the payer account of your organisation to identify the RDS clusters in your organisation that will be impacted by the extended support and the estimated additional cost. Check the README for additional details as to which database engines it will scan and provide estimations for. (#193)

**s3-restore-and-copy-progress-monitoring**

[s3-restore-and-copy-progress-monitoring](https://aws-oss.beachgeek.co.uk/3o3) this is a very comprehensive and polished repo that provides an example of how you can restore data that you have stored in S3, providing you a single visualised dashboard to monitor the restore and copy progress within a defined scope.(#189)

**sample-service-quotas-replicator-for-aws**

[sample-service-quotas-replicator-for-aws](https://aws-oss.beachgeek.co.uk/4d5) is a code repo that contains sample code for the AWS Quota Replicator (AQR) tool, which demonstrates how to build a solution for comparing and managing service quotas across AWS accounts and regions. This tool was proudly built with the assistance of AWS Q Developer by **Kirankumar Chandrashekar** and **Gopinath Jagadesan**.   This tool hopes to simplify one of the most challenging aspects of AWS multi-account management: service quotas. The AWS Service Quotas Replicator helps you: 1/ Compare quotas across accounts/regions with visual indicators, 2/ Request quota increases with just a few clicks, 3/ Track request status in real-time, and 4/ Identify critical quota gaps before they impact your workloads. (#210)

**service-screener-v2**

[service-screener-v2](https://aws-oss.beachgeek.co.uk/3ol) Service Screener is a tool for AWS customers to analyse their AWS accounts against best practices for architecture. It provides an easy-to-use report with recommendations across various areas like cost optimisation and security, highlighting quick fixes that are cost-effective and downtime-free. Service Screener checks environments against the Well-Architected framework and other standards, such as the Foundational Technical Review and Startup Security Baseline, offering a comprehensive, stylish report that's cost-free and easy to understand, often running within minutes. Check out the README for lots of examples and explainer videos. (#190)

**tailpipe**

[tailpipe](https://aws-oss.beachgeek.co.uk/492) is an open source SIEM for instant log insights from our friends at Turbot, powered by DuckDB. It allows you to analyse millions of events in seconds, right from your terminal. Check out the README that includes more info include a nice video demo of this in works. Bob Tordella (who we have featured many times over the years in this newsletter), has put together a blog post too, which is a must read. Go check it out - [Query AWS CloudTrail Logs Locally with SQL](https://aws-oss.beachgeek.co.uk/494) (#207)


**terraform-aws-vulne-soldier**

[terraform-aws-vulne-soldier](https://aws-oss.beachgeek.co.uk/47f) is a new project from Victor Omolayo that provides an AWS EC2 vulnerability remediation tool designed to automate the process of patching nodes managed by AWS Systems Manager. He has put together a blog post, [vulne-soldier: AWS EC2 Vulnerability Remediation Tool](https://aws-oss.beachgeek.co.uk/47g) that provides more detail on the project and background as to why he created it as well as how to get started. (#206)

### Java, Kotlin, Scala, OpenJDK

**java-on-aws**

[java-on-aws](https://aws-oss.beachgeek.co.uk/3lx) is a fantastic resource for all Java developers who want to dive deeper on how to deploy their Java applications on AWS. Taking a sample application, the workshop looks at how you can containerise it, and then deploy it across a number of different compute environments - from serverless to Kubernetes.(#187) 

**powertools-lambda-kotlin**

[powertools-lambda-kotlin](https://aws-oss.beachgeek.co.uk/3dv) This project demonstrates the Lambda for Powertools Kotlin module deployed using Serverless Application Model with Gradle running the build. This example is configured for Java 8 only; in order to use a newer version, check out the Gradle configuration guide in the main project README. You can also use sam init to create a new Gradle-powered Powertools application - choose to use the AWS Quick Start Templates, and then Hello World Example with Powertools for AWS Lambda, Java 17 runtime, and finally gradle. (#178)

**serverless-java-container**

[serverless-java-container](https://aws-oss.beachgeek.co.uk/3my) this repo provides a Java wrapper to run Spring, Spring Boot, Jersey, and other apps inside AWS Lambda. Serverless Java Container natively supports API Gateway's proxy integration models for requests and responses, you can create and inject custom models for methods that use custom mappings. Check out the supporting blog post from Dennis Kieselhorst, [Re-platforming Java applications using the updated AWS Serverless Java Container](https://aws-oss.beachgeek.co.uk/3mz). (#188)

### Networking

**create-and-delete-ngw**

[create-and-delete-ngw](https://aws-oss.beachgeek.co.uk/3x2) This project contains source code and supporting files for a serverless application that allocates an Elastic IP address, creates a NAT Gateway, and adds a route to the NAT Gateway in a VPC route table. The application also deletes the NAT Gateway and releases the Elastic IP address. The process to create and delete a NAT Gateway is orchestrated by an AWS Step Functions State Machine, triggered by an EventBridge Scheduler. The schedule can be defined by parameters during the SAM deployment process.(#198)

**eks-shared-subnets**

[eks-shared-subnets](https://aws-oss.beachgeek.co.uk/3k2) this sample code demonstrates how a central networking team in an enterprise can create and share the VPC Subnets from their own AWS Account with other Workload Specific accounts. So that, Application teams can deploy and manage their own EKS clusters and related resources in those Subnets. (#184)

**networking-costs-calculator**

[networking-costs-calculator](https://aws-oss.beachgeek.co.uk/44t) provide a useful sample code for a networking costs calculator, helping to estimate the networking costs such as Data Transfer, Transit Gateway Attachments, NAT Gateways etc. The calculator has two main components: a serverless backend part, that uses the AWS Price List Query APIs to get the updated prices for the relevant networking services. These prices are cached in a DynamoDB table, and a ReactJS frontend web application, that is the user interface for estimating the costs for various networking services (hosted with S3 and CloudFront).(#204) 

**route53-hostedzone-migrator**

[route53-hostedzone-migrator](https://aws-oss.beachgeek.co.uk/3nc) is a handy script will help you to automate the migration of an AWS Route 53 hosted zone from an AWS account to another one. It will follow all the needed steps published in the official AWS Route 53 documentation regarding the migration of a hosted zone.(#188)

**rust-s3-cdn**

[rust-s3-cdn](https://aws-oss.beachgeek.co.uk/3j4) provides a Least Recently Used (LRU) cached proxy for AWS S3 written in Rust. I actually had to look up [LRU](https://helpful.knobs-dials.com/index.php/Cache_and_proxy_notes#Least_Recently_Used_(LRU)) as this was a new term for me. The repo provides a couple of reasons why you might want to use this tool, as well as helping you be aware of caveats too. (#183)

**terraform-aws-alternat**

[terraform-aws-alternat](https://aws-oss.beachgeek.co.uk/3xx) simplifies how you can deploy high availability implementation of AWS NAT instances, which may help you to reduce your AWS costs if you need to provide internet access within your VPC's. It is worth checking out the README which provides details and comparisons on using this approach vs NAT Gateways. (#199)

**trading-latency-benchmark**

[trading-latency-benchmark](https://aws-oss.beachgeek.co.uk/3d1) This repository contains a network latency test stack that consists of Java based trading client and Ansible playbooks to coordinate distributed tests. Java based trading client is designed to send limit and cancel orders, allowing you to measure round-trip times of the network communication. (#177)

**vpcshark**

[vpcshark](https://aws-oss.beachgeek.co.uk/47q) is a recent project from AWS Hero Aidan Steele, that provides a Wireshark extcap to make ad hoc mirroring of AWS EC2 traffic easier. Check out the README to find out some more details as to why he open sourced this project.(#206)

### Observability

**observability-solution-kit**

[observability-solution-kit](https://aws-oss.beachgeek.co.uk/3d3) this repository is the Ollyv sdk. The Ollyv sdk offers a simple way to configure its behaviour through *.properties files, which are environment-specific. Currently code is provide from NodeLambda ✨ · NodeExpress 👟 · JavaSpring 🦚 (#177)

**prometheus-rds-exporter**

[prometheus-rds-exporter](https://aws-oss.beachgeek.co.uk/3mx) is a project from Vincent Mercier that provides a Prometheus exporter for AWS RDS. Check out the README, it is very detailed and well put together. It provides a lot of information on how they built this, examples of configurations as well as detailed configuration options. (#188)

**s3-restore-and-copy-progress-monitoring**

[s3-restore-and-copy-progress-monitoring](https://aws-oss.beachgeek.co.uk/3o3) this is a very comprehensive and polished repo that provides an example of how you can restore data that you have stored in S3, providing you a single visualised dashboard to monitor the restore and copy progress within a defined scope.(#189)

### Security

**apeman**

[apeman](https://aws-oss.beachgeek.co.uk/43n) is a new tool that helps security people to map and visualise AWS identity attack paths. The README provides detailed instructions on how to get up and running, but I also found the blog post, [ Project Apeman : A Comprehensive Guide To Graph-Based AWS Security Analysis](https://aws-oss.beachgeek.co.uk/43o) very helpful.(#203)

**auth-pep-pdp**

[auth-pep-pdp](https://aws-oss.beachgeek.co.uk/48s) is another solution from AWS Hero Jimmy Dahlqvist to add to the already great selection he shares in his repo. It provides code to help you implement PDP (Policy Decision Point) and a PEP (Policy Enforcement Point). We will build an simple API and use Amazon API Gateway and Lambda Authorizer as the PEP. It gets extra credit as its using Cedar for the policies.  To help you get started with this project, you can check out the excellent README as well as the supporting blog post, [PEP and PDP for Secure Authorization with AVP ](https://aws-oss.beachgeek.co.uk/48r).(#207)

**aws-lint-iam-policies**

[aws-lint-iam-policies](https://aws-oss.beachgeek.co.uk/3pe)  runs IAM policy linting checks against either a single AWS account or all accounts of an AWS Organization. Reports on policies that violate security best practices or contain errors. Supports both identity-based and resource-based policies. Optionally dumps all policies analysed. The actual linting is performed by the AWS IAM Access Analyzer policy validation feature, which is mostly known for showing recommendations when manually editing IAM policies on the AWS Console UI. The repo provides additional blog posts to help you get started, as well as more details on how this works with supporting resources (#191)

**aws-managed-kms-keys**

[aws-managed-kms-keys](https://aws-oss.beachgeek.co.uk/4es) more goodness from the folks at **Fog Security**, this time providing a repo that provides a listing of AWS Managed KMS Keys and their associated policies in /reference_key_policies. There's a periodic scheduled job that will run and update the listings and data. The README does note that *"AWS managed keys are a legacy key type that are no longer being created for new AWS services as of 2021"* so you might not expect to see more AWS managed keys outside of the ones already listed (#211)

**aws-mine**

[aws-mine](https://aws-oss.beachgeek.co.uk/43k) is a project from Steven Smiley that should interest security folk. It provides a [honey](https://uk.norton.com/blog/iot/what-is-a-honeypot) token system for AWS, that allows you to create AWS access keys that can be placed in various places to tempt bad guys. If used, you will be notified within ~4 minutes. You can then investigate that asset to determine if it may have been compromised. (#203)

**aws-nitro-enclaves-eif-build-action**

[aws-nitro-enclaves-eif-build-action](https://aws-oss.beachgeek.co.uk/3pj) is a new project from AWS Hero Richard Fan that uses a number of tools to help you build a reproducible AWS Nitro Enclaves EIF (Enclave Image File). This GitHub Action use kaniko and Amazon Linux container with nitro-cli, and provides examples of how you can use other tools such as sigstore to sign artefacts as well. (#191)

**aws-rotate-key**

[aws-rotate-key](https://aws-oss.beachgeek.co.uk/3sd) is a project from AWS Community Builder  Stefan Sundin, that helps you implement security good practices around periodically regenerating your API keys. This command line tool simplifies the rotation of those access keys as defined in your local ~/.aws/credentials file. Check out the README for plenty of helpful info and examples of how you might use this. (#194)

**aws-scps-for-sandbox-and-training-accounts**

[aws-scps-for-sandbox-and-training-accounts](https://aws-oss.beachgeek.co.uk/3pf) is a collection of example Service Control Policies (SCPs) that are useful for sandbox and training AWS accounts. The SCPs deny API calls that change baseline account settings (contacts, billing, tax settings, etc.), have long-term financial effects (purchases and reservations) or operate outside allow-listed AWS regions or services. (#191)

**aws-secret-inject**

[aws-secret-inject](https://aws-oss.beachgeek.co.uk/3pg) this handy command line tool from Quincy Mitchell allows you to inject AWS Secrets or SSM Parameters into your configuration files (.env, or whatever you like to call your configuration files these days). The README contains examples of how you can use this. Very handy indeed. (#191)

**aws-secretsmanager-agent**

[aws-secretsmanager-agent](https://aws-oss.beachgeek.co.uk/41i) is a local HTTP service that you can install and use in your compute environments to read secrets from Secrets Manager and cache them in memory. The Secrets Manager Agent can retrieve and cache secrets in memory so that your applications can consume secrets directly from the cache. That means you can fetch the secrets your application needs from the localhost instead of making calls to Secrets Manager. The Secrets Manager Agent can only make read requests to Secrets Manager - it can't modify secrets. The Secrets Manager Agent uses the AWS credentials you provide in your environment to make calls to Secrets Manager. The Secrets Manager Agent offers protection against Server Side Request Forgery (SSRF) to help improve secret security. You can configure the Secrets Manager Agent by setting the maximum number of connections, the time to live (TTL), the localhost HTTP port, and the cache size.(#202)

**avp-toy-store-sample**

[avp-toy-store-sample](https://aws-oss.beachgeek.co.uk/3jw) is a great sample project if you want to explore Cedar, and how this fits in with Amazon Verified Permissions. This sample web application demonstrates authentication and policy-based authorization for different user types to an imaginary toy store. The toy store takes orders online and then send them to customers through multiple warehouses. This application is used by warehouses to help sending orders to customers. The application uses Amazon Cognito for authentication and uses Amazon Verified Permissions for policy-based authorization. Additionally, the application uses API-Gateway as the front-door to the application, and Lambda to process requests. (#184)

**aws-sdk-python-signers**

[aws-sdk-python-signers](https://aws-oss.beachgeek.co.uk/3x3) AWS SDK Python Signers provides stand-alone signing functionality. This enables users to create standardised request signatures (currently only SigV4) and apply them to common HTTP utilities like AIOHTTP, Curl, Postman, Requests and urllib3. This project is currently in an Alpha phase of development. There likely will be breakages and redesigns between minor patch versions as we collect user feedback. We strongly recommend pinning to a minor version and reviewing the changelog carefully before upgrading. Check out the README for details on how to use the signing module. (#198)

**aws-waf-for-event-analysis-dashboard**

[aws-waf-for-event-analysis-dashboard](https://aws-oss.beachgeek.co.uk/3ls) finding the information you need during security incidents is what this project aims to help with. During major online events like live broadcasts, security teams need a fast and clear understanding of attack patterns and behaviours to distinguish between normal and malicious traffic flows. The solution outlined here allows filtering flow logs by "Client IP", "URI", "Header name", and "Header value" to analyse these fields and pinpoint values specifically associated with attack traffic versus normal traffic. For example, the dashboard can identify the top header values that are atypical for normal usage. The security team can then create an AWS WAF rule to block requests containing these header values, stopping the attack. This project demonstrates using AWS Glue crawlers to categorise and structure WAF flow log data and Amazon Athena for querying. Amazon Quicksight is then employed to visualise query results in a dashboard. Once deployed, the dashboard provides traffic visualisation similar to the example graphs shown in Images folder in under project , empowering security teams with insight into attacks and defence.(#186)

**awskillswitch**

[awskillswitch](https://aws-oss.beachgeek.co.uk/3gr) is an open sourced tool from Jeffrey Lyon that is worth checking out. AWS Kill Switch is a Lambda function (and proof of concept client) that an organisation can implement in a dedicated "Security" account to give their security engineers the ability to delete IAM roles or apply a highly restrictive service control policy (SCP) on any account in their organisation. Make sure you check out the README for full details, but this looks like it might be one of those tools that are useful to have in the back pocket in times of need. (#181)

**awsesh**

[awsesh](https://aws-oss.beachgeek.co.uk/4ad) is a command line tool for managing your AWS SSO sessions for those of you who are using AWS IAM Identity Centre (IdC). Whilst the repo provides pre built binaries for Mac, Linux, and Windows, you can also build from source too. There is a brief README with videos showing how this works - it is pretty simple to get up and running. If you are not using a vanity name for your IdC, then just use the prefix you see on the IdC dashboard when configuring this tool.(#208)

**awsviz**

[awsviz](https://aws-oss.beachgeek.co.uk/3z1) is a super nice little tool from Bour Mohamed Abdelhadi, that helps you quickly visualy your IAM policies. You can check out the hosted version of [awsviz](https://aws-oss.beachgeek.co.uk/3z3) and there are some sample policies to show you what you can expect. Check out the[ use cases doc](https://aws-oss.beachgeek.co.uk/3z4) to see why you might want to try this tool out. (#200)

**cdk-sops-secrets**

[cdk-sops-secrets](https://aws-oss.beachgeek.co.uk/4ex) helps you create secret values in AWS with infrastructure-as-code easily by providing a CDK construct library that facilitate syncing SOPS-encrypted secrets to AWS Secrets Manager and SSM Parameter Store. It enables secure storage of secrets in Git repositories while allowing seamless synchronisation and usage within AWS. Even large sets of SSM Parameters can be created quickly from a single file. Detailed README with plenty of examples of how you can use this. Very nice.(#211)

**cedar-access-control-for-k8s**

[cedar-access-control-for-k8s](https://aws-oss.beachgeek.co.uk/47k) is a very very cool project from Micah Hausler, that extends Cedar to the Kubernetes control plane and allows you to implement fine grain policies in Cedar that allow you to have much greater control and flexibility of authorisation within your Kubernetes environments. If you are using Kubernetes, then reviewing this project is a must. Check out the video in the Videos section at the end for more info, where Micah walks you through how this works in more detail. (#206)

**cedar-antlr-grammar**

[cedar-antlr-grammar](https://aws-oss.beachgeek.co.uk/3n0) - ANTLR (ANother Tool for Language Recognition) is a powerful parser generator for reading, processing, executing, or translating structured text or binary files. It's widely used to build languages, tools, and frameworks. From a grammar, ANTLR generates a parser that can build and walk parse trees. AWS Hero Ian Mckay has created one for Cedar. (#188)

**cedar-go**

[cedar-go](https://aws-oss.beachgeek.co.uk/3qf) provides the Go implementation of the Cedar policy language. Check out the README for a quick example of how to use Cedar within your Go applications, and am looking forward to seeing how Go developers start to incorporate this into their applications.(#192)

**chaos-machine**

[chaos-machine](https://aws-oss.beachgeek.co.uk/4ao) is a complete chaos engineering workflow that enables customers to run controlled chaos experiments and test hypotheses related to system behaviour. Chaos Machine uses metric and alarm data from both Amazon CloudWatch and Prometheus as inputs to evaluate system behaviour before, during, and after the experiment. The Chaos Machine provides a simple, consistent way to organise and execute chaos experiments, and is appropriate to use for both building and conducting ad-hoc experiments or integrating into more sophisticated automation pipelines. Chaos Machine uses the AWS Fault Injection Service (FIS) to run controlled experiments, and AWS Step Functions and AWS Lambda for orchestration and execution.(#208)

**config-rds-ca-expiry**

[config-rds-ca-expiry](https://aws-oss.beachgeek.co.uk/3z7) provides sample code to create a custom AWS Config rule to detect expiring CA certificates. Everyone loves TLS certs, but we all hate it when we realise that stuff has broken because they expired. It can happen to anyone, so check this out and make sure you are proactively managing your certs on your Amazon RDS instances, and how this is different to the out of the box notifications you already get with Amazon RDS. (#200)

**CloudConsoleCartographer**

[CloudConsoleCartographer](https://aws-oss.beachgeek.co.uk/3uk) is a project that was released at Black Hat Asia on April 18, 2024, Cloud Console Cartographer is a framework for condensing groupings of cloud events (e.g. CloudTrail logs) and mapping them to the original user input actions in the management console UI for simplified analysis and explainability. It helps you detect signals from the noise more efficiently, which is always important when you are dealing with security incidents. (#196)

**CloudGrappler**

[CloudGrappler](https://aws-oss.beachgeek.co.uk/3qb) is a purpose-built tool designed for effortless querying of high-fidelity and single-event detections related to well-known threat actors in AWS. Andi Ahmeti has put together a blog post, [Introducing CloudGrappler: A Powerful Open-Source Threat Detection Tool for Cloud Environments](https://aws-oss.beachgeek.co.uk/3qc), that provides an overview of how this works with examples.(#192)

**cloud-snitch**

[cloud-snitch](https://aws-oss.beachgeek.co.uk/4bo) - is an essential learning tool for anyone using AWS that helps you better understand  your AWS account activity. Inspired by the MacOS tool "Little Snitch", this will visually show you your AWS activity, allow you to share and collaborate with others, summarise your AWS activity through a number of different lenses and help you uncover suspicious behaviour. You have to check out the README to get a look at some of the screenshots of this tool in use, looks super interesting.(#209)

**csr-builder-for-kms**

[csr-builder-for-kms](https://aws-oss.beachgeek.co.uk/40h) provides a Python library for creating and signing X.509 certificate signing requests (CSRs) with KMS Keys. (#201)

**dAWShund**

[dAWShund](https://aws-oss.beachgeek.co.uk/4bj) is a suite of tools to enumerate, evaluate and visualise the access conditions between different resources in your AWS environments. Perhaps the most critical component of an AWS infrastructure is the policy document describing the actions allowed or denied to a resource. This can get overwhelming sometimes, so tools like this will provide you with the help you need to keep on top of it. Check out the README for details on usage. (#209)

**duplicate-rule-detection-tool**

[duplicate-rule-detection-tool](https://aws-oss.beachgeek.co.uk/3oq) is a project to assess the current active AWS Config rules with potential duplicate scope in an AWS account. Our goal is to help customers can make informed decisions on how to streamline their AWS Config rules and reduce complexity. Plenty of examples and detailed breakdown of how this works in the README, so give it a look. (#190)

**finders-keypers**

[finders-keypers](https://aws-oss.beachgeek.co.uk/4et) is a command line tool that will explore your AWS account and kook for direct connections for KMS Keys and resources in your account. This tool supports both AWS Customer Managed KMS Keys and AWS Managed KMS Keys (with some additional details in the README that you should check out). Typical use cases where this tool is helpful includes security and audit for KMS Key and Resources, Data Protection with Encryption, discovering blast radius of a specific KMS Key, changing a KMS Key or rotating key material, checking Default Settings in AWS that create new resources with the KMS Key, and audit of your resources that a KMS Key may grant access to. README provide examples of how to run the cli against some of those use cases. Very nice tool indeed.(#211)


**gen-ai-cve-patching**

[gen-ai-cve-patching](https://aws-oss.beachgeek.co.uk/3vy) This repository introduces an innovative automated remediation pipeline, designed to effectively address vulnerabilities detected by AWS ECR Inspector. By leveraging Generative AI through Amazon Bedrock's in-context learning, this solution significantly enhances the security posture of application development workflows. The architecture integrates with CI/CD processes, offering a comprehensive and automated approach to vulnerability management. The architecture diagram provided illustrates the solution's key components and their interactions, ensuring a holistic vulnerability remediation strategy.(#197)

**iam-collect**

[iam-collect](https://aws-oss.beachgeek.co.uk/4ew) is a tool from **David Kerber** that helps you collect IAM information from all your AWS organization, accounts, and resources. This is built to run out of the box in simple use cases, and also work in terribly oppressive environments with a little more configuration. If you want to analyze IAM data at scale this is what you've been looking for.(#211)

**iam-lens**

[iam-lens](https://aws-oss.beachgeek.co.uk/4ev) is another tool from **David Kerber**, which builds upon iam-collect, and helps you evaluate your AWS IAM policies offline. It. helps you get visibility into the IAM permissions in your AWS Organizations and accounts. It will use your actual AWS IAM policies (downloaded via iam-collect) and evaluate the effective permissions. Hat tip to Eduard Agavriloae whose [social media](https://www.linkedin.com/posts/activity-7336269672012525569-C_Ao/) message tipped me off. (#211)

**kye**

[kye](https://aws-oss.beachgeek.co.uk/4br), also known as "Know Your Enemy", is a tool that analyses IAM Role trust policies and S3 bucket policies in your AWS account to identify third-party vendors with access to your resources. It compares the AWS account IDs found in these policies against a reference list of known AWS accounts from fwd:cloudsec to identify the vendors behind these accounts. Simple to install, the README provides example configurations and instructions on how to get started. (#209)

**OpenSecOps-Org**

[OpenSecOps-Org](https://aws-oss.beachgeek.co.uk/4bq) is the repo that holds the various sub projects that are part of [opensecops.org](https://www.opensecops.org/), that provide open source tools and solutions that help you seamlessly integrate into your operations, providing robust security frameworks and operational excellence. I have not actually looked at this in detail, but just skimming it looks very comprehensive. Split across Foundation that helps you setup your AWS against industry standard good practices, and Soar that looks more at continuous monitoring, automated incident handling and remediation of security issues.(#209)

**powerpipe**

[powerpipe](https://aws-oss.beachgeek.co.uk/3qd) is dashboards and benchmarks as code. Use it to visualise any data source, and run compliance benchmarks and controls, for effective decision-making and ongoing compliance monitoring. As with all the Turbot open source projects, excellent documentation, and they have included a video that provides a demo of this at work. (#192)

**pristup**

[pristup](https://aws-oss.beachgeek.co.uk/3se) is along similar lines to the previous project, except this project from my colleague Dark Mesaros, provides a way to generate temporary AWS Console sign-in URLs. The purpose of this is to enable your users that do not have AWS Console access, temporary access to it without the need for a username and password. As with all of Darko's projects, excellent documentation and examples abound in the README. (#194)

**RunaVault**

[RunaVault](https://aws-oss.beachgeek.co.uk/4eq) is a secure, serverless password management application built using AWS free-tier services and a React frontend. It enables users to create, manage, and share encrypted secrets (e.g., passwords) with individuals or groups, leveraging AWS Cognito for authentication, DynamoDB for storage, and KMS for encryption. Check out the repo for screenshots of what the app looks like as well as more technical implementation details (which you can deploy via OpenTofu/Terraform). I wish more READMEs would do this, but they also include some estimated costs of what it might cost to run this project in your AWS Account.(#211)

**s3-prefix-level-kms-keys**

[s3-prefix-level-kms-keys](https://aws-oss.beachgeek.co.uk/3os) is a demo of an approach to enforce Prefix level KMS keys on S3. At the moment, S3 supports default bucket keys that is used automatically to encrypt objects to that bucket. But no such feature exists for prefixes, (i.e) you might want to use different keys for different prefixes within the same bucket (rather than one key for the entire bucket). This project shows a potential solution on how to enforce prefix level KMS keys.(#190)

**s3-presignedurl-staticips-endpoint-with-cdk**

[s3-presignedurl-staticips-endpoint-with-cdk](https://aws-oss.beachgeek.co.uk/3k5) this solution simplifies access to Amazon S3 by creating secure, custom presigned URLs for object downloads through a single endpoint with a unique domain and static IPs. The use case involves users following an IP and domain Allowlist firewall policy, limiting API access to specific domains and IPs. The architecture employs key AWS services, including AWS Global Accelerator, Amazon API Gateway, AWS Lambda, Application Load Balancer(ALB), VPC Endpoint, and Amazon S3. This design centralizes the API for generating presigned URLs and the S3 endpoint under a single domain, linked to an AWS Global Accelerator with two static IPs. Consequently, users can effortlessly request presigned URLs and download S3 objects through a unified domain endpoint with static IPs. This architecture is especially beneficial for customers with strict policies or compliance requirements, such as those in the public, medical, and finance sectors. (#184)

**sample-code-for-a-secure-vault-using-aws-nitro-enclaves**

[sample-code-for-a-secure-vault-using-aws-nitro-enclaves](https://aws-oss.beachgeek.co.uk/43c) - This repository contains a sample secure vault solution built using AWS Nitro Enclaves, a feature available exclusively through the AWS Nitro System hypervisor on supported Amazon EC2 instances. A vault solution is useful when you need to ensure sensitive data (such as Protected Health Information (PHI)/Personally Identifiable Information (PII)) is properly secured at rest and can only be decrypted through approved channels. Check out more details about how cool Nitro Enclaves are by reading the supporting documentation for this project, [About the AWS Nitro Enclaves Vault](https://aws-oss.beachgeek.co.uk/43b).(#203)

**secret**

[secret](https://aws-oss.beachgeek.co.uk/4ez) is from **AWS Hero Matthew Bonig** and provides a CDK Construct to create Secrets Manager secrets without unexpectedly recreating them (avoiding the issue when you update the generateSecretString property, the secret gets recreated!). Detailed README including the design philosophy. (#211)

**security-hardened-amis-for-eks**

[security-hardened-amis-for-eks](https://aws-oss.beachgeek.co.uk/4d1) provides a fully automated solution to create security-hardened Amazon EKS AMIs that comply with either CIS Level 1 or Level 2 standards. This solution will help you if you are looking for guidance on how to generate a CIS-hardened AMI for EKS, as well as if you are encountering issues with workloads running on your own custom, CIS-hardened AMIs. (#210)

**security-hub-compliance-analyzer**

[security-hub-compliance-analyzer](https://aws-oss.beachgeek.co.uk/41s) this repo provides a compliance analysis tool which enables organisations to more quickly articulate their compliance posture and also generate supporting evidence artefacts. Security Hub Compliance Analyzer (SHCA) generates artefacts in support of Department of Defense Risk Management Framework (RMF) Information System accreditation. Utilising Amazon Web Services provided documentation, mapping NIST800-53-Rev-5 Controls to AWS Security Hub Security Control IDs, SHCA requests the current environment compliance from Security Hub and generates a zip file stored in Amazon S3 containing discrete artefacts in CSV, JSON, OCSF providing SecOps with artefacts to import into the RMF tool.(#202)

**sigv4a-signing-examples**

[sigv4a-signing-examples](https://aws-oss.beachgeek.co.uk/1jl) provides a useful set of examples that show examples of sigv4 implementations. Josh Hart, maintainer of this repo, has put together a companion repo that has a set of sigv4 examples in different frameworks and using the sdk or from scratch. Josh is also looking for community contributions in other frameworks, so if you are working on a framework and want to combine efforts, drop Josh a note.(#195)

**sra-verify**

[sra-verify](https://aws-oss.beachgeek.co.uk/4fl)  is a security assessment tool that automates the verification of AWS Security Reference Architecture (SRA) implementations across multiple AWS accounts and regions. It provides detailed findings and actionable remediation steps to ensure your AWS environment follows security best practices. The tool performs automated security checks across multiple AWS services including CloudTrail, GuardDuty, IAM Access Analyzer, AWS Config, Security Hub, and S3. It supports multi-account environments and can run checks specific to management, audit, and log archive accounts while providing detailed findings with remediation guidance. **Jeremy Schiefer, Justin Kontny, **and **Matt Nispel** have put together a blog post that provides more info about this project, so check out [Introducing SRA Verify – an AWS Security Reference Architecture assessment tool](https://aws-oss.beachgeek.co.uk/4fm) (#212)

**sso-sync-to-amazon-rds**

[sso-sync-to-amazon-rds](https://aws-oss.beachgeek.co.uk/3e2) This project sets up AWS Lambda functions, Amazon EventBridge rule, Amazon VPC Endpoint for AWS IAM Identity Center (successor to AWS Single Sign-On), the related Security Groups and permissions necessary to automatically provision database users to the Amazon Relational Database Service (Amazon RDS) cluster using AWS Cloud Development Kit (AWS CDK). When a new user is created in IAM Identity Center and the user belongs to the group specified in a IAM_IDC_GROUP_NAME variable, EventBridge rule will trigger the Lambda function. The Lambda function will create a new user in a specified Amazon RDS cluster. The user will then be able to login to the database using their SSO username and IAM credentials. Adding a user to the configured group will trigger the Lambda function as well. (#178)

**StsSamlDriver**

[StsSamlDriver](https://aws-oss.beachgeek.co.uk/49d) is A Python-based SAML authentication handler for AWS STS that allows you to get temporary credentials using SAML to the AWS CLI, or an application written using an AWS SDK without the need to screen scrape or emulate a browser.(#207)

**tailpipe**

[tailpipe](https://aws-oss.beachgeek.co.uk/492) is an open source SIEM for instant log insights from our friends at Turbot, powered by DuckDB. It allows you to analyse millions of events in seconds, right from your terminal. Check out the README that includes more info include a nice video demo of this in works. Bob Tordella (who we have featured many times over the years in this newsletter), has put together a blog post too, which is a must read. Go check it out - [Query AWS CloudTrail Logs Locally with SQL](https://aws-oss.beachgeek.co.uk/494) (#207)

**terraform-aws-vulne-soldier**

[terraform-aws-vulne-soldier](https://aws-oss.beachgeek.co.uk/47f) is a new project from Victor Omolayo that provides an AWS EC2 vulnerability remediation tool designed to automate the process of patching nodes managed by AWS Systems Manager. He has put together a blog post, [vulne-soldier: AWS EC2 Vulnerability Remediation Tool](https://aws-oss.beachgeek.co.uk/47g) that provides more detail on the project and background as to why he created it as well as how to get started. (#206)

**threat-designer**

[threat-designer](https://aws-oss.beachgeek.co.uk/4d7)  is a GenerativeAI application designed to automate and streamline the threat modelling process for secure system design. Harnessing the power of large language models (LLMs), it analyzes system architectures, identifies potential security threats, and generates detailed threat models. By automating this complex and time-intensive task, Threat Designer empowers developers and security professionals to seamlessly incorporate security considerations from the earliest stages of development, enhancing both efficiency and system resilience. Check out the README as it has more detail, including sample screen shots of what you can expect. (#210)

**validate-aws-policies**

[validate-aws-policies](https://aws-oss.beachgeek.co.uk/42r) is a Python command line tool from  Alejandro Velez to validate aws policies using boto3 and Access Analyzer API. You can check out his supporting blog post (including demo video) [Continuous Delivery applied to Authorization with IAM Identity Center and AWS IAM Access Analyzer – Part 1](https://aws-oss.beachgeek.co.uk/42s) that shows you how you can incorporate this tool as part of your CI/CD pipeline.(#203)

**vpcshark**

[vpcshark](https://aws-oss.beachgeek.co.uk/47q) is a recent project from AWS Hero Aidan Steele, that provides a Wireshark extcap to make ad hoc mirroring of AWS EC2 traffic easier. Check out the README to find out some more details as to why he open sourced this project.(#206)

**vscode-iam-service-principal-snippets**

[vscode-iam-service-principal-snippets](https://aws-oss.beachgeek.co.uk/495) is the latest VSCode plugin from AWS Community Builder Danny Steenman ([his fifth](https://www.linkedin.com/posts/dannysteenman_today-marks-a-small-but-significant-milestone-activity-7292070581011660800-iwbT/)). This VS Code extension provides autocompletion of all AWS services that can be used as Service Principals in your IAM policies. (#207)

**ww-project-eks-goat**

[www-project-eks-goat](https://aws-oss.beachgeek.co.uk/4bp) this repo provides an immersive workshop on AWS ECR & EKS Security designed to take participants through real-world scenarios of attacking and defending Kubernetes clusters hosted on AWS EKS. This workshop provides a comprehensive approach, from understanding the anatomy of attacks on EKS clusters using AWS ECR to deploying robust defense mechanisms. Participants will learn how to backdoor AWS ECR image & exploit misconfigurations and vulnerabilities within AWS EKS, followed by the implementation of best security practices to safeguard the environment.This workshop is tailored for security professionals, cloud engineers, and DevOps teams looking to enhance their understanding of offensive and defensive Kubernetes security strategies. (#209)

**yes3-scanner**

[yes3-scanner](https://aws-oss.beachgeek.co.uk/4er) is the first in a series of projects from the folks at **Fog Security**, which scans an AWS Account for potential S3 security issues such as access issues such as Public Access, preventative S3 Security Settings, additional security such as encryption, and Ransomware Protection, Data Protection, and Recovery.(#211)

### Storage

**amazon-s3-glacier-archive-data-delete**

[amazon-s3-glacier-archive-data-delete](https://aws-oss.beachgeek.co.uk/3lr) Amazon S3 Glacier Archive (data) Delete solution provides an automated workflow to delete ALL of your data in an S3 Glacier Vault. This solution only applies to Amazon S3 Glacier Vault Archives. Within S3 Glacier, data is stored as an Archive within a Vault. This solution does not apply to objects in Glacier Deep Archive, Glacier Flexible Retrieval, and Glacier Instant Retrieval stored in an Amazon S3 Bucket. Good README with clear guidance and overview of how this works.(#186)

**ebs-bootstrap**

[ebs-bootstrap](https://aws-oss.beachgeek.co.uk/3mg) is a very handy tool from Lasith Koswatta Gamage that solves a very specific problem. Lasith reached out to explain more about the "itch" that needed to be scratched. ebs-bootstrap is a tool that provides a safe and as-code approach for managing block devices on AWS EC2. If you need precise and consistent control over your EBS volumes when attaching them to your EC2 Nitro based instances, you need to check out this project. The README provides some additional example configurations, and there is a blog post in the works which I will share once it has been published. (#187)

**ebsight**
                                                                 
[ebsight](https://aws-oss.beachgeek.co.uk/490) is a Python-based analysis tool developed by Stephen Jones in response to AWS's new EBS snapshot size reporting feature. This tool leverages the newly introduced FullSnapshotSizeInBytes field to provide comprehensive insights into EBS volume usage, performance metrics, and cost optimisation opportunities. After you have checked the repo out and gone through the README, why not read the supporting blog post, [Unleash the Power of EBSight for Optimal AWS Storage Management](https://aws-oss.beachgeek.co.uk/491) (#207)

**eks-auto-mode-ebs-migration-tool**

[eks-auto-mode-ebs-migration-tool](https://aws-oss.beachgeek.co.uk/4d9) is a tool you can use to migrate a Persistent Volume Claim from a standard EBS CSI StorageClass (ebs.csi.aws.com) to the Amazon EKS Auto EBS CSI StorageClass (ebs.csi.eks.amazonaws.com) or vice-versa. Check out the README for some important details about using this tool.(#210)

**fsx-to-s3-int**

[fsx-to-s3-int](https://aws-oss.beachgeek.co.uk/4a9) is a new tool from Steve Jones that helps AWS users analyse their FSx for NetApp ONTAP (FSxN) volume usage patterns to estimate potential cost savings when migrating to Amazon S3 Intelligent-Tiering. It collects and analyses metrics such as storage usage, access patterns, and operations to provide insights for data migration planning. Steve has put together a blog post that helps you get started, so go check out [Unlocking Cloud Savings: Your Guide to FSx and S3 Intelligent-Tiering with Python Magic!](https://aws-oss.beachgeek.co.uk/4aa) (#208)


**git-remote-s3**

[git-remote-s3](https://aws-oss.beachgeek.co.uk/44s) is a neat tool that provides you with the ability to use Amazon S3 as a [Git Large File Storage (LFS)](https://git-lfs.com/) remote provider. It provides an implementation of a git remote helper to use S3 as a serverless Git server. The README provides good examples of how to set this up and example git commands that allow you to use this setup. This is pretty neat, and something I am going to try out for myself in future projects. (#204)

**robinzhon**

[robinzhon](https://aws-oss.beachgeek.co.uk/4gk) is a high-performance Python library from **Robin Quintero** for fast, concurrent S3 object downloads. Check out the README for details on how to get started. Robin shared why he created this project:
> Recently at work I have faced that we need to pull a lot of files from S3 but the existing solutions are slow so I was thinking in ways to solve this and that's why I decided to create robinzhon. The main purpose of robinzhon is to download high amounts of S3 Objects without having to do extensive manual work trying to achieve optimisations. I know that you can implement your own concurrent approach to try to improve your download speed but robinzhon can be 3 times faster even 4x if you start to increase the max_concurrent_downloads but you must be careful because AWS can start to fail due to the amount of requests. (#212)

**s3-delta-download**
 
[ s3-delta-download](https://aws-oss.beachgeek.co.uk/4af) is a new tool for interacting with your Amazon S3 buckets, that will help anyone who is encountering current limitations of the 's3 sync', specifically that it does not yet support using a non-directory key prefix. This means that it will only download those files it does not already have, skipping files that are not already in your local directory. The README provide a nice example of how this works, and also covers installation dependencies you need (it requires .NET SDK or later).(#208)

**s3-diff-uploader**

[s3-diff-uploader](https://aws-oss.beachgeek.co.uk/3z0) is the latest project from open source good guy Damon Cortesi, that came about from some [experimentations](https://www.linkedin.com/posts/dacort_i-wanted-to-experiment-recently-with-incremental-activity-7206314345599832065--95_
) he was doing with incremental uploads of compressed files to S3. He decided to publish a simple proof-of-concept CLI tool that demonstrates how you can both append and compress file uploads to S3. The result so far, is it uses UploadPartCopy and the fact that you can concatenate gzip chunks to reduce the amount of data you need to upload directly. (#200)

**s3-fast-list**

[s3-fast-list](https://aws-oss.beachgeek.co.uk/3k6) is a rust based tool that concurrently list Amazon S3 bucket with ListObjectsV2 API. Check out the README for use cases as to when s3-fast-list is going to help you out (against existing options you have) (#184)

**s3-migrate**

[s3-migrate](https://aws-oss.beachgeek.co.uk/4bh) is a project that came onto my radar thanks to a message from Luciano Mammino, and provides a command line tool to help you move all your objects between s3-compatible storage systems. It can migrate objects from one S3-compatible storage bucket to another, resuming interrupted transfers using a SQLite state file. This project is (currently) experimental so use with care, and is intended for a one-off migration, not to keep two buckets in sync. Check out the README for more details on how to configure and use this tool. (#209) 


**s3-prefix-level-kms-keys**

[s3-prefix-level-kms-keys](https://aws-oss.beachgeek.co.uk/3os) is a demo of an approach to enforce Prefix level KMS keys on S3. At the moment, S3 supports default bucket keys that is used automatically to encrypt objects to that bucket. But no such feature exists for prefixes, (i.e) you might want to use different keys for different prefixes within the same bucket (rather than one key for the entire bucket). This project shows a potential solution on how to enforce prefix level KMS keys.(#190)

**s3-presignedurl-staticips-endpoint-with-cdk**

[s3-presignedurl-staticips-endpoint-with-cdk](https://aws-oss.beachgeek.co.uk/3k5) this solution simplifies access to Amazon S3 by creating secure, custom presigned URLs for object downloads through a single endpoint with a unique domain and static IPs. The use case involves users following an IP and domain Allowlist firewall policy, limiting API access to specific domains and IPs. The architecture employs key AWS services, including AWS Global Accelerator, Amazon API Gateway, AWS Lambda, Application Load Balancer(ALB), VPC Endpoint, and Amazon S3. This design centralizes the API for generating presigned URLs and the S3 endpoint under a single domain, linked to an AWS Global Accelerator with two static IPs. Consequently, users can effortlessly request presigned URLs and download S3 objects through a unified domain endpoint with static IPs. This architecture is especially beneficial for customers with strict policies or compliance requirements, such as those in the public, medical, and finance sectors. (#184)

**s3-restore-and-copy-progress-monitoring**

[s3-restore-and-copy-progress-monitoring](https://aws-oss.beachgeek.co.uk/3o3) this is a very comprehensive and polished repo that provides an example of how you can restore data that you have stored in S3, providing you a single visualised dashboard to monitor the restore and copy progress within a defined scope.(#189)


**s3-small-object-compaction**

[s3-small-object-compaction](https://aws-oss.beachgeek.co.uk/3mk) This solution deploys a serverless application to combine ("compact") small objects stored in a given Amazon S3 prefix into a single larger file. Larger files enable cost effective use of S3 storage tiers that have a minimum billable object size (e.g. 128 KB). It can also improve performance when querying data directly with Amazon Athena. The sample code is written using the AWS Cloud Development Kit in Python.(#187)

**s3-vectors-mcp**

[s3-vectors-mcp](https://aws-oss.beachgeek.co.uk/4gl) hot off the announcement from the AWS Summit New York, **AWS Community Hero Kazuaki Morita** has put together this repo that provides tools for interacting with AWS S3 Vectors service. This server enables AI assistants to embed text using Amazon Bedrock models and store/query vector embeddings in S3 Vectors indexes. Check out the README for details of the specific features as well as how to configure this MCP Server.(#212)

**s3grep**

[s3grep](https://aws-oss.beachgeek.co.uk/4gw) is the latest essential tool from Damon Cortesi that provides a parallel CLI tool for searching logs and unstructured content in Amazon S3 buckets. It supports .gz decompression, progress bars, and robust error handling—making it ideal for cloud-native log analysis. (#212)


**s3vectors-embed-cli**

[s3vectors-embed-cli](https://aws-oss.beachgeek.co.uk/4gy) is a standalone command-line tool that simplifies the process of working with vector embeddings in S3 Vectors. You can create vector embeddings for your data using Amazon Bedrock and store and query them in your S3 vector index using single commands. This tool facilitates semantic similarity search on media in Amazon S3 via AWS Bedrock and Amazon S3 Vectors. Check out the README for the supported commands, installation and configuration details as well as some quick starts to help you understand how this all works. (#212)


**shuk**

[shuk](https://aws-oss.beachgeek.co.uk/3r4) my colleague Darko Mesaros has been experimenting with Rust, and he has created shuk, a file sharing tool (in Rust) for Amazon S3. Run the tool with any file you want to upload, and it will generated a pre-signed URL ready for you to use. Very much alpha, so keep watching (and if you feel so inclined, contribute). (#193)

**stree**

[stree](https://aws-oss.beachgeek.co.uk/3o1) this project from Takafumi Miyanaga is a CLI tool designed to visualize the directory tree structure of an S3 bucket.
By inputting an S3 bucket/prefix and utilizing various flags to customize your request, you can obtain a colorized or non-colorized directory tree right in your terminal. Whether it's for verifying the file structure, sharing the structure with your team, or any other purpose, stree offers an easy and convenient way to explore your S3 buckets. (#189)

# AWS Services

**amazon-gamelift-agent**

[amazon-gamelift-agent](https://aws-oss.beachgeek.co.uk/3ur) is a Java application that is used to launch game server processes on Amazon GameLift fleets. This application registers a compute resource for an existing Amazon GameLift fleet using the RegisterCompute API. The application also calls the GetComputeAuthToken API to fetch an authorisation token for the compute resource, using it to make a web socket connection to the Amazon GameLift service.(#196)

**appfabric-data-analytics**

[appfabric-data-analytics](https://aws-oss.beachgeek.co.uk/3k3) this project enables you to maintain logs from various SaaS applications and provides the ability to search and display log data. This solution leverages AWS AppFabric to create a data repository that you can query with Amazon Athena. While customers can obtain normalized and enriched SaaS audit log data (OCSF) from AppFabric, many prefer not only to forward these logs to a security tool. Some have the requirement to preserve logs for post-incident analysis, while others need to utilize the logs for tracking SaaS subscription and license usage. Additionally, some customers aim to analyze user activity to discover patterns. This project establishes a data pipeline that empowers customers to construct dashboards on top of it. (#184)

**amazon-bedrock-slack-gateway**

[amazon-bedrock-slack-gateway](https://aws-oss.beachgeek.co.uk/3z8) lets you use Amazon Bedrock's generative AI to enable Slack channel members to access your organisations data and knowledge sources via conversational question-answering. You can connect to your organisation data via data source connectors and integrate it with Slack Gateway for Amazon Bedrock to enable access to your Slack channel members. It allows your users to converse with Amazon Bedrock using Slack Direct Message (DM) to ask questions and get answers based on company data, get help creating new content such as emails, and performing tasks. You can also invite it to participate in your team channels. In a channel users can ask it questions in a new message, or tag it in a thread at any point. Get it to provide additional data points, resolve a debate, or summarise the conversation and capture next steps.(#200)

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

**distill-cli**

[distill-cli](https://aws-oss.beachgeek.co.uk/3yz) is a new project from Amazon CTO Dr Werner Vogels, which uses Amazon Transcribe and Amazon Bedrock to create summaries of your audio recordings (e.g., meetings, podcasts, etc.) directly from the command line. Distill CLI takes a dependency on Amazon Transcribe, and as such, supports the following media formats: AMR, FLAC, M4A, MP3, MP4, Ogg, WebM, WAV. It is great to feature this latest project, with the previous one being featured in [#197](https://community.aws/content/2gPNtsdSfQRIpmbUrNyPrjUg54D/aws-open-source-newsletter-197). To go with this repo, there is a post too, [Introducing Distill CLI: An efficient, Rust-powered tool for media summarization](https://aws-oss.beachgeek.co.uk/3yy) where Werner shares his experience building this tool in Rust, and provides some closing thoughts too. (#200)

**Drag-and-Drop-Email-Designer**

[Drag-and-Drop-Email-Designer](https://aws-oss.beachgeek.co.uk/469) looks like a neat project that provides a way of designing email templates that you can use with the Send with SES project. Check out the README for some visuals on what this looks like. (#205)


**duplicate-rule-detection-tool**

[duplicate-rule-detection-tool](https://aws-oss.beachgeek.co.uk/3oq) is a project to assess the current active AWS Config rules with potential duplicate scope in an AWS account. Our goal is to help customers can make informed decisions on how to streamline their AWS Config rules and reduce complexity. Plenty of examples and detailed breakdown of how this works in the README, so give it a look. (#190)

**ebs-bootstrap**

[ebs-bootstrap](https://aws-oss.beachgeek.co.uk/3mg) is a very handy tool from Lasith Koswatta Gamage that solves a very specific problem. Lasith reached out to explain more about the "itch" that needed to be scratched. ebs-bootstrap is a tool that provides a safe and as-code approach for managing block devices on AWS EC2. If you need precise and consistent control over your EBS volumes when attaching them to your EC2 Nitro based instances, you need to check out this project. The README provides some additional example configurations, and there is a blog post in the works which I will share once it has been published. (#187)

**gen-ai-on-eks**

[gen-ai-on-eks](https://aws-oss.beachgeek.co.uk/3d2) this repository aims to showcase how to finetune a FM model in Amazon EKS cluster using, JupyterHub to provision notebooks and craft both serving and training scripts, RayOperator to manage Ray Clusters and Karpenter to manage Node Scaling. (#177)

**qgis-amazonlocationservice-plugin**

[qgis-amazonlocationservice-plugin](https://aws-oss.beachgeek.co.uk/3xp) is a new open source plugin from AWS Hero Yasunori Kirimoto that uses the functionality of Amazon Location Service for  the Geographic Information System (GIS), a user friendly Open Source application licensed under the GNU General Public License. You can find out more by reading his post, [Amazon Location Service Plugin for QGIS released in OSS](https://aws-oss.beachgeek.co.uk/3xo) (#199)

**makit-llm-lambda**

[makit-llm-lambda ](https://aws-oss.beachgeek.co.uk/3cx)this repo from Martyn Kilbryde is an example of how you can run a Large Language Model (LLM) inside an AWS Lambda Function.  Whilst the code will help you deploy to AWS Lambda, it can be ran locally inside Docker for testing as well. The function contains the full LLM model and the code to use the model, allowing basic text generation from a HTTP call into it. (#177)

**newsletter-manager-template**

[newsletter-manager-template](https://aws-oss.beachgeek.co.uk/3sa) is a project from AWS Community Builder Matteo Depascale that provides backend service orchestrations for newsletter builders. (#194)

**observability-solution-kit**

[observability-solution-kit](https://aws-oss.beachgeek.co.uk/3d3) this repository is the Ollyv sdk. The Ollyv sdk offers a simple way to configure its behaviour through *.properties files, which are environment-specific. Currently code is provide from NodeLambda ✨ · NodeExpress 👟 · JavaSpring 🦚 (#177)

**resource-explorer-with-organizations**

[resource-explorer-with-organizations](https://aws-oss.beachgeek.co.uk/3dy) you may have a use cases where you are eager to find lingering resources, or resources that were not at their optimal settings. By utilising Resource Explorer and Step Functions, you can gather all the necessary information from these accounts, and use them to create a report to gain a wider understanding of the state of your AWS accounts. As of this release, the limitation of Resource Explorer is that it is done on a per account basis. However, the README provides details of a workaround to deploy this to all your accounts in our AWS Organization using StackSets. The use case shown in the repo shows you how you can find resources in an multiple AWS accounts over multiple regions, and generating an Excel Document displaying the Account it belongs to, Name, Resource Type, and ARN of the resource. The repo provides details of how you can deploy this tool, so make sure you check that out too. (#178)

**s3-fast-list**

[s3-fast-list](https://aws-oss.beachgeek.co.uk/3k6) is a rust based tool that concurrently list Amazon S3 bucket with ListObjectsV2 API. Check out the README for use cases as to when s3-fast-list is going to help you out (against existing options you have) (#184)

**s3-presignedurl-staticips-endpoint-with-cdk**

[s3-presignedurl-staticips-endpoint-with-cdk](https://aws-oss.beachgeek.co.uk/3k5) this solution simplifies access to Amazon S3 by creating secure, custom presigned URLs for object downloads through a single endpoint with a unique domain and static IPs. The use case involves users following an IP and domain Allowlist firewall policy, limiting API access to specific domains and IPs. The architecture employs key AWS services, including AWS Global Accelerator, Amazon API Gateway, AWS Lambda, Application Load Balancer(ALB), VPC Endpoint, and Amazon S3. This design centralizes the API for generating presigned URLs and the S3 endpoint under a single domain, linked to an AWS Global Accelerator with two static IPs. Consequently, users can effortlessly request presigned URLs and download S3 objects through a unified domain endpoint with static IPs. This architecture is especially beneficial for customers with strict policies or compliance requirements, such as those in the public, medical, and finance sectors. (#184)

**sagemaker-mlflow**

[sagemaker-mlflow](https://aws-oss.beachgeek.co.uk/3ya)  This plugin generates Signature V4 headers in each outgoing request to the Amazon SageMaker with MLflow capability, determines the URL of capability to connect to tracking servers, and registers models to the SageMaker Model Registry. It generates a token with the SigV4 Algorithm that the service will use to conduct Authentication and Authorization using AWS IAM.(#200)

**sample-code-for-a-secure-vault-using-aws-nitro-enclaves**

[sample-code-for-a-secure-vault-using-aws-nitro-enclaves](https://aws-oss.beachgeek.co.uk/43c) - This repository contains a sample secure vault solution built using AWS Nitro Enclaves, a feature available exclusively through the AWS Nitro System hypervisor on supported Amazon EC2 instances. A vault solution is useful when you need to ensure sensitive data (such as Protected Health Information (PHI)/Personally Identifiable Information (PII)) is properly secured at rest and can only be decrypted through approved channels. Check out more details about how cool Nitro Enclaves are by reading the supporting documentation for this project, [About the AWS Nitro Enclaves Vault](https://aws-oss.beachgeek.co.uk/43b).(#203)

**terraform-aws-ecr-watch**

[terraform-aws-ecr-watch](https://aws-oss.beachgeek.co.uk/3f2) is a project out of the folks from Porsche, when they are not busy designing super fast cars, their engineers are busy creating useful open source tools for folks to use. This project is a Terraform module to configure an AWS ECR Usage Dashboard based on AWS CloudWatch log insight queries with data fetched from AWS CloudTrail. (#180)

**user-behavior-insights**

[user-behavior-insights](https://aws-oss.beachgeek.co.uk/3yx) This repository contains the OpenSearch plugin for the User Behavior Insights (UBI) capability. This plugin facilitates persisting client-side events (e.g. item clicks, scroll depth) and OpenSearch queries for the purpose of analyzing the data to improve search relevance and user experience.(#200)

# Open Source projects on AWS

**apeman**

[apeman](https://aws-oss.beachgeek.co.uk/43n) is a new tool that helps security people to map and visualise AWS identity attack paths. The README provides detailed instructions on how to get up and running, but I also found the blog post, [ Project Apeman : A Comprehensive Guide To Graph-Based AWS Security Analysis](https://aws-oss.beachgeek.co.uk/43o) very helpful.(#203)

**bedrock-litellm**

[bedrock-litellm](https://aws-oss.beachgeek.co.uk/43m) is an awesome project that provides a way of proxying requests in the OpenAI format, so that they will work with Amazon Bedrock. OpenAI is often one of the default options for integrating various generative AI tools and libraries, and now you have a way of being able to point those to use foundational models managed by Amazon Bedrock. It uses [litellm](https://www.litellm.ai/) to do this, and is deployed in a Kubernetes cluster.(#203)

**bluesky-pds-cdk**

[bluesky-pds-cdk](https://aws-oss.beachgeek.co.uk/47x) if you are looking to deploy a self-hosted a fully containerized, serverless Bluesky Personal Data Server (PDS) on AWS, then this is the repo for you. It provides an opinionated AWS CDK construct that makes deploying this on AWS a breeze. (#206)

**cdk-vscode-server**

[cdk-vscode-server](https://aws-oss.beachgeek.co.uk/47j) is a new CDK construct from Manuel Vogel that provides a speed way to provision VSCode servers on AWS. Check out [his LinedIn post here](https://www.linkedin.com/posts/manuel-vogel_aws-cdk-cdkconstruct-ugcPost-7285010738505523201-YbLw/?utm_source=share&utm_medium=member_ios) for more details, as well as the detailed README. I have done this in the past with CloudFormation (check out my [gist here](https://gist.github.com/094459/0dc0eefcffbbc2c843e11e96940c2011)) but will be switching over to this construct from now on. (#206)

**cumuli-aws-console-chat**

[cumuli-aws-console-chat](https://aws-oss.beachgeek.co.uk/3j2)  is an open source Chrome extension that provides similar functionality to Amazon Q. The Cumuli Chrome extension adds a side panel with LLM chat to all AWS pages, and lets you add screenshots of the console to your queries to get context-aware responses. It's similar to Amazon Q but uses GPT-4 Turbo with vision. Check out the repo that includes a demo video of it in action.(#183)

**deploy-langfuse-on-ecs-with-fargate**

[deploy-langfuse-on-ecs-with-fargate](https://aws-oss.beachgeek.co.uk/43j) This repository contains the AWS CDK Python code for deploying the Langfuse application using Amazon Elastic Container Registry (ECR) and Amazon Elastic Container Service (ECS). If you are not familiar with Langfuse, it is is an open-source LLM engineering platform that helps teams collaboratively debug, analyze, and iterate on their LLM applications.(#203)

**ha-aws-cost**

[ha-aws-cost](https://aws-oss.beachgeek.co.uk/44i)  is a project from Diego Marques for folks who use [Home Assistant](https://www.home-assistant.io/) (a very popular open source home automation system), and provides custom component interacts with AWS to get the month to date cost and forecast cost and make it available on Home Assistant. It adds two new entities 1/Month to date cost: The current cost of your AWS account, and 2/Forecasted month costs: The forecasted cost based in your current consumption. Check out Diego's[ post on LinkedIn](https://www.linkedin.com/posts/diego7marques_aws-cost-my-first-home-assistant-integration-activity-7239741496013340672-cCSY/) that provides some more background to this project.(#204)

**mlspace**

[mlspace](https://aws-oss.beachgeek.co.uk/3r8) provides code that will help you deploy [MLSpace](https://aws-oss.beachgeek.co.uk/3r9) into your AWS account. [MLSpace](https://aws-oss.beachgeek.co.uk/3r9) is an open source no-hassle tool for data science, machine learning and deep learning, and has pre-made environments for pytorch, tensorflow and everything else you might need. (#193)

**node-red-contrib-aws-bedrock**

[node-red-contrib-aws-bedrock](https://aws-oss.beachgeek.co.uk/4ae) is a project from my colleague Kelsea Blackweel that create a new node type within node-red (my favourite open source project ever!) that allows you to integrate Amazon Bedrock into your flows. (#208)


**pagemosaic-cms**

[pagemosaic-cms](https://aws-oss.beachgeek.co.uk/3tb) is an open-source platform optimized for AWS to efficiently host static websites. It simplifies the process of creating, managing, and publishing content online with an emphasis on cost-effectiveness and efficient use of AWS resources. You can check out the project home page for more details [here](https://aws-oss.beachgeek.co.uk/3tk).(#195)

**ragna**

[ragna](https://aws-oss.beachgeek.co.uk/3f3) this is a repo I put together to show you how you can add Amazon Bedrock models from Anthropic and Meta within the Ragna tool. I blogged last week about this [#179](https://dev.to/aws/unboxing-ragna-getting-hands-on-and-making-it-to-work-with-amazon-bedrock-7k3) but I have put together this repo that shows the actual code as I had received quite a few DMs, and as a bonus, I have also added the recently announced Llama2 13B model from Meta. To help with this, a new blog post, [Adding Amazon Bedrock Llama2 as an assistant in Ragna](https://dev.to/aws/adding-amazon-bedrock-llama2-as-an-assistant-in-ragna-pdl) will help you get this all up and running. There is also lots of useful info in the project README. (#180)

**sample-ollama-server**

[sample-ollama-server](https://aws-oss.beachgeek.co.uk/4bv) is a project that folks who are interested in or already using ollama need to check out. This repo provides a AWS CloudFormation template to provision NVIDIA GPU EC2 instances with Ollama and Open WebUI, and include access to Amazon Bedrock foundation models (FMs). The solution can be deployed as a website for LLM interaction through Open WebUI, or as application development environment with Amazon DCV server.(#209)

**smart-assistant-agent**

[smart-assistant-agent](https://aws-oss.beachgeek.co.uk/3rd) is a project from AWS Community Builder Darya Petrashka that provides a solution to building an AWS Bedrock agent acting as a Telegram chat assistant. Check out the README for example videos of what this can do, as well as very detailed deployment instructions. (#193)

**streamlit-bedrock-claude-sample**

[streamlit-bedrock-claude-sample](https://aws-oss.beachgeek.co.uk/437) - I have featured Gary Stafford's open source projects and blog posts regularly in this newsletter. Gary has built a number of simple Streamlit applications to make it easy access the latest models and features of Amazon Web Services (AWS) Amazon Bedrock as part of several talks, workshops, and demonstrations he has done.  As part these, he has put together a simple Streamlit application that uses the Amazon Bedrock boto3 Python SDK to call the latest Anthropic Claude 3 family of multimodal foundation models. The application accepts a system and user prompt and generates a text-based response. The Streamlit app can be easily modified to incorporate new Bedrock features or as a starting point for your own applications. (#203)

**symfony-bref-starter**

[symfony-bref-starter](https://aws-oss.beachgeek.co.uk/3ta) is a starter kit for symfony projects using bref / serverless to allow easy deployments. bref is a fantastic and well tested way of running PHP applications via serverless, handling easy deployments and environments on AWS. Check out the README for some very cool stats around bref usage. The repo provides good examples and everything you need to get started.(#195)

**weaviate-on-eks**

[weaviate-on-eks](https://aws-oss.beachgeek.co.uk/3d4) this repository includes sample code that can be used to deploy and configure an instance of the [Weaviate](https://aws-oss.beachgeek.co.uk/3d5) distributed vector database on EKS. (#177)

**whisperx-on-aws-lambda**

[whisperx-on-aws-lambda](https://aws-oss.beachgeek.co.uk/49a) is a project from Vincent Claes that shows you how you can run [WhisperX](https://aws-oss.beachgeek.co.uk/49b) (one of the most versatile and feature-rich Whisper variation that provides fast automatic speech recognition) on AWS Lambda - WhisperX goes serverless! (#207)

**zero-downtime-deployment-tofu**

[zero-downtime-deployment-tofu](https://aws-oss.beachgeek.co.uk/438) is a repo from AWS Community Build Jorge Tovar that contains code examples using OpenTofu that shows how you can achieve zero downtime using a number of different approaches. Check out the supporting blog post for more details, [Zero Downtime Deployment in AWS with Tofu/Terraform and SAM](https://aws-oss.beachgeek.co.uk/439). This is this weeks essential repo to check out, and a good opportunity to learn about and become familiar with the different techniques and options you have. (#203)


# Demos and Samples

**2025-bootiful-aws-ai**

[2025-bootiful-aws-ai](https://aws-oss.beachgeek.co.uk/4aj) is just what you need if you are looking to explore the wonderful world of MCP in Java. James and Josh have put together this simple demo that uses the recently GAd Spring Boot AI libraries to make creating MCP clients and servers a breeze.(#208)

**a2a-advisory-trading**

[a2a-advisory-trading](https://aws-oss.beachgeek.co.uk/4gv) is a comprehensive demo of a serverless multi-agent trading advisory system built on AWS, leveraging Google's Agent2Agent Protocol using a2a SDK, Strands Agent, and built-in MCP tools from Strands SDK to deliver personalized investment analysis, risk assessment, and secure trade execution. This project serves as a reference implementation demonstrating how to design and deploy multi-agent systems using Google's Agent2Agent Protocol on AWS through a serverless architecture, powered by Strands Agent and Amazon Bedrock. It showcases patterns for building agent networks while leveraging cloud-native services. (#212)


**ai-driven-sql-generation**

[ai-driven-sql-generation](https://aws-oss.beachgeek.co.uk/3w0) this sample code from AWS Community Builder Hardik Singh Behl uses Amazon Bedrock with Spring AI to convert natural language queries to SQL queries, using Anthropic's Claude 3 Haiku model.(#197)

**amazon-bedrock-audio-summarizer**

[amazon-bedrock-audio-summarizer](https://aws-oss.beachgeek.co.uk/3w2) This project provides an automated way to transcribe and summarise audio files using AWS. We use Amazon S3, AWS Lambda, Amazon Transcribe, and Amazon Bedrock (with Claude 3 Sonnet), to create text transcripts and summaries from uploaded audio recordings. (#197)

**amazon-bedrock-serverless-prompt-chaining**

[amazon-bedrock-serverless-prompt-chaining](https://aws-oss.beachgeek.co.uk/47w) is an essential resource for AWS developers keen to discover examples of how to build complex, serverless, and highly scalable generative AI applications with prompt chaining and Amazon Bedrock.(#206)

**amazon-bedrock-slack-gateway**

[amazon-bedrock-slack-gateway](https://aws-oss.beachgeek.co.uk/3z8) lets you use Amazon Bedrock's generative AI to enable Slack channel members to access your organisations data and knowledge sources via conversational question-answering. You can connect to your organisation data via data source connectors and integrate it with Slack Gateway for Amazon Bedrock to enable access to your Slack channel members. It allows your users to converse with Amazon Bedrock using Slack Direct Message (DM) to ask questions and get answers based on company data, get help creating new content such as emails, and performing tasks. You can also invite it to participate in your team channels. In a channel users can ask it questions in a new message, or tag it in a thread at any point. Get it to provide additional data points, resolve a debate, or summarise the conversation and capture next steps.(#200)

**amazon-eks-chaos**

[amazon-eks-chaos](https://aws-oss.beachgeek.co.uk/4ar) this repo shows the steps involved to implement running chaos experiments on micro services in Amazon EKS using AWS Fault Injection Simulator with ChaosMesh and LitmusChaos. You can read about this in the supporting blog post,[ Chaos engineering on Amazon EKS using AWS Fault Injection Simulator](https://aws-oss.beachgeek.co.uk/4as).(#208)

**amazon-eks-running-webassembly**

[amazon-eks-running-webassembly](https://aws-oss.beachgeek.co.uk/3th) This repository contains code for building custom Amazon EKS AMIs using HashiCorp Packer. The AMIs include necessary binaries and configurations to enable you to run WebAssembly workloads in an EKS cluster and are based on Amazon Linux 2023. The runtimes used in the AMIs are Spin and WasmEdge. The respective containerd-shims are used for both runtimes. Deploying the cluster is done using Hashicorp Terraform. After the cluster is created, RuntimeClasses and example workloads are deployed to the cluster. If you are exploring Wasm, then this is for you. (#195)


**amplify-godot-engine-sample**

[amplify-godot-engine-sample](https://aws-oss.beachgeek.co.uk/44u) for the past few years, Godot has been one of the most active and popular open source projects. If you are not familiar with it, it provides a game engine that allows you to build 2D and 3D games (currently half way through a Udemy course, and loving it!).  If you wanted to know how you can use AWS Amplify with Godot, this repo provides some sample code using some of the Godot demo projects. (#204)

**aws-chatbot-fargate-python**

[aws-chatbot-fargate-python](https://aws-oss.beachgeek.co.uk/43f) is a new repo from AWS Hero Ran Isenberg that deploys a Streamlit Chatbot in an AWS Fargate-based ESC cluster web application using AWS CDK (Cloud Development Kit). The infrastructure includes an ECS cluster, Fargate service, Application Load Balancer, VPC, and WAF and includes security best practices with CDK-nag verification. The chatbot is based on an implementation by Streamlit and the initial prompt is that the chatbot is me, Ran the builder, a serverless hero and attempts to answer as me. The Chatbot uses custom domain (you can remove it or change it to your own domain) and assume an OpenAI token exists in the account in the form of a secrets manager secret for making API calls to OpenAI.(#203)

**aws-agentic-document-assistant**

[aws-agentic-document-assistant](https://aws-oss.beachgeek.co.uk/3gu) The Agentic Documents Assistant is an LLM assistant that provides users with easy access to information and insights stored across their business documents, through natural conversations and question answering. It supports answering factual questions by retrieving information directly from documents using semantic search with the popular RAG design pattern. Additionally, it answers analytical questions by translating user questions into SQL queries and running them against a database of entities extracted from the documents with a batch process. It is also able to answer complex multi-step questions by combining different tools and data sources using an LLM agent design pattern.(#181)

**aws-clean-rooms-lab**

[aws-clean-rooms-lab ](https://aws-oss.beachgeek.co.uk/3j5)is a workshop from AWS Security Hero Richard Fan that  will walk you through the setup of AWS Clean Rooms so you can try its different features. Richard wrote about this repo in his write up [Start building my AWS Clean Rooms lab](https://aws-oss.beachgeek.co.uk/3j6), which you should read to help you get started. This is a work in progress, but there is still a lot of stuff to get stuck into so worth checking out if AWS Clean Rooms is something that you are exploring. (#183)

**aws-genai-rfpassistant**

[aws-genai-rfpassistant](https://aws-oss.beachgeek.co.uk/43a) this repository contains the code and infrastructure as code for a Generative AI-powered Request for Proposal (RFP) Assistant leveraging Amazon Bedrock and AWS Cloud Development Kit (CDK). This could be very hand if responding to RFP's is something that you do and you want to look at ways of optimising your approach. The documentation in the repo is very comprehensive. I have not tried this one out for myself, but I have been involved in both writing and reviewing RFPs in the past, so understand the pain that led to the creation of this project. (#203)

**aws-eks-udp-telemetry**

[aws-eks-udp-telemetry](https://aws-oss.beachgeek.co.uk/44m) is a project that I have tried to build myself many moons ago. Did you know that many of the amazing (and very realistic) racing games for consoles allow you to export telemetry data? When I found this out many moons ago, I put together some simple code that showed some simple telemetry (speed, tyre temperature). I kind of stopped there with plans to do great things, but as is often the way, the project was left half finished. So I am delighted that this project from AWS Community Builder Amador Criado provides a more complete version, including a blog post ([Ingesting F1 Telemetry UDP real-time data in AWS EKS](https://aws-oss.beachgeek.co.uk/44n)) where he dives into the details of how everything works. (#204)

**aws-piday2024**

[aws-piday2024 ](https://aws-oss.beachgeek.co.uk/3r3)my colleague Suman Debnath has put together this AWS Pi Day 2024 repository, where you can explore various applications and examples using Amazon Bedrock, fine-tuning, and Retrieval-Augmented Generation (RAG). (#193)

**aws-real-time-web-shop-analytics**

[aws-real-time-web-shop-analytics](https://aws-oss.beachgeek.co.uk/3sh) this project delivers a code sample to experiment with real-time web analytics with a simplified online shop as web front, Apache Flink application for real-time pattern detection using Amazon Managed Service for Apache Flink, and an OpenSearch Dashboard to visualise the results using Amazon OpenSearch Service. This application is capable of ingesting click, add to cart, and purchase event from the online shop including the user that can be specified on the interface. Then the clickstream data is analysed for predefined patterns such as removing duplicates, add to cart events that are not followed by a purchase event within a specified timeframe etc. The results are then visualised in the OpenSearch Dashboard.(#194)

**aws-serverless-example-for-webxr**

[aws-serverless-example-for-webxr](https://aws-oss.beachgeek.co.uk/3si) This repository contains an example solution on how to enhance your WebXR applications using AWS Serverless Services, providing scalable, efficient, and seamless user experiences. (#194)

**big-data-summarization-using-griptape-bedrock-redshift**

[big-data-summarization-using-griptape-bedrock-redshift](https://aws-oss.beachgeek.co.uk/3gv) I have looked at Griptape in other blog posts, so it was nice to see this repo that provides sample code and instructions for a Big data summarisation example using this popular open-source library, together with Amazon Bedrock and Amazon Redshift. In this sample,  TitanXL LLM is used to summarise but Anthropic's Claude v2 is also used to drive the application. This application sample demonstrates how data can be pulled from Amazon Redshift and then passed to the summarisation model. The driving model is isolated from the actual data and uses the tools provided to it to orchestrate the application. (#181)

**Bedrock-MEAI-Sample**

[Bedrock-MEAI-Sample](https://aws-oss.beachgeek.co.uk/4ai) is just what you need if you are looking to integrate Amazon Bedrock with .NET MAUI, providing sample code that demonstrates how to create a chat client in a .NET MAUI mobile app.(#208)

**build-an-agentic-llm-assistant**

[build-an-agentic-llm-assistant](https://aws-oss.beachgeek.co.uk/43e) this repo provides code that you can follow along as part of the "Build an agentic LLM assistant on AWS" workshop. This hands-on workshop, aimed at developers and solution builders, trains you on how to build a real-life serverless LLM application using foundation models (FMs) through Amazon Bedrock and advanced design patterns such as: Reason and Act (ReAct) Agent, text-to-SQL, and Retrieval Augemented Generation (RAG). It complements the Amazon Bedrock Workshop by helping you transition from practicing standalone design patterns in notebooks to building an end-to-end llm serverless application. Check out the README for additional links to the workshop text, as well as more details on how this repo works.(#203)

**build-neptune-graphapp-cdk**

[build-neptune-graphapp-cdk](https://aws-oss.beachgeek.co.uk/3z9) this repo provides a quick example on how to build a graph application with Amazon Neptune and AWS Amplify. (#200)

**content-based-item-recommender**

[content-based-item-recommender](https://aws-oss.beachgeek.co.uk/44r) provides some example code the helps you deploy a content-based recommender system. It is called "content-based" as it bases the recommendation based on the matching between the input's content and the items' content from your database. This uses prompt to large-language models (LLM) and vector search to perform the recommendation. (#204)

**cost-news-slack-bot**

[cost-news-slack-bot](https://aws-oss.beachgeek.co.uk/3or) is a tool written in Python that read an RSS feed and selectively publish articles, based on keywords, to Slack via Webhook.  In the example, the tool checks the AWS 'What's New' RSS feed every minute for announcements related to cost optimisation. Perfect for customising and using it for your own use cases. (#190)

**deploy-crewai-agents-terraform**

[deploy-crewai-agents-terraform](https://aws-oss.beachgeek.co.uk/4d6) is a project designed to help you perform security audits and generate reports for your AWS infrastructure using a multi-agent AI system, leveraging the powerful and flexible framework provided by CrewAI. The AWS Security Auditor Crew architecture combines CrewAI's multi-agent framework with AWS services to provide comprehensive security auditing capabilities. The system can be deployed locally or to AWS using Terraform, with Amazon Bedrock powering the AI agents. (#210)


**deploy-langfuse-on-ecs-with-fargate**

[deploy-langfuse-on-ecs-with-fargate](https://aws-oss.beachgeek.co.uk/43j) This repository contains the AWS CDK Python code for deploying the Langfuse application using Amazon Elastic Container Registry (ECR) and Amazon Elastic Container Service (ECS). If you are not familiar with Langfuse, it is is an open-source LLM engineering platform that helps teams collaboratively debug, analyze, and iterate on their LLM applications.(#203)

**distill-cli**

[distill-cli](https://aws-oss.beachgeek.co.uk/3yz) is a new project from Amazon CTO Dr Werner Vogels, which uses Amazon Transcribe and Amazon Bedrock to create summaries of your audio recordings (e.g., meetings, podcasts, etc.) directly from the command line. Distill CLI takes a dependency on Amazon Transcribe, and as such, supports the following media formats: AMR, FLAC, M4A, MP3, MP4, Ogg, WebM, WAV. It is great to feature this latest project, with the previous one being featured in [#197](https://community.aws/content/2gPNtsdSfQRIpmbUrNyPrjUg54D/aws-open-source-newsletter-197). To go with this repo, there is a post too, [Introducing Distill CLI: An efficient, Rust-powered tool for media summarization](https://aws-oss.beachgeek.co.uk/3yy) where Werner shares his experience building this tool in Rust, and provides some closing thoughts too. (#200)

**domain-specific-genai-chatbot-with-serverless**

[domain-specific-genai-chatbot-with-serverless](https://aws-oss.beachgeek.co.uk/3t9) This sample demonstrates how to build a domain specific chatbot using serverless and Bedrock. The chatbot employs Retrieval-Augments Generation (RAG) and chat history to provide context for the LLM to answer. (#195)

**eks_demo**

[eks_demo](https://aws-oss.beachgeek.co.uk/4ah) is a nice demo repo from former colleague Seth Elliot that provides a ready to go set of AWS resources via Terraform, that deploys an Amazon EKS cluster with a sample guestbook application using a number of supporting AWS services (check out the repo for more info). It is intended as an easy to understand environment for you to study and learn. (#208)

**fmbench-orchestrator**

[fmbench-orchestrator](https://aws-oss.beachgeek.co.uk/467) this repo is a tool designed to automate the deployment and management of FMBench for benchmarking on Amazon EC2, Amazon SageMaker and Amazon Bedrock. In case of benchmarking on EC2, we could benchmark on multiple instances simultaneously, and these instances can be of different instance types (so you could run g6e, p4de and a trn1 instances via the same config file), in different AWS regions and also test multiple FMBench config files. This orchestrator automates the creation of Security Groups, Key Pairs, EC2 instances, runs FMBench for a specific config, retrieves the results, and shuts down the instances after completion. Thus it simplifies the benchmarking process (no more manual creation of SageMaker Notebooks, EC2 instances and cleanup, downloading results folder) and ensures a streamlined and scalable workflow. Very detailed README that provides much more details on how this works. (#205)

**genai-asl-avatar-generator**

[genai-asl-avatar-generator](https://aws-oss.beachgeek.co.uk/405) this repo provides code that demonstrates the power of a number of AWS services working in concert to enable seamless translation from speech/text to American Sign Language (ASL) avatar animations.  Check out the supporting blog post, [Generative AI-powered American Sign Language Avatars](https://aws-oss.beachgeek.co.uk/406), where Suresh Poopandi walks through the project and code and how it all hangs together. (#201)

**gen-ai-cve-patching**

[gen-ai-cve-patching](https://aws-oss.beachgeek.co.uk/3vy) This repository introduces an innovative automated remediation pipeline, designed to effectively address vulnerabilities detected by AWS ECR Inspector. By leveraging Generative AI through Amazon Bedrock's in-context learning, this solution significantly enhances the security posture of application development workflows. The architecture integrates with CI/CD processes, offering a comprehensive and automated approach to vulnerability management. The architecture diagram provided illustrates the solution's key components and their interactions, ensuring a holistic vulnerability remediation strategy.(#197)

**generative-ai-newsletter-app**

[generative-ai-newsletter-app](https://aws-oss.beachgeek.co.uk/3vx) is a ready-to-use serverless solution designed to allow users to create rich newsletters automatically with content summaries that are AI-generated. The application offers users the ability to influence the generative AI prompts to customise how content is summarised such as the tone, intended audience, and more. Users can stylise the HTML newsletter, define how frequently newsletters are created and share the newsletters with others.(#197)

**generative-bi-using-rag**

[generative-bi-using-rag](https://aws-oss.beachgeek.co.uk/40f) is a comprehensive framework designed to enable Generative BI capabilities on customised data sources (RDS/Redshift) hosted on AWS. It offers the following key features:

* Text-to-SQL functionality for querying customised data sources using natural language.
* User-friendly interface for adding, editing, and managing data sources, tables, and column descriptions.
* Performance enhancement through the integration of historical question-answer ranking and entity recognition.
* Customise business information, including entity information, formulas, SQL samples, and analysis ideas for complex business problems. 
* Add agent task splitting function to handle complex attribution analysis problems.
* Intuitive question-answering UI that provides insights into the underlying Text-to-SQL mechanism.
* Simple agent design interface for handling complex queries through a conversational approach.

(#201)

**generate-s3-accelerate-presigned-url**

[generate-s3-accelerate-presigned-url](https://aws-oss.beachgeek.co.uk/3tg) this sample project demonstrates how to generate an Amazon S3 pre-signed URL with S3 Transfer Acceleration, using Amazon API Gateway REST API and AWS Lambda function. The Lambda function, composed in Java 21, is responsible for generating a presigned URL to allow customers to upload a single file into S3, with S3 Transfer Acceleration enabled, to speed up content transfers to Amazon S3 securely, over long distances. The API is protected by IAM authentication, to protect against non-authenticated users.(#195)

**gen-ai-on-eks**

[gen-ai-on-eks](https://aws-oss.beachgeek.co.uk/3d2) this repository aims to showcase how to finetune a FM model in Amazon EKS cluster using, JupyterHub to provision notebooks and craft both serving and training scripts, RayOperator to manage Ray Clusters and Karpenter to manage Node Scaling. (#177)

**genai-sports-commentary**

[genai-sports-commentary](https://aws-oss.beachgeek.co.uk/3tj) is a repo that appeals to my inner spots fan, and looks at how you can apply generative AI and large language models, to create highly personalized contents for sport fans. In our example, we'll use a foundation model to generate play by play commentary based on American Football game data synthetically created (in reality, the data could be directly sourced from the stadiums, or cloud). We'll instruct the model to generate variety of commentary using different prompts. For instance, create prompts to the model to generate commentary in particular commentary writing style, or a particular language that the fans would prefer.(#195)

**get-the-news-rss-atom-feed-summary**

[get-the-news-rss-atom-feed-summary](https://aws-oss.beachgeek.co.uk/3l6) is a fantastic demo that demonstrates some of the capabilities that using large language models can help you introduce into your applications. The demo code provides a summary of the most recent news from an RSS or Atom feed using Amazon Bedrock. (#185)


**golang-url-shortener**

[golang-url-shortener](https://aws-oss.beachgeek.co.uk/3j3) is a project that you can build from Furkan Gulsen that deploys a URL shortener service, built with Go and Hexagonal Architecture, leverages a serverless approach for efficient scalability and performance. It uses a variety of AWS services to provide a robust, maintainable, and highly available URL shortening service. Are URL Shortners the new todo app? Not sure but I really like the quality of the documentation of this project, and like I did last year with implementing a serverless web analytics solution, I am going to check this project out and see if it would be a good replacement for the tool I currently use, YOURLS. Check out the discussion on reddit [here](https://www.reddit.com/r/aws/comments/18nnfix/url_shortener_hexagonal_serverless_architecture/).(#183)

**hackernews-converse-api-demo**

[hackernews-converse-api-demo](https://aws-oss.beachgeek.co.uk/41j) this repo provides some sample code on how you can use the Amazon Bedrock Converse API, using an example of summarising comments from a Hacker News thread. It is a simple example, but shows you how easy it is to incorporate generative AI in your own applications. You can check out the supporting blog post too, [Save time reading Hacker News comments using Converse API](https://aws-oss.beachgeek.co.uk/41l) (#202)

**improve-employee-productivity-using-genai**

[improve-employee-productivity-using-genai](https://aws-oss.beachgeek.co.uk/43h) is an innovative code sample designed to elevate the efficiency and effectiveness of writing tasks through the integration of AI technologies. Tailored for AWS users, this assistant example utilises Amazon Bedrock and generative AI models to automate the creation of initial templates, that can also be customised for your own needs. Users can input both text and images, benefiting from the multimodal capabilities of cutting-edge AI like the Claude 3 family, which supports dynamic interaction with multiple data types. The README is very comprehensive and covers not only how to set up and configure this project, but also has lots of great short videos of it in action. (#203)

**Lambda-MCP-Server**

[Lambda-MCP-Server](https://aws-oss.beachgeek.co.uk/4bl) - this project from my colleague Mike Chambers demonstrates a powerful and developer-friendly way to create serverless MCP (Model Context Protocol) tools using AWS Lambda. It showcases how to build a stateless, serverless MCP server with minimal boilerplate and an excellent developer experience. The included client demonstrates integration with Amazon Bedrock, using the Bedrock Converse API and Amazon Nova Pro to build an intelligent agent.(#209)

**maplibregljs-amazon-location-service-route-calculators-starter**

[maplibregljs-amazon-location-service-route-calculators-starter](https://aws-oss.beachgeek.co.uk/3p9) is a new repo from AWS Hero Yasunori Kirimoto that provides an example of how you can start routing with MapLibre GL JS and Amazon Location Service. He has also put together a blog post to help get you start, [Building a Route Search Function with Amazon Location SDK and API Key Function ](https://aws-oss.beachgeek.co.uk/3pa) (#191)

**makit-llm-lambda**

[makit-llm-lambda ](https://aws-oss.beachgeek.co.uk/3cx)this repo from Martyn Kilbryde is an example of how you can run a Large Language Model (LLM) inside an AWS Lambda Function.  Whilst the code will help you deploy to AWS Lambda, it can be ran locally inside Docker for testing as well. The function contains the full LLM model and the code to use the model, allowing basic text generation from a HTTP call into it. (#177)

**multi-tenant-chatbot-using-rag-with-amazon-bedrock**

[multi-tenant-chatbot-using-rag-with-amazon-bedrock](https://aws-oss.beachgeek.co.uk/3c5) provides a solution for building a multi-tenant chatbot with Retrieval Augmented Generation (RAG). RAG is a common pattern where a general-purpose language model is queried with a user question along with additional contextual information extracted from private documents. To help you understand and deploy the code, check out the supporting blog post from Farooq Ashraf, Jared Dean, and Ravi Yadav, [Build a multi-tenant chatbot with RAG using Amazon Bedrock and Amazon EKS](https://aws-oss.beachgeek.co.uk/3c6) (#177)

**news-clustering-and-summarization**

[news-clustering-and-summarization](https://aws-oss.beachgeek.co.uk/44l) this repository contains code for a near real-time news clustering and summarisation solution using AWS services like Lambda, Step Functions, Kinesis, and Bedrock. It demonstrates how to efficiently process, embed, cluster, and summarise large volumes of news articles to provide timely insights for financial services and other industries. This solution aims to launch a news Event feature that clusters related news stories into summaries, providing customers with near real-time updates on unfolding events. This augmented news consumption experience will enable users to easily follow evolving stories while maximising relevance and reducing the firehose of information for articles covering the same event. By tailoring news clusters around key events, this application can improve customer satisfaction and engagement. Detailed docs will help you get up and running in no time. (#204)


**newsletter-manager-template**

[newsletter-manager-template](https://aws-oss.beachgeek.co.uk/3sa) is a project from AWS Community Builder Matteo Depascale that provides backend service orchestrations for newsletter builders. (#194)

**nova_virtual_stylist**

[nova_virtual_stylist](https://aws-oss.beachgeek.co.uk/4gr) provides a super fun demo that showcases the power Amazon Nova Canvas and Nova Reel that allows you to upload your photo and clothing items (or generate them with AI), and then see how they look with virtual try-on, then create stunning fashion videos to showcase your style. I have seen this project in action, and its pretty cool so defo add this to your list of projects to try out. (#212)

**public-file-browser-for-amazon-s3**

[public-file-browser-for-amazon-s3](https://aws-oss.beachgeek.co.uk/3qk) allows customers to create a simple PUBLIC file repository using Amazon S3 and Amazon CloudFront. This sample code deploys a website and a public files S3 bucket which can be loaded with any files they wish to publish publicly online.(#192)

**quarkus-bedrock-demo**

[quarkus-bedrock-demo](https://aws-oss.beachgeek.co.uk/3cv) This is a sample project from my colleague Denis Traub, based on [work from Vini](https://aws-oss.beachgeek.co.uk/3b2)  , that demonstrates how to access Amazon Bedrock from a Quarkus application deployed on AWS Lambda. (#177)

**real-time-social-media-analytics-with-generative-ai**

[real-time-social-media-analytics-with-generative-ai](https://aws-oss.beachgeek.co.uk/3x5) this repo helps you to build and deploy an AWS Architecture that is able to combine streaming data with GenAI using Amazon Managed Service for Apache Flink and Amazon Bedrock. (198)

**reinvent-session-concierge**

[reinvent-session-concierge](https://aws-oss.beachgeek.co.uk/3gq) is potentially a very useful tool for those of you heading out to re:Invent, and wanting to make sure that you make the most of your time there by attending the sessions of most interest to you. This project uses Amazon Bedrock Titan text embeddings stored in a PostgreSQL database to enable generative AI queries across the re:Invent session data. It combines both semantic search and traditional queries. I am going to be trying it out later today to help me plan my online viewing. (#181)

**sample-agents-with-nova-act-and-mcp**

[sample-agents-with-nova-act-and-mcp](https://aws-oss.beachgeek.co.uk/4bm) - This repository demonstrates how to build intelligent web automation agents using Amazon Nova Act integrated with MCP (Model Context Protocol). MCP provides a standardised way to connect AI models to different data sources and tools - think of it like a "USB-C port for AI applications." The repo provides requirements for running these demos as well as an overview of the different example use cases.(#209)

**sample-aws-dax-go-v2**

[sample-aws-dax-go-v2](https://aws-oss.beachgeek.co.uk/4an)  is the official AWS DAX SDK for the Go programming language. In addition to this, [sample-aws-dax-go-v2-sample](https://aws-oss.beachgeek.co.uk/4al)is some sample code / application showing how to use DynamoDB Accelerator (DAX) Go Language SDK Client.(#208)

**sample-code-for-a-secure-vault-using-aws-nitro-enclaves**

[sample-code-for-a-secure-vault-using-aws-nitro-enclaves](https://aws-oss.beachgeek.co.uk/43c) - This repository contains a sample secure vault solution built using AWS Nitro Enclaves, a feature available exclusively through the AWS Nitro System hypervisor on supported Amazon EC2 instances. A vault solution is useful when you need to ensure sensitive data (such as Protected Health Information (PHI)/Personally Identifiable Information (PII)) is properly secured at rest and can only be decrypted through approved channels. Check out more details about how cool Nitro Enclaves are by reading the supporting documentation for this project, [About the AWS Nitro Enclaves Vault](https://aws-oss.beachgeek.co.uk/43b).(#203)

**sample-genai-underwriting-workbench-demo**

[sample-genai-underwriting-workbench-demo](https://aws-oss.beachgeek.co.uk/4h1) a demonstration project showcasing the power of Amazon Bedrock and Claude 3.7 Sonnet in transforming life insurance underwriting workflows. This solution leverages intelligent document processing to streamline the underwriting process by automatically extracting, analyzing, and making accessible critical information from insurance applications and related documents. (#212)

**Sample-Model-Context-Protocol-Demos**

[Sample-Model-Context-Protocol-Demos](https://aws-oss.beachgeek.co.uk/4bt) - this repo provides a list of examples of how to use Model Context Protocol (MCP) with AWS. Keep an eye on this repo as it will grow as MCP is the new hotness at the moment.(#209)

**serverless-genai-food-analyzer-app**

[serverless-genai-food-analyzer-app](https://aws-oss.beachgeek.co.uk/3x6) provides code for a personalised GenAI nutritional web application for your shopping and cooking recipes built with serverless architecture and generative AI capabilities. It was first created as the winner of the AWS Hackathon France 2024 and then introduced as a booth exhibit at the AWS Summit Paris 2024. You use your cell phone to scan a bar code of a product to get the explanations of the ingredients and nutritional information of a grocery product personalised with your allergies and diet. You can also take a picture of food products and discover three personalised recipes based on their food preferences. The app is designed to have minimal code, be extensible, scalable, and cost-efficient. It uses Lazy Loading to reduce cost and ensure the best user experience. (#198)

**serverless-rss-filtered-feed-gen**

[serverless-rss-filtered-feed-gen](https://aws-oss.beachgeek.co.uk/3dz) This is a configurable serverless solution that generates filtered rss feeds and makes them public accessible. Defined RSS sources are read at a given interval and new filtered feeds are generated and stored. The architecture uses a minimum number of AWS services to keep it easy to maintain and cost-effective. (#178)

**scaling-with-karpenter**

[scaling-with-karpenter](https://aws-oss.beachgeek.co.uk/44o) is a project from AWS Community Builder Romar Cablao that provides a demo of how Karpenter autoscale your Kubernetes clusters. To help you get started with the code, check out his supporting blog post, [Scaling & Optimizing Kubernetes with Karpenter - An AWS Community Day Talk](https://aws-oss.beachgeek.co.uk/44p).(#204)

**slackrock**

[slackrock](https://aws-oss.beachgeek.co.uk/3yw) is a conversational AI assistant powered by Amazon Bedrock & your favorite cutting-edge frontier models. The project is focused on cost efficiency & simplicity, while supporting a wide variety of AI models with differing strengths & weaknesses to fit the widest array of use cases. Converse with your favourite LLMs without ever leaving Slack! (#200)

**song-identification-on-aws**

[song-identification-on-aws](https://aws-oss.beachgeek.co.uk/3qj) This repo contains sample code that demonstrates how you can "fingerprint" your songs, and then detect the presence of your songs in either stored audio files like MP3s, or within streaming media. The underlying idea is to convert audio data into a spectrogram, and then isolate important markers within the spectrogram that will allow us to identify music. Roughly 10000 to 25000 fingerprints will be created for an average length song. Each fingerprint is stored as a large integer. See the blog post for more details about how the system works. (#192)

**strands-a2a-demo**

[strands-a2a-demo](https://aws-oss.beachgeek.co.uk/4gq) this repo provides a demo that showcases multiple AI agents using the complete official A2A (agent-to-agent) Python SDK patterns from google-a2a/a2a-samples. The agents use the Strands Agents Python SDK for their LLM implementation, fully integrated with the official A2A SDK patterns. This project features a unified a2a-client with multiple modes of operation, including web-based UI, voice interaction capabilities, and comprehensive testing.(#212)

**strands-agent-on-lambda**

[strands-agent-on-lambda](https://aws-oss.beachgeek.co.uk/4gp) this repo contains a sample implementation of user-aware AI Agent and MCP Server running on AWS Lambda. The sample implements an AI-based Travel Agent for a fictitious corporation AcmeCorp.(#212)

**strands-serverless**

[strands-serverless](https://aws-oss.beachgeek.co.uk/4f1) is some sample code from my colleague **Didier Durand** that provides a serverless implementation of Strands Agents. This first experiment is the composition of Strands Agents with Chainlit (web-based interface) in one bundle. It delivers a complete AI agent in one Docker image running as an Amazon Lightsail virtual cloud server. (#211)

**streamlit-bedrock-claude-sample**

[streamlit-bedrock-claude-sample](https://aws-oss.beachgeek.co.uk/437) - I have featured Gary Stafford's open source projects and blog posts regularly in this newsletter. Gary has built a number of simple Streamlit applications to make it easy access the latest models and features of Amazon Web Services (AWS) Amazon Bedrock as part of several talks, workshops, and demonstrations he has done.  As part these, he has put together a simple Streamlit application that uses the Amazon Bedrock boto3 Python SDK to call the latest Anthropic Claude 3 family of multimodal foundation models. The application accepts a system and user prompt and generates a text-based response. The Streamlit app can be easily modified to incorporate new Bedrock features or as a starting point for your own applications. (#203)

**svdxt-sagemaker-huggingface**

[svdxt-sagemaker-huggingface](https://aws-oss.beachgeek.co.uk/3uo) is the latest demo repo from regular contributor Gary Stafford, that showcases some of the cool stuff Gary has been writing about in the generative AI space. This time he takes a look at the emerging field of generating videos through Stability AI's Stable Video Diffusion XT (SVT-XT). This foundation model is a diffusion model that takes in a still image as a conditioning frame and generates a video from it.(#196)

**swift-chat**

[swift-chat](https://aws-oss.beachgeek.co.uk/47y)  is a fast and responsive AI chat application developed with React Native and powered by Amazon Bedrock. With its minimalist design philosophy and robust privacy protection, it delivers real-time streaming conversations and AI image generation capabilities across Android, iOS, and macOS platforms. Check out the README for plenty of in depth details including sample screenshots from mobile simulators. Essential for any Swift developer (#206)

**terraform-f1-telemetry-infra**

[terraform-f1-telemetry-infra](https://aws-oss.beachgeek.co.uk/462) this project aims to develop a UDP listener to capture, parse, and visualise this data in real-time using various AWS services, including ECS, Elastic Load Balancing, and Route 53. A number of racing games on leading consoles provide the ability to send telemetry data to a target IP address, which this project will then ingest. Very cool stuff.(#205)

**twinmaker-dynamicscenes-crossdock-demo**

[twinmaker-dynamicscenes-crossdock-demo](https://aws-oss.beachgeek.co.uk/44q) provides code to create a demonstration of the AWS IoT TwinMaker dynamic scenes feature using a 'cross-dock' warehouse as an example. Using this demonstration code, the environment allows the simulation of goods on pallets entering the warehouse at the inbound docks, transition through sorting and then on to the outbound dock. (#204)

**valkey-python-demo**

[valkey-python-demo](https://aws-oss.beachgeek.co.uk/41k) provides some sample code that shows you how you can connect to a Valkey server using three different types of client. Existing Redis clients, the Valkey client, and the all new GLIDE client too. I put together a quick blog post on how I put this code together, so check it out - [Using Amazon Q Developer to update Valkey client code](https://aws-oss.beachgeek.co.uk/41m) (#202)

**valkey-finch**

[valkey-finch](https://aws-oss.beachgeek.co.uk/41n) is a quick recipe on how to run Valkey in a container using Finch. It did not work out of the box for me, and I had to figure out how to get it working. Now you can save yourself the trouble and check out this configuration. I also put a short blog on this, so check out [Getting started with Valkey and Finch](https://aws-oss.beachgeek.co.uk/40u) (#202)

**video-understanding-solution**

[video-understanding-solution](https://aws-oss.beachgeek.co.uk/3sj) This is a deployable solution to help save your time in understanding videos without having to watch every video. You can upload videos and this solution can generate AI-powered summary and entities extraction for each video. It also supports Q&A about the video like "What is funny about the video?", "How does Jeff Bezos look like there?", and "What shirt did he wear?". You can also search for videos using semantic search e.g. "Amazon's culture and history". This solution extracts information from visual scenes, audio, visible texts, and detected celebrities or faces in the video. It leverages an LLM which can understand visual and describe the video frames. You can upload videos to your Amazon Simple Storage Service (S3) bucket bucket by using AWS console, CLI, SDK, or other means (e.g. via AWS Transfer Family). This solution will automatically trigger processes including call to Amazon Transcribe for voice transcription, call to Amazon Rekognition to extract the objects visible, and call to Amazon Bedrock with Claude 3 model to extract scenes and visually visible text. The LLM used can perform VQA (visual question answering) from images (video frames), which is used to extract the scene and text. This combined information is used to generate the summary and entities extraction as powered by generative AI with Amazon Bedrock. The UI chatbot also uses Amazon Bedrock for the Q&A chatbot. The summaries, entities, and combined extracted information are stored in S3 bucket, available to be used for further custom analytics. (#194)

**webapp-form-builder**

[webapp-form-builder ](https://aws-oss.beachgeek.co.uk/43d) - this repo was built to accelerate the development of web forms on the frontend using the AWS Cloudscape Design System. Cloudscape is an open source design system to create web applications. It was built for and is used by Amazon Web Services (AWS) products and services. This solution provides you with a sample application that utilises components of the Cloudscape Design System that are commonly used in web-forms where users are required to input data.  Check out the more expansive README for more details of how this works and how to get started.(#203)

**whats-new-summary-notifier**

[whats-new-summary-notifier](https://aws-oss.beachgeek.co.uk/3x4) is a demo repo that lets you build a generative AI application that summarises the content of AWS What's New and other web articles in multiple languages, and delivers the summary to Slack or Microsoft Teams. (#198)

**wio-from-diagram-to-code-with-amazon-q-developer**

[wio-from-diagram-to-code-with-amazon-q-developer](https://aws-oss.beachgeek.co.uk/4a8) this project from AWS Community Builder Olivier Lemaitre that demonstrates how you can generate diagrams from an application code, but also how to generate code from diagrams using Amazon Q Developer in the Visual Studio Code IDE. (#208)

**youtube-video-summarizer-with-bedrock**

[youtube-video-summarizer-with-bedrock](https://aws-oss.beachgeek.co.uk/3j7) is a example project from Zied Ben Tahar that uses large language models to create a YouTube video summariser, allowing you to sift through You Tube videos and get an high level summary of them, allowing you to make better decisions as to whether you want to spend more time watching the video.  Zied has also put together a supporting blog post, [AI powered video summariser with Amazon Bedrock](https://aws-oss.beachgeek.co.uk/3j8) that provides everything you need to get this project up and running for yourself. (#183)

**zero-downtime-deployment-tofu**

[zero-downtime-deployment-tofu](https://aws-oss.beachgeek.co.uk/438) is a repo from AWS Community Build Jorge Tovar that contains code examples using OpenTofu that shows how you can achieve zero downtime using a number of different approaches. Check out the supporting blog post for more details, [Zero Downtime Deployment in AWS with Tofu/Terraform and SAM](https://aws-oss.beachgeek.co.uk/439). This is this weeks essential repo to check out, and a good opportunity to learn about and become familiar with the different techniques and options you have. (#203)

# Industry use cases

**garnet-framework**

[garnet-framework](https://aws-oss.beachgeek.co.uk/3e1) Garnet is an open-source framework for building scalable, reliable and interoperable platforms leveraging open standards, FIWARE open source technology and AWS Cloud services. It supports the development and integration of smart and efficient solutions across multiple domains such as Smart Cities, Regions and Campuses, Energy and Utilities, Agriculture, Smart Building, Automotive and Manufacturing. The repo provides code and links to the dedicated documentation site to help you get started. (#178)

**geo-location-api**

[geo-location-api](https://aws-oss.beachgeek.co.uk/3k0) is a project for the .NET developers out there, that provides a NET Web API for retrieving geolocations. The  geolocation data is provided by MaxMind GeoLite2. (#184)

**res**

[res](https://aws-oss.beachgeek.co.uk/3f0) Research and Engineering Studio on AWS (RES) is an open source, easy-to-use web-based portal for administrators to create and manage secure cloud-based research and engineering environments. Using RES, scientists and engineers can visualise data and run interactive applications without the need for cloud expertise. With just a few clicks, scientists and engineers can create and connect to Windows and Linux virtual desktops that come with pre-installed applications, shared data, and collaboration tools they need. With RES, administrators can define permissions, set budgets, and monitor resource utilisation through a single web interface. RES virtual desktops are powered by Amazon EC2 instances and NICE DCV. RES is available at no additional charge. You pay only for the AWS resources needed to run your applications. (#180)









