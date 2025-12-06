# PROMPT ENGINEERING AI AGENT
An AI agent that helps researchers and developers learn, design, and optimize prompts for language models effectively.

## AGENT INITIALIZATION:

1. Install Python3.12.10 - https://www.python.org/downloads/release/python-31210/
2. Install python dependencies
	pip install -r requirements.txt
3. Install Ollama - https://ollama.com/download. 
   Execute below command in the terminal to download the phi model.
	ollama pull phi
   Ollama offers various models. Please browse through the models here - https://ollama.com/search. Based on the selected LLM, .env file must be updated and the model must be downloaded.
4. Execute the PromptVA_LangChain.py file

## NOTES:

1. Databases Folder - Has the snapshots of the database collection. Note that any change made it to this databases will not impact the Model responses. These are just for viewing purposes.

## IDEA

Use of Large Language Models are exponentially increasing in real world applications. This has in turn triggered the importance of effective prompt engineering significantly. While users in the technological/AI space may have sufficient knowledge to design high-quality prompts independently, many lack the experience or knowledge to do so, especially with the rapid evolution of prompting strategies.
This project aims to build an LLM powered Prompt Engineering Virtual Assistant that supports users in learning, designing, and optimizing prompts for Language models. The Virtual Assistant is capable of:
- Suggesting best prompting techniques for various NLP tasks
- Explain and provide examples for different prompt techniques
- Recommend prompt templates for different techniques
- Provide information on latest trends and research in the field of Prompt Engineering

Using this assistant will empower users with knowledge and tools needed to engage with LLMs effectively and boost productivity, consistency, and quality in prompt design. This will ultimately contribute to making AI user friendly to public and contributing to higher-quality research.

## ARCHITECTURE
Below is the detailed description of the architecture of Prompt Engineering Virtual Assistant:
-	Databases: All the databases are hosted on MongoDB Atlas and accessed locally on the computer using MongoDB Compass. Security Testing was performed to ensure that these databases are not corrupted through user prompts. The details on this will be explained more in the coming sections. The assistant has access to three databases:
    -	Prompt Techniques: This database contains the best prompting technique for a vast variety of NLP tasks. While many articles are available online, there was no dataset that had this information. This database was hence compiled by going through various research papers. Below is a snapshot of few entries in the database.
    -	Prompt Templates: This database has the ideal template for numerous prompting techniques. This database has also been customized by gathering information from various sources. Below is an overview of the information stored in Prompt Templates database.
    -	User History: This database is used to log the user interaction with the assistant. All the queries, responses, timestamps are stored within this database. This database plays a key role in Prompt Caching implementation and will be discussed in the later sections of the report. 

-	LLM Backend - Two models were experimented with for this project – Phi-2 (2.7B), Mistral (7B). The models are hosted on Ollama and executed locally.
-	Orchestration Layer - The orchestration layer is built using LangChain agent. The agent used for this project is a Zero-Shot ReAct agent, an agent that thinks step-by-step and decides when to use tools without being fine tuned for specific tool usage. The tools provided to the agent are independent functions designed specifically for a task. While the overall workflow is governed by the LangChain agent, a few heuristic conditions are incorporated to guide the agent toward appropriate tool usage. The tools designed for this assistant are:
  - Prompt Template Identifier: Gets ready-made templates from the database – Prompt Templates for known prompting techniques.
  - Prompt Technique Recommender: Recommends best prompting strategy for a given NLP task from the database – Prompt Technique
  - User Interaction Logger: Saves the query and response history to User History database.
  - Web Search Tool: Performs DuckDuckGo search for up-to-date information from the web.
  - User Prompt Metadata Generator: Generates metadata for the user prompt. This metadata helps in designing the system prompt to the agent.

-	Web Search - The assistant is equipped to perform Web Search to gather up-to-date information on Prompt Engineering. The web search is triggered when information to requested regarding the latest trends, or research in conferences/companies. The assistant uses DuckDuckGo search tool provided by LangChain package.
-	User Interface - A simple chat interface is implemented with example prompt provided. The user interface is implemented using Gradio. 

