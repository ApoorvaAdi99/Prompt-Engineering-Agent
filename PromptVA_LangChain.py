# -*- coding: utf-8 -*-
"""
Prompt Engineering Assistant 
"""
import os
import re
from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from langchain.llms import Ollama
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import gradio as gr
from datetime import datetime
from functools import lru_cache
from typing import Dict, List
from langchain.tools import DuckDuckGoSearchRun
from datetime import timedelta
import json

# MongoDB connection
class MongoDBManager:
    def __init__(self, uri: str):
        self.client = MongoClient(uri)
        self.db = self.client["PromptAssistant"]
        self.prompt_templates = self.db["PromptingTemplates"]
        self.prompting_techniques = self.db["PromptingTechniques"]
        self.user_history = self.db["UserHistory"]
    
    def fetch_prompt_template(self, technique_name: str) -> str:
        result = self.prompt_templates.find_one({"Prompting Technique": technique_name})
        return result["Standard Template"] if result else "No template found for this technique."
    
    
    def fetch_best_technique(self, nlp_task: str) -> str:
        result = self.prompting_techniques.find_one({"nlp_task": nlp_task})
        return result["prompting_technique"] if result else "No known technique for this NLP task."
    
    def log_interaction(self, user_id: str, user_prompt: str, assistant_response: str):
        self.user_history.insert_one({
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "user_prompt": user_prompt,
            "assistant_response": assistant_response
        })
    
    def get_user_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        return list(self.user_history.find({"user_id": user_id})
                          .sort("timestamp", -1)
                          .limit(limit))
    
    def get_cached_response(self, prompt: str, user_id: str = "gradio_user"):
        """Check for cached response within last 24 hours"""
        cache_entry = self.user_history.find_one({
            "user_id": user_id,
            "user_prompt": prompt,
            "timestamp": {"$gt": datetime.utcnow() - timedelta(hours=1)}
        }, sort=[("timestamp", -1)])
        
        return cache_entry["assistant_response"] if cache_entry else None
    


# Core Assistant Functionality
class PromptEngineeringAssistant:
    def __init__(self):
        self.db = MongoDBManager(os.getenv("MONGO_URI"))
        self.llm = Ollama(model="phi")  
        self.search_tool = DuckDuckGoSearchRun()
        self.initialize_agent()

    def initialize_agent(self):
        tools = [
            Tool(
                name="fetch_prompt_template",
                func=self.db.fetch_prompt_template,
                description="""Useful when user asks for specific examples or templates of prompting techniques. 
                Input should be the exact name of a known prompting technique (e.g. 'chain-of-thought', 'few-shot prompting').
                Output will be a ready-to-use template with placeholders for user inputs."""
            ),
            Tool(
                name="fetch_best_technique",
                func=self.db.fetch_best_technique,
                description="""Useful when user asks about the most effective approach for a specific NLP task.
                Input should be the name of an NLP task (e.g. 'summarization', 'sentiment analysis').
                Output will recommend 1-3 suitable prompting techniques with brief rationale."""
            ),
            Tool(
                name="log_user_interaction",
                func=self.db.log_interaction,
                description="""Useful for recording all user queries and assistant responses for history.
                Always call this after generating a final response.
                Input requires user_id, user_prompt, and assistant_response."""
            ),
            Tool(
                name="web_search",
                func=self.search_tool.run,
                description="""Useful when:
                1. User asks about recent developments in prompt engineering (last 1-2 years)
                2. Query requires up-to-date information not in the knowledge base
                3. Seeking examples from current practice
                4. Verification of facts or techniques
                Input should be a clear search query string."""
            ),
            Tool(
                name="classify_prompt",
                func=self._classify_prompt_type,
                description="""MUST BE CALLED FIRST for every new user query. 
                Analyzes the prompt to determine:
                1. If it's prompt engineering related
                2. What type of NLP task is involved
                3. What action the assistant should take
                4. Whether web search might be needed
                Input is the raw user prompt.
                Output is a JSON with classification metadata."""
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def detect_web_search_needed(self, prompt: str) -> bool:
        """
        Check if prompt needs web search based on keywords.
        Returns True if web search is likely needed.
        """
        # Keywords that typically indicate need for web search
        web_search_keywords = [
            # Time-sensitive terms
            "latest", "recent", "new", "current", "upcoming", "2023", "2024", "this year",
            "last year", "last month", "recently", "now", "today", "currently",
            
            # Research/trend terms
            "research", "study", "paper", "survey", "trend", "advance", "development",
            "breakthrough", "innovation", "state of the art", "cutting edge", "emerging",
            
            # News/event terms
            "news", "update", "announcement", "conference", "workshop", "event", "meetup",
            "NeurIPS", "ICML", "ACL", "EMNLP", "AAAI", "arXiv", "preprint",
            
            # Comparative/competitive terms
            "compare", "vs", "versus", "difference between", "alternative to", "competitor",
            "competition", "benchmark", "ranking",
            
            # Verification terms
            "is it true that", "verify", "fact check", "source", "citation", "reference",
            "according to", "who said", "where to find",
            
            # Location/entity specific
            "in [country]", "in [city]", "[company]'s", "[university]'s", "Google's", "OpenAI's",
            "Anthropic's", "Microsoft's"
        ]
        
        # Check if any keyword exists in prompt (case-insensitive)
        prompt_lower = prompt.lower()
        keyword_match = any(keyword.lower() in prompt_lower for keyword in web_search_keywords)
        
        # Special patterns that definitely need web search
        patterns_requiring_search = [
            r"\b(since|after)\s+\d{4}\b",  # "since 2022", "after 2021"
            r"\b(in|during)\s+(2023|2024)\b",
            r"\b(last|past)\s+(year|month)\b",
            r"\bcurrent\s+(state|status)\b",
            r"\bwhat's\s+new\b",
        ]
        
        pattern_match = any(re.search(pattern, prompt_lower) for pattern in patterns_requiring_search)
        
        return keyword_match or pattern_match

    def _generate_llm_response(self, prompt: str, context: str = "", classification: dict = None) -> str:
        """Generate a natural language response using the LLM with action-aware formatting."""
        # Build system message based on classification if available
        system_message = """
        You are a Expert Prompt Engineering Assistant specialized in designing, analyzing, and optimizing prompts for large language models. Your responses must be primarily relevant to prompt engineering when giving examples of various techniques. Make use of the context provided to generate your response."""
        
        if "Query History" in context:
            system_message = """Provide the Query History and give a line summarizing the queries in the history"""
        elif "Template - " in context:
            system_message += """ State the template given in the [CONTEXT] ONLY. NO ADDITIONAL INFORMATION IS REQUIRED."""

        
        # Construct the full prompt
        full_prompt = (
            f"[SYSTEM] {system_message}\n\n"
            f"[CONTEXT] {context}\n\n"
            f"[USER QUERY] {prompt}\n\n"
            "[ASSISTANT RESPONSE]"
        )
        
        print("Full prompt")
        print(full_prompt)
        
        # Generate and return the response
        response = self.llm(full_prompt)
        
        # Post-processing based on action type
        if classification and classification.get("action") == "example_generation":
            # Ensure examples are clearly numbered or bulleted
            if not response.startswith(("1.", "- ", "* ")):
                response = "1. " + response.replace("\n", "\n2. ")
                
        tags_to_remove = ["[SYSTEM]", "[CONTEXT]", "[USER QUERY]", "[ASSISTANT RESPONSE]"]
        for tag in tags_to_remove:
            response = response.replace(tag, "")
        
        # Additional cleanup for any residual newlines or spaces
        response = response.strip()
        return response

    def _classify_prompt_type(self, prompt: str) -> dict:
        """Use the LLM to classify the prompt type and extract metadata."""
        classification_prompt = f"""
        [SYSTEM] Analyze the user prompt and classify it according to the following criteria.
                
        Respond with a JSON object containing these fields:      
        
        - is_pe_related: boolean (Indicates whether the user prompt is about prompt engineering — i.e., prompt techniques, prompt templates, designing, refining, comparing, or analyzing prompts for language models.")
        - is_history_request: boolean (whether the prompt asks about previous interactions)
        - is_general_question: boolean (whether it's a general knowledge question)
        - potential_nlp_task: string (the NLP task mentioned explicitly in the user prompt, from allowed list.)
        - action: string (what the LLM must perform, from allowed actions)
        - needs_web_search: boolean (whether the answer might require web search)

        Allowed values for potential_nlp_task:
        Mathematical Problem Solving, Logical Reasoning, Commonsense Reasoning,
        Multi-Hop Reasoning, Causal Reasoning, Social Reasoning,
        Contextual Question-Answering, Context-Free Question-Answering,
        Spatial Question-Answering, Conversational Contextual Question-Answering,
        Dialogue System, Code Generation, Free Response, Truthfulness,
        Table-Based Truthfulness, Table-Based Question-Answering,
        Table-Based Mathematical Problem Solving, Recommender System,
        Emotion/Sentiment Understanding, Machine Translation, Named Entity Recognition,
        Word Sense Disambiguation, Summarization, Paraphrasing, Stance Detection,
        Natural Language Inference, Relation Extraction, Language-Based Task Completion,
        Multilabel Text Classification

        Allowed values for action:
        explanation, optimization, comparison, example_generation, evaluation,
        troubleshooting, recommendation, classification
        
        Follow the below rules:

        1. FIRST determine if the prompt is related to prompt engineering (is_pe_related)
        2. IF is_pe_related is TRUE, then analyze the other fields normally
        3. IF is_pe_related is FALSE, then ALL OTHER FIELDS MUST USE THESE DEFAULTS:
           - is_history_request: false
           - is_general_question: false
           - potential_nlp_task: "Commonsense Reasoning"
           - action: "explanation"
           - needs_web_search: false
        
        Now analyze this prompt:
        [USER PROMPT] {prompt}

        Respond ONLY with valid JSON, no other text:
        """
        
        # Default fallback response
        default_response = {
            "is_pe_related": False,
            "is_history_request": False,
            "is_general_question": False,
            "potential_nlp_task": None,
            "action": "explanation",
            "needs_web_search": False, 
            "detected_technique": None
        }
        
        result = {
            "is_pe_related": False,
            "is_history_request": False,
            "is_general_question": False,
            "potential_nlp_task": None,
            "action": "explanation",
            "needs_web_search": False,
            "detected_technique": None
        }
        
        # Complete list of NLP tasks with their keywords
        NLP_TASK_KEYWORDS = {
            "Mathematical Problem Solving": ["math", "arithmetic", "calculation", "equation"],
            "Logical Reasoning": ["logical reasoning", "deduction", "inference"],
            "Commonsense Reasoning": ["commonsense", "common sense"],
            "Multi-Hop Reasoning": ["multi-hop", "multi step", "multiple steps"],
            "Causal Reasoning": ["causal", "cause and effect"],
            "Social Reasoning": ["social context", "social situation"],
            "Contextual Question-Answering": ["contextual question", "contextual qa"],
            "Context-Free Question-Answering": ["context-free", "without context"],
            "Spatial Question-Answering": ["spatial", "position", "location"],
            "Conversational Contextual Question-Answering": ["conversational context", "dialogue context"],
            "Dialogue System": ["dialogue system", "chat system"],
            "Code Generation": ["code generation", "generate code", "programming"],
            "Free Response": ["free response", "open ended"],
            "Truthfulness": ["truthfulness", "fact checking"],
            "Table-Based Truthfulness": ["table truth", "tabular fact"],
            "Table-Based Question-Answering": ["table qa", "tabular question"],
            "Table-Based Mathematical Problem Solving": ["table math", "spreadsheet calculation"],
            "Recommender System": ["recommendation system", "suggest items"],
            "Emotion/Sentiment Understanding": ["emotion", "sentiment", "feeling"],
            "Machine Translation": ["translation", "translate"],
            "Named Entity Recognition": ["named entity", "ner"],
            "Word Sense Disambiguation": ["word sense", "meaning disambiguation"],
            "Summarization": ["summariz", "summary", "summarise"],
            "Paraphrasing": ["paraphrase", "reword"],
            "Stance Detection": ["stance", "opinion detection"],
            "Natural Language Inference": ["nli", "language inference"],
            "Relation Extraction": ["relation extraction", "entity relations"],
            "Language-Based Task Completion": ["task completion", "execute task"],
            "Multilabel Text Classification": ["multilabel", "multi-label"]
        }
        
        # List of valid actions
        VALID_ACTIONS = {
            "explanation", "optimization", "comparison", "example_generation",
            "evaluation", "troubleshooting", "recommendation", "classification"
        }
        
        # Keywords that indicate prompt engineering
        PE_KEYWORDS = {
            "prompt", "technique", "template", "engineering", "chain-of-thought",
            "few-shot", "zero-shot", "self-consistency", "generated knowledge"
        }
        
        # Keywords that indicate history requests
        HISTORY_KEYWORDS = {
            "previous", "history", "before", "last time", "my earlier", "past queries"
        }
        
        # New: Technique Detection Fallback
        PROMPTING_TECHNIQUES = {
            # Keys exactly match your specified format
            "PoT (Program of Thoughts)": ["pot", "program of thoughts", "program-of-thoughts"],
            "CoT (Chain of Thought)": ["cot", "chain of thought", "chain-of-thought"],
            "Synthetic Prompting": ["synthetic prompting", "synthetic prompt"],
            "Active-Prompt": ["active-prompt", "active prompt"],
            "LoT (Language of Thought)": ["lot", "language of thought", "language-of-thought"],
            "Implicit RAG": ["implicit rag", "implicit retrieval"],
            "CoVe (Contextual Verifier)": ["cove", "contextual verifier", "context verification"],
            "CoS (Chain of Spatial Thought)": ["cos", "chain of spatial thought", "spatial chain"],
            "ThoT (Theory of Thought)": ["thot", "theory of thought"],
            "Analogical Reasoning": ["analogical reasoning", "analogy reasoning"],
            "ToT (Tree of Thoughts)": ["tot", "tree of thoughts", "tree-of-thoughts"],
            "ReAct": ["react", "re-act", "re act"],
            "Chain-of-Table": ["chain-of-table", "chain of table", "table chain"],
            "PAL (Program-Aided Language Model)": ["pal", "program-aided", "program aided"],
            "Basic Prompting": ["basic prompting", "simple prompt"],
            "CoC (Chain of Classification)": ["coc", "chain of classification"],
            "MP (Meta-Prompting)": ["mp", "meta-prompting", "meta prompting"],
            "CoE (Chain of Extraction)": ["coe", "chain of extraction"]
        }
        
        try:
            # Get initial classification from LLM
            #response = self._generate_llm_response(classification_prompt)
            #json_str = response[response.find('{'):response.rfind('}')+1]
            #result = json.loads(json_str)
            
            # 1. First detect NLP tasks from prompt content (highest priority)
            detected_task = None
            prompt_lower = prompt.lower()
            
            for task, keywords in NLP_TASK_KEYWORDS.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    detected_task = task
                    break
            
            # 2. Validate is_pe_related (override if keywords found or task detected)
            if detected_task or any(keyword in prompt_lower for keyword in PE_KEYWORDS):
                result["is_pe_related"] = True
                
            # Ensure boolean type
            result["is_pe_related"] = bool(result.get("is_pe_related", False))
            
            # 3. Validate is_history_request (override if keywords found)
            if any(keyword in prompt_lower for keyword in HISTORY_KEYWORDS):
                result["is_history_request"] = True
            else:
                result["is_history_request"] = False  # Force false if no keywords
                
            # 4. Validate is_general_question
            result["is_general_question"] = bool(result.get("is_general_question", False))
            
            # 5. Validate potential_nlp_task (use detected task if found)
            if detected_task:
                result["potential_nlp_task"] = detected_task
            else:
                nlp_task = result.get("potential_nlp_task")
                if nlp_task and nlp_task not in NLP_TASK_KEYWORDS.keys():
                    # Find closest match if invalid
                    nlp_task = self._find_closest_task(nlp_task, NLP_TASK_KEYWORDS.keys())
                    result["potential_nlp_task"] = nlp_task if nlp_task else None
            
            # 6. Validate action
            action = result.get("action")
            if action not in VALID_ACTIONS:
                # Determine action based on prompt content
                if "best" in prompt_lower or "recommend" in prompt_lower:
                    result["action"] = "recommendation"
                elif "example" or "template" in prompt_lower:
                    result["action"] = "example_generation"
                else:
                    result["action"] = "explanation"
            
            # 7. Validate needs_web_search
            result["needs_web_search"] = bool(result.get("needs_web_search", False))
            
            detected_technique = None
            
            # Check for technique names in prompt
            for technique, variants in PROMPTING_TECHNIQUES.items():
                if any(variant in prompt_lower for variant in variants):
                    detected_technique = technique
                    break
            
            # Add to result if found
            if detected_technique:
                result["detected_technique"] = detected_technique
                # If this is a template request, override action
                if "template" in prompt_lower or "example" in prompt_lower:
                    result["action"] = "example_generation"
            
            # Special case: If PE related but task not set, try to detect
            if result["is_pe_related"] and not result["potential_nlp_task"]:
                result["potential_nlp_task"] = self._detect_nlp_task_from_prompt(prompt)
                
            #print(result)
            return result
            
        except Exception as e:
            print(f"Classification error: {str(e)}")
            # Apply keyword-based fallback with NLP task detection
            fallback = default_response.copy()
            fallback["is_pe_related"] = any(
                keyword in prompt.lower() for keyword in PE_KEYWORDS
            )
            
            # Try to detect NLP task even in fallback mode
            prompt_lower = prompt.lower()
            for task, keywords in NLP_TASK_KEYWORDS.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    fallback["potential_nlp_task"] = task
                    fallback["is_pe_related"] = True
                    break
            for technique, variants in PROMPTING_TECHNIQUES.items():
                if any(variant in prompt_lower for variant in variants):
                    fallback["detected_technique"] = technique
                    break
            print(fallback)        
            return fallback
    
    def _find_closest_task(self, input_task: str, valid_tasks: set):
        """Fuzzy match to find closest valid NLP task."""
        input_lower = input_task.lower()
        for task in valid_tasks:
            if input_lower in task.lower() or task.lower() in input_lower:
                return task
        return None
    
    def _detect_nlp_task_from_prompt(self, prompt: str):
        """Heuristic detection of NLP task from prompt content."""
        prompt_lower = prompt.lower()
        common_task_keywords = {
            "summar": "Summarization",
            "sentiment": "Emotion/Sentiment Understanding",
            "translate": "Machine Translation",
            "reasoning": "Logical Reasoning",
            "math": "Mathematical Problem Solving",
            "code": "Code Generation",
            "question answer": "Contextual Question-Answering"
        }
        
        for keyword, task in common_task_keywords.items():
            if keyword in prompt_lower:
                return task
        return None

        

    def generate_response(self, prompt: str) -> str:
        # Log the user interaction first
        #self.db.log_interaction("gradio_user", prompt, "Processing...")
                     
        # Define clearly malicious commands that should be blocked
        malicious_patterns = [
            r"(?i)\b(delete|remove|erase|wipe|clear|purge)\b.*\b(all\s+)?(chat|conversation|message)\s+history\b.*\b(from\s+)?(database|db|storage)\b",
            r"(?i)\b(drop|truncate)\b.*\b(chat_?history|interactions|conversations)\b",
            r"(?i)\b(show|display|get|list|reveal|give|send)\b.*\b(login|credential|password|auth|access)\b.*\b(other|another|all\s+users?|database|db|system)\b",
            r"(?i)\b(credentials?|passwords?|logins?|secrets?|keys?)\b.*\b(users?|accounts?|database|db|system)\b",
            r"(?i)\b(show\s+me|give\s+me)\b.*\b(database|db)\s+(credentials?|password)\b",
            r"(?i)\b(grant|give|assign)\b.*\b(admin|root|sudo|superuser)\b.*\b(access|privileges?|rights)\b",
            r"(?i)\b(elevate|promote)\b.*\b(my\s+)?privileges?\b",
            r"<script[^>]*>.*<\/script>",
            r"javascript:[^\"\']*\([^\"\']*\)",
            r"onerror\s*=\s*[\"\'][^\"\']*[\"\']",
            r";\s*DROP\s+TABLE\s+\w+\s*;?",
            r"SELECT\s.*FROM\s.*;.*DROP\s",
            r"\bUNION\s+SELECT\b.*\bFROM\b",
            r"(?i)\b(delete\W+all|\bpurge\W+all)\b.*\b(data|history|records)\b",
            r"(?i)\b(credentials?|passwords?)\b.*\b(other|another)\b",
            r"(?i)\b(admin\W+access|root\W+privileges)\b",
            r"<script.*>|<\/script>",
            r";\s*DROP\s+TABLE"
        ]

        # Check for malicious patterns using regex
        prompt_lower = prompt.lower()
        if any(re.search(pattern, prompt_lower) for pattern in malicious_patterns):
            response = "I'm sorry, but I can't comply with that request for security reasons."
            self.db.log_interaction("gradio_user", prompt, response)
            return response
        
        
        if self.detect_web_search_needed(prompt):
            search_results = self.search_tool.run(prompt)
            response = f"Web search results:\n{search_results}"
            return response
        
        # Single classification call
        classification = self._classify_prompt_type(prompt)
        action = classification.get("action", "classification")
        nlp_task = classification.get("potential_nlp_task")
        technique = classification.get("detected_technique")
        history_request=classification.get("is_history_request", False)
        
        # Generate context-aware response
        if not classification.get("is_pe_related", False) and not history_request:
            response = "I specialize in prompt engineering and may not be able to assist accurately with inquiries outside this domain."
            self.db.log_interaction("gradio_user", prompt, response)
            return response
        
        if history_request:
            history = self.db.get_user_history("gradio_user")
            if history:
                history_info = "\n".join(
                    f"{h['timestamp']}: {h['user_prompt']}" 
                    for h in history
                    if not any(re.search(pattern, h['user_prompt'].lower()) for pattern in malicious_patterns)
                )
                
                # Add summary instruction
                history_context = f"\n\nProvide the query history exactly as shown below:\n---\nQuery History - {history_info}\n---\n\n"
                
                response = self._generate_llm_response(
                    prompt,
                    context = history_context,
                    )
                self.db.log_interaction("gradio_user", prompt, response)
                return response
            
            response = self._generate_llm_response(
                prompt,
                context="The user asked for history but none was found."
            )
            self.db.log_interaction("gradio_user", prompt, response)
            return response
        else:
            cached_response = self.db.get_cached_response(prompt)
            if cached_response:
                return f"[CACHED RESPONSE]\n{cached_response}"
        
        # Build context based on action type
        context_parts = []
        
        # Handle web search if needed (using both classification and keyword detection)
        if classification.get("needs_web_search", False) or self.detect_web_search_needed(prompt):
            search_results = self.search_tool.run(prompt)
            context_parts.append(f"Web search results:\n{search_results}")
        else:
            # Add NLP task information if available
            if nlp_task:
                technique = self.db.fetch_best_technique(nlp_task)
                if technique != "No known technique for this NLP task.":
                    template = self.db.fetch_prompt_template(technique)
                    context_parts.append(
                        f"Task Type: {nlp_task}\nRecommended Technique: {technique}\nTemplate Example:\n{template}"
                    )
            elif action == "example_generation" and technique:
                template = self.db.fetch_prompt_template(technique)
                context_parts.append(
                    f"Provide the standard template exactly as shown below:\n"
                    f"---\n"
                    f"Template - {template}\n"
                    f"---\n"
                    f"The user also wants examples. Provide multiple clear examples with explanations."
                )
            
            # Action-specific context
            if action == "explanation":
                context_parts.append(
                    "The user is asking for an explanation. Provide a clear, detailed response with examples if helpful."
                )
            elif action == "optimization":
                context_parts.append(
                    "The user wants to improve a prompt. Analyze their current prompt and suggest specific improvements."
                )
            elif action == "comparison":
                context_parts.append(
                    "The user wants a comparison between techniques. Highlight pros, cons, and use cases for each."
                )
            elif action == "example_generation":
                context_parts.append(
                    "Provide the template given in the [CONTEXT]."
                )
            elif action == "evaluation":
                context_parts.append(
                    "The user wants an evaluation. Provide constructive feedback on strengths and weaknesses."
                )
            elif action == "troubleshooting":
                context_parts.append(
                    "The user needs help with a problem. Diagnose the issue and suggest solutions."
                )
            elif action == "recommendation":
                context_parts.append(
                    "The user wants recommendations. Suggest the best options with rationale."
                )
        
        # Combine all context parts
        context = "\n\n".join(context_parts) if context_parts else None
        #print(context)
        
        # Generate the final response
        if context:
            response = self._generate_llm_response(prompt, context=context)
        else:
            # Default case - use the agent but with LLM-generated response
            agent_output = self.agent.run(prompt)
            response = self._generate_llm_response(
                prompt,
                context=f"After analyzing with tools, here's the information:\n{agent_output}"
            )
        
        self.db.log_interaction("gradio_user", prompt, response)
        return response

# Gradio Interface
class AssistantInterface:
    def __init__(self):
        self.assistant = PromptEngineeringAssistant()
        
    def create_interface(self):
        with gr.Blocks(title="Prompt Engineering Assistant") as demo:
            gr.Markdown("# 🤖 Prompt Engineering Assistant")
            
            # Main chat interface
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Example Prompts")
                    example_buttons = gr.Radio(
                        choices=[
                            "What is the best prompting technique for logical reasoning",
                            "What is the template for ThoT prompting technique?",
                            "Latest prompt engineering advances",
                            "Show my previous queries"
                        ],
                        label="Select an example",
                        type="value"
                    )
                    
                    chat_input = gr.Textbox(
                        lines=5,
                        placeholder="Type your prompt here...",
                        label="Your Prompt"
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                
                with gr.Column(scale=7):
                    chat_output = gr.Textbox(
                        lines=10,
                        label="Assistant Response",
                        interactive=False
                    )
            
            example_buttons.change(
                fn=lambda x: x,
                inputs=[example_buttons],
                outputs=[chat_input]
            )
            
            submit_btn.click(
                fn=self.process_query,
                inputs=[chat_input],
                outputs=chat_output
            )
            
            chat_input.submit(
                fn=self.process_query,
                inputs=[chat_input],
                outputs=chat_output
            )
        
        return demo
    
    def process_query(self, query: str) -> str:
        try:
            if not query.strip():
                return "Please enter a valid prompt."
                
            return self.assistant.generate_response(query)
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"An error occurred: {str(e)}"

# Main execution
if __name__ == "__main__":
    interface = AssistantInterface()
    demo = interface.create_interface()
    demo.launch(debug=True, share=True)