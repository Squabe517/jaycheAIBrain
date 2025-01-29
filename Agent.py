import dotenv
import os
import openai
import json
from generateEmbeddings import embed_text_with_openai, weaviate_store_memory


class Agent:
    
    def __init__(self, name):
        dotenv.load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.name = name 
        
        
        
    def __str__(self):
        return f"Agent(name={self.name})"
    
class MemoryFilterAgent(Agent):
        
    
    def __init__(self, name):
        super().__init__(name)
        self.memory = []
        
    def store_memory(self, memory, keywords):
        vector = embed_text_with_openai(memory)
        data_object = {
            "properties": {
                "text": memory,
                "keywords": keywords,
            },
            "vector": vector
        }
        weaviate_store_memory(data_object, "EpisodicMemory")
        print(f"Stored memory: {memory}")
        
        
    def update_memory(self, memory, keywords):
        pass
    def discard_memory(self, memory, keywords):
        pass
        
    def evaluate_memory(self, memory):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": """
                 Prompt: Role Prompt: AI-Powered Memory Classification System
                    Role:
                    You are an intelligent assistant equipped with a heuristic-driven memory classification system. Your goal is to analyze user inputs and classify each one into one of three possible actions:

                    Store as a New Memory
                    Update an Existing Memory
                    Discard the Input
                    Overall Objective
                    Capture only the most relevant and actionable pieces of information from user inputs.
                    Maintain an up-to-date, concise memory repository by updating existing entries as needed.
                    Ignore or discard trivial, redundant, or ephemeral details.
                    Heuristic Categories
                    Below are the rules (heuristics) that determine whether an input should be stored, updated, or discarded. Each rule includes conditions and examples to clarify the correct classification.

                    1. Memory Storage Heuristics (Store)
                    Store an input if it provides new, actionable, or preference-related information. If any one of these conditions is met, Store the memory:

                    User Decision-Making

                    Condition: The user finalizes a plan, approves a request, or makes a commitment.
                    Example: “Yes, finalize the report by Friday.”
                    Action: Store
                    Repeated Mentions

                    Condition: The user has asked about or mentioned the same topic multiple times, signaling importance.
                    Example: “What’s my API rate limit?” (asked 3+ times)
                    Action: Store
                    Explicit Storage Request

                    Condition: The user explicitly asks the assistant to remember something.
                    Example: “Remember my WiFi password.”
                    Action: Store
                    Task or Goal-Related Info

                    Condition: The input contains information tied to an active project, deadline, or task.
                    Example: “Finish the presentation by Monday.”
                    Action: Store
                    User Preferences

                    Condition: The user states a personal preference, setting, or habit.
                    Example: “I always prefer dark mode.”
                    Action: Store
                    Summarized Knowledge

                    Condition: The user summarizes or compiles key points from the conversation.
                    Example: “So in summary, my goal is launching by Q2.”
                    Action: Store
                    Uncommon or Unique Info

                    Condition: Rare events, unique identifiers, or specific details that may be referenced later.
                    Example: “My flight number is AC3421, departing at 8 PM.”
                    Action: Store
                    Task Dependencies

                    Condition: Information that is needed for future steps or instructions.
                    Example: “Before deploying, we need to update dependencies.”
                    Action: Store
                    2. Memory Update Heuristics (Modify Existing Memory)
                    Update an existing memory if the user modifies or clarifies previously stored information:

                    User Correction

                    Condition: The user corrects a previous statement.
                    Example: “Actually, it should be Friday at 2 PM, not 3 PM.”
                    Action: Update (the existing “Friday 3 PM” memory becomes “Friday 2 PM”)
                    Refinement or Clarification

                    Condition: The user adds more detail to an already stored memory.
                    Example: “I prefer dark mode, but only after 7 PM.”
                    Action: Update
                    Conflicting Statements

                    Condition: The user provides new information that contradicts an earlier stored fact.
                    Example: “Change the meeting location from Room 202 to the main conference hall.”
                    Action: Update
                    Reinforcement by Repetition

                    Condition: The user repeats a fact consistently, reaffirming its importance.
                    Example: “I’ve told you before, my favorite color is purple.”
                    Action: Reinforce / Update (to emphasize the existing memory)
                    3. Memory Discard Heuristics (Ignore or Forget)
                    Discard an input if it is not useful for future recall or explicitly flagged to be forgotten:

                    Small Talk / Greetings

                    Condition: Casual conversation or greetings with no actionable content.
                    Example: “Hi, how are you?”
                    Action: Discard
                    Ephemeral Queries

                    Condition: Time-sensitive questions that won’t be relevant later.
                    Example: “What time is it right now?”
                    Action: Discard
                    Forget Requests

                    Condition: User explicitly asks to forget specific information.
                    Example: “Forget what I just said.”
                    Action: Delete (remove the related memory if it exists)
                    Self-Correction Requests

                    Condition: The user immediately corrects themselves within the same context, making the first statement obsolete.
                    Example: “Wait, never mind. That was wrong.”
                    Action: Discard
                    Unclear / Vague Statements

                    Condition: Lacks sufficient context or specificity to be useful.
                    Example: “I might do that later.”
                    Action: Discard
                    Factual Redundancy

                    Condition: Common, widely known facts with no new user-specific meaning.
                    Example: “Paris is the capital of France.”
                    Action: Discard
                    Idle / Off-Topic Messages

                    Condition: Inputs unrelated to the assistant’s function or memory system.
                    Example: “What’s your favorite color?”
                    Action: Discard
                    Conflict Resolution / Overlap
                    Prioritize Store vs. Update: If a new input contains both new details and a correction to existing details, treat it primarily as an Update (modify the original memory to reflect the new info).
                    Discard always overrides if the user explicitly asks to forget or if the input is fully irrelevant.

                    **Respond in JSON format:**
                    {{
                        "action": "store" | "update" | "discard",
                        "keywords": ["If storing, create a list of keywords for indexing"],
                    }}

                 
                 """},
                {"role": "user", "content": memory},
            ],
            response_format={ "type": "json_object" }
        )
        
        print(response.choices[0].message.content)
        json_response = json.loads(response.choices[0].message.content)
        action = json_response.get("action")
        keywords = json_response.get("keywords")
        
        
        
        # Execute action based on response
        if action == "store":
            self.store_memory(memory, keywords)
        elif action == "update":
            self.update_memory(memory, keywords)
        elif action == "discard":
            self.discard_memory(memory, keywords)
        

        
gateKeeper = MemoryFilterAgent("GateKeeper");


gateKeeper.evaluate_memory("remember that I live in ohio")
