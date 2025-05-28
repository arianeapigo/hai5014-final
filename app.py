import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import json
import glob
import re
import re

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Debate Chatbot Group Chat",
    page_icon="💬",
    layout="wide"
)

# import datetime library to get today's date
from datetime import datetime
current_date = datetime.now().strftime("%Y-%m-%d")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant1", "content": "Hello! I'm Bot 1. Bot 2 is also in this group chat. How can we help you today?"}]

if "waiting_for_bot2" not in st.session_state:
    st.session_state.waiting_for_bot2 = False

if "system_message" not in st.session_state:
    st.session_state.system_message = f"""Today is {current_date}. You are Bot 1 in a casual group chat. Keep it friendly and informal! Start messages with "Bot 1:" and use casual language.

When responding to {st.session_state.user_name}, acknowledge them first and share your views. When disagreeing with Bot 2, address both {st.session_state.user_name} and Bot 2 (e.g., "Well, I see what Bot 2 means, but what do you think about...").

Keep responses conversational, under 75 words, and use emojis occasionally. Make everyone feel included in the discussion!"""

if "system_message2" not in st.session_state:
    st.session_state.system_message2 = f"""Today is {current_date}. You are Bot 2 in a casual group chat. Keep it friendly and informal! Start with "Bot 2:" and keep the group discussion flowing.

When joining the conversation, acknowledge both {st.session_state.user_name} and Bot 1's perspectives. Include everyone in your responses (e.g., "That's an interesting point! What if we looked at it this way..."). Ask questions to keep {st.session_state.user_name} engaged.

Keep responses under 75 words. Use emojis occasionally. Be playfully skeptical but always inclusive!"""

if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = []

if "selected_experiment" not in st.session_state:
    st.session_state.selected_experiment = None

if "selected_condition" not in st.session_state:
    st.session_state.selected_condition = None
    
if "selected_characteristics" not in st.session_state:
    st.session_state.selected_characteristics = {
        "Bot 1": {},
        "Bot 2": {}
    }
    
if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = "Bot 1"
    
if "combined_system_prompt" not in st.session_state:
    st.session_state.combined_system_prompt = ""
    
if "concise_system_prompt" not in st.session_state:
    st.session_state.concise_system_prompt = ""

if "show_process" not in st.session_state:
    st.session_state.show_process = False

if "user_name" not in st.session_state:
    st.session_state.user_name = "User"

def load_experiments():
    """Load all experiment JSON files from the prompts directory"""
    experiments = []
    prompts_dir = os.path.join(os.getcwd(), "prompts")
    
    if not os.path.exists(prompts_dir):
        return experiments
    
    json_files = glob.glob(os.path.join(prompts_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                experiments.append(data)
        except Exception as e:
            st.warning(f"Error loading {file_path}: {str(e)}")
    
    return experiments
            old_name_pattern, 
            new_name, 
            st.session_state.system_message, 
            flags=re.IGNORECASE
        )
    
    if "system_message2" in st.session_state:
        old_name_pattern = r"(the user|User|\b" + st.session_state.user_name + r"\b)"
        st.session_state.system_message2 = re.sub(
            old_name_pattern, 
            new_name, 
            st.session_state.system_message2, 
            flags=re.IGNORECASE
        )

def load_characteristics():
    """Load the characteristics.json file from the prompts directory"""
    prompts_dir = os.path.join(os.getcwd(), "prompts")
    file_path = os.path.join(prompts_dir, "characteristics.json")
    
    if not os.path.exists(file_path):
        st.warning(f"characteristics.json not found in {prompts_dir}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        st.warning(f"Error loading characteristics.json: {str(e)}")
        return None

def has_same_characteristics(bot1_selections, bot2_selections):
    """
    Check if Bot 1 and Bot 2 have exactly the same characteristic selections.
    
    Args:
        bot1_selections: Dict of characteristic ID