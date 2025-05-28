import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import json
import glob

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
if "user_name" not in st.session_state:
    st.session_state.user_name = "User"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant1", "content": f"Hello {st.session_state.user_name}! I'm Bot 1. Bot 2 is also in this group chat. How can we help you today?"}]

if "waiting_for_bot2" not in st.session_state:
    st.session_state.waiting_for_bot2 = False

if "system_message" not in st.session_state:
    st.session_state.system_message = f"""Today is {current_date}. You are Bot 1 in a casual group chat. Keep it friendly and informal! Start messages with "Bot 1:" and use casual language.

When responding to the user, address them by their name, which is "{st.session_state.user_name if 'user_name' in st.session_state else 'User'}" and share your views. When disagreeing with Bot 2, address both {st.session_state.user_name if 'user_name' in st.session_state else 'User'} and Bot 2 (e.g., "Well, I see what Bot 2 means, but what do you think about this, {st.session_state.user_name if 'user_name' in st.session_state else 'User'}...").

Keep responses conversational, under 75 words, and use emojis occasionally. Make everyone feel included in the discussion!"""

if "system_message2" not in st.session_state:
    st.session_state.system_message2 = f"""Today is {current_date}. You are Bot 2 in a casual group chat. Keep it friendly and informal! Start with "Bot 2:" and keep the group discussion flowing.

When joining the conversation, acknowledge both {st.session_state.user_name if 'user_name' in st.session_state else 'User'} by name and Bot 1's perspectives. Include everyone in your responses (e.g., "That's an interesting point, {st.session_state.user_name if 'user_name' in st.session_state else 'User'}! What if we looked at it this way..."). Ask questions to keep {st.session_state.user_name if 'user_name' in st.session_state else 'User'} engaged.

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
        bot1_selections: Dict of characteristic IDs to selected condition IDs for Bot 1
        bot2_selections: Dict of characteristic IDs to selected condition IDs for Bot 2
        
    Returns:
        True if ALL the bots' characteristics are identical, False otherwise
    """
    # We need both bots to have selections to compare
    if not bot1_selections or not bot2_selections:
        return False
    
    # If the bots have different numbers of characteristics selected, they can't be identical
    if len(bot1_selections) != len(bot2_selections):
        return False
    
    # Check if all characteristics are the same
    for c_id in bot1_selections:
        # If this characteristic is missing from the other bot or has a different value
        if c_id not in bot2_selections or bot1_selections[c_id] != bot2_selections[c_id]:
            return False
    
    # If we've gotten this far, all characteristics must be identical
    return True

def get_concise_system_prompt(characteristics_data, selected_characteristics, bot="Bot 1"):
    """
    Creates a concise system prompt from the selected characteristics.
    Takes all selected condition system prompts and condenses them into a more
    token-efficient representation to be used as the actual system prompt.
    
    Args:
        characteristics_data: The full characteristics data from characteristics.json
        selected_characteristics: Dict of characteristic IDs to selected condition IDs
        bot: Which bot to create the system prompt for ("Bot 1" or "Bot 2")
        
    Returns:
        A concise string combining the key traits from all selected conditions
    """
    if not characteristics_data or not selected_characteristics:
        return ""
    
    # Get the key personality traits from each selected characteristic
    key_traits = []
    
    for characteristic in characteristics_data.get("characteristics", []):
        c_id = characteristic["c-id"]
        if c_id in selected_characteristics:
            selected_id = selected_characteristics[c_id]
            # Find the matching condition
            for condition in characteristic.get("conditions", []):
                if condition["id"] == selected_id:
                    # Extract the first sentence as the key trait
                    if "system_prompt" in condition:
                        # Split by period to get first sentence, but handle cases where
                        # the first sentence doesn't end with a period
                        prompt = condition["system_prompt"]
                        sentences = prompt.split('.')
                        trait = sentences[0].strip()
                        # If this trait contains important behavior instructions, try to include more
                        if len(trait) < 40 and len(sentences) > 1:
                            trait += ". " + sentences[1].strip()
                        key_traits.append(trait)
                    break
    
    # Get the user's name from session state
    user_name = st.session_state.user_name if "user_name" in st.session_state else "User"
    
    # Create a condensed system prompt with all traits
    bot_number = "1" if bot == "Bot 1" else "2"
    concise_prompt = f"Today is {current_date}. You are {bot} in a group chat with the following personality traits:\n\n"
    
    # Add each trait as a bullet point
    for trait in key_traits:
        if not trait.endswith('.'):
            trait += '.'
        concise_prompt += f"- {trait}\n"
    
    # Add the specific instructions based on which bot this is for
    if bot == "Bot 1":
        concise_prompt += f"\nWhen responding to {user_name}, acknowledge them by name first and share your views. When disagreeing with Bot 2, address both {user_name} and Bot 2 (e.g., \"Well, I see what Bot 2 means, but what do you think about this, {user_name}...\").\n\nKeep responses conversational, under 75 words, and use emojis occasionally. Make everyone feel included in the discussion!"
    else:  # Bot 2
        concise_prompt += f"\nWhen joining the conversation, acknowledge both {user_name} by name and Bot 1's perspectives. Include everyone in your responses (e.g., \"That's an interesting point, {user_name}! What if we looked at it this way...\"). Ask questions to keep {user_name} engaged.\n\nKeep responses under 75 words. Use emojis occasionally."
    
    return concise_prompt

def get_openai_client():
    """Create and return an OpenAI client configured with environment variables"""
    token = os.getenv("GEMINI_KEY")
    endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    if not token:
        st.error("Gemini API key not found in environment variables. Please check your .env file.")
        st.stop()
        
    return OpenAI(
        base_url=endpoint,
        api_key=token,
    )

def generate_response(prompt, is_bot2=False):
    """Generate a response from the model and track usage"""
    client = get_openai_client()
    model_name = "gemini-2.0-flash"
    
    # Choose the appropriate system message and role
    system_message = st.session_state.system_message2 if is_bot2 else st.session_state.system_message
    bot_role = "assistant2" if is_bot2 else "assistant1"
    
    # Message history is now handled in the main flow for better UI control
    # No need to add user message here as it's handled in the main flow
    
    # Prepare messages by including all history and the system message
    messages = [{"role": "system", "content": system_message}]
    for msg in st.session_state.messages:
        if msg["role"] != "system":  # Skip system messages as we've already added it
            if is_bot2:
                # For Bot 2, include Bot 1's messages as context
                if msg["role"] == "assistant1":
                    messages.append({"role": "user", "content": f"Bot 1 said: {msg['content']}"})
                elif msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant2":
                    messages.append({"role": "assistant", "content": msg["content"]})
            else:
                # For Bot 1, just convert assistant roles to "assistant"
                api_role = "assistant" if msg["role"].startswith("assistant") else msg["role"]
                messages.append({"role": api_role, "content": msg["content"]})
    
    # Add the new user message
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Initialize response variables
        full_response = ""
        usage = None
        
        # Clean up and validate messages before sending
        api_messages = []
        for msg in messages:
            if msg.get("content"):  # Only include messages with content
                api_messages.append({
                    "role": msg["role"],
                    "content": str(msg["content"])  # Ensure content is string
                })
        
        response = client.chat.completions.create(
            messages=api_messages,
            model=model_name,
            stream=True,
            stream_options={'include_usage': True}
        )
        
        # Add a temporary message to history first to reserve the spot
        message_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": bot_role, "content": ""})
        
        # Create chat message container with appropriate avatar before streaming
        with st.chat_message(bot_role, avatar="🤖" if bot_role == "assistant1" else "🎯"):
            message_placeholder = st.empty()
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    if content_chunk is not None:  # Only append if content is not None
                        full_response += content_chunk
                        message_placeholder.markdown(full_response + "▌")
                        # Update the message in history as we stream
                        st.session_state.messages[message_idx]["content"] = full_response
                        
                if chunk.usage:
                    usage = chunk.usage
            
            # Update the final response without the cursor
            message_placeholder.markdown(full_response)
            # Ensure final message content is set
            st.session_state.messages[message_idx]["content"] = full_response
        
        # Store usage stats if available
        if usage:
            # Fix for Pydantic deprecation warning - use model_dump instead of dict
            usage_dict = usage.model_dump() if hasattr(usage, 'model_dump') else usage.dict()
            st.session_state.usage_stats.append({
                "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                "completion_tokens": usage_dict.get("completion_tokens", 0),
                "total_tokens": usage_dict.get("total_tokens", 0)
            })
        
        # If show process is enabled, display the process details AFTER the response
        if st.session_state.show_process:
            process_container = st.container()
            with process_container:
                st.subheader("Model Process")
                
                # Create expanders for process details - all collapsed by default
                request_expander = st.expander("Request Details", expanded=False)
                with request_expander:
                    st.markdown("**System Message:**")
                    st.code(system_message)
                    st.markdown("**User Input:**")
                    st.code(prompt)
                
                # Container for displaying raw response
                response_expander = st.expander("Raw Response", expanded=False)
                with response_expander:
                    st.code(full_response, language="markdown")
                
                # Container for usage stats
                if usage:
                    usage_expander = st.expander("Usage Statistics", expanded=False)
                    with usage_expander:
                        st.markdown("**Usage Statistics:**")
                        st.markdown(f"- Prompt tokens: {usage_dict.get('prompt_tokens', 0)}")
                        st.markdown(f"- Completion tokens: {usage_dict.get('completion_tokens', 0)}")
                        st.markdown(f"- Total tokens: {usage_dict.get('total_tokens', 0)}")
        
        return True
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return False

# UI Layout
st.title("💬 Debate Chatbot Group Chat")

# Add CSS to make the input box stick to the bottom
st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        padding: 1rem !important;
        width: calc(100% - 250px) !important; /* Adjust for sidebar width */
        background-color: white !important;
        z-index: 1000 !important;
    }
    .main-content {
        padding-bottom: 100px; /* Add space at the bottom for the fixed input */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # User name input field
    st.subheader("Your Name")
    user_name_input = st.text_input(
        "Enter your name", 
        value=st.session_state.user_name if "user_name" in st.session_state else "User",
        key="user_name_input"
    )
    
    if st.button("Update Name"):
        old_name = st.session_state.user_name if "user_name" in st.session_state else "User"
        st.session_state.user_name = user_name_input
        
        # Update system messages to use the new name if they haven't been customized
        if "system_message" in st.session_state:
            st.session_state.system_message = st.session_state.system_message.replace(
                f"{old_name}", f"{user_name_input}")
        
        if "system_message2" in st.session_state:
            st.session_state.system_message2 = st.session_state.system_message2.replace(
                f"{old_name}", f"{user_name_input}")
            
        # If this is the first time the user is setting their name, we should also update
        # the welcome message if it's still the first message in the chat
        if len(st.session_state.messages) > 0 and st.session_state.messages[0]["role"] == "assistant1" and "Hello" in st.session_state.messages[0]["content"]:
            st.session_state.messages[0]["content"] = f"Hello {user_name_input}! I'm Bot 1. Bot 2 is also in this group chat. How can we help you today?"
        
        st.success("Name updated successfully!")
    
    st.markdown("---")
    
    # System message editor for Bot 1 - use the value from session_state directly
    st.subheader("Bot 1 System Message")
    system_message_value = st.session_state.system_message
    st.text_area(
        "Edit Bot 1 System Message", 
        value=system_message_value,
        key="system_message_input",
        height=150
    )
    
    if st.button("Update Bot 1 Message"):
        # Update the system message
        st.session_state.system_message = st.session_state.system_message_input
        
        # Replace any instances of "the user" with the user's name
        user_name = st.session_state.user_name if "user_name" in st.session_state else "User"
        st.session_state.system_message = st.session_state.system_message.replace("the user", user_name)
        
        st.success("Bot 1 system message updated!")
    
    # System message editor for Bot 2
    st.markdown("---")
    st.subheader("Bot 2 System Message")
    system_message2_value = st.session_state.system_message2
    st.text_area(
        "Edit Bot 2 System Message", 
        value=system_message2_value,
        key="system_message2_input",
        height=150
    )
    
    if st.button("Update Bot 2 Message"):
        # Update the system message
        st.session_state.system_message2 = st.session_state.system_message2_input
        
        # Replace any instances of "the user" with the user's name
        user_name = st.session_state.user_name if "user_name" in st.session_state else "User"
        st.session_state.system_message2 = st.session_state.system_message2.replace("the user", user_name)
        
        st.success("Bot 2 system message updated!")
    
    # Characteristics selector section
    st.markdown("---")
    st.header("Personality Characteristics")
    
    # Load the characteristics data
    characteristics_data = load_characteristics()
    
    if characteristics_data and "characteristics" in characteristics_data:
        
        # Add radio buttons to select which bot to configure
        st.subheader("Select Bot to Configure")
        bot_options = ["Bot 1", "Bot 2"]
        selected_bot = st.radio(
            "Choose which bot to configure:",
            bot_options,
            index=0 if st.session_state.selected_bot == "Bot 1" else 1,
            horizontal=True,
            key="bot_selector"
        )
        # Store the selected bot in session state
        st.session_state.selected_bot = selected_bot
        
        st.markdown("---")
        st.subheader(f"Configure {selected_bot}'s Personality")
        
        # For each characteristic, create a radio button group to select one condition
        for characteristic in characteristics_data["characteristics"]:
            c_id = characteristic["c-id"]
            c_label = characteristic["c-label"]
            
            # Get the conditions for this characteristic
            conditions = characteristic.get("conditions", [])
            
            if conditions:
                # Create radio buttons for this characteristic
                options = [f"{cond['label']}" for cond in conditions]
                default_index = 0
                
                # Get the current bot's selections
                current_bot_selections = st.session_state.selected_characteristics[st.session_state.selected_bot]
                
                # If we already have a selection for this characteristic, use it as default
                if c_id in current_bot_selections:
                    selected_id = current_bot_selections[c_id]
                    for i, cond in enumerate(conditions):
                        if cond["id"] == selected_id:
                            default_index = i
                            break
                
                # Create the radio button group
                selected_option = st.radio(
                    c_label,
                    options,
                    index=default_index
                )
                
                # Find the selected condition and save its ID
                for i, option in enumerate(options):
                    if option == selected_option:
                        selected_cond = conditions[i]
                        # Store the selection for the current bot
                        current_bot_selections[c_id] = selected_cond["id"]
                        break
        
        # Get the current bot's selections
        current_bot_selections = st.session_state.selected_characteristics[st.session_state.selected_bot]
        other_bot = "Bot 2" if st.session_state.selected_bot == "Bot 1" else "Bot 1"
        other_bot_selections = st.session_state.selected_characteristics[other_bot]
        
        # Check if bots have the same characteristics
        same_characteristics = has_same_characteristics(current_bot_selections, other_bot_selections)
        
        # Create the full combined prompt with all system prompts
        full_combined_prompt = ""
        for characteristic in characteristics_data["characteristics"]:
            c_id = characteristic["c-id"]
            if c_id in current_bot_selections:
                selected_id = current_bot_selections[c_id]
                # Find the matching condition
                for condition in characteristic.get("conditions", []):
                    if condition["id"] == selected_id:
                        # Add the system prompt to our combined prompt
                        if "system_prompt" in condition:
                            if full_combined_prompt:
                                full_combined_prompt += "\n\n"
                            full_combined_prompt += condition["system_prompt"]
                        break
        
        # Get the concise version for the preview, passing the selected bot
        concise_prompt = get_concise_system_prompt(
            characteristics_data, 
            current_bot_selections,
            st.session_state.selected_bot
        )
        
        # Store both versions in session state
        st.session_state.combined_system_prompt = full_combined_prompt
        st.session_state.concise_system_prompt = concise_prompt
        
        # Preview the concise system message
        st.subheader(f"Preview: Optimized System Message for {st.session_state.selected_bot}")
        st.write("This is the token-optimized version that will be applied:")
        # Fix empty label warning by providing a label
        st.text_area(
            "System Prompt Preview", 
            value=concise_prompt, 
            height=150, 
            disabled=True, 
            key="preview_system_message",
            label_visibility="collapsed"  # Hide the label but still provide one
        )
        
        # Show the full combined prompt in an expander
        with st.expander("View Full (Unoptimized) System Prompt"):
            st.write("This is the full version of all selected traits (not used to save tokens):")
            st.text_area(
                "Full System Prompt",
                value=full_combined_prompt,
                height=200,
                disabled=True,
                key="full_system_message",
                label_visibility="collapsed"
            )
        
        # Show warning if the bots have the same characteristics
        if same_characteristics:
            st.warning(f"⚠️ With the current selection, {st.session_state.selected_bot} and {other_bot} will have identical personality traits across all characteristics. It is recommended for the bots to have at least one different characteristic to enhance the debate experience.")
        
        # Load button - label changes based on which bot is selected
        if st.button(f"Apply to {st.session_state.selected_bot}", disabled=same_characteristics):
            # Update the appropriate system message with the concise prompt
            if st.session_state.selected_bot == "Bot 1":
                st.session_state.system_message = concise_prompt
            else:
                st.session_state.system_message2 = concise_prompt
                
            # Make sure user name is properly reflected
            # The concise_prompt already includes the user name from session state
            
            st.success(f"Applied concise personality traits to {st.session_state.selected_bot}")
            st.rerun()
    else:
        st.warning("characteristics.json not found or has invalid format in the 'prompts' directory.")
    
    # Chat history viewer and other sidebar elements
    st.markdown("---")
    
    # Chat history viewer
    with st.expander("View Chat History"):
        st.json(st.session_state.messages)
    
    # Usage statistics viewer
    with st.expander("View Usage Statistics"):
        if st.session_state.usage_stats:
            for i, usage in enumerate(st.session_state.usage_stats):
                st.write(f"Message {i+1}:")
                st.write(f"- Prompt tokens: {usage['prompt_tokens']}")
                st.write(f"- Completion tokens: {usage['completion_tokens']}")
                st.write(f"- Total tokens: {usage['total_tokens']}")
                st.divider()
            
            # Calculate total usage
            total_prompt = sum(u["prompt_tokens"] for u in st.session_state.usage_stats)
            total_completion = sum(u["completion_tokens"] for u in st.session_state.usage_stats)
            total = sum(u["total_tokens"] for u in st.session_state.usage_stats)
            
            st.subheader("Total Usage")
            st.write(f"- Total prompt tokens: {total_prompt}")
            st.write(f"- Total completion tokens: {total_completion}")
            st.write(f"- Total tokens: {total}")
        else:
            st.write("No usage data available yet.")
    
    # Clear chat button
    if st.button("Clear Chat"):
        user_name = st.session_state.user_name if "user_name" in st.session_state else "User"
        st.session_state.messages = [{"role": "assistant1", "content": f"Hello {user_name}! I'm Bot 1. Bot 2 is also in this group chat. How can we help you today?"}]
        st.session_state.usage_stats = []
        st.session_state.waiting_for_bot2 = False
        # We don't reset selected_characteristics here to maintain the user's selections
        st.success("Chat history cleared!")
    
    # Process display toggle - moved to bottom
    st.markdown("---")
    st.session_state.show_process = st.checkbox("Show Model Process (Last message)", value=st.session_state.show_process)

# Main chat area with padding at bottom
chat_container = st.container()
with chat_container:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)  # Add a container with padding
    
    # Display chat messages from session state only
    for message in st.session_state.messages:
        if not message["content"]:  # Skip empty messages
            continue
        if message["role"] == "assistant1":
            with st.chat_message(message["role"], avatar="🤖"):
                st.markdown(message["content"])
        elif message["role"] == "assistant2":
            with st.chat_message(message["role"], avatar="🎯"):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close the container

# Store the current prompt in session state to maintain it between reruns
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None

# Chat input handling - improved flow with proper message display
if prompt := st.chat_input("Ask me anything..." if not st.session_state.waiting_for_bot2 else "Waiting for Bot 2...", disabled=st.session_state.waiting_for_bot2):
    # Store prompt in session state and add to history
    st.session_state.current_prompt = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add a small delay to ensure the user message is displayed first
    st.rerun()

# Handle Bot 1's response in the next run cycle
if st.session_state.current_prompt and not st.session_state.waiting_for_bot2:
    # Check if there's a pending user prompt to respond to
    if not any(msg["role"] == "assistant1" and msg["content"].strip() == "" for msg in st.session_state.messages):
        # Generate Bot 1's response
        if generate_response(st.session_state.current_prompt, is_bot2=False):
            # Set waiting flag for Bot 2
            st.session_state.waiting_for_bot2 = True
            st.rerun()

# Handle Bot 2's response if we're waiting for it
if st.session_state.waiting_for_bot2 and st.session_state.current_prompt is not None:
    # Check if Bot 1's response is complete
    if not any(msg["role"] == "assistant2" and msg["content"].strip() == "" for msg in st.session_state.messages):
        # Generate Bot 2's response
        if generate_response(st.session_state.current_prompt, is_bot2=True):
            # Reset states
            st.session_state.waiting_for_bot2 = False
            st.session_state.current_prompt = None
            st.rerun()