import requests
import json
import os
from typing import Optional


class CompanyPolicyChatbot:
    def __init__(self, policy_file_path: str = "company_policy.txt", model: str = "llama3:latest"):
        """
        Initialize the company policy chatbot.

        Args:
            policy_file_path: Path to the company policy text file
            model: Ollama model name (default: llama3)
        """
        self.policy_file_path = policy_file_path
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.policy_content = ""
        self.conversation_history = []
        self.available_models = []

        # Load company policy
        self.load_policy()

        # Check available models and suggest alternatives
        self.check_available_models()

    def load_policy(self) -> None:
        """Load the company policy from the text file."""
        try:
            with open(self.policy_file_path, 'r', encoding='utf-8') as file:
                self.policy_content = file.read()
            print(f"✅ Company policy loaded successfully from {self.policy_file_path}")
        except FileNotFoundError:
            print(f"❌ Error: Could not find {self.policy_file_path}")
            print("Please ensure the company_policy.txt file exists in the current directory.")
            exit(1)
        except Exception as e:
            print(f"❌ Error loading policy file: {str(e)}")
            exit(1)

    def check_available_models(self) -> None:
        """Check what models are available in Ollama."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model['name'] for model in data.get('models', [])]

                # Suggest faster alternatives if current model might be slow
                fast_models = ['mistral', 'phi', 'gemma:2b', 'tinyllama']
                available_fast = [m for m in self.available_models if any(fast in m for fast in fast_models)]

                if self.model not in self.available_models:
                    print(f"⚠️  Warning: Model '{self.model}' not found.")
                    if available_fast:
                        print(f"💡 Suggestion: Try a faster model like: {', '.join(available_fast[:3])}")
                    print(f"📋 Available models: {', '.join(self.available_models[:5])}")
                elif any('llama' in self.model.lower() and '70b' in self.model.lower() for m in [self.model]):
                    print("⚠️  You're using a large model that may be slow.")
                    if available_fast:
                        print(f"💡 For faster responses, consider: {', '.join(available_fast[:3])}")

        except Exception:
            pass  # Silent fail, not critical

    def switch_model(self, new_model: str) -> bool:
        """
        Switch to a different model.

        Args:
            new_model: Name of the new model to use

        Returns:
            True if switch was successful
        """
        if new_model in self.available_models:
            old_model = self.model
            self.model = new_model
            print(f"🔄 Switched from {old_model} to {new_model}")
            return True
        else:
            print(f"❌ Model {new_model} is not available.")
            return False

    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def query_ollama_simple(self, prompt: str) -> Optional[str]:
        """
        Simple, fast query with minimal context for quick responses.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response or None if there was an error
        """
        try:
            # Very minimal prompt for speed
            simple_prompt = f"Answer briefly based on company policy:\n\nQuestion: {prompt}\n\nAnswer:"

            payload = {
                "model": self.model,
                "prompt": simple_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200,  # Very short responses
                    "num_ctx": 1024,  # Small context
                    "top_p": 0.5
                }
            }

            response = requests.post(
                self.ollama_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # Short timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            return None

        except Exception:
            return None

    def query_ollama(self, prompt: str, use_streaming: bool = False) -> Optional[str]:
        """
        Send a query to Ollama and get the response.

        Args:
            prompt: The prompt to send to the model
            use_streaming: Whether to use streaming response

        Returns:
            The model's response or None if there was an error
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": use_streaming,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused responses
                    "top_p": 0.8,
                    "num_ctx": 4096,  # Context window
                    "num_predict": 512,  # Limit response length
                    "stop": ["\n\nUSER:", "USER QUESTION:", "\n\nQUESTION:"]
                }
            }

            if use_streaming:
                return self._handle_streaming_response(payload)
            else:
                response = requests.post(
                    self.ollama_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60  # Reasonable timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    print(f"❌ Error: Ollama returned status code {response.status_code}")
                    return None

        except requests.exceptions.Timeout:
            print("❌ Error: Request timed out. Trying simplified approach...")
            return self.query_ollama_simple(
                prompt.split("QUESTION: ")[-1].split("\n")[0] if "QUESTION: " in prompt else prompt)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error communicating with Ollama: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return None

    def _handle_streaming_response(self, payload) -> Optional[str]:
        """Handle streaming response from Ollama."""
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
                stream=True
            )

            if response.status_code == 200:
                full_response = ""
                print("\n🤖 Bot: ", end="", flush=True)

                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                chunk = data['response']
                                print(chunk, end="", flush=True)
                                full_response += chunk
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue

                print()  # New line after streaming
                return full_response.strip()
            else:
                return None

        except Exception as e:
            print(f"❌ Streaming error: {str(e)}")
            print("💡 Try switching to a faster model or check your system resources.")
            return None

    def create_policy_prompt(self, user_question: str) -> str:
        """
        Create a prompt that includes relevant policy context and user question.

        Args:
            user_question: The user's question about company policies

        Returns:
            A formatted prompt for the model
        """
        # Try to find relevant sections (simple keyword matching)
        relevant_sections = self.find_relevant_sections(user_question)

        if relevant_sections:
            policy_context = relevant_sections
        else:
            # If no specific sections found, use truncated version
            policy_context = self.policy_content[:2000] + "..." if len(
                self.policy_content) > 2000 else self.policy_content

        prompt = f"""You are a company policy assistant. Answer the user's question based on the policy information provided.

POLICY INFORMATION:
{policy_context}

QUESTION: {user_question}

Provide a clear, concise answer based on the policy information. If the information isn't sufficient, say so.

ANSWER:"""

        return prompt

    def find_relevant_sections(self, question: str) -> str:
        """
        Find relevant sections of the policy based on keywords in the question.

        Args:
            question: The user's question

        Returns:
            Relevant sections of the policy
        """
        question_lower = question.lower()
        policy_lines = self.policy_content.split('\n')
        relevant_lines = []

        # Common policy keywords
        keywords = {
            'vacation': ['vacation', 'holiday', 'time off', 'pto', 'paid time'],
            'sick': ['sick', 'illness', 'medical leave', 'health'],
            'remote': ['remote', 'work from home', 'wfh', 'telecommute'],
            'expense': ['expense', 'reimbursement', 'receipt', 'travel'],
            'dress': ['dress', 'attire', 'clothing', 'appearance'],
            'conduct': ['conduct', 'behavior', 'harassment', 'discrimination'],
            'performance': ['performance', 'review', 'evaluation', 'goals'],
            'training': ['training', 'development', 'education', 'learning'],
            'benefits': ['benefits', 'insurance', 'healthcare', 'retirement']
        }

        # Find matching keywords
        matched_categories = []
        for category, terms in keywords.items():
            if any(term in question_lower for term in terms):
                matched_categories.append(category)

        # Extract relevant sections
        for i, line in enumerate(policy_lines):
            line_lower = line.lower()

            # Check if line contains relevant keywords
            if matched_categories:
                for category in matched_categories:
                    if any(term in line_lower for term in keywords[category]):
                        # Include surrounding context
                        start = max(0, i - 2)
                        end = min(len(policy_lines), i + 5)
                        relevant_lines.extend(policy_lines[start:end])
                        break

            # Also include section headers and important lines
            if (line.strip() and
                    (line.startswith('#') or
                     line.isupper() or
                     any(keyword in line_lower for keyword_list in keywords.values() for keyword in keyword_list))):
                relevant_lines.append(line)

        # Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in relevant_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

        relevant_text = '\n'.join(unique_lines)

        # Limit length to prevent timeouts
        if len(relevant_text) > 1500:
            relevant_text = relevant_text[:1500] + "..."

        return relevant_text if relevant_text.strip() else ""

    def chat(self) -> None:
        """Start the interactive chat session."""
        print("\n🤖 Company Policy Chatbot")
        print("=" * 50)
        print("Hi! I'm here to help you understand our company policies.")
        print("Ask me any questions about policies, procedures, or guidelines.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Commands: '/models' to see available models, '/model <name>' to switch")
        print("=" * 50)

        # Check Ollama connection
        if not self.check_ollama_connection():
            print("❌ Error: Cannot connect to Ollama.")
            print("Please make sure Ollama is running (ollama serve) and try again.")
            return

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n🤖 Goodbye! Feel free to ask about policies anytime.")
                    break

                if not user_input:
                    print("Please enter a question about company policies.")
                    continue

                # Special commands
                if user_input.lower().startswith('/model '):
                    new_model = user_input[7:].strip()
                    self.switch_model(new_model)
                    continue
                elif user_input.lower() == '/models':
                    print(f"📋 Available models: {', '.join(self.available_models)}")
                    print(f"🔧 Current model: {self.model}")
                    print("💡 Use '/model <name>' to switch models")
                    continue

                # Show thinking indicator
                print("\n🤖 Bot: Looking up policy information...")

                # Create prompt with relevant policy context
                prompt = self.create_policy_prompt(user_input)

                # Get response from Ollama (try non-streaming first)
                response = self.query_ollama(prompt, use_streaming=False)

                if response:
                    print(f"\n🤖 Bot: {response}")

                    # Store conversation
                    self.conversation_history.append({
                        "question": user_input,
                        "answer": response
                    })
                else:
                    print("\n🤖 Bot: I'm sorry, I couldn't process your question right now.")
                    print("💡 Try switching to a faster model with '/models' and '/model <name>'")

            except KeyboardInterrupt:
                print("\n\n🤖 Goodbye! Feel free to ask about policies anytime.")
                break

    def ask_question(self, question: str) -> str:
        """
        Ask a single question programmatically (useful for integration).

        Args:
            question: The question to ask

        Returns:
            The bot's response
        """
        prompt = self.create_policy_prompt(question)
        response = self.query_ollama(prompt)
        return response if response else "I'm sorry, I couldn't process your question right now."


def main():
    """Main function to run the chatbot."""
    try:
        # Initialize the chatbot
        chatbot = CompanyPolicyChatbot()

        # Start the chat session
        chatbot.chat()

    except Exception as e:
        print(f"❌ Error starting chatbot: {str(e)}")


if __name__ == "__main__":
    main()

# Example usage for programmatic access:
"""
# Initialize chatbot
bot = CompanyPolicyChatbot("company_policy.txt")

# Ask questions programmatically
response = bot.ask_question("What is the vacation policy?")
print(response)

response = bot.ask_question("How many sick days do I get?")
print(response)
"""