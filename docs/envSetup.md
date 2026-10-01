Local setup instructions
Install dependencies
Make sure you have Python 3.8+ installed.

You can install all required packages using the provided requirements.txt file:

pip install -r requirements.txt

If you're using smolagents, install it separately:

pip install smolagents

For other options like pydantic-ai or npcsh[lite], refer to their documentation.

Create .env File
Add your OpenAI-compatible API key:

UDACITY_OPENAI_API_KEY=your_openai_key_here

This project uses a custom OpenAI-compatible proxy hosted at https://openai.vocareum.com/v1(opens in a new tab).
