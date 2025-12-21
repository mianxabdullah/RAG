import os
import groq

api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    print("Please set GROQ_API_KEY environment variable")
else:
    client = groq.Groq(api_key=api_key)
    try:
        models = client.models.list()
        print("Available Groq models:")
        print("-" * 50)
        for model in models.data:
            print(f"- {model.id}")
    except Exception as e:
        print(f"Error: {e}")

