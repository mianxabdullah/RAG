import os
api_key = os.environ.get("GROQ_API_KEY", "")
if api_key:
    print(f"GROQ_API_KEY is set (length: {len(api_key)} characters)")
    print(f"  First 10 chars: {api_key[:10]}...")
else:
    print("GROQ_API_KEY is not set")
    print("\nTo set it:")
    print("  Windows PowerShell: $env:GROQ_API_KEY = 'your-key'")
    print("  Windows CMD: set GROQ_API_KEY=your-key")
    print("  Linux/Mac: export GROQ_API_KEY='your-key'")

