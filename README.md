# Groq for Claude Code: An Anthropic-Compatible Proxy

This server acts as a bridge, enabling you to use **Claude Code** with **Groq's lightning-fast inference API**. It translates API requests and responses between the Anthropic format (used by Claude Code) and the Groq format (via LiteLLM), allowing seamless integration.

Especially useful to use with the new Kimi-K2 moonshot model.

![Claude Code with Groq Proxy](image.png)

## Features

- **Claude Code Compatibility with Groq**: Directly use the Claude Code CLI with Groq's optimized models.
- **Seamless Model Mapping**: Intelligently maps Claude Code model requests (e.g., `haiku`, `sonnet`, `opus` aliases) to your chosen Groq models.
- **LiteLLM Integration**: Leverages LiteLLM for robust and flexible interaction with the Groq API.
- **Enhanced Streaming Support**: Handles streaming responses from Groq with robust error recovery for malformed chunks and API errors.
- **Complete Tool Use for Claude Code**: Translates Claude Code's tool usage (function calling) to and from Groq's format, with robust handling of tool results.
- **Advanced Error Handling**: Provides specific and actionable error messages for common Groq API issues with automatic fallback strategies.
- **Resilient Architecture**: Gracefully handles Groq API instability with smart retry logic and fallback to non-streaming modes.
- **Diagnostic Endpoints**: Includes `/health` and `/test-connection` for easier troubleshooting of your setup.
- **Token Counting**: Offers a `/v1/messages/count_tokens` endpoint compatible with Claude Code.


## Prerequisites

- A Groq API key.
- Python 3.8+.
- Claude Code CLI installed (e.g., `npm install -g @anthropic-ai/claude-code`).

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/wearedevx/groq-for-claude-code.git # Or your fork
    cd groq-for-claude-code
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your Groq API key. You can also customize model mappings and server settings:
    ```dotenv
    # Required: Your Groq API key
    GROQ_API_KEY="your-groq-api-key"

    # Optional: Model mappings for Claude Code aliases
    BIG_MODEL="moonshotai/kimi-k2-instruct"    # For 'sonnet' or 'opus' requests
    SMALL_MODEL="qwen/qwen3-32b" # For 'haiku' requests
    
    # Optional: Server settings
    HOST="0.0.0.0"
    PORT="8082"
    LOG_LEVEL="WARNING"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Optional: Performance and reliability settings
    MAX_TOKENS_LIMIT="16384"           # Max tokens for Groq responses
    REQUEST_TIMEOUT="90"              # Request timeout in seconds
    MAX_RETRIES="2"                   # LiteLLM retries to Groq
    MAX_STREAMING_RETRIES="12"         # Streaming-specific retry attempts
    
    # Optional: Streaming control (use if experiencing issues)
    FORCE_DISABLE_STREAMING="false"     # Disable streaming globally
    EMERGENCY_DISABLE_STREAMING="false" # Emergency streaming disable
    ```

5.  **Run the server**:
    The `server.py` script includes a `main()` function that starts the Uvicorn server:
    ```bash
    python server.py
    ```
    For development with auto-reload (restarts when you save changes to `server.py`):
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8082 --reload
    ```
    You can view all startup options, including configurable environment variables, by running:
    ```bash
    python server.py --help
    ```

## Usage with Claude Code

1.  **Start the Proxy Server**: Ensure the Groq proxy server (this application) is running (see step 5 above).

2.  **Configure Claude Code to Use the Proxy**:
    Set the `ANTHROPIC_BASE_URL` environment variable when running Claude Code:
    ```bash
    ANTHROPIC_BASE_URL=http://localhost:8082 claude
    ```
    Replace `localhost:8082` if your proxy is running on a different host or port.

## How It Works: Powering Claude Code with Groq

1.  **Claude Code Request**: You issue a command or prompt in the Claude Code CLI.
2.  **Anthropic Format**: Claude Code sends an API request (in Anthropic's Messages API format) to the proxy server's address (`http://localhost:8082`).
3.  **Proxy Translation (Anthropic to Groq)**: The proxy server:
    *   Receives the Anthropic-formatted request.
    *   Validates it and maps any Claude model aliases (like `claude-3-sonnet...`) to the corresponding Groq model specified in your `.env` (e.g., `moonshotai/kimi-k2-instruct`).
    *   Translates the message structure, content blocks, and tool definitions into a format LiteLLM can use with the Groq API.
4.  **LiteLLM to Groq**: LiteLLM sends the prepared request to the target Groq model using your `GROQ_API_KEY`.
5.  **Groq Response**: Groq processes the request and sends its response back through LiteLLM.
6.  **Proxy Translation (Groq to Anthropic)**: The proxy server:
    *   Receives the Groq response from LiteLLM (this can be a stream of events or a complete JSON object).
    *   Handles streaming errors and malformed chunks with intelligent recovery.
    *   Converts Groq's output (text, tool calls, stop reasons) back into the Anthropic Messages API format that Claude Code expects.
7.  **Response to Claude Code**: The proxy sends the Anthropic-formatted response back to your Claude Code client, which then displays the result or performs the requested action.

## Model Mapping for Claude Code

To ensure Claude Code's model requests are handled correctly by Groq:

- Requests from Claude Code for model names containing **"haiku"** (e.g., `claude-3-haiku-20240307`) are mapped to the Groq model specified by your `SMALL_MODEL` environment variable (default: `qwen/qwen3-32b`).
- Requests from Claude Code for model names containing **"sonnet"** or **"opus"** (e.g., `claude-3-sonnet-20240229`, `claude-3-opus-20240229`) are mapped to the Groq model specified by your `BIG_MODEL` environment variable (default: `moonshotai/kimi-k2-instruct`).

The server maintains a list of known Groq models. If a recognized Groq model is requested by the client without the `groq/` prefix, the proxy will add it.

## Endpoints

- `POST /v1/messages`: The primary endpoint for Claude Code to send messages to Groq. It's fully compatible with the Anthropic Messages API specification that Claude Code uses.
- `POST /v1/messages/count_tokens`: Allows Claude Code to estimate the token count for a set of messages, using Groq's tokenization.
- `GET /health`: Returns the health status of the proxy, including API key configuration, streaming settings, and basic API key validation.
- `GET /test-connection`: Performs a quick API call to Groq to verify connectivity and that your `GROQ_API_KEY` is working.
- `GET /`: Root endpoint providing a welcome message, current configuration summary (models, limits), and available endpoints.

## Error Handling & Troubleshooting

### Common Issues and Solutions

**Streaming Errors (malformed chunks):**
- The proxy automatically handles malformed JSON chunks from Groq
- If streaming becomes unstable, set `FORCE_DISABLE_STREAMING=true` as a temporary fix
- Increase `MAX_STREAMING_RETRIES` for more resilient streaming

**Groq 500 Internal Server Errors:**
- The proxy automatically retries with exponential backoff
- These are temporary Groq API issues that resolve automatically
- Check `/health` endpoint to monitor API status

**Connection Timeouts:**
- Increase `REQUEST_TIMEOUT` if experiencing frequent timeouts
- Check your internet connection and firewall settings
- Use `/test-connection` endpoint to verify API connectivity

**Rate Limiting:**
- Monitor your Groq console for quota limits
- The proxy will provide specific rate limit guidance in error messages

### Emergency Mode

If you experience persistent issues:
```bash
# Disable streaming temporarily
export EMERGENCY_DISABLE_STREAMING=true

# Or force disable all streaming
export FORCE_DISABLE_STREAMING=true
```

## Logging

The server provides detailed logs, which are especially useful for understanding how Claude Code requests are translated for Groq and for monitoring error recovery. Logs are colorized in TTY environments for easier reading. Adjust verbosity with the `LOG_LEVEL` environment variable:

- `DEBUG`: Detailed request/response logging and error recovery steps
- `INFO`: General operation logging
- `WARNING`: Error recovery and fallback notifications (recommended)
- `ERROR`: Only errors and failures
- `CRITICAL`: Only critical failures

## Performance Tips

- **Model Selection**: Use `moonshotai/kimi-k2-instruct` for everything, it's too good!
- **Streaming**: Keep streaming enabled for better interactivity; the proxy handles errors automatically
- **Timeouts**: Increase `REQUEST_TIMEOUT` for complex requests that need more processing time
- **Retries**: Adjust `MAX_STREAMING_RETRIES` based on your network stability

## Contributing

Contributions, issues, and feature requests are welcome! Please submit them on the GitHub repository.

Areas where contributions are especially valuable:
- Additional Groq model support
- Performance optimizations
- Enhanced error recovery strategies
- Documentation improvements

## Thanks

This project was heavily inspired by and builds upon the foundational work of the [claude-code-proxy by @1rgs](https://github.com/1rgs/claude-code-proxy). Their original proxy was instrumental in demonstrating the viability of such a bridge.

Special thanks to the community for testing and feedback on error handling improvements.