# API Setup
Our agents can be deployed with either OpenAI API or your local models.

## [Option 1] Using OpenAI API
Accessing the OpenAI API requires the API key, which you could get from [here](https://platform.openai.com/account/api-keys). We here provide instructions for different OS.

### Unix-like System (Linux / MacOS)
```bash
echo 'export OPENAI_API_KEY="your_api_key"' >> ~/.zshrc

# # If you are using other proxy services like Azure [TODO]
# echo 'export OPENAI_API_BASE_URL="your_base_url"' >> ~/.zshrc # (Optional)

# Let the change take place
source ~/.zshrc
```

Replace `~/.zshrc` with `~/.bashrc` if you are using bash.

### Windows
If you are using Command Prompt:
```bash
set OPENAI_API_KEY="your_api_key"

# If you are using other proxy services
# set OPENAI_API_BASE_URL="your_base_url" # (Optional)
```
Or if you are using PowerShell:
```powershell
$env:OPENAI_API_KEY="your_api_key"

# If you are using other proxy services
$env:OPENAI_API_BASE_URL="your_base_url" # (Optional)
```
These commands on Windows will set the environment variable for the duration of that particular Command Prompt or PowerShell session only. You may use `setx` or change the system properties dialog for the change to take place in all the new sessions.

### General method

Create a file named `.env` in your project directory, with the following setting.

```bash
OPENAI_API_KEY=<your-openai-api-key>
```

Then, load the environment variables in your python script:

```python
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```


## [Option 2] Using other APIs

If you are using other APIs that are not provided by OpenAI, you can refer to [Models/Using Models by API calling](../key_modules/models.md#using-models-by-api-calling)

## [Option 3] Using Local Models
If you are using local models, you can refer to [Models/Using Local Models](../key_modules/models.md#using-on-device-open-source-models)

