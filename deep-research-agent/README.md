### Installation

1. Create a virtual environment and install dependencies:

```bash
# Using uv (recommended)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

2. Set up your environment variables:

```bash
cp .env.example .env
# Edit .env and add your API keys and ark model endpoint.
```
