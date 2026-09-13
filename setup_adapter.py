"""STEP 1 Integration Adapter"""

# Models
MODELS = [
    {"id": "gpt-4.1", "name": "GPT-4.1"},
    {"id": "gpt-4o-mini", "name": "GPT-4o-mini"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5"},
    {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Lite"},
    {"id": "gpt-oss-120b", "name": "Groq 120B"},
    {"id": "gpt-oss-20b", "name": "Groq 20B"},
    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
    {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5"},
    {"id": "traditional-generator", "name": "Traditional"},
]

TOTAL_EXPERIMENT_SAMPLES = 1350

def get_experiment_config():
    return {
        "total_samples": TOTAL_EXPERIMENT_SAMPLES,
        "producers": len(MODELS),
        "estimated_cost_usd": 50.0,
    }

if __name__ == "__main__":
    config = get_experiment_config()
    print(f"✅ STEP 1 Setup")
    print(f"   Models: {config['producers']}")
    print(f"   Samples: {config['total_samples']:,}")
    print(f"   Cost: ${config['estimated_cost_usd']:.2f}")