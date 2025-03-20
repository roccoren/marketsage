# Azure OpenAI Integration with StockScan

This document provides detailed information on using Azure OpenAI with StockScan for LLM-powered stock prediction and analysis.

## Prerequisites

Before using the Azure OpenAI integration, you need:

1. An Azure account with an Azure OpenAI Service resource
2. Access to GPT models in your Azure OpenAI deployment
3. Your Azure OpenAI API key
4. The Azure OpenAI endpoint URL
5. A deployment name for your GPT model

## Setup

### Azure OpenAI Resource

1. Create an Azure OpenAI resource in the [Azure Portal](https://portal.azure.com/)
2. Deploy a GPT-4 or similar model in your Azure OpenAI resource
3. Note down your:
   - API key
   - Endpoint URL
   - Deployment name

### Environment Configuration

Set up your environment variables:

```bash
# Set Azure OpenAI API key
export OPENAI_API_KEY=your-azure-openai-api-key
```

## Usage

### Command Line Arguments

When running StockScan with Azure OpenAI integration, use the following command line arguments:

```bash
python examples/azure_openai_example.py \
  --azure_endpoint https://your-resource-name.openai.azure.com/ \
  --azure_deployment your-gpt4-deployment-name \
  --symbols AAPL,MSFT,NVDA \
  --start_date 2022-01-01 \
  --end_date 2023-01-01
```

Additional options:

- `--azure_api_version`: API version (default: 2023-05-15)
- `--rdagent_ui`: Start the RDAgent UI for visualization
- `--output_dir`: Directory for output files

### Using in Python Code

You can integrate Azure OpenAI in your custom Python code:

```python
from stockscan.evolution.rdagent_manager import RDAgentManager
from stockscan.pipeline import Pipeline

# Create RDAgent manager with Azure OpenAI configuration
rdagent_manager = RDAgentManager(
    log_dir='./logs/rdagent',
    api_key=your_api_key,  # From environment or directly provided
    chat_model='gpt-4',    # This will be mapped to your Azure deployment
    azure_endpoint='https://your-resource-name.openai.azure.com/',
    azure_deployment='your-gpt4-deployment-name',
    azure_api_version='2023-05-15'
)

# Create pipeline with Azure OpenAI integration
pipeline = Pipeline(
    provider_uri='~/.qlib/qlib_data/us_data',
    region='us',
    output_dir='./output',
    log_dir='./logs',
    rdagent_api_key=your_api_key,
    azure_endpoint='https://your-resource-name.openai.azure.com/',
    azure_deployment='your-gpt4-deployment-name',
    azure_api_version='2023-05-15',
    chat_model='gpt-4'
)

# Use the pipeline for factor evolution and model optimization
pipeline.prepare_data(
    instruments=['AAPL', 'MSFT', 'NVDA'],
    start_time='2022-01-01',
    end_time='2023-01-01'
)

results = pipeline.run_pipeline(
    max_iterations=3,
    stopping_threshold=0.01
)
```

## What RDAgent Does

The RDAgent integration uses LLMs to:

1. **Evolve Technical Factors**: Generate and optimize custom technical indicators for better prediction accuracy
2. **Optimize Models**: Tune model hyperparameters and architectures based on backtest results
3. **Analyze Market Patterns**: Provide natural language insights into market trends

## Performance Considerations

- Azure OpenAI has different rate limits and pricing than direct OpenAI API
- Consider the token usage and model capabilities when configuring iterations
- Set reasonable `max_iterations` values (3-5 is often sufficient)
- Monitor costs through the Azure Portal

## Troubleshooting

### Common Issues

**Error: Azure OpenAI API key not found**
- Ensure your `OPENAI_API_KEY` environment variable is set correctly

**Error: Invalid deployment name**
- Verify your deployment name matches exactly what's in the Azure Portal

**Error: Azure OpenAI resource not found**
- Confirm your endpoint URL is correct
- Check that your API key has access to the specified resource

**Error: RDAgent not installed**
- Install RDAgent with `pip install rdagent`

## Best Practices

1. **Start with smaller datasets** when testing Azure OpenAI integration
2. **Use more powerful models** (GPT-4) for complex factor evolution
3. **Save evolved factors** for reuse to minimize API calls
4. **Set appropriate stopping thresholds** to avoid unnecessary iterations
5. **Monitor token usage** to control costs

## Examples

See the `examples/azure_openai_example.py` script for a complete working example. 