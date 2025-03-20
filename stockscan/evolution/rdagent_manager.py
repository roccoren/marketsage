"""
RDAgentManager - Handles integration with RDAgent for factor and model evolution
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import json
import subprocess
import pandas as pd
import tempfile
import logging
from dotenv import load_dotenv
import sys

logger = logging.getLogger(__name__)

class RDAgentManager:
    """
    Manager class for RDAgent integration
    """
    
    def __init__(self, 
               log_dir: str = './logs/rdagent', 
               api_key: Optional[str] = None, 
               chat_model: str = 'gpt-4-turbo',
               azure_endpoint: Optional[str] = None,
               azure_deployment: Optional[str] = None,
               azure_api_version: str = '2023-05-15'):
        """
        Initialize the RDAgentManager
        
        Parameters
        ----------
        log_dir : str
            Directory for RDAgent logs
        api_key : Optional[str]
            API key for RDAgent (if None, will use from .env file)
        chat_model : str
            LLM model to use for RDAgent
        azure_endpoint : Optional[str]
            Azure OpenAI endpoint URL (if None, will use OpenAI API)
        azure_deployment : Optional[str]
            Azure OpenAI deployment name (if None, will use OpenAI API)
        azure_api_version : str
            Azure OpenAI API version to use
        """
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Store the configurations for later use
        self.api_key = api_key
        self.chat_model = chat_model
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_api_version = azure_api_version
        
        # Set API key if provided
        if api_key is not None:
            os.environ['OPENAI_API_KEY'] = api_key
            
        # Determine if we're using Azure OpenAI
        self.use_azure = azure_endpoint is not None and azure_deployment is not None
        if self.use_azure:
            # Configure the environment following RDAgent documented approach
            # https://rdagent.readthedocs.io/en/latest/installation_and_configuration.html#azure-openai
            
            # Set main flag
            os.environ['USE_AZURE'] = 'True'
            
            # Set embedding model settings
            os.environ['EMBEDDING_OPENAI_API_KEY'] = api_key or os.environ.get('OPENAI_API_KEY', '')
            os.environ['EMBEDDING_AZURE_API_BASE'] = azure_endpoint
            os.environ['EMBEDDING_AZURE_API_VERSION'] = azure_api_version
            os.environ['EMBEDDING_MODEL'] = 'text-embedding-3-small'
            
            # Set chat model settings
            os.environ['CHAT_OPENAI_API_KEY'] = api_key or os.environ.get('OPENAI_API_KEY', '')
            os.environ['CHAT_AZURE_API_BASE'] = azure_endpoint
            os.environ['CHAT_AZURE_API_VERSION'] = azure_api_version
            os.environ['CHAT_MODEL'] = azure_deployment
            
            # Log the configuration
            logger.info(f"Using Azure OpenAI deployment: {azure_deployment}")
        else:
            # Using standard OpenAI API
            os.environ['CHAT_MODEL'] = chat_model
        
        # Check if RDAgent is installed
        self.is_available = True
        try:
            # Try to import rdagent module instead of running subprocess
            import importlib
            rdagent_module = importlib.import_module('rdagent')
            logger.info("RDAgent is installed and available.")
        except ImportError:
            self.is_available = False
            logger.warning("RDAgent is not installed. Please install it with 'pip install rdagent'")
            logger.warning("RDAgent features will be disabled.")
            
    def evolve_factors(self, 
                     data_path: str, 
                     output_path: str, 
                     existing_factors: Optional[Dict[str, str]] = None, 
                     backtest_results: Optional[Dict[str, float]] = None,
                     max_iterations: int = 5) -> Dict[str, str]:
        """
        Evolve factors using RDAgent
        
        Parameters
        ----------
        data_path : str
            Path to the data
        output_path : str
            Path to save the output
        existing_factors : Optional[Dict[str, str]]
            Existing factors to start with
        backtest_results : Optional[Dict[str, float]]
            Results from previous backtests to guide evolution
        max_iterations : int
            Maximum number of iterations
            
        Returns
        -------
        Dict[str, str]
            Dictionary of evolved factors
        """
        if not self.is_available:
            logger.warning("RDAgent is not available. Returning existing factors.")
            return existing_factors or {}
            
        os.makedirs(output_path, exist_ok=True)
        
        # Save existing factors if provided
        if existing_factors:
            existing_factors_path = os.path.join(output_path, 'existing_factors.json')
            with open(existing_factors_path, 'w') as f:
                json.dump(existing_factors, f, indent=4)
                
        # Save backtest results if provided
        if backtest_results:
            backtest_results_path = os.path.join(output_path, 'backtest_results.json')
            with open(backtest_results_path, 'w') as f:
                json.dump(backtest_results, f, indent=4)
                
        # Set up environment variables
        env_vars = os.environ.copy()
        if self.use_azure:
            env_vars['USE_AZURE'] = 'True'
            env_vars['EMBEDDING_OPENAI_API_KEY'] = self.api_key or os.environ.get('OPENAI_API_KEY', '')
            env_vars['EMBEDDING_AZURE_API_BASE'] = self.azure_endpoint
            env_vars['EMBEDDING_AZURE_API_VERSION'] = self.azure_api_version 
            env_vars['EMBEDDING_MODEL'] = 'text-embedding-3-small'
            env_vars['CHAT_OPENAI_API_KEY'] = self.api_key or os.environ.get('OPENAI_API_KEY', '')
            env_vars['CHAT_AZURE_API_BASE'] = self.azure_endpoint
            env_vars['CHAT_AZURE_API_VERSION'] = self.azure_api_version
            env_vars['CHAT_MODEL'] = self.azure_deployment
            
            # Create local .env file in the output directory
            env_file_path = os.path.join(output_path, '.env')
            try:
                with open(env_file_path, 'w') as f:
                    f.write(f"USE_AZURE=True\n")
                    f.write(f"EMBEDDING_OPENAI_API_KEY={env_vars['EMBEDDING_OPENAI_API_KEY']}\n")
                    f.write(f"EMBEDDING_AZURE_API_BASE={env_vars['EMBEDDING_AZURE_API_BASE']}\n")
                    f.write(f"EMBEDDING_AZURE_API_VERSION={env_vars['EMBEDDING_AZURE_API_VERSION']}\n")
                    f.write(f"EMBEDDING_MODEL={env_vars['EMBEDDING_MODEL']}\n")
                    f.write(f"CHAT_OPENAI_API_KEY={env_vars['CHAT_OPENAI_API_KEY']}\n")
                    f.write(f"CHAT_AZURE_API_BASE={env_vars['CHAT_AZURE_API_BASE']}\n") 
                    f.write(f"CHAT_AZURE_API_VERSION={env_vars['CHAT_AZURE_API_VERSION']}\n")
                    f.write(f"CHAT_MODEL={env_vars['CHAT_MODEL']}\n")
                logger.info(f"Created local .env file in {output_path}")
            except Exception as e:
                logger.warning(f"Failed to create local .env file: {e}")
        
        # Create a batch script for Windows or shell script for Unix
        if os.name == 'nt':
            # Windows batch script
            script_path = os.path.join(output_path, 'run_rdagent_factors.bat')
            with open(script_path, 'w') as f:
                # Set environment variables
                for key, value in env_vars.items():
                    if key.startswith(('USE_', 'EMBEDDING_', 'CHAT_', 'OPENAI_')):
                        f.write(f'SET {key}={value}\n')
                
                # Execute RDAgent command
                f.write(f'rdagent fin_factor')
                f.write(f' --data_path "{data_path}"')
                f.write(f' --output_path "{output_path}"')
                f.write(f' --log_dir "{self.log_dir}"')
                f.write(f' --evolving_n {max_iterations}\n')
                
                # Add optional arguments
                if existing_factors:
                    f.write(f' --existing_factors "{existing_factors_path}"')
                if backtest_results:
                    f.write(f' --backtest_results "{backtest_results_path}"')
        else:
            # Unix shell script
            script_path = os.path.join(output_path, 'run_rdagent_factors.sh')
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n\n")
                
                # Set environment variables
                for key, value in env_vars.items():
                    if key.startswith(('USE_', 'EMBEDDING_', 'CHAT_', 'OPENAI_')):
                        f.write(f'export {key}="{value}"\n')
                
                # Execute RDAgent command
                f.write(f'rdagent fin_factor')
                f.write(f' --data_path "{data_path}"')
                f.write(f' --output_path "{output_path}"')
                f.write(f' --log_dir "{self.log_dir}"')
                f.write(f' --evolving_n {max_iterations}')
                
                # Add optional arguments
                if existing_factors:
                    f.write(f' --existing_factors "{existing_factors_path}"')
                if backtest_results:
                    f.write(f' --backtest_results "{backtest_results_path}"')
            
            # Make the script executable
            os.chmod(script_path, 0o755)
            
        logger.info(f"Created RDAgent factors script at {script_path}")
        
        # Also create a Python script as an alternative approach
        py_script_path = os.path.join(output_path, 'run_rdagent_factors.py')
        with open(py_script_path, 'w') as f:
            f.write("#!/usr/bin/env python\n")
            f.write("import os\n")
            f.write("import sys\n")
            f.write("import json\n")
            f.write("import logging\n")
            f.write("import subprocess\n\n")
            
            # Set up environment variables
            f.write("# Set up environment variables\n")
            for key, value in env_vars.items():
                if key.startswith(('USE_', 'EMBEDDING_', 'CHAT_', 'OPENAI_')):
                    f.write(f"os.environ['{key}'] = '{value}'\n")
            
            f.write("\n# Set up arguments\n")
            f.write(f"data_path = '{data_path}'\n")
            f.write(f"output_path = '{output_path}'\n")
            f.write(f"log_dir = '{self.log_dir}'\n")
            f.write(f"evolving_n = {max_iterations}\n\n")
            
            # Setup paths to optional files
            if existing_factors:
                f.write(f"existing_factors_path = '{existing_factors_path}'\n")
            if backtest_results:
                f.write(f"backtest_results_path = '{backtest_results_path}'\n")
            
            f.write("# Set up logging\n")
            f.write("logging.basicConfig(level=logging.INFO)\n\n")
            
            # Add the command to run
            f.write("# Build rdagent command\n")
            f.write("cmd = [\n")
            f.write("    'rdagent', 'fin_factor',\n")
            f.write("    '--data_path', data_path,\n")
            f.write("    '--output_path', output_path,\n")
            f.write("    '--log_dir', log_dir,\n")
            f.write("    '--evolving_n', str(evolving_n)\n")
            f.write("]\n\n")
            
            # Add optional arguments
            if existing_factors:
                f.write("# Add existing factors path\n")
                f.write("cmd.extend(['--existing_factors', existing_factors_path])\n")
            if backtest_results:
                f.write("# Add backtest results path\n")
                f.write("cmd.extend(['--backtest_results', backtest_results_path])\n")
            
            # Execute the command
            f.write("\n# Execute rdagent command\n")
            f.write("try:\n")
            f.write("    print('Starting RDAgent factor evolution...')\n")
            f.write("    result = subprocess.run(cmd, check=True, capture_output=True, text=True)\n")
            f.write("    print('RDAgent stdout:', result.stdout)\n")
            f.write("    if result.stderr:\n")
            f.write("        print('RDAgent stderr:', result.stderr)\n")
            f.write("    print('RDAgent factor evolution completed')\n")
            f.write("except subprocess.CalledProcessError as e:\n")
            f.write("    print(f'Error running RDAgent: {e}')\n")
            f.write("    print(f'Error output: {e.stderr}')\n")
            f.write("    sys.exit(1)\n")
            f.write("except Exception as e:\n")
            f.write("    print(f'Error running RDAgent: {e}')\n")
            f.write("    sys.exit(1)\n")
        
        # Make the script executable
        os.chmod(py_script_path, 0o755)
        logger.info(f"Created RDAgent factor Python script at {py_script_path}")
        
        # Run the script
        try:
            logger.info(f"Running RDAgent factor script: {script_path}")
            if os.name == 'nt':
                result = subprocess.run(
                    [script_path],
                    check=True, 
                    capture_output=True, 
                    text=True,
                    shell=True,
                    env=env_vars
                )
            else:
                result = subprocess.run(
                    [script_path],
                    check=True, 
                    capture_output=True, 
                    text=True,
                    env=env_vars
                )
            
            # Log the output
            logger.info(f"RDAgent factor script stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"RDAgent factor script stderr: {result.stderr}")
            
            # Parse output to get the evolved factors
            factors_path = os.path.join(output_path, 'evolved_factors.json')
            if os.path.exists(factors_path):
                with open(factors_path, 'r') as f:
                    evolved_factors = json.load(f)
                return evolved_factors
            else:
                # Check for any other JSON file that might contain factors
                import glob
                json_files = glob.glob(os.path.join(output_path, '*.json'))
                for file_path in json_files:
                    if file_path != existing_factors_path and file_path != (backtest_results_path if backtest_results else "") and 'factor' in os.path.basename(file_path).lower():
                        try:
                            with open(file_path, 'r') as f:
                                potential_factors = json.load(f)
                            # Check if this looks like a factors dictionary
                            if isinstance(potential_factors, dict) and len(potential_factors) > 0:
                                logger.info(f"Using alternative factors from {file_path}")
                                return potential_factors
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Error running RDAgent factor script: {e}")
            # Fall back to the Python script if the shell script fails
            try:
                logger.info(f"Falling back to Python script: {py_script_path}")
                result = subprocess.run(
                    [sys.executable, py_script_path],
                    check=True, 
                    capture_output=True, 
                    text=True,
                    env=env_vars
                )
                
                # Log the output
                logger.info(f"RDAgent factor Python script stdout: {result.stdout}")
                if result.stderr:
                    logger.warning(f"RDAgent factor Python script stderr: {result.stderr}")
                
                # Check for output files again
                factors_path = os.path.join(output_path, 'evolved_factors.json')
                if os.path.exists(factors_path):
                    with open(factors_path, 'r') as f:
                        evolved_factors = json.load(f)
                    return evolved_factors
            except Exception as e2:
                logger.error(f"Error running fallback Python script: {e2}")
                return existing_factors or {}
            
        return existing_factors or {}
        
    def optimize_model(self, 
                     model_config: Dict[str, Any], 
                     data_path: str, 
                     output_path: str, 
                     backtest_results: Optional[Dict[str, float]] = None,
                     max_iterations: int = 5,
                     azure_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize model using RDAgent
        
        Parameters
        ----------
        model_config : Dict[str, Any]
            Current model configuration
        data_path : str
            Path to the data
        output_path : str
            Path to save the output
        backtest_results : Optional[Dict[str, float]]
            Results from previous backtests to guide optimization
        max_iterations : int
            Maximum number of iterations
        azure_config : Optional[Dict[str, Any]]
            Override Azure OpenAI configuration
            
        Returns
        -------
        Dict[str, Any]
            Optimized model configuration
        """
        if not self.is_available:
            logger.warning("RDAgent is not available. Returning original model config.")
            return model_config
            
        os.makedirs(output_path, exist_ok=True)
        
        # Save model config
        model_config_path = os.path.join(output_path, 'model_config.json')
        
        # Check if model_config_path already exists, don't overwrite it
        if not os.path.exists(model_config_path):
            with open(model_config_path, 'w') as f:
                # Handle non-JSON serializable values
                sanitized_config = {}
                for k, v in model_config.items():
                    try:
                        json.dumps({k: v})  # Test if serializable
                        sanitized_config[k] = v
                    except (TypeError, OverflowError):
                        sanitized_config[k] = str(v)
                json.dump(sanitized_config, f, indent=4)
        
        # Save backtest results if provided
        if backtest_results:
            backtest_results_path = os.path.join(output_path, 'backtest_results.json')
            with open(backtest_results_path, 'w') as f:
                json.dump(backtest_results, f, indent=4)
        
        # Set up Azure OpenAI environment if provided or if using class defaults
        use_azure = azure_config is not None or self.use_azure
        env_vars = os.environ.copy()
        
        if use_azure:
            # Use provided config or class defaults
            config = azure_config or {
                'use_azure': True,
                'azure_endpoint': self.azure_endpoint,
                'azure_api_key': self.api_key,
                'azure_api_version': self.azure_api_version,
                'azure_deployment': self.azure_deployment
            }
            
            # Configure environment variables for Azure OpenAI
            env_vars['USE_AZURE'] = 'True'
            
            # Embedding model settings
            env_vars['EMBEDDING_OPENAI_API_KEY'] = config.get('azure_api_key', self.api_key) or os.environ.get('OPENAI_API_KEY', '')
            env_vars['EMBEDDING_AZURE_API_BASE'] = config.get('azure_endpoint', self.azure_endpoint)
            env_vars['EMBEDDING_AZURE_API_VERSION'] = config.get('azure_api_version', self.azure_api_version)
            env_vars['EMBEDDING_MODEL'] = 'text-embedding-3-small'
            
            # Chat model settings
            env_vars['CHAT_OPENAI_API_KEY'] = config.get('azure_api_key', self.api_key) or os.environ.get('OPENAI_API_KEY', '')
            env_vars['CHAT_AZURE_API_BASE'] = config.get('azure_endpoint', self.azure_endpoint)
            env_vars['CHAT_AZURE_API_VERSION'] = config.get('azure_api_version', self.azure_api_version)
            env_vars['CHAT_MODEL'] = config.get('azure_deployment', self.azure_deployment)
            
            # Create local .env file in the output directory
            env_file_path = os.path.join(output_path, '.env')
            try:
                with open(env_file_path, 'w') as f:
                    f.write(f"USE_AZURE=True\n")
                    f.write(f"EMBEDDING_OPENAI_API_KEY={env_vars['EMBEDDING_OPENAI_API_KEY']}\n")
                    f.write(f"EMBEDDING_AZURE_API_BASE={env_vars['EMBEDDING_AZURE_API_BASE']}\n")
                    f.write(f"EMBEDDING_AZURE_API_VERSION={env_vars['EMBEDDING_AZURE_API_VERSION']}\n")
                    f.write(f"EMBEDDING_MODEL={env_vars['EMBEDDING_MODEL']}\n")
                    f.write(f"CHAT_OPENAI_API_KEY={env_vars['CHAT_OPENAI_API_KEY']}\n")
                    f.write(f"CHAT_AZURE_API_BASE={env_vars['CHAT_AZURE_API_BASE']}\n") 
                    f.write(f"CHAT_AZURE_API_VERSION={env_vars['CHAT_AZURE_API_VERSION']}\n")
                    f.write(f"CHAT_MODEL={env_vars['CHAT_MODEL']}\n")
                logger.info(f"Created local .env file in {output_path}")
            except Exception as e:
                logger.warning(f"Failed to create local .env file: {e}")
        
        # Create a batch script for Windows or shell script for Unix
        if os.name == 'nt':
            # Windows batch script
            script_path = os.path.join(output_path, 'run_rdagent.bat')
            with open(script_path, 'w') as f:
                # Set environment variables
                for key, value in env_vars.items():
                    if key.startswith(('USE_', 'EMBEDDING_', 'CHAT_', 'OPENAI_')):
                        f.write(f'SET {key}={value}\n')
                
                # Execute RDAgent command
                f.write(f'rdagent fin_model')
                f.write(f' --data_path "{data_path}"')
                f.write(f' --output_path "{output_path}"')
                f.write(f' --model_config "{model_config_path}"')
                f.write(f' --log_dir "{self.log_dir}"')
                f.write(f' --evolving_n {max_iterations}\n')
                
                # Add backtest results if provided
                if backtest_results:
                    f.write(f' --backtest_results "{backtest_results_path}"')
        else:
            # Unix shell script
            script_path = os.path.join(output_path, 'run_rdagent.sh')
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n\n")
                
                # Set environment variables
                for key, value in env_vars.items():
                    if key.startswith(('USE_', 'EMBEDDING_', 'CHAT_', 'OPENAI_')):
                        f.write(f'export {key}="{value}"\n')
                
                # Execute RDAgent command
                f.write(f'rdagent fin_model')
                f.write(f' --data_path "{data_path}"')
                f.write(f' --output_path "{output_path}"')
                f.write(f' --model_config "{model_config_path}"')
                f.write(f' --log_dir "{self.log_dir}"')
                f.write(f' --evolving_n {max_iterations}')
                
                # Add backtest results if provided
                if backtest_results:
                    f.write(f' --backtest_results "{backtest_results_path}"')
            
            # Make the script executable
            os.chmod(script_path, 0o755)
        
        logger.info(f"Created RDAgent script at {script_path}")
        
        # Also create a Python script as an alternative approach
        py_script_path = os.path.join(output_path, 'run_rdagent.py')
        with open(py_script_path, 'w') as f:
            f.write("#!/usr/bin/env python\n")
            f.write("import os\n")
            f.write("import sys\n")
            f.write("import json\n")
            f.write("import logging\n")
            f.write("import subprocess\n\n")
            
            # Set up environment variables
            f.write("# Set up environment variables\n")
            for key, value in env_vars.items():
                if key.startswith(('USE_', 'EMBEDDING_', 'CHAT_', 'OPENAI_')):
                    f.write(f"os.environ['{key}'] = '{value}'\n")
            
            f.write("\n# Set up arguments\n")
            f.write(f"data_path = '{data_path}'\n")
            f.write(f"output_path = '{output_path}'\n")
            f.write(f"model_config_path = '{model_config_path}'\n")
            f.write(f"log_dir = '{self.log_dir}'\n")
            f.write(f"evolving_n = {max_iterations}\n\n")
            
            # Add backtest results if provided
            if backtest_results:
                f.write(f"backtest_results_path = '{backtest_results_path}'\n\n")
            
            f.write("# Set up logging\n")
            f.write("logging.basicConfig(level=logging.INFO)\n\n")
            
            # Add the command to run
            f.write("# Build rdagent command\n")
            f.write("cmd = [\n")
            f.write("    'rdagent', 'fin_model',\n")
            f.write("    '--data_path', data_path,\n")
            f.write("    '--output_path', output_path,\n")
            f.write("    '--model_config', model_config_path,\n")
            f.write("    '--log_dir', log_dir,\n")
            f.write("    '--evolving_n', str(evolving_n)\n")
            f.write("]\n\n")
            
            # Add optional arguments
            if backtest_results:
                f.write("# Add backtest results path\n")
                f.write("cmd.extend(['--backtest_results', backtest_results_path])\n\n")
            
            # Execute the command
            f.write("# Execute rdagent command\n")
            f.write("try:\n")
            f.write("    print('Starting RDAgent optimization...')\n")
            f.write("    result = subprocess.run(cmd, check=True, capture_output=True, text=True)\n")
            f.write("    print('RDAgent stdout:', result.stdout)\n")
            f.write("    if result.stderr:\n")
            f.write("        print('RDAgent stderr:', result.stderr)\n")
            f.write("    print('RDAgent optimization completed')\n")
            f.write("except subprocess.CalledProcessError as e:\n")
            f.write("    print(f'Error running RDAgent: {e}')\n")
            f.write("    print(f'Error output: {e.stderr}')\n")
            f.write("    sys.exit(1)\n")
            f.write("except Exception as e:\n")
            f.write("    print(f'Error running RDAgent: {e}')\n")
            f.write("    sys.exit(1)\n")
        
        # Make the script executable
        os.chmod(py_script_path, 0o755)
        logger.info(f"Created RDAgent Python script at {py_script_path}")
        
        # Run the script
        try:
            logger.info(f"Running RDAgent script: {script_path}")
            if os.name == 'nt':
                result = subprocess.run(
                    [script_path],
                    check=True, 
                    capture_output=True, 
                    text=True,
                    shell=True,
                    env=env_vars
                )
            else:
                result = subprocess.run(
                    [script_path],
                    check=True, 
                    capture_output=True, 
                    text=True,
                    env=env_vars
                )
            
            # Log the output
            logger.info(f"RDAgent script stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"RDAgent script stderr: {result.stderr}")
            
            # Parse output to get the optimized model config
            optimized_config_path = os.path.join(output_path, 'optimized_model.json')
            if os.path.exists(optimized_config_path):
                with open(optimized_config_path, 'r') as f:
                    optimized_config = json.load(f)
                return optimized_config
            else:
                # Check for any other JSON file that might contain model parameters
                import glob
                json_files = glob.glob(os.path.join(output_path, '*.json'))
                for file_path in json_files:
                    if file_path != model_config_path and file_path != (backtest_results_path if backtest_results else ""):
                        try:
                            with open(file_path, 'r') as f:
                                potential_config = json.load(f)
                            # Check if this looks like a model config
                            if isinstance(potential_config, dict) and len(potential_config) > 0:
                                logger.info(f"Using alternative model config from {file_path}")
                                return potential_config
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Error running RDAgent script: {e}")
            # Fall back to the Python script if the shell script fails
            try:
                logger.info(f"Falling back to Python script: {py_script_path}")
                result = subprocess.run(
                    [sys.executable, py_script_path],
                    check=True, 
                    capture_output=True, 
                    text=True,
                    env=env_vars
                )
                
                # Log the output
                logger.info(f"RDAgent Python script stdout: {result.stdout}")
                if result.stderr:
                    logger.warning(f"RDAgent Python script stderr: {result.stderr}")
                
                # Parse output to get the optimized model config
                optimized_config_path = os.path.join(output_path, 'optimized_model.json')
                if os.path.exists(optimized_config_path):
                    with open(optimized_config_path, 'r') as f:
                        optimized_config = json.load(f)
                    return optimized_config
            except Exception as e2:
                logger.error(f"Error running fallback Python script: {e2}")
                return model_config
            
        return model_config
        
    def start_ui(self, port: int = 19899) -> Optional[subprocess.Popen]:
        """
        Start the RDAgent UI
        
        Parameters
        ----------
        port : int
            Port to run the UI on
            
        Returns
        -------
        Optional[subprocess.Popen]
            Process object for the UI, or None if RDAgent is not available
        """
        if not self.is_available:
            logger.warning("RDAgent is not available. Cannot start UI.")
            return None
            
        # Check port availability
        self._check_port_availability(port)
        
        # Start the UI
        try:
            process = subprocess.Popen([
                'rdagent', 'ui',
                '--port', str(port),
                '--log_dir', self.log_dir
            ])
            
            print(f"RDAgent UI started on http://localhost:{port}")
            return process
        except Exception as e:
            logger.error(f"Error starting RDAgent UI: {e}")
            return None
        
    def _check_port_availability(self, port: int):
        """
        Check if a port is available
        
        Parameters
        ----------
        port : int
            Port to check
        """
        if not self.is_available:
            return
            
        try:
            subprocess.run(['rdagent', 'health_check', '--port', str(port)],
                          check=True, 
                          capture_output=True)
            print(f"Port {port} is available")
        except subprocess.CalledProcessError:
            print(f"Warning: Port {port} may not be available")
            
    def check_docker(self) -> bool:
        """
        Check if Docker is installed and running
        
        Returns
        -------
        bool
            True if Docker is installed and running
        """
        if not self.is_available:
            logger.warning("RDAgent is not available. Skipping Docker check.")
            return False
            
        try:
            subprocess.run(['docker', 'info'], 
                          check=True, 
                          capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: Docker is not installed or not running. RDAgent requires Docker for code execution.")
            return False 