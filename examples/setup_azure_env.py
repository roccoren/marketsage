#!/usr/bin/env python
"""
Setup Azure OpenAI Environment - Helper to set up Azure OpenAI environment variables
"""
import os
import sys
import getpass
from pathlib import Path
from dotenv import set_key, dotenv_values

def main():
    """Main function"""
    print("Azure OpenAI Environment Setup Helper")
    print("=====================================")
    print("This script will help you set up your Azure OpenAI environment variables.")
    print("These variables will be saved to a .env file in your current directory.")
    print()
    
    # Determine .env file path
    env_path = Path('.env')
    
    # Load existing values if present
    existing_values = {}
    if env_path.exists():
        existing_values = dotenv_values(env_path)
        print(f"Found existing .env file at {env_path.absolute()}")
    
    # Get Azure OpenAI API key
    default_api_key = existing_values.get('OPENAI_API_KEY', '')
    if default_api_key:
        print(f"Current API key: {default_api_key[:4]}...{default_api_key[-4:]}")
        
    api_key = getpass.getpass("Enter your Azure OpenAI API key (press Enter to keep current): ")
    if not api_key and not default_api_key:
        print("Error: API key is required.")
        return 1
    
    if not api_key:
        api_key = default_api_key
    
    # Get Azure OpenAI endpoint
    default_endpoint = existing_values.get('OPENAI_API_BASE', '')
    if default_endpoint:
        print(f"Current endpoint: {default_endpoint}")
        
    endpoint = input(f"Enter your Azure OpenAI endpoint URL (press Enter to keep current): ")
    if not endpoint and not default_endpoint:
        print("Error: Endpoint URL is required.")
        return 1
    
    if not endpoint:
        endpoint = default_endpoint
    
    # Get Azure OpenAI deployment name
    default_deployment = existing_values.get('OPENAI_DEPLOYMENT_NAME', '')
    if default_deployment:
        print(f"Current deployment name: {default_deployment}")
        
    deployment = input(f"Enter your Azure OpenAI deployment name (press Enter to keep current): ")
    if not deployment and not default_deployment:
        print("Error: Deployment name is required.")
        return 1
    
    if not deployment:
        deployment = default_deployment
    
    # Get Azure OpenAI API version
    default_api_version = existing_values.get('OPENAI_API_VERSION', '2023-05-15')
    if default_api_version:
        print(f"Current API version: {default_api_version}")
        
    api_version = input(f"Enter your Azure OpenAI API version (press Enter to keep current): ")
    if not api_version:
        api_version = default_api_version
    
    # Get LLM model
    default_chat_model = existing_values.get('CHAT_MODEL', 'gpt-4')
    if default_chat_model:
        print(f"Current chat model: {default_chat_model}")
        
    chat_model = input(f"Enter your LLM model name (press Enter to keep current): ")
    if not chat_model:
        chat_model = default_chat_model
    
    # Create or update .env file
    print("\nWriting environment variables to .env file...")
    
    # Create file if it doesn't exist
    if not env_path.exists():
        env_path.touch()
    
    # Set variables
    set_key(env_path, 'OPENAI_API_KEY', api_key)
    set_key(env_path, 'OPENAI_API_TYPE', 'azure')
    set_key(env_path, 'OPENAI_API_BASE', endpoint)
    set_key(env_path, 'OPENAI_API_VERSION', api_version)
    set_key(env_path, 'OPENAI_DEPLOYMENT_NAME', deployment)
    set_key(env_path, 'CHAT_MODEL', chat_model)
    
    print("\nAzure OpenAI environment variables have been set up successfully!")
    print(f".env file created at {env_path.absolute()}")
    print("\nYou can now run the Azure OpenAI example with:")
    print("python examples/azure_openai_trio_test.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 