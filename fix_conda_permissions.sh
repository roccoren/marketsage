#!/bin/bash
set -e

echo "Creating local conda package cache directories..."

# Create local conda package and cache directories
mkdir -p ./.conda_pkgs
mkdir -p ./.conda_cache
mkdir -p ./.conda_env

# Configure conda to use these local directories
echo "Configuring conda to use local directories..."

# Create or update .condarc file directly instead of using conda config command
cat > ~/.condarc << EOF
pkgs_dirs:
  - $(pwd)/.conda_pkgs
envs_dirs:
  - $(pwd)/.conda_env
  - $HOME/.conda/envs
cache_dir: $(pwd)/.conda_cache
channels:
  - conda-forge
  - defaults
EOF

# Clean conda caches to prevent using the permission-denied locations
echo "Cleaning conda caches..."
rm -rf ~/Library/Caches/conda-anaconda-tos
rm -rf ~/.conda/pkgs
rm -rf ~/.conda/cache

# Initialize conda with the new configuration
echo "Initializing conda with new configuration..."
source $(conda info --base)/etc/profile.d/conda.sh
conda init bash
conda init zsh

echo "Done! Conda is now configured to use local directories in your project."
echo "You can now run ./setup_conda_env.sh to create your environment." 