from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Optional dependencies
extras_require = {
    "ml": [
        "scikit-learn>=1.0.0",
        "tensorflow>=2.12.0",
        "keras>=2.12.0"
    ],
    "dev": [
        "black>=23.3.0",
        "flake8>=6.0.0",
        "isort>=5.12.0",
        "mypy>=1.3.0",
        "pre-commit>=3.3.0",
        "pytest>=7.3.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0"
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "nbsphinx>=0.9.0"
    ]
}

setup(
    name="stockscan",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A LEAN-based trading and backtesting platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stockscan",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/stockscan/issues",
        "Documentation": "https://stockscan.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/stockscan",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "stockscan=stockscan.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)