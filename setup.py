from setuptools import setup, find_packages

setup(
    name="stockscan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.35",
        "pythonnet>=3.0.1",
        "clr>=1.0.3",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.1.0",
            "flake8>=7.0.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A stock market data analysis tool using LEAN",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stockscan",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)