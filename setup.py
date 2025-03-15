from setuptools import setup, find_packages

setup(
    name="market_system",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "jsonschema>=4.17.0",
        "psutil>=5.9.0",
        "python-dateutil>=2.8.0"
    ],
    python_requires=">=3.7",
)