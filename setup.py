from setuptools import setup, find_packages

setup(
    name="stockscan",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        line.strip()
        for line in open("config/requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "stockscan=src.main:main",
        ],
    },
)