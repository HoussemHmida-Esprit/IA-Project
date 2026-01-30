from setuptools import setup, find_packages

setup(
    name="ia-project",
    version="0.1.0",
    packages=find_packages(include=["utils", "utils.*", "pages", "pages.*", "models", "models.*"]),
    python_requires=">=3.10",
)
