from setuptools import setup, find_packages

setup(
    name="koarami_kurumi",
    version="0.0.1",
    description="An AI VTuber named Koarami Kurumi.",
    author="koalitynuts",
    packages=find_packages(),
    long_description=open("README.md").read(),
    log_description_content_type="text/markdown",
    include_package_data=True,
)

