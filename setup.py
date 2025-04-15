from setuptools import setup, find_packages

# This setup.py file is maintained for backward compatibility.
# For modern Python projects, the pyproject.toml file is the preferred way to configure a project.

setup(
    name="heihachi",
    version="0.1.0",
    description="A high-performance audio analysis framework for electronic music with focus on neurofunk and drum & bass",
    author="Kundai Sachikonye",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "pyyaml>=6.0",
        "h5py>=3.7.0",
        "psutil>=5.9.0",
        "numba>=0.56.0",
        "tqdm>=4.64.0",
        "fastdtw>=0.3.4",
        "madmom>=0.16.1",
        "networkx>=3.0",
    ],
    include_package_data=True,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/heihachi",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)
