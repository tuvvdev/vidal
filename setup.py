from setuptools import setup, find_packages
import vidal

setup(
    name="vidal",
    version=".".join(vidal.__version__),
    description="Video To Images",
    author="Sang Pil Yoo, Vo Van Tu",
    author_email="ysp9714@gmail.com",
    url="",
    download_url="",
    install_requires=["opencv-python", 
                        "tqdm",
                        "numpy",
                        "grpc",
                        "kazoo",
                        "bs4",
                        "pandas",
                        "tensorflow"],
    entry_points={"console_scripts": ["vidal=vidal.cutter:main"]},
    packages=find_packages(exclude=["docs", "test*"]),
    keywords=["video", "frames"],
    python_requires=">=3.6",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
