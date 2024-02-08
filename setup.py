from setuptools import setup, find_packages

setup(
    name='lfd',
    version='1.0.0',
    description='An library for ML prototyping',
    entry_points={'console_scripts': []},
    python_requires='>=3.9.18',
    packages=find_packages(),
    install_requires=[
        "pandas==2.0.3",
        "numpy==1.23.0",
        "dash==2.11.1",
        "dash-bootstrap-components==1.4.1",
        "dash_auth==2.0.0",
        "ipykernel==6.23.3",
        "scikit-learn==1.3.0",
        "xgboost==1.7.6",
        "shap==0.41.0",
        "pyarrow==12.0.1",
        "jupyter==1.0.0",
        "tabulate==0.9.0",
        "openpyxl==3.1.2",
        "memory_profiler==0.61.0",
        "nbformat==5.9.0",
        "xlsxwriter==3.1.2",
        "lxml==4.9.2",
        "diskcache==5.6.1",
        "tensorflow==2.12.0",
        "opencv-python==4.8.0.74",
        "tensorflow_hub==0.13.0"
    ],
    package_data={'lfd': ['visualize/board/assets/*', 'config/*.html']},
    include_package_data=True
)