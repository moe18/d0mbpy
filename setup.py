from setuptools import setup, find_packages

setup(
    name='d0mbpy',
    version='0.1',
    packages=find_packages(),
    description='d0mbpy is your go-to study tool for exploring the fascinating world of NumPy and linear algebra. Dive into numerical computing at your own pace, and dont be afraid to take it slow – because d0mbpy certainly does!',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Moe Chabot',
    author_email='chabotmoe1@gmail.com',
    url='https://github.com/moe18/d0mbpy',
    license='LICENSE',
    install_requires=[
        'numpy'
    ],
)
