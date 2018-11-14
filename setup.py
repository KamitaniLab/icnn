'''Setup script for icnn.'''


from setuptools import setup


if __name__ == '__main__':
    setup(name='icnn',
          version='1.1.0',
          description='Inverting CNN',
          author='Shen Guo-Hua, Kei Majima',
          author_email='kamitanilab.contact@gmail.com',
          url='https://github.com/KamitaniLab/icnn',
          license='MIT',
          packages=['icnn'],
          install_requires=['numpy', 'scipy'])
