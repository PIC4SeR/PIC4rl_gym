import platform
from setuptools import setup, find_packages

install_requires = [
    "cpprb>=8.1.1",
    "setuptools>=41.0.0",
    "numpy>=1.16.0",
    "joblib",
    "future",
    "scipy",
    "scikit-image"]

tf_version = "2.10"  # Default Version, 2.11 and latest version give unsolved errors in td3

compatible_tfp = {"2.10": ["tensorflow-probability~=0.18"],
		  "2.9": ["tensorflow-probability~=0.17"],
		  "2.8": ["tensorflow-probability~=0.16"],
		  "2.7": ["tensorflow-probability~=0.15"],
		  "2.6": ["tensorflow-probability~=0.14"],
		  "2.5": ["tensorflow-probability~=0.15"],
		  "2.4": ["tensorflow-probability~=0.12"],
                  "2.3": ["tensorflow-probability~=0.11"],
                  "2.2": ["tensorflow-probability~=0.10"],
                  "2.1": ["tensorflow-probability~=0.8"],
                  "2.0": ["tensorflow-probability~=0.8"]}
compatible_tfa = {"2.10": ["tensorflow-addons~=0.18"],
		  "2.9": ["tensorflow-addons~=0.17"],
		  "2.8": ["tensorflow-addons~=0.16"],
		  "2.7": ["tensorflow-addons~=0.15"],
		  "2.6": ["tensorflow-addons~=0.14"],
		  "2.5": ["tensorflow-addons~=0.15"],
		  "2.4": ["tensorflow-addons~=0.13"],
                  "2.3": ["tensorflow-addons~=0.13"],
                  "2.2": ["tensorflow-addons==0.11"],
                  "2.1": ["tensorflow-addons~=0.9"],
                  "2.0": ["tensorflow-addons~=0.6"]}
install_requires.append(*compatible_tfp[tf_version])

if not (platform.system() == 'Windows' and tf_version == "2.0"):
    # tensorflow-addons does not support tf2.0 on Windows
    install_requires.append(*compatible_tfa[tf_version])


compatible_gym = {
    "2.0": "gym[atari]<0.21.0",
    "2.1": "gym[atari]<0.21.0"
}
gym_version = compatible_gym.get(tf_version, "gym[atari]>=0.21.0")

extras_require = {
    "tf": ["tensorflow>=2.0.0"],
    "tf_gpu": ["tensorflow-gpu>=2.0.0"],
    "examples": [gym_version, "opencv-python"],
    "test": ["coveralls", gym_version, "matplotlib", "opencv-python", "future"]
}

setup(
    name="tf2rl",
    version="1.1.5",
    description="Deep Reinforcement Learning for TensorFlow2",
    url="https://github.com/keiohta/tf2rl",
    author="Kei Ohta",
    author_email="dev.ohtakei@gmail.com",
    license="MIT",
    packages=find_packages("."),
    install_requires=install_requires,
    extras_require=extras_require)
