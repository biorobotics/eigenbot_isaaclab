# Eigenbot IsaacLab

Windows setup
```
uv venv --python 3.11 biorobotics

biorobotics\Scripts\activate
```

Ensure that everything below is run in the virtual environment, and agree to any agreements prompted.

```
uv pip install --upgrade pip

uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

```
isaacsim        # Test if everything is installed correctly.
```

```
git clone https://github.com/biorobotics/eigenbot_isaaclab.git

cd eigenbot_isaaclab\IsaacLab

.\isaaclab.bat -i
```

Use `uv pip install` to fix any broken dependencies. The most common are:
```
uv pip install torchaudio==2.7.0
```