FROM kaggle/python-gpu-build:latest

# install lint and static analysis tools
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt