FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY environment.yml pyproject.toml ./
RUN pip install --no-cache-dir h5py scipy einops pyyaml tqdm pytest tensorboard
COPY . .
RUN pip install -e .
