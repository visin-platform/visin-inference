# The model server for the Visin label editor. Run it on the machine with the GPU:
#
#   docker build -t visin-infer .
#   docker run --rm --gpus all -p 127.0.0.1:8000:8000 -v visin-infer-weights:/data \
#     visin-infer --allow-origin https://visin.example.com
#
# Publish the port on 127.0.0.1 only (as above): inside the container the server
# listens on all interfaces, and the host binding is what keeps other machines out.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY visin_infer ./visin_infer
RUN pip install --no-cache-dir '.[models]'

# Model weights are downloaded on first start into the working directory; a
# volume here keeps them across runs.
WORKDIR /data
EXPOSE 8000
ENTRYPOINT ["visin-infer", "serve", "--host", "0.0.0.0", "--port", "8000"]
