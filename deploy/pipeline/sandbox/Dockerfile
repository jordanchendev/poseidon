# Sandbox image for custom strategy blocks (poseidon.pipeline.sandbox).
#
# Python + numpy and the three sandbox modules; no pip, no project code, no secrets.
# The runtime restrictions (no network, read-only rootfs, uid 65534, caps dropped,
# memory/cpu/pids limits) are applied by strategy-sandboxd at `docker run` time
# (poseidon.pipeline.sandbox.broker.docker_run_argv); USER here is a second layer.
#
# Build on the docker host from the repo root (Dockerfile.dockerignore trims the context):
#   docker build -f deploy/pipeline/sandbox/Dockerfile \
#     -t poseidon-pipeline-sandbox:$(git rev-parse --short HEAD) .
FROM python:3.13-slim@sha256:57b29a7c050c4e679e4bea572e14b3965cf4b5839de2da49d55e49ffc4024f39

COPY deploy/pipeline/sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && pip uninstall -y pip
# no setuid/setgid binaries (no-new-privileges already blocks them; belt and braces)
RUN find / -xdev -perm /6000 -type f -exec chmod a-s {} +

ARG SITE=/usr/local/lib/python3.13/site-packages
COPY src/poseidon/__init__.py ${SITE}/poseidon/__init__.py
COPY src/poseidon/pipeline/__init__.py ${SITE}/poseidon/pipeline/__init__.py
COPY src/poseidon/pipeline/sandbox/__init__.py src/poseidon/pipeline/sandbox/protocol.py \
     src/poseidon/pipeline/sandbox/static_check.py src/poseidon/pipeline/sandbox/runner.py \
     ${SITE}/poseidon/pipeline/sandbox/
RUN python -m compileall -q ${SITE}/poseidon && chmod -R a-w ${SITE}/poseidon

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/tmp

USER 65534:65534
WORKDIR /tmp
# -P: no cwd on sys.path; -s: no user site; -B: no .pyc writes. (-I would drop PYTHONHASHSEED.)
ENTRYPOINT ["python", "-P", "-s", "-B", "-m", "poseidon.pipeline.sandbox.runner"]
