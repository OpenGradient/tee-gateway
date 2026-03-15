# Build nitriding from source
FROM golang:1.24 as builder

WORKDIR /

# Clone the repository and build the stand-alone nitriding executable.
ARG NITRIDING_COMMIT=2b7dfefaee56819681b7f5a4ee8d66a417ad457d
RUN git clone https://github.com/brave/nitriding-daemon.git && \
    cd nitriding-daemon && git checkout ${NITRIDING_COMMIT}
ARG TARGETARCH
RUN ARCH=${TARGETARCH} make -C nitriding-daemon/ nitriding

# Copy startup script into builder for permission setting
COPY scripts/start.sh /bin/
RUN chown root:root /bin/start.sh
RUN chmod 0755 /bin/start.sh

# ---------- Final image ----------
FROM python:3.12.10-slim-bullseye

# API keys are NOT set here — they are injected at runtime via POST /v1/keys
# after the enclave starts, keeping PCR measurements stable across deployments.

# Install necessary tools
RUN echo 'Dir::Log "/dev/null";' > /etc/apt/apt.conf.d/00no-log \
    && echo 'Dir::Log::Terminal "";' >> /etc/apt/apt.conf.d/00no-log \
    && echo 'Dir::Log::History "";' >> /etc/apt/apt.conf.d/00no-log \
    && ln -sf /dev/null /var/log/dpkg.log \
    && ln -sf /dev/null /var/log/alternatives.log \
    && apt-get update -qq && apt-get install -y --no-install-recommends \
    wget \
    tar \
    build-essential \
    python3-dev \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /var/cache/ldconfig/aux-cache \
    && find /usr/lib/python3.9 -name "*.pyc" -delete \
    && find /usr/lib/python3.9 -name "__pycache__" -type d -delete

# Copy nitriding and startup script from builder
COPY --from=builder /nitriding-daemon/nitriding /bin/nitriding
COPY --from=builder /bin/start.sh /bin/start.sh

# Install uv for deterministic dependency installation from lockfile
COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /usr/local/bin/uv

# Install Python dependencies from lockfile (exact versions + hashes)
COPY pyproject.toml uv.lock /app/
ENV PYTHONDONTWRITEBYTECODE=1 UV_SYSTEM_PYTHON=1
RUN cd /app && uv sync --frozen --no-dev --no-install-project

# Copy the tee_gateway package
COPY tee_gateway /app/tee_gateway

# Set working directory to /app so `python -m tee_gateway` resolves correctly
WORKDIR /app

# Expose ports:
#   443  - nitriding (external TLS, proxied from EC2 host)
#   8080 - nitriding internal API (/enclave/ready, /enclave/hash)
#   8000 - Flask/connexion app (internal, proxied by nitriding to 443)
EXPOSE 443
EXPOSE 8080
EXPOSE 8000

CMD ["/bin/start.sh"]
