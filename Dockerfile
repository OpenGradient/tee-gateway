# Build nitriding from source
FROM golang:latest as builder

WORKDIR /

# Clone the repository and build the stand-alone nitriding executable.
RUN git clone https://github.com/brave/nitriding-daemon.git
ARG TARGETARCH
RUN ARCH=${TARGETARCH} make -C nitriding-daemon/ nitriding

# Copy startup script into builder for permission setting
COPY scripts/start.sh /bin/
RUN chown root:root /bin/start.sh
RUN chmod 0755 /bin/start.sh

# ---------- Final image ----------
FROM python:3.12-slim-bullseye

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

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1
RUN pip install --no-cache-dir --only-binary=:all: --no-compile -r /app/requirements.txt

# Copy the openapi_server package
COPY openapi_server /app/openapi_server

# Set working directory to /app so `python -m openapi_server` resolves correctly
WORKDIR /app

# Expose ports:
#   443  - nitriding (external TLS, proxied from EC2 host)
#   8080 - nitriding internal API (/enclave/ready, /enclave/hash)
#   8000 - Flask/connexion app (internal, proxied by nitriding to 443)
EXPOSE 443
EXPOSE 8080
EXPOSE 8000

CMD ["/bin/start.sh"]
