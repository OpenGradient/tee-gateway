# Build nitriding from source
FROM golang:latest as builder

WORKDIR /

# Clone the repository and build the stand-alone nitriding executable.
RUN git clone https://github.com/brave/nitriding-daemon.git
ARG TARGETARCH
RUN ARCH=${TARGETARCH} make -C nitriding-daemon/ nitriding

# Copy application files into builder for permission setting
COPY start.sh /bin/
COPY server.py /bin/
RUN chown root:root /bin/start.sh /bin/server.py
RUN chmod 0755 /bin/start.sh /bin/server.py

# ---------- Final image ----------
FROM python:3.12-slim-bullseye

# Environment keys for LLMs
ENV OPENAI_API_KEY=
ENV GOOGLE_API_KEY=
ENV ANTHROPIC_API_KEY=
ENV XAI_API_KEY=

# Install necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    tar \
    build-essential \
    python3-dev \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy nitriding and scripts from builder
COPY --from=builder /nitriding-daemon/nitriding /bin/nitriding
COPY --from=builder /bin/start.sh /bin/start.sh
COPY --from=builder /bin/server.py /bin/server.py

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the openapi_server package
COPY openapi_server /app/openapi_server

# Set working directory to /app so `python -m openapi_server` resolves correctly
WORKDIR /app

# Expose ports:
#   443  - nitriding (external TLS)
#   8080 - Flask/connexion app (internal, proxied by nitriding)
#   8000 - server.py LLM backend (internal only, temporary)
EXPOSE 443
EXPOSE 8080
EXPOSE 8000

CMD ["/bin/start.sh"]
