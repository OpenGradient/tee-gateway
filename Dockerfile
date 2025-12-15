# Build nitriding from source
FROM golang:latest as builder

WORKDIR /

# Clone the repository and build the stand-alone nitriding executable.
RUN git clone https://github.com/brave/nitriding-daemon.git
ARG TARGETARCH
RUN ARCH=${TARGETARCH} make -C nitriding-daemon/ nitriding

# Use the intermediate builder image to add our files.
# This is necessary to avoid intermediate layers that contain inconsistent file permissions.
COPY server.py start.sh /bin/
RUN chown root:root /bin/server.py /bin/start.sh
RUN chmod 0755 /bin/server.py /bin/start.sh

FROM python:3.12-slim-bullseye

# Environment keys for LLMs
ENV OPENAI_API_KEY=
ENV GOOGLE_API_KEY=

# Install necessary tools
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Copy nitriding and application files from builder
COPY --from=builder /nitriding-daemon/nitriding /bin/nitriding
COPY --from=builder /bin/server.py /bin/server.py
COPY --from=builder /bin/start.sh /bin/start.sh

# Copy requirements file into final image
COPY requirements.txt /app/requirements.txt

# Install the required Python packages.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set working directory
WORKDIR /bin

# Expose ports
EXPOSE 443
EXPOSE 8000

CMD ["start.sh"]
