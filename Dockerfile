# A Go base image is enough to build nitriding reproducibly.
# We use a specific instead of the latest image to ensure reproducibility.
FROM golang:1.23 as builder

WORKDIR /

# Clone the repository and build the stand-alone nitriding executable.
RUN git clone https://github.com/brave/nitriding-daemon.git
ARG TARGETARCH
RUN ARCH=${TARGETARCH} make -C nitriding-daemon/ nitriding

# Use the intermediate builder image to add our files.  This is necessary to
# avoid intermediate layers that contain inconsistent file permissions.
COPY server.py start.sh utils.py /bin/
COPY storage/storage.py storage/__init__.py /bin/storage/
RUN chown root:root /bin/server.py /bin/start.sh
RUN chmod 0755      /bin/server.py /bin/start.sh

FROM python:3.12-slim-bullseye

# Set environment variables for IPFS
ENV IPFS_VERSION=0.19.1
ENV IPFS_PATH=/root/.ipfs
ENV LIBP2P_FORCE_PNET=1

# Install necessary tools for IPFS and clean up
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*


# Copy all our files to the final image.
COPY --from=builder /nitriding-daemon/nitriding /bin/start.sh /bin/server.py /bin/utils.py /bin/
COPY --from=builder /bin/storage/__init__.py /bin/storage/storage.py /bin/storage/

# Copy requirements file into final image
COPY requirements.txt /app/requirements.txt

# Install the required Python packages.
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set working directory
WORKDIR /bin

# Expose port 8000 for flask server
EXPOSE 443
EXPOSE 8000

CMD ["start.sh"]
