FROM tensorflow/tensorflow:latest-gpu

LABEL maintainer="Tristan Webb <tristan.webb@intel.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-tk


RUN pip --no-cache-dir install \
        scikit-image \
        stl \
        tqdm \
        visdom

COPY entry.sh entry.sh

COPY src src

CMD ["./entry.sh"]
