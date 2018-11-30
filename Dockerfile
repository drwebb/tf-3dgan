FROM tensorflow/tensorflow:latest-gpu-py3

LABEL maintainer="Tristan Webb <tristan.webb@intel.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-tk \
        python3-tk \
        git


RUN pip3 --no-cache-dir install \
        scikit-image \
        stl \
        future \
        tqdm \
        visdom

RUN git clone https://github.com/scikit-image/scikit-image.git

WORKDIR /notebooks/scikit-image

RUN pip3 install -e .

WORKDIR /notebooks

COPY entry.sh entry.sh

COPY src src

CMD ["./entry.sh"]
