FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

RUN pip install pytorch-pretrained-bert

RUN apt-get update && apt-get install screen

WORKDIR /workspace

# Switch to sciuto user
ARG UNAME=sciuto
ARG UID=186819
ARG GID=11169
ARG NB_GROUP=MLO-unit
RUN groupadd $NB_GROUP -g $GID
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME
USER $UNAME

ENTRYPOINT ["/bin/bash"]