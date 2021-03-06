============================================
SageMaker MXNet Containers Build Script
============================================

Users can use docker_image_creator.py to build the Docker images with .whl files from either their local machine or the internet. To run this program, execute the command:

::

    python docker_image_creator.py  <mxnet_version> <python_version> <gpu|cpu> <binary_path>

::

binary_path can be either a URL or a path on your machine to the .whl framework binary.
To use nvidia-docker instead of docker to build the image, run the command:

::

    python docker_image_creator.py  <mxnet_version> <python_version> <gpu|cpu> <binary_path> --nvidia-docker

::

The default Docker repository the final image will be placed in is 'preprod-mxnet'. To set the repository to a custom value, run the command:

::

    python docker_image_creator.py  <mxnet_version> <python_version> <gpu|cpu> <binary_path> --final-image-repository <name>

::

The default tag the final image will have is '<framework_version>-<processor_type>-<py_version>' (i.e. 1.1.0-gpu-py2).
To customize the tag(s) set, run the command:

::

    python docker_image_creator.py  <mxnet_version> <python_version> <gpu|cpu> <binary_path> --final-image-tags <tag1> <tag2> ...

::

Here is an example for MXNet 1.1.0 GPU with a binary from PyPi:

::

    python docker_image_creator.py 1.1.0 2.7.0 gpu https://files.pythonhosted.org/packages/a0/dc/3198a0277d2321474ebd0b06dba6baf44484b00c253f4ead98763a21f634/mxnet_cu90-1.1.0.post0-py2.py3-none-manylinux1_x86_64.whl
     --nvidia-docker --final-image-repository custom_repo_name --final-image-tags custom_tag1 custom_tag2

::

Notes
~~~~~

- `Only versions that have Dockerfiles in this repository can be built`
- `Python version must have 3 sections (i.e. 2.7.0 or 3.6.0)`
- `Framework version must have 3 sections (i.e. 1.1.0)`
- `Build script only builds docker images from Dockerfiles in the main branch of the repos mentioned above`
