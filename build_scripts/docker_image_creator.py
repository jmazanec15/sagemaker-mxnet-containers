""" Script to create Sagemaker MXNet Docker images

    Usage:
        python docker_image_creator.py binary_path gpu|cpu mxnet_version python_version
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

def create_docker_image(binary_path, processor, framework_version, python_version, final_image_repository, final_image_tags):
    """ Function builds a docker image

    Args:
        processor (str): gpu or cpu
        framework_version (str): mxnet version i.e 1.1.0
        python_version (str): (i.e. 3.6.5 or 2.7.4)
        binary_path (str): path to where the binary is.
        final_image_repository (str): name of final repo. If None, 'preprod-mxnet' will be used
        final_image_tags (list(str)): list of tag names for final image. If set to empty, default tag will be used
    """
    # Initialize commonly used variables
    py_v = 'py{}'.format(python_version.split('.')[0]) # i.e. py2
    base_docker_path = '{}/../docker/{}/base'.format(PATH_TO_SCRIPT, framework_version, processor)
    if framework_version == '1.0.0' or framework_version == '0.12.1':
        final_docker_path = '{}/../docker/{}/final/{}'.format(PATH_TO_SCRIPT, framework_version, py_v)
    else:
        final_docker_path = '{}/../docker/{}/final'.format(PATH_TO_SCRIPT, framework_version)

    # Get binary file
    print('Getting binary...')
    binary_filename = binary_path.split('/')[-1]
    if os.path.isfile(binary_path):
        shutil.copyfile(binary_path, '{}/{}'.format(final_docker_path, binary_filename))
    else:
        with open('{}/{}'.format(final_docker_path, binary_filename), 'wb') as binary_file:
            subprocess.call(['curl', binary_path], stdout=binary_file)

    # Build base image
    if framework_version == '1.0.0' or framework_version == '0.12.1':
        print('Building base image...')
        image_name = 'mxnet-base:{}-{}-{}'.format(framework_version, processor,  py_v)
        subprocess.call([DOCKER, 'build', '-t', image_name, '-f', 'Dockerfile.{}'.format(processor), '.'], cwd=base_docker_path)

    #  Build final image
    print('Building final image...')
    subprocess.call(['python', 'setup.py', 'sdist'], cwd='{}/..'.format(PATH_TO_SCRIPT))
    output_file = glob.glob('{}/../dist/sagemaker_mxnet_container-*.tar.gz'.format(PATH_TO_SCRIPT))[0]
    output_filename = output_file.split('/')[-1]
    shutil.copyfile(output_file, '{}/{}'.format(final_docker_path, output_filename))

    final_image_repository = final_image_repository if final_image_repository else 'preprod-mxnet'
    final_image_tags = final_image_tags if final_image_tags else ['{}-{}-{}'.format(framework_version, processor, py_v)]
    command_list = [DOCKER, 'build']
    for tag in final_image_tags:
        command_list.append('-t')
        command_list.append('{}:{}'.format(final_image_repository, tag))

    command_list.extend(['--build-arg', 'py_version={}'.format(py_v[-1]),
                    '--build-arg', 'framework_installable={}'.format(binary_filename),
                    '-f', 'Dockerfile.{}'.format(processor), '.'])
    subprocess.call(command_list, cwd=final_docker_path)

if __name__ == '__main__':
    # Parse command line options
    parser = argparse.ArgumentParser(description='Build Sagemaker MXNet Docker Images')
    parser.add_argument('binary_path', help='path to the binary')
    parser.add_argument('processor_type', choices=['cpu', 'gpu'], help='gpu if you would like to use GPUs or cpu')
    parser.add_argument('framework_version', help='MXNet framework version (i.e. 1.8.0)')
    parser.add_argument('python_version', help='Python version to be used (i.e. 2.7.0)')
    parser.add_argument('--nvidia-docker', action='store_true', help='Enables nvidia-docker usage over docker usage')
    parser.add_argument('--final-image-repository', default=None, help='Name of final repo the image is stored in')
    parser.add_argument('--final-image-tags', default=[], nargs='+', help='List of tag names for final image')
    args = parser.parse_args()

    # Set value for docker
    DOCKER = 'nvidia-docker' if args.nvidia_docker else 'docker'

    # Sets PATH_TO_SCRIPT so that command can be run from anywhere
    PATH_TO_SCRIPT = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Build image
    create_docker_image(args.binary_path, args.processor_type, args.framework_version, args.python_version,
                        args.final_image_repository, args.final_image_tags)
