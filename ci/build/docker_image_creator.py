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

def build_docker_image(framework_version, python_version, processor, binary_path, final_image_repository,
                        final_image_tags, docker, main_directory_path):
    """ Function builds TF docker image

    Args:
        framework_version (str): mxnet version (i.e 1.1.0)
        python_version (str): (i.e. 3.6.5 or 2.7.4)
        processor (str): gpu or cpu
        binary_path (str): path to where the binary is.
        final_image_repository (str): name of final repo.
        final_image_tags (list(str)): list of tag names for final image
        docker (str): either nvidia-docker or docker
        main_directory_path (str):  absolute path to where the sagemaker-mxnet-repo
    """
    # Initialize commonly used variables
    py_v = 'py{}'.format(python_version.split('.')[0]) # i.e. py2

    # Path's depends on version for sagemaker-mxnet-repository
    if framework_version == '1.0.0' or framework_version == '0.12.1':
        base_docker_path = os.path.join(main_directory_path, 'docker', framework_version, 'base')
        final_docker_path = os.path.join(main_directory_path, 'docker', framework_version, 'final', py_v)
    else:
        final_docker_path = os.path.join(main_directory_path, 'docker', framework_version, 'final')

    # Get binary file
    print('Getting binary...')
    if os.path.isfile(binary_path):
        binary_filename = os.path.basename(binary_path)
        shutil.copyfile(binary_path, os.path.join(final_docker_path, binary_filename))
    else:
        binary_filename = binary_path.split('/')[-1]
        with open(os.path.join(final_docker_path, binary_filename), 'wb') as binary_file:
            subprocess.call(['curl', binary_path], stdout=binary_file)

    # Build base image
    if framework_version == '1.0.0' or framework_version == '0.12.1':
        print('Building base image...')
        image_name = 'mxnet-base:{}-{}-{}'.format(framework_version, processor,  py_v)
        subprocess.call([docker, 'build', '-t', image_name, '-f', 'Dockerfile.{}'.format(processor), '.'],
                        cwd=base_docker_path)

    #  Build final image
    print('Building final image...')
    subprocess.call(['python', 'setup.py', 'sdist'], cwd=main_directory_path)
    tar_file = glob.glob(os.path.join(main_directory_path, 'dist', 'sagemaker_mxnet_container-*.tar.gz'))[0]
    tar_filename = tar_file.split('/')[-1]
    shutil.copyfile(tar_file, os.path.join(final_docker_path, tar_filename))

    final_command_list = [docker, 'build']
    for tag in final_image_tags:
        final_command_list.append('-t')
        final_command_list.append('{}:{}'.format(final_image_repository, tag))

    final_command_list.extend(['--build-arg', 'py_version={}'.format(py_v[-1]), '--build-arg',
                               'framework_installable={}'.format(binary_filename), '-f',
                               'Dockerfile.{}'.format(processor), '.'])
    subprocess.call(final_command_list, cwd=final_docker_path)

def main():
    # Parse command line options
    parser = argparse.ArgumentParser(description='Build Sagemaker MXNet Docker Images')
    parser.add_argument('framework_version', help='Framework version (i.e. 1.1.0)')
    parser.add_argument('python_version', help='Python version to be used (i.e. 2.7.0)')
    parser.add_argument('processor_type', choices=['cpu', 'gpu'], help='gpu if you would like to use GPUs or cpu')
    parser.add_argument('binary_path', help='Path to the binary.')
    parser.add_argument('--nvidia-docker', action='store_true', help='Enables nvidia-docker usage over docker')
    parser.add_argument('--final-image-repository', default='preprod-mxnet',
                        help='Name of final docker repo the image is stored in')
    parser.add_argument('--final-image-tags', default=[], nargs='+', help='List of tag names for final image')
    args = parser.parse_args()

    # Arguments used in build functions
    docker = 'nvidia-docker' if args.nvidia_docker else 'docker'
    main_directory_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')
    if args.final_image_tags:
        final_image_tags = args.final_image_tags
    else:
        final_image_tags = ['{}-{}-py{}'.format(args.framework_version, args.processor_type, args.python_version[0])]

    # Build the image
    build_docker_image(args.framework_version, args.python_version, args.processor_type, args.binary_path,
                        args.final_image_repository, final_image_tags, docker, main_directory_path)

if __name__ == '__main__':
    main()
