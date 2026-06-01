from setuptools import find_packages, setup

package_name = "leroi_serial_messenger"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Leo Zhieh",
    maintainer_email="notes.leozhieh@gmail.com",
    description="Serial bridge between Jetson ROS 2 and VEX V5 brain.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "serial_messenger_node = leroi_serial_messenger.serial_messenger_node:main",
        ],
    },
)