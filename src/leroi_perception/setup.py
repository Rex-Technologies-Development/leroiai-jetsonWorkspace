from setuptools import find_packages, setup

package_name = "leroi_perception"

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
    description="Perception package for OAK camera and YOLO-based object detection.",
    license="Apache-2.0",
    # tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception_node = leroi_perception.perception_node:main",
            "route_planner_node = leroi_perception.route_planner_node:main",
            "route_review_node = leroi_perception.route_review_node:main",
        ],
    },
)