from launch import LaunchDescription
from launch_ros.actions import Node


MODEL_PATH = "/home/leroi-ultio/leroiai-jetsonWorkspace/models/yolo/best.pt"


def generate_launch_description():
    oak_front = Node(
        package="leroi_perception",
        executable="perception_node",
        name="oak_front_perception_node",
        output="screen",
        parameters=[
            {
                "camera_name": "oak_front",
                "camera_type": "oak",
                "camera_index": 0,
                "camera_width": 640,
                "camera_height": 480,
                "camera_fps": 30,
                "oak_mxid": "",

                # Keep false until the OAK stream works and your model exists.
                "enable_yolo": False,
                "yolo_model_path": MODEL_PATH,
                "yolo_confidence": 0.35,
                "yolo_image_size": 640,
                "yolo_device": "0",
                "yolo_use_half": False,

                "publish_rate_hz": 10.0,
                "publish_every_frame": True,
            }
        ],
    )

    return LaunchDescription([
        oak_front,
    ])