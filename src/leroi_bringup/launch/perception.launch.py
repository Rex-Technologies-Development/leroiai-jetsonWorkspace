# from launch import LaunchDescription
# from launch_ros.actions import Node


# MODEL_PATH = "/home/leroi-ultio/leroiai-jetsonWorkspace/models/yolo/best.pt"


# def generate_launch_description():
#     oak_front = Node(
#         package="leroi_perception",
#         executable="perception_node",
#         name="oak_front_perception_node",
#         output="screen",
#         parameters=[
#             {
#                 "camera_name": "oak_front",
#                 "camera_type": "oak",
#                 "camera_index": 0,
#                 "camera_width": 640,
#                 "camera_height": 480,
#                 "camera_fps": 30,
#                 "oak_mxid": "",

#                 # Keep false until the OAK stream works and your model exists.
#                 "enable_yolo": False,
#                 "yolo_model_path": MODEL_PATH,
#                 "yolo_confidence": 0.35,
#                 "yolo_image_size": 640,
#                 "yolo_device": "0",
#                 "yolo_use_half": False,

#                 "publish_rate_hz": 10.0,
#                 "publish_every_frame": True,
#             }
#         ],
#     )

#     return LaunchDescription([
#         oak_front,
#     ])


from launch import LaunchDescription
from launch_ros.actions import Node
import os

YOLO_MODEL_PATH = os.path.expanduser(
    "~/leroiai-jetsonWorkspace/models/yolo/best_fp16.engine"
)

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
                "oak_with_depth": True,

                "enable_yolo": True,
                "yolo_model_path": YOLO_MODEL_PATH,
                "yolo_confidence": 0.35,
                "yolo_image_size": 320,
                "yolo_device": "0",
                "yolo_use_half": False,
                "yolo_rate_hz": 15.0,
                "yolo_class_names": '{"0": "blue_block", "1": "blue_block_in_goal", "2": "red_block", "3": "red_block_in_goal"}',

                "enable_goal_view": False,
                "goal_view_port": 8082,
                "enable_preview_stream": True,
                "preview_port": 8080,
                "preview_jpeg_quality": 60,
                "draw_detections": True,
                "preview_rate_hz": 30.0,
                "preview_stream_fps": 15.0,
                "status_rate_hz": 2.0,

                "enable_map_stream": True,
                "map_stream_port": 8081,
            }
        ],
    )

    return LaunchDescription([
        oak_front,
    ])
