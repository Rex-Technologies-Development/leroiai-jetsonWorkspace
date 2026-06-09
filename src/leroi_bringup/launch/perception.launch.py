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

                "camera_offset_x_in": -0.25,  # camera mounted 0.25in left of robot center
                "camera_offset_y_in": 5.91,    # camera 5.91in forward of robot center
                "camera_offset_z_in": 13.75,   # lens 13 3/4in off the ground
                # Physical tilt is 19deg, but the piecewise depth calibration in
                # yolo_detector already maps depth -> true horizontal distance, so
                # set 0 here to avoid double-correcting the forward distance.
                "camera_tilt_deg": 0.0,

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

    route_planner = Node(
        package="leroi_perception",
        executable="route_planner_node",
        name="route_planner_node",
        output="screen",
        parameters=[
            {
                "alliance_color": "blue",
                # Plan from the pose perception ships in the detection packet
                # (assumed origin when no odom) so routes appear without the V5
                # brain streaming odometry yet.
                "require_fresh_odom": False,
                # If a block passes the filters, always route to it -- no value
                # cutoff to bob across (kills the go/no-go flicker).
                "min_route_value": 0.0,
            }
        ],
    )

    return LaunchDescription([
        oak_front,
        route_planner,
    ])
