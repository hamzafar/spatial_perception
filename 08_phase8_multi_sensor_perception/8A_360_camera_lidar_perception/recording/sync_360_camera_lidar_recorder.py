import carla
import cv2
import csv
import time
import numpy as np

from pathlib import Path
from queue import Queue

import random


# ==================================================
# USER CONFIGURATION
# ==================================================

TARGET_FPS = 10

RECORD_SECONDS = 60

TARGET_FRAMES = (
    TARGET_FPS *
    RECORD_SECONDS
)

# ==================================================

def carla_image_to_numpy(image):
    """
    Convert a CARLA BGRA image to a BGR NumPy array.
    """

    image_np = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    image_np = image_np.reshape(
        (
            image.height,
            image.width,
            4
        )
    )

    return image_np[:, :, :3]


def save_image(image, directory, frame_id):
    """
    Save a NumPy image as a JPEG file.
    """

    filename = (
        directory /
        f"frame_{frame_id:06d}.jpg"
    )

    cv2.imwrite(
        str(filename),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            90
        ]
    )

def main():

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(
        10.0
    )

    world = client.get_world()

    original_settings = (
        world.get_settings()
    )

    settings = world.get_settings()

    settings.synchronous_mode = True

    settings.fixed_delta_seconds = (
        1.0 / TARGET_FPS
    )

    world.apply_settings(
        settings
    )


    blueprint_library = (
        world.get_blueprint_library()
    )

    # ----------------------------------
    # Vehicle
    # ----------------------------------

    vehicle_bp = (
        blueprint_library.filter(
            "vehicle.tesla.model3"
        )[0]
    )

    vehicle = None

    spawn_points = (
        world.get_map()
        .get_spawn_points()
    )

    random.shuffle(
        spawn_points
    )

    for spawn_point in spawn_points:

        try:

            vehicle = world.spawn_actor(
                vehicle_bp,
                spawn_point
            )

            print(
                f"Spawn Location: "
                f"{spawn_point.location}"
            )

            break

        except RuntimeError:

            continue

    if vehicle is None:

        print(
            "No free spawn point found"
        )

        return

    vehicle.set_autopilot(
        True
    )

    print(
        f"Vehicle spawned: "
        f"{vehicle.id}"
    )

    # ----------------------------------
    # Camera
    # ----------------------------------

   
    # Front Camera
    
    front_bp = blueprint_library.find("sensor.camera.rgb")

    front_bp.set_attribute("sensor_tick", "0.0")
    front_bp.set_attribute("role_name", "rgb_front")

    # Rear Camera
    
    rear_bp = blueprint_library.find("sensor.camera.rgb")

    rear_bp.set_attribute("sensor_tick", "0.0")
    rear_bp.set_attribute("role_name", "rgb_rear")

    # Left Camera
    
    left_bp = blueprint_library.find("sensor.camera.rgb")

    left_bp.set_attribute("sensor_tick", "0.0")
    left_bp.set_attribute("role_name", "rgb_left")

    # Right Camera
    
    right_bp = blueprint_library.find("sensor.camera.rgb")

    right_bp.set_attribute("sensor_tick", "0.0")
    right_bp.set_attribute("role_name", "rgb_right")


    front_transform = carla.Transform(
        carla.Location(x=-1.5, z=2.4),
        carla.Rotation(yaw=0)
    )

    rear_transform = carla.Transform(
        carla.Location(x=-1.5, z=2.4),
        carla.Rotation(yaw=180)
    )

    left_transform = carla.Transform(
        carla.Location(x=-1.5, z=2.4),
        carla.Rotation(yaw=-90)
    )

    right_transform = carla.Transform(
        carla.Location(x=-1.5, z=2.4),
        carla.Rotation(yaw=90)
    )

    front_camera = world.spawn_actor(front_bp, front_transform, attach_to=vehicle)

    rear_camera = world.spawn_actor(rear_bp, rear_transform, attach_to=vehicle)

    left_camera = world.spawn_actor(left_bp, left_transform, attach_to=vehicle)

    right_camera = world.spawn_actor(right_bp, right_transform, attach_to=vehicle)

    print(f"Front Camera spawned: " f"{front_camera.id}")
    print(f"Rear Camera spawned: " f"{rear_camera.id}")
    print(f"Left Camera spawned: " f"{left_camera.id}")
    print(f"Right Camera spawned: " f"{right_camera.id}")

    front_queue = Queue()
    rear_queue  = Queue()
    left_queue  = Queue()
    right_queue = Queue()

    front_camera.listen(
        front_queue.put
    )

    rear_camera.listen(
        rear_queue.put
    )

    left_camera.listen(
        left_queue.put
    )

    right_camera.listen(
        right_queue.put
    )


    # ----------------------------------
    # LiDAR
    # ----------------------------------

    lidar_bp = (
        blueprint_library.find(
            "sensor.lidar.ray_cast"
        )
    )

    lidar_bp.set_attribute(
        "range",
        "50"
    )

    lidar_bp.set_attribute(
        "channels",
        "32"
    )

    lidar_bp.set_attribute(
        "points_per_second",
        "56000" 
    )   #"56000"

    lidar_bp.set_attribute(
        "rotation_frequency",
        "10"
    )

    lidar_bp.set_attribute(
        "upper_fov",
        "10"
    )

    lidar_bp.set_attribute(
        "lower_fov",
        "-30"
    )

    lidar_bp.set_attribute(
        "sensor_tick",
        "0.0"
    )

    lidar_transform = (
        carla.Transform(
            carla.Location(
                x=0.0,
                z=2.4
            )
        )
    )

    lidar_sensor  = world.spawn_actor(
        lidar_bp,
        lidar_transform,
        attach_to=vehicle
    )

    print(
        f"LiDAR spawned: "
        f"{lidar_sensor.id}"
    )

    lidar_queue = Queue()

    lidar_sensor.listen(
        lidar_queue.put
    )

    # ----------------------------------
    # Recording Directory
    # ----------------------------------

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    session_dir = Path(
        f"recordings/session_{timestamp}"
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    front_dir = (
        session_dir / "front"
    )

    rear_dir = (
        session_dir / "rear"
    )

    left_dir = (
        session_dir / "left"
    )

    right_dir = (
        session_dir / "right"
    )

    lidar_dir = (
        session_dir / "lidar"
    )

    front_dir.mkdir(
        exist_ok=True
    )

    rear_dir.mkdir(
        exist_ok=True
    )

    left_dir.mkdir(
        exist_ok=True
    )

    right_dir.mkdir(
        exist_ok=True
    )

    lidar_dir.mkdir(
        exist_ok=True
    )

    csv_file = open(
        session_dir / "timestamps.csv",
        "w",
        newline=""
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow([
        "frame_id",
        "front_frame",
        "front_timestamp",
        "rear_frame",
        "rear_timestamp",
        "left_frame",
        "left_timestamp",
        "right_frame",
        "right_timestamp",
        "lidar_frame",
        "lidar_timestamp"
    ])


    # ----------------------------------
    # First Image
    # ----------------------------------

    # print("First Tick") #debug

    world.tick()

    first_front = (
        front_queue.get()
    )

    first_rear = (
        rear_queue.get()
    )

    first_left = (
        left_queue.get()
    )

    first_right = (
        right_queue.get()
    )

    # print("First Images received ") #debug

    first_lidar = (
        lidar_queue.get()
        )
    # print("First Lidar received") #debug



    print(
        f"LiDAR points: "
        f"{len(first_lidar.raw_data) // 16}"
    )



    print(
        "\n=== Recorder Configuration ==="
    )

    print(
        f"Target FPS      : "
        f"{TARGET_FPS}"
    )

    print(
        f"Record Seconds  : "
        f"{RECORD_SECONDS}"
    )

    print(
        f"Target Frames   : "
        f"{TARGET_FRAMES}"
    )

    print(
        f"Camera Width    : "
        f"{first_front.width}"
    )

    print(
        f"Camera Height   : "
        f"{first_front.height}"
    )

    print(
        f"\nRecording directory:\n"
        f"{session_dir}\n"
    )

    start_time = time.time()

    received_images = 0

    try:
         
        for frame_id in range(
            1,
            TARGET_FRAMES + 1
        ):

            if frame_id % 50 == 0: #debug
                print(vehicle.get_location())

            if frame_id == 1:

                front = first_front
                rear = first_rear
                left = first_left
                right = first_right
                lidar = first_lidar

            else:
                # print("Tick onward") #debug

                world.tick()

                front = front_queue.get()
                rear = rear_queue.get()
                left = left_queue.get()
                right = right_queue.get()
                lidar = lidar_queue.get()
                # print("Lidar received onward") #debug


            received_images += 1

            front_np = carla_image_to_numpy(front)
            rear_np = carla_image_to_numpy(rear)
            left_np = carla_image_to_numpy(left)
            right_np = carla_image_to_numpy(right)

            # if not lidar_queue.empty():
            lidar_points = np.frombuffer(
                lidar.raw_data,
                dtype=np.float32
            )

            lidar_points = lidar_points.reshape(
                (-1, 4)
            )
            lidar_filename = (
                lidar_dir /
                f"frame_{frame_id:06d}.npy"
            )

            # print(
            #     lidar_filename
            # )

            np.save(
                lidar_filename,
                lidar_points
            )



            save_image(
                front_np,
                front_dir,
                frame_id
            )

            save_image(
                rear_np,
                rear_dir,
                frame_id
            )

            save_image(
                left_np,
                left_dir,
                frame_id
            )

            save_image(
                right_np,
                right_dir,
                frame_id
            )

  
            csv_writer.writerow([
                frame_id,

                front.frame,
                front.timestamp,

                rear.frame,
                rear.timestamp,

                left.frame,
                left.timestamp,

                right.frame,
                right.timestamp,

                lidar.frame,
                lidar.timestamp
            ])

            

            if frame_id % 100 == 0:

                print(
                    f"Saved={frame_id}"
                )

    finally:

        duration = (
            time.time() -
            start_time
        )

        csv_file.close()


        front_camera.stop()
        front_camera.destroy()

        rear_camera.stop()
        rear_camera.destroy()

        left_camera.stop()
        left_camera.destroy()

        right_camera.stop()
        right_camera.destroy()

        lidar_sensor.stop()
        lidar_sensor.destroy()


        vehicle.set_autopilot(
            False
        )

        time.sleep(1)

        vehicle.destroy()

        world.apply_settings(
            original_settings
        )

        print(
            "\n===== Recording Summary ====="
        )

        print(
            f"Images Received : "
            f"{received_images}"
        )

        print(
            f"Saved Frames : "
            f"{TARGET_FRAMES}"
        )

        print(
            f"Elapsed Time : "
            f"{duration:.2f} sec"
        )

        print(
            f"Dataset Path : "
            f"{session_dir}"
        )

        print(
            "============================="
        )


if __name__ == "__main__":
    main()