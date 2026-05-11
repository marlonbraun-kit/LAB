"""Launch the UR3 pick-and-place pipeline driven by MoveIt2.

Three modes (selected via launch arguments):
  default                — execute on the real UR3 robot, no RViz.
  rviz:=true             — home simulation: mock_components hardware + RViz.
  rviz:=true debug:=true — real robot + RViz with the Trajectory display
                           and the manager step-gate (Enter between phases).

Stack started:
  robot_state_publisher        URDF -> TF
  ros2_control_node            controller_manager
  joint_state_broadcaster      publishes /joint_states
  joint_trajectory_controller  FollowJointTrajectory action server (arm)
  gripper_controller           Float64MultiArray command (gripper fingers)
  move_group                   MoveIt2 planning + execution
  planning_scene_manager_node  publishes detected cans as CollisionObjects
  pick_place_manager_node      state machine (calls MoveGroup action)
  rviz_visualizer_node         markers + place-slot visualisation
  depth_camera_node            simulated camera source (only when fake_camera:=true)
  rviz2                        only when rviz:=true

External topic contract (the teammates' nodes publish these on the real robot):
  /target_can_pose        ur3_interfaces/CanDetectionArray
                          - source="top"   -> from above-camera (positions only)
                          - source="front" -> from front-camera (positions + class_name)
  /human_proximity        std_msgs/Float32   in [0, 1] (0=danger, 1=safe)

Operator inputs (same in every mode):
  /pick_command           std_msgs/String   e.g.  "coke,mahou"
  /clear_place_zone       std_msgs/Empty    resets the 2x2 place slot map
"""
import os
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _load_yaml(path):
    try:
        with open(path, 'r') as fh:
            return yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        print(f'[pick_place_moveit.launch] failed to load {path}: {e}')
        return {}


def generate_launch_description():
    pkg = 'ur3_project'
    pkg_share = get_package_share_directory(pkg)

    rviz_arg = LaunchConfiguration('rviz')
    debug_arg = LaunchConfiguration('debug')
    fake_camera_arg = LaunchConfiguration('fake_camera')

    declared_args = [
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='If true, start RViz alongside the pipeline.'),
        DeclareLaunchArgument(
            'debug', default_value='false',
            description=('If true (and rviz:=true), execute on the real robot, '
                         'show the trajectory ghost in RViz, and gate every '
                         'motion behind Enter presses on stdin.')),
        DeclareLaunchArgument(
            'fake_camera', default_value='false',
            description=("'true' to start depth_camera_node as a simulated "
                         "/target_can_pose source; 'false' to never start it; "
                         "'auto' (default) starts it only in home-sim mode "
                         "(rviz:=true and debug:=false).")),
        DeclareLaunchArgument(
            'robot_ip', default_value='169.254.12.28',
            description='IP address of the UR3 controller (ignored in fake-hardware mode).'),
        DeclareLaunchArgument(
            'kinematics_params',
            default_value='/home/marlon/my_robot_calibration.yaml',
            description='Path to the robot-specific kinematic calibration YAML.'),
    ]

    use_fake_hardware = PythonExpression([
        "'true' if '", rviz_arg, "'.lower() == 'true' and '",
        debug_arg, "'.lower() != 'true' else 'false'"
    ])

    # On real hardware, hold the FSM in WAIT_FOR_ROBOT until the URCap
    # External Control program is running and the reverse interface is up.
    # In fake-hardware mode, that topic never publishes, so disable the gate.
    wait_for_robot_program = PythonExpression([
        "'true' if '", use_fake_hardware, "'.lower() != 'true' else 'false'"
    ])

    # fake_camera resolution: 'auto' -> equal to use_fake_hardware logic.
    fake_camera_enabled = PythonExpression([
        "'true' if ('", fake_camera_arg, "'.lower() == 'true' or "
        "('", fake_camera_arg, "'.lower() == 'auto' and "
        "'", rviz_arg, "'.lower() == 'true' and "
        "'", debug_arg, "'.lower() != 'true')) else 'false'"
    ])

    rviz_config_file = PythonExpression([
        "'", os.path.join(pkg_share, 'rviz', 'moveit_pick_place_debug.rviz'),
        "' if '", debug_arg, "'.lower() == 'true' else '",
        os.path.join(pkg_share, 'rviz', 'moveit_pick_place_sim.rviz'), "'",
    ])

    urdf_xacro = PathJoinSubstitution([FindPackageShare(pkg), 'urdf', 'ur3_camera_gripper.urdf.xacro'])
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ', urdf_xacro,
        ' sim_gazebo:=false',
        ' sim_ignition:=false',
        ' use_fake_hardware:=', use_fake_hardware,
        ' robot_ip:=', LaunchConfiguration('robot_ip'),
        ' kinematics_params:=', LaunchConfiguration('kinematics_params'),
        ' script_filename:=/opt/ros/humble/share/ur_robot_driver/resources/ros_control.urscript',
        ' output_recipe_filename:=/opt/ros/humble/share/ur_robot_driver/resources/rtde_output_recipe.txt',
        ' input_recipe_filename:=/opt/ros/humble/share/ur_robot_driver/resources/rtde_input_recipe.txt',
    ])
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    srdf_xacro = PathJoinSubstitution([FindPackageShare(pkg), 'urdf', 'ur3.srdf.xacro'])
    robot_description_semantic_content = Command([FindExecutable(name='xacro'), ' ', srdf_xacro])
    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(robot_description_semantic_content, value_type=str)
    }

    kinematics_yaml = _load_yaml(os.path.join(pkg_share, 'config', 'moveit_kinematics.yaml'))
    robot_description_kinematics = {'robot_description_kinematics': kinematics_yaml}

    ompl_yaml = _load_yaml(os.path.join(pkg_share, 'config', 'moveit_ompl_planning.yaml'))
    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_yaml,
    }

    joint_limits_yaml = _load_yaml(os.path.join(pkg_share, 'config', 'moveit_joint_limits.yaml'))
    joint_limits = {'robot_description_planning': joint_limits_yaml}

    moveit_controllers = _load_yaml(os.path.join(pkg_share, 'config', 'moveit_controllers.yaml'))

    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 10.0,
        'trajectory_execution.allowed_goal_duration_margin': 10.0,
        'trajectory_execution.allowed_start_tolerance': 0.05,
    }
    planning_scene_monitor = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    ros2_controllers_path = os.path.join(pkg_share, 'config', 'ur3_controllers.yaml')

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, ros2_controllers_path],
        output='log',
    )

    def spawner(name):
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name, '--controller-manager', '/controller_manager'],
            output='log',
        )

    spawn_jsb = spawner('joint_state_broadcaster')
    spawn_jtc = spawner('scaled_joint_trajectory_controller')
    # Zimmer HRC-03 is driven via /io_and_status_controller/set_io.
    # The gripper_controller (position controller for the finger joints)
    # is kept solely so RViz can visualise open/close — pick_place_manager
    # publishes a Float64MultiArray with the same value to drive the
    # mock_components-backed gripper joints.
    spawn_io = spawner('io_and_status_controller')
    spawn_gripper = spawner('gripper_controller')

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            planning_pipelines,
            moveit_controllers,
            trajectory_execution,
            planning_scene_monitor,
            joint_limits,
            {'planning_plugin': 'ompl_interface/OMPLPlanner'},
            {'warehouse_plugin': 'warehouse_ros_sqlite::DatabaseConnection'},
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
        ],
        output='log',
    )

    planning_scene_manager = Node(
        package=pkg, executable='planning_scene_manager_node.py', output='screen',
    )
    pick_place_manager = Node(
        package=pkg, executable='pick_place_manager_node.py', output='screen',
        parameters=[{
            'debug_step': debug_arg,
            'wait_for_robot_program': wait_for_robot_program,
        }],
        emulate_tty=True,
    )
    depth_camera = Node(
        package=pkg, executable='depth_camera_node.py', output='log',
        condition=IfCondition(
            PythonExpression([
                "'true' if ('", rviz_arg, "'.lower() == 'true' and '",
                debug_arg, "'.lower() == 'true') or ('",
                rviz_arg, "'.lower() == 'false' and '",
                debug_arg, "'.lower() == 'false') else 'false'"
            ])
        ),
    )
    rviz_visualizer = Node(
        package=pkg, executable='rviz_visualizer_node.py', output='log',
    )

    jtc_after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_jtc, spawn_gripper, spawn_io])
    )
    move_group_after_jtc = RegisterEventHandler(
        OnProcessExit(target_action=spawn_jtc, on_exit=[
            move_group,
            TimerAction(period=5.0, actions=[
                planning_scene_manager, pick_place_manager,
                depth_camera, rviz_visualizer, rviz,
            ]),
        ])
    )

    return LaunchDescription(declared_args + [
        rsp, controller_manager, spawn_jsb, jtc_after_jsb, move_group_after_jtc,
    ])
