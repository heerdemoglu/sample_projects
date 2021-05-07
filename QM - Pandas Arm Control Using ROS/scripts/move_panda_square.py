#!/usr/bin/env python

################################################################################
# Code developed by Hasan Emre Erdemoglu, 200377106
# Code developed as a part of
# Advanced Robotic Systems Course (QMUL-ECS7004P)
# Inspired from the documentation at:
# http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/
# move_group_python_interface/move_group_python_interface_tutorial.html
################################################################################

import rospy
import math
import time
import copy
import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Float64
import std_msgs

def executeActionSrv(data):
    try:
        print('\n')
        print('--------------------------------------------------------------')
        print('Move Panda - Initialize robot to starting condition.')
        print('Move Panda - Received square size: s = %s' % data.data)
        print('--------------------------------------------------------------')

        # Part 2a: Move Robot to a Starting Configuration:
        # Initialize starting configuration: (In Joint Space!)
        # This is done to avoid kinematic singularity (rank deficiency), which
        # is the case with initial config of Panda arm.
        start_config = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]

        # Move to the config, then set in place by stop() command:
        group.go(start_config, wait = True)
        group.stop() # set in place after movement is complete.

        # Part 2b: Plan a Cartesian Path:
        print('--------------------------------------------------------------')
        print('Move Panda - Planning motion trajectory.')
        print('--------------------------------------------------------------')
        waypoints = [] # Construct a list holding all waypoints:

        # The goal is to draw a square with given size in xy Cartesian Plane:
        # Comments below assume bird-eye view on the xy plane:
        # Deep copy is used to copy value, not reference, normally a C++ impl.
        # detail, however followed within the tutorials.
        # Start from current position:
        robot_pos = group.get_current_pose().pose

        # From top left corner, do down: (Bottom Left)
        robot_pos.position.y -= data.data
        waypoints.append(copy.deepcopy(robot_pos))

        # From bottom left corner, go left: (Bottom Right)
        robot_pos.position.x += data.data
        waypoints.append(copy.deepcopy(robot_pos))

        # From bottom right corner, go up: (Top Right)
        robot_pos.position.y += data.data
        waypoints.append(copy.deepcopy(robot_pos))

        # From top right corner, go left, complete square: (Top Left)
        robot_pos.position.x -= data.data
        waypoints.append(copy.deepcopy(robot_pos))

        # Part 2c: Show Planned Trajectory on RViz:
        # Used in the tutorials - Cartesian path interpolated @ 1cm  w/ eef_step
        # Jump threshold is disabled.
        print('--------------------------------------------------------------')
        print('Move Panda - Showing motion trajectory.')
        print('--------------------------------------------------------------')
        (plan, fraction) = group.compute_cartesian_path(waypoints,
                                        eef_step = 0.01, jump_threshold = 0.0)

        # Construct moveit message for displaying the trajectory:
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()

        # Display from robot's initial state
        display_trajectory.trajectory_start = robot.get_current_state()

        # Append the plan to the trajectory, 1 cm interpol. steps, no jumps
        display_trajectory.trajectory.append(plan)

        # Publish the trajectory
        disp_traj_publ.publish(display_trajectory);

        # For better visualization add 5 seconds delay between plan and exec.
        time.sleep(5)

        # Part 2d-e: Execute Planned Trajectory and wait for next message:
        print('--------------------------------------------------------------')
        print('Move Panda - Executing planned trajectory.')
        print('Move Panda - Waiting for next input.')
        print('--------------------------------------------------------------')
        group.execute(plan, wait=True)

    except rospy.ServiceException, e:
        print("Service call failed: %s" % e)

def move_panda_square():
    # Initialize the node with its name:
    rospy.init_node('move_panda_square', anonymous=True)

    print('--------------------------------------------------------------')
    print('Move Panda - Waiting for desired size of square trajectory.')
    print('--------------------------------------------------------------')

    # Subscribe to sq_size from square_size_generator:
    # Again message is a single Float64, so directly embedded
    # here as is. (Not the best practice)
    rospy.Subscriber('sq_size', Float64, executeActionSrv)

    # Run indefinitely:
    rospy.spin()

if __name__ == "__main__":

    # Print statement in a similar way to the given screencast.
    print('--------------------------------------------------------------')
    print('Move Panda - Initializing.')
    print('--------------------------------------------------------------')
    # Initialize MoveIt Related Functions: (Using MoveIt tutorials)
    # Instantiate RobotCommander (Outer level interface):
    robot = moveit_commander.RobotCommander()

    # Instantiate move group commander (interfaces robot joints)
    # Panda Arm is used in this assignment
    group = moveit_commander.MoveGroupCommander('panda_arm')

    # This will be used to publish trajectories to RViz (Visualization)
    disp_traj_publ = rospy.Publisher('/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory,
                                    queue_size = 20)

    # Run the script:
    move_panda_square()
