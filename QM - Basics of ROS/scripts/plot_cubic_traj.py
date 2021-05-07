#!/usr/bin/env python

import rospy

from AR_week4_test.msg import *
from std_msgs.msg import Float64

# Create 3 new ROS topics for position, velocity, acceleration:
pos_pub = rospy.Publisher('pos_traj', Float64, queue_size=0)
vel_pub = rospy.Publisher('vel_traj', Float64, queue_size=0)
acc_pub = rospy.Publisher('acc_traj', Float64, queue_size=0)

def callback(data):

	# Set time component: - each time when a new set of coeffs come start over.
	t = 0.0
	while t <= data.tf:
		# Trajectory calculations: - Follow from W4-S15
		pos_msg = data.a0 * 1 + data.a1 * t + data.a2 * t**2 + data.a3 * t**3
		vel_msg = data.a1 * 1 + 2 * data.a2 * t + 3 * data.a3 * t**2
		acc_msg = 2 * data.a2 + 6 * data.a3 * t

		# Publish them:
		pos_pub.publish(pos_msg)
		vel_pub.publish(vel_msg)
		acc_pub.publish(acc_msg)

		# Update time:
		t = t + 0.00025

def plot_trajectory():
    # Initialize node:
	rospy.init_node('plot_cubic_traj', anonymous=True)

	# Subscribe to traj_coeffs from cubic_traj_planner
	rospy.Subscriber('traj_coeffs', cubic_traj_coeffs, callback)
	# Keep going:
	rospy.spin()


if __name__ == '__main__':
	try:
		plot_trajectory()
	except rospy.ROSInterruptException:
		pass
