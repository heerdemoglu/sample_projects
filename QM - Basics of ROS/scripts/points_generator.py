#!/usr/bin/env python

import rospy
import random

from AR_week4_test.msg import cubic_traj_params

# Static variables: Position and Velocity should not exceed in +- range.
P_MAX = 10
V_MAX = 10

# First construct the publisher:
pub = rospy.Publisher('points_msg', cubic_traj_params, queue_size=0)

# Start with an initial message
message = cubic_traj_params()

message.p0 = random.uniform(-P_MAX, P_MAX)
message.pf = random.uniform(-P_MAX, P_MAX)

# Velocity Generation:
message.v0 = random.uniform(-V_MAX, V_MAX)
message.vf = random.uniform(-V_MAX, V_MAX)


# Publish generated points:
def generate_points():

	rospy.init_node('points_generator', anonymous=True)
	rate = rospy.Rate(0.05)



	# Embed to message right away, no need to have seperate parameters
	# Start from where you are left off, continue
	while not rospy.is_shutdown():
		# Position Generation:
		message.p0 = message.pf
		message.pf = random.uniform(-P_MAX, P_MAX)

		# Velocity Generation:
		message.v0 = message.vf
		message.vf = random.uniform(-V_MAX, V_MAX)

		# Initial and Final Time Generation:
		dt = random.random() * 5 + 5 	# Random real number between 5 and 10
		message.t0 = 0 					# Always have to be 0
		message.tf = message.t0 + dt

		# Publish the message:
		pub.publish(message)

		rate.sleep()

if __name__ == '__main__':
	try:
		generate_points()
	except rospy.ROSInterruptException:
		pass
