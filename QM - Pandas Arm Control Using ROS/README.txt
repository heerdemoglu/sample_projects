Code developed by Hasan Emre Erdemoglu, 200377106
Code developed as a part of Advanced Robotic Systems Course (QMUL-ECS7004P)

Preliminary Assumptions & possible hotfixes
1) This code segment assumes that you have ROS Melodic and MoveIt installed in your machine. (The screencast provided with this project uses ROS Melodic and Ubuntu 18.04.)
2) If the .py files in the scripts folder are not executable, run following commands
   from the path where .py scripts are located (./ar_week10_test/scripts)
      * chmod +x square_size_generator.py
      * chmod +x move_panda_square.py

To run the package:
1) Download and install the following packages in the src folder of the catkin workspace:
   More information can be found in "ENVIRONMENT" section of the laboratory document.
      * git clone -b melodic-devel https://github.com/ros-planning/panda_moveit_config.git
      * git clone -b melodic-devel https://github.com/ros-planning/moveit_tutorials.git
2) Unzip "ar_week10_test.zip" as "ar_week10_test" folder in your catkin workspace.
3) Build the catkin workspace.
4) In four seperate terminals use the following commands in order:
    * roslaunch panda_moveit_config demo.launch
    * rosrun ar_week10_test square_size_generator.py
    * rosrun ar_week10_test move_panda_square.py
    * rosrun rqt_plot rqt_plot

A sample execution can be found from [this]https://drive.google.com/file/d/1Diz3J90EEyqcM2jgk9R8m1E4WNIWtgWU/view?usp=sharing) link.
