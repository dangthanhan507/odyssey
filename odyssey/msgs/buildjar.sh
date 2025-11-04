#!/bin/sh

# try to automatically determine where the LCM java file is
LCM_JAR="/usr/share/java/lcm.jar"

lcm-gen -j ./*.lcm
lcm-gen -p ./*.lcm


javac -cp $LCM_JAR lcm_msgs/*.java

jar cf my_types.jar lcm_msgs/*.class

export CLASSPATH=/root/amazon_ws/odyssey/odyssey/msgs/my_types.jar:$CLASSPATH