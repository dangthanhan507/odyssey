# odyssey

Since people like adding weird library names, here's one of them.

Odyssey (noun) - a long wandering or voyage usually marked by many changes of fortune.

This library should be my one-stop shop for running code on the kuka hardware. 

## Docker

remember to pull docker image for ubuntu 24.04:

```bash
docker pull ubuntu:24.04
```

## LCM Messages

```bash
lcm-gen -p example_t.lcm
```

Create custom lcm messages

```bash
lcm-gen -j ./path_to_lcm_files/*.lcm
javac -cp /usr/share/java/lcm.jar lcm_msgs/*.java
jar cf my_types.jar lcm_msgs/*.class
```

**NOTE**: this is assuming we are using Docker which installed lcm-dev into `/usr/share/java/lcm.jar`.