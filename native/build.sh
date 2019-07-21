#!/bin/bash

g++ -o libannoy.jnilib -fPIC -lc -shared -I. -I /home/jotschi/workspaces/mesh/annoy/src/ -I /opt/jvm/java8/include/ -I /opt/jvm/java8/include/linux/ com_spotify_annoy_jni_base_AnnoyIndexImpl.cpp
