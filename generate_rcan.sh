#!/bin/bash

for i in {0..2}
do
  python data_generation_rcan.py --img_n $((i * 300 + 0)) --obj_m2 1.8 --boundary 2.5 --samples_scene 800 --n_walls 6
done
