#!/bin/bash

for j in `seq $1 $2` ; do 
    scancel $j
    echo  $j
done