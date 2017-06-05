#!/bin/bash

p=9876
c=$(( $RANDOM % 9 + 1 ))
echo "port=$p, computer=$c"
ssh -L $p:localhost:$p tryglaw \
	-t ssh -L $p:localhost:$p hera \
	-t ssh -L $p:localhost:$p lab110-0$c
