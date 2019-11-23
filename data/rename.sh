#!/bin/bash

for f in *.BMP; do
	mv -- "$f" "${f%.BMP}.bmp"
done
