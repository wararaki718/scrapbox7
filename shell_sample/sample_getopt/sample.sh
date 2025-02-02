#!/bin/bash

while getopts a:b:c: OPT; do
    case $OPT in
        a) echo "a -> ${OPTARG}";;
        b) echo "b -> ${OPTARG}";;
        c) echo "c -> ${OPTARG}";;
        :) echo "none args";;
        ?) echo "none defined";;
    esac
done

echo "DONE"
