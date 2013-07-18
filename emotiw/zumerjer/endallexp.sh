#!/bin/bash

if [ -e "$HOME/.cache/search_space_exp.pid" ]; then
    for pid in `cat $HOME/.cache/search_space_exp.pid`; do
        `kill -1 $pid`
        if [ "$?" -ne "0" ]; then
            echo "Failed to clear running experiments: couldn't kill $pid"
            exit 1
        fi 
    done

    echo '' > "$HOME/.cache/search_space_exp.pid"
    echo "Experiment status cleared successfully"
    exit 0

else
    echo "No experiment has been run"
    exit 0
fi

