#!/bin/bash

if [ ! -e $HOME/.cache/search_space_exp.pid ]; then
    exit 1;
fi

max=`cat $HOME/.cache/search_space_exp.pid | wc -l`
if ! (($1<max)); then
    echo "There are only ${max} experiments running."
    exit 1 ;
fi

exit `cat $HOME/.cache/search_space_exp.pid | awk "NR==(($1+1))" | xargs kill -1`
