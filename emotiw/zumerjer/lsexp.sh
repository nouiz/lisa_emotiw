#!/bin/bash

if [ ! -e "$HOME/.cache/search_space_exp.pid" ]; then
    exit 1;
fi

i=0
for line in `cat $HOME/.cache/search_space_exp.pid`; do
    status=`ps -p $line -o state=`
    
    if [ 0 -ne $? ]; then
        status='DONE'; 
    fi

    echo "["${line}":$status] experiment "${i}
    i=$((i+1)); 
done
