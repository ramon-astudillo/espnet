for machine in $(ls /export/ | grep ^b);do
    if ping -c 1 $machine &> /dev/null;then
        echo $machine
    fi    
done #| tail -n 4
