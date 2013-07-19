from sys import argv
from os import system
import subprocess
import os

params = ['lr', 'momentum', 'type', 'units', 'pieces', 'norm_reg', 'batch_size']

if __name__ == '__main__':
    experiment = os.path.abspath(argv[1])
    prepath, _ = os.path.split(experiment)

    default_machine = 'localhost'

    devices = []
    rules = {}
    exp_pid = []

    add_devices = False
    add_rules = None

    for x in argv[2:]:
        if add_devices:
            if x.startswith('-'):
                add_devices = False
            else:
                devices.append(x)
        elif add_rules is not None:
            if x.startswith('-'):
                add_rules = None
            else:
                rules[add_rules].append(x)

        if x == '-d':
            add_devices = True
        elif (x[2:] in params or x[2:] == 'host') and x[0:2] == '--':
            add_rules = x[2:]
            rules[add_rules] = []

    if len(devices) == 0:
        devices.append('gpu')

    if 'host' not in rules:
        rules['host'] = []
        rules['host'].append(default_machine)

    max_len = max(len(rules[x]) for x in rules)

    for x in rules:
        while len(rules[x]) < max_len:
            rules[x].append(rules[x][-1])

    while len(devices) < max_len:
        devices.append(devices[-1])

    write_params = ""
    for i in xrange(max_len):
        exp_params = os.path.join(prepath, 'experiment_' + str(i) + '_params.py')
        exp_out = os.path.join(prepath, 'experiment_' + str(i) + '_output.txt')
        exp_err = os.path.join(prepath, 'experiment_' + str(i) + '_err.txt')

        write_params = "if [ -e '" + exp_params + "' ]; then rm '" + exp_params
        write_params += "'; fi; if [ -e '" + exp_out + "' ]; then rm '"
        write_params += exp_out + "'; fi; "
        write_params += "if [ -e '" + exp_err + "' ]; then rm '" + exp_err + "'; fi; sed "

        for x in rules:
            if x == 'host':
                continue
            write_params += "-e 's/\\(" + str(x) + "=\\).*$/\\1" + rules[x][i] + "/g' "

        system(write_params + " < experiment_params.py > " + exp_params)
        
        has_device = system('grep -i "device=" ' + exp_params)
        has_host = system('grep -i "device=" ' + exp_params)

        if has_device != 0: #EXIT_SUCCESS => entry found
            system('echo "device=\\\"' + devices[i] + '\\\"" >> ' + exp_params)
        else:
            system("sed 's/\\(device=\\).*$/\\1\\\"" + devices[i] + "\\\"/g' < {} > {}".format(exp_params, exp_params))
        if has_host != 0: #EXIT_SUCCESS => entry found
            system('echo "host=\\\"' + rules['host'][i] + '\\\"" >> ' + exp_params)
        else:
            system("sed 's\\(host=\\).*$/\\1\\\"" + rules['host'][i] + "\\\"/g' < {} > {}".format(exp_params, exp_params))

        flag = 'THEANO_FLAGS="device='+devices[i]+',mode=FAST_RUN,floatX=float32"'
        command = flag + ' python ' + experiment + ' ' + str(i)
        if rules['host'][i] != 'localhost':
            exp_pid.append(subprocess.Popen(['ssh',
                                            rules['host'][i],
                                            command],
                                            stdout=open(exp_out, mode='w'),
                                            stderr=open(exp_err, mode='w')))
        else:
            exp_pid.append(subprocess.Popen(command,
                                            shell=True,
                                            stdout=open(exp_out, mode='w'),
                                            stderr=open(exp_err, mode='w')))

        fout = open(os.path.join(os.environ['HOME'], '.cache', 'search_space_exp.pid')
        for pid in exp_pid:
            fout.write(str(pid) + '\n')

