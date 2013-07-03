from sys import argv
from os import system
import subprocess
import curses
import atexit

params = ['lr', 'momentum', 'type', 'units', 'pieces', 'norm_reg']

def cleanup(wnd=None):
    if wnd is not None:
        curses.nocbreak()
        curses.echo()
        wnd.keypad(0)
        curses.endwin()

if __name__ == '__main__':
    atexit.register(cleanup)
    experiment = argv[1]
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
        exp_params = 'experiment_' + str(i) + '_params.py'
        exp_out = 'experiment_' + str(i) + '_output.txt'

        write_params = "if [ -e '" + exp_params + "' ]; then rm '" + exp_params
        write_params += "'; fi; if [ -e '" + exp_out + "' ]; then rm '"
        write_params += exp_out + "'; fi; sed "

        for x in rules:
            if x == 'host':
                continue
            write_params += "-e 's/\\(" + str(x) + "=\\).*$/\\1" + rules[x][i] + "/g' "

        system(write_params + " < experiment_params.py > " + exp_params)
        flag = 'THEANO_FLAGS="device='+devices[i]+',mode=FAST_RUN,floatX=float32"'
        command = flag + ' python ' + experiment + ' ' + str(i)
        if rules['host'][i] != 'localhost':
            exp_pid.append(subprocess.Popen(['ssh',
                                            rules['host'][i],
                                            command],
                                            stdout=open(exp_out, mode='w'),
                                            stderr=open('/dev/null', mode='w')))
        else:
            exp_pid.append(subprocess.Popen(command,
                                            shell=True,
                                            stdout=open(exp_out, mode='w'),
                                            stderr=open('/dev/null', mode='w')))

    wnd = curses.initscr()
    curses.echo()
    curses.cbreak()
    wnd.nodelay(True)
    wnd.keypad(1)
    height, width = wnd.getmaxyx()
    scr = curses.newpad(len(exp_pid) + 2, width)
    scr.nodelay(True)

    curr_pos = 0
    cmd = ""

    display_status = []

    for idx, popen in enumerate(exp_pid):
        scr.addstr(idx, 0, '[' + str(popen.pid) + '] Experiment ' + str(idx))
        display_status.append(False)
    scr.addstr(len(exp_pid), 0, 'KILL> ')
    scr.redrawwin()
    scr.refresh(curr_pos, 0, 0, 0, height, width)
    wnd.refresh()

    while True:

        height, width = wnd.getmaxyx()
        scr.refresh(curr_pos, 0, 0, 0, height, width)
        all_done = reduce(lambda x, y: x and y, [p.returncode is not None for p in exp_pid])

        for idx, popen in enumerate(exp_pid):
            ret_val = popen.poll()

            if ret_val is not None and display_status[idx] is False:
                scr.addstr(idx, 0, '[T:'+str(ret_val)+'] ' + 'Experiment ' + str(idx) + '\n')
                scr.redrawln(idx, 1)
                display_status[idx] = True

        if all_done:
            scr.addstr(len(exp_pid), 0, 'DONE!\n')
            scr.redrawln(len(exp_pid), 1)
            height, width = wnd.getmaxyx()
            scr.refresh(curr_pos, 0, 0, 0, height, width)

            ch = wnd.getch()
            while -1 == ch:
                ch = wnd.getch()

            cleanup(wnd)
            break

        ch = wnd.getch()
        if -1 < ch < 256:
            if chr(ch) == '\n':
                if cmd.isdigit() and int(cmd) in xrange(len(exp_pid)) and exp_pid[int(cmd)].poll() is None:
                    exp_pid[int(cmd)].kill()
                elif len(cmd) > 0 and cmd[0] == 'q':  #note: startswith does not work: "str object has no attribute startswith()"
                    for p in exp_pid:
                        try:
                            p.kill()
                        except:
                            pass
                    cleanup(wnd)
                    break
                cmd = ""
            else:
                cmd += chr(ch)

            scr.addstr(len(exp_pid), 0, 'KILL> ' + cmd + '\n') 
            scr.redrawln(len(exp_pid), 1)
        elif ch == curses.KEY_UP:
            curr_pos = min(len(exp_pid), curr_pos+1)
        elif ch == curses.KEY_DOWN:
            curr_pos = max(0, curr_pos-1)
        elif ch == curses.KEY_BACKSPACE:
            cmd = cmd[:-1]
            scr.addstr(len(exp_pid), 0, 'KILL> ' + cmd + '\n') 
            scr.redrawln(len(exp_pid), 1)

        scr.move(len(exp_pid), len('KILL> ' + cmd))
