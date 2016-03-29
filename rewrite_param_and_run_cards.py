#!/usr/bin/env python

import sys
from math import sqrt

def usage(appname):
    print("Usage: ")
    print("    {} <dir to cards> <H+ mass>".format(appname))

def write_param_card(cardpath, mhc):
    mhc = float(mhc)
    mh1 = 125.
    mh2 = sqrt(mhc**2 + 80.4**2)
    mh3 = mh2

    param_card = [line.rstrip('\n') for line in open(cardpath + "/param_card.dat")]
    new_card = []
    mass_block = False
    for line in param_card:
        if "Block mass" in line:
            mass_block = True
        if mass_block:
            if "mh1" in line:
                line = "   25 {:e} #mh1".format(mh1)
            if "mh2" in line:
                line = "   35 {:e} #mh2".format(mh2)
            if "mh3" in line:
                line = "   36 {:e} #mh3".format(mh3)
            if "mhc" in line:
                line = "   37 {:e} #mhc".format(mhc)
        if "Block" in line and "mass" not in line:
            mass_block = False
        new_card.append(line)
    with open(cardpath + "/param_card.dat", 'w') as f:
        for line in new_card:
            f.write(line + '\n')

def write_run_card(cardpath, scale):
    scale = float(scale)

    run_card = [line.rstrip('\n') for line in open(cardpath + "/run_card.dat")]
    new_card = []
    for line in run_card:
        skip = False
        for c in line:
            if c == " ":
                pass
            elif c == "#":
                skip = True
                break
            else:
                skip = False
                break
        if not skip:
            if " = fixed_ren_scale" in line:
                line = " True = fixed_ren_scale"
            if " = fixed_fac_scale" in line:
                line = " True = fixed_fac_scale"
            if " = scale" in line:
                line = "{:f} = scale ! fixed renormalization scale".format(scale)
            if " = dsqrt_q2fact1" in line:
                line = "{:f} = dsqrt_q2fact1 ! fixed fact scale for pdf1".format(scale)
            if " = dsqrt_q2fact2" in line:
                line = "{:f} = dsqrt_q2fact2 ! fixed fact scale for pdf2".format(scale)
            if " = dynamical_scale_choice" in line:
                line = "-1 = dynamical_scale_choice"
        new_card.append(line)
    with open(cardpath + "/run_card.dat", 'w') as f:
        for line in new_card:
            f.write(line + '\n')

def main(argv):
    if len(argv) < 3 or argv[1] == "help":
        sys.exit(usage(argv[0]))

    cardpath = argv[1]
    mhc = float(argv[2])
    scale = mhc/3

    write_param_card(cardpath, mhc)    
    write_run_card(cardpath, scale)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
