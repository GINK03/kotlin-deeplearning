#! /usr/bin/python3

import glob
import os

jars = ":".join(glob.glob("jars/*.jar"))
print(jars)
targets = " ".join(glob.glob("*.kt"))
os.system("kotlin -cp {jars} MnistKt".format(jars=jars) )

