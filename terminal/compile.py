#! /usr/bin/python3

import glob
import os

jars = ":".join(glob.glob("jars/*.jar"))
print(jars)
targets = " ".join(glob.glob("*.kt"))
os.system("kotlinc {targets} mnist.kt -cp {jars} -include-runtime -d jars/kotlin.jar".format(targets=targets, jars=jars) )

