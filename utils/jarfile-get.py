
import glob
import os
import sys
def find_local():
  text = os.popen("find . ../../deeplearning4j/ | grep .jar").read()
  for line in text.split("\n"):
    if line == "":
      continue
    os.system("cp {line} ../terminal/jars".format(line=line) )
    print( text )

if __name__ == '__main__':
  if '--find_local' in sys.argv: 
    find_local()
