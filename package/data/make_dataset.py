
import os
import sys
import math
import pickle
import json

def char_index():
  char_index = {}
  for ch in list( open("pg100.txt").read() ):
    if char_index.get(ch) is None:
      char_index[ch] = len(char_index)

  stream = "\n".join( ["%s:%d"%(char, index) for char, index in char_index.items()] )
  open("char_index.txt","w").write( stream )
  open("char_index.json", "w").write( json.dumps( char_index ) )

def cropper():
  char_index = json.loads( open("char_index.json", "r").read() ) 
  stream     = open("pg100.txt", "r").read()
  length     = len( stream ) 
  crop_size  = 30 

  with open("DATASET.txt", "w") as f:
    
    for e, i in enumerate( range(length - 30 - 1) ):
      if e%100 == 0:
        print("now iter,", e, length )

      if e>10000:
        break
      inputs = stream[i:i+crop_size]
      output = stream[i+1:i+crop_size+1]
      if "\n" in inputs or "\n" in output:
        continue
      # swap to int
      first = " ".join( ["%d"%char_index[ch] for ch in list(inputs)] )
      last  = " ".join( ["%d"%char_index[ch] for ch in list(output)] )
      f.write( "%s,%s\n"%(first, last) )
      
   

   

if __name__ == '__main__':
  if '--step1' in sys.argv:
    char_index()

  if '--step2' in sys.argv:
    cropper()
