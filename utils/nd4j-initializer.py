
import os
import sys
import math

def main():
  os.system("wget http://central.maven.org/maven2/org/nd4j/nd4j-native/0.8.0/nd4j-native-0.8.0.jar")
  os.system("wget http://central.maven.org/maven2/org/nd4j/nd4j-api/0.8.0/nd4j-api-0.8.0.jar")
  os.system("wget http://central.maven.org/maven2/org/nd4j/nd4j-jblas/0.4-rc3.6/nd4j-jblas-0.4-rc3.6.jar")
  os.system("wget http://central.maven.org/maven2/org/nd4j/nd4j-native-platform/0.8.0/nd4j-native-platform-0.8.0.jar")
  os.system("wget http://central.maven.org/maven2/org/nd4j/nd4j-base64/0.8.0/nd4j-base64-0.8.0.jar")
  os.system("wget http://central.maven.org/maven2/org/nd4j/nd4j-x86/0.4-rc3.8/nd4j-x86-0.4-rc3.8.jar")
  os.system("")
  os.system("wget http://central.maven.org/maven2/org/slf4j/slf4j-api/1.7.25/slf4j-api-1.7.25.jar")
  os.system("wget http://central.maven.org/maven2/org/slf4j/slf4j-simple/1.7.25/slf4j-simple-1.7.25.jar")
  
  os.system("mv *.jar ../terminal/jars")

if __name__ == '__main__':
  main()
