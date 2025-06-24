import cv2
from lle import LLE


env1 = LLE.from_str("""
                     G  L2S . . S0 S1 S2 S3 . . . .  . 
  .   .  . . .  .  .  .  . . . .  .   
  .   .  . . .  .  .  .  . . . .  .   
  @   @  . . .  .  .  .  . . . .  .   
 L0E  .  . . .  .  .  @  @ @ @ @  @   
  .   .  . . .  .  .  .  . . . .  .   
  .   .  . . .  .  .  .  . . . . L1W
  .   .  . . .  .  .  @  . G . .  .   
  .   .  . . .  .  .  @  @ @ @ @  @   
  .   .  . . .  .  .  .  . . . .  .   
  .   .  . . G  .  .  .  . . . .  .   
  X   X  X X X  X  X  X  X X X X  G   """).build()

img = env1.get_image()
cv2.imwrite("lvl6.png", img)

env2 = LLE.from_str("""  G   @  . . S0 S1 S2 S3 . . . .  . 
  .   .  . . .  .  .  .  . . . .  .   
  .   .  . . .  .  .  .  . . . .  .   
  @   @  . . .  .  .  .  . . . .  .   
  @   .  . . .  .  .  @  @ @ @ @  @   
  .   .  . . .  .  .  .  . . . .  .   
  .   .  . . .  .  .  .  . . . .  @
  .   .  . . .  .  .  @  . G . .  .   
  .   .  . . .  .  .  @  @ @ @ @  @   
  .   .  . . .  .  .  .  . . . .  .   
  .   .  . . G  .  .  .  . . . .  .   
  X   X  X X X  X  X  X  X X X X  G   """).build()
img = env2.get_image()
cv2.imwrite("lvl6-no-lasers.png", img)
