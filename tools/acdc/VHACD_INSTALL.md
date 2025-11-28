
## Collision decomposition
You need to install trimesh & vhacd 

Installing vhacd:
```
git clone https://github.com/kmammou/v-hacd.git
cd v-hacd/app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
export PATH=$PATH:<install path>/v-hacd/app/build
```
Make a little change on trimesh's vhacd code:
```
vim <path to trimesh>/trimesh/interfaces/vhacd.py
# vim /home/yandan/anaconda3/envs/atiss/lib/python3.8/site-packages/trimesh/interfaces/vhacd.py
```
change line 43 
```
argstring = ' --input $MESH_0 --output $MESH_POST --log $SCRIPT'
```
into 
```
argstring = ' $MESH_0 '
```
Then run `python collision_decompose.py`. You may need to modify the `base_dir` in the code, which is the path of the scene.
For example, compute collision decomposition:
```
objdir=output_scene/obj
for entry in `ls $objdir`; do  #109
    python utils/collision_decompose.py $entry
done
```
## Collision decomposition
You need to install trimesh & vhacd 

Installing vhacd:
```
## Collision decomposition
You need to install trimesh & vhacd 

Installing vhacd:
```
git clone https://github.com/kmammou/v-hacd.git
cd v-hacd/app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
export PATH=$PATH:<install path>/v-hacd/app/build
```
Make a little change on trimesh's vhacd code:
```
vim <path to trimesh>/trimesh/interfaces/vhacd.py
# vim /home/yandan/anaconda3/envs/atiss/lib/python3.8/site-packages/trimesh/interfaces/vhacd.py
```
change line 43 
```
argstring = ' --input $MESH_0 --output $MESH_POST --log $SCRIPT'
```
into 
```
argstring = ' $MESH_0 '
```
Then run `python collision_decompose.py`. You may need to modify the `base_dir` in the code, which is the path of the scene.
For example, compute collision decomposition:
```
objdir=output_scene/obj
for entry in `ls $objdir`; do  #109
    python utils/collision_decompose.py $entry
done
```
## Collision decomposition
You need to install trimesh & vhacd 

Installing vhacd:
```
git clone https://github.com/kmammou/v-hacd.git
cd v-hacd/app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
export PATH=$PATH:<install path>/v-hacd/app/build
```
Make a little change on trimesh's vhacd code:
```
vim <path to trimesh>/trimesh/interfaces/vhacd.py
# vim /home/yandan/anaconda3/envs/atiss/lib/python3.8/site-packages/trimesh/interfaces/vhacd.py
```
change line 43 
```
argstring = ' --input $MESH_0 --output $MESH_POST --log $SCRIPT'
```
into 
```
argstring = ' $MESH_0 '
```
Then run `python collision_decompose.py`. You may need to modify the `base_dir` in the code, which is the path of the scene.
For example, compute collision decomposition:
```
objdir=output_scene/obj
for entry in `ls $objdir`; do  #109
    python utils/collision_decompose.py $entry
done
```
git clone https://github.com/kmammou/v-hacd.git
cd v-hacd/app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
export PATH=$PATH:<install path>/v-hacd/app/build
```
Make a little change on trimesh's vhacd code:
```
vim <path to trimesh>/trimesh/interfaces/vhacd.py
# vim /home/yandan/anaconda3/envs/atiss/lib/python3.8/site-packages/trimesh/interfaces/vhacd.py
```
change line 43 
```
argstring = ' --input $MESH_0 --output $MESH_POST --log $SCRIPT'
```
into 
```
argstring = ' $MESH_0 '
```
Then run `python collision_decompose.py`. You may need to modify the `base_dir` in the code, which is the path of the scene.
For example, compute collision decomposition:
```
objdir=output_scene/obj
for entry in `ls $objdir`; do  #109
    python utils/collision_decompose.py $entry
done
```