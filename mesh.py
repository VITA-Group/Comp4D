import subprocess
import os

src = '../4dfy/output/fourdfy_stage_2_low_vram/'
dst = './data/8w/'
os.makedirs(dst, exist_ok=True)

for obj_dir in os.listdir(src):
    key = obj_dir.split('@')[0]
    obj_path = os.path.join(src, obj_dir, 'save/it25000-export/model.obj')
    if not os.path.exists(obj_path):
        print(f'no object file at {obj_path}')
        continue

    dst_path = os.path.join(dst, key+'.ply')
    if os.path.exists(dst_path):
        continue
    
    command = f'python mesh2ply_8w.py {obj_path} {dst_path}'
    result = subprocess.run(command, shell=True)
    print(result)