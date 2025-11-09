# This file was created by chatgpt
# It is used to rotate images to create a larger data set for our positive annotations
# Run from the cascade-training directory


import cv2
import numpy as np
import os
import math

input_file = "roost_info.dat"         # original annotations (eg. roost_info.dat or warrior_info.dat)
output_file = "roost_info_rotated.dat"  # (eg. roost_info_rotated.dat or warrior_info_rotated.dat)
output_dir = "rotated-images"   # (eg. images/roost-positive-rotated or images/warrior-positive-rotated)
os.makedirs(output_dir, exist_ok=True)

angles = range(0, 360, 3)          # desired rotation angles in degrees

def corners_of_box(x, y, w, h):
    return np.array([[x, y],
                     [x + w, y],
                     [x, y + h],
                     [x + w, y + h]], dtype=np.float32)

def rotate_corners(corners, M):
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])         # Nx3
    rotated = (M @ corners_h.T).T                  # Nx2
    return rotated

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if not parts:
            continue
        img_path = parts[0]
        num_objs = int(parts[1])
        boxes = [list(map(int, parts[i:i+4])) for i in range(2, 2 + num_objs*4, 4)]

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: cannot read {img_path}, skipping.")
            continue
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        for angle in angles:
            # 1) Compute rotation matrix for center
            angle_rad = math.radians(angle)
            M0 = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)  # 2x3

            # 2) Find bounding box of the rotated image by rotating the 4 image corners
            img_corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
            ones = np.ones((4,1), dtype=np.float32)
            img_corners_h = np.hstack([img_corners, ones])
            rotated_img_corners = (M0 @ img_corners_h.T).T   # 4x2

            x_min, y_min = rotated_img_corners.min(axis=0)
            x_max, y_max = rotated_img_corners.max(axis=0)

            # 3) Compute size of the new canvas required
            new_w = int(math.ceil(x_max - x_min))
            new_h = int(math.ceil(y_max - y_min))

            # 4) We need to translate the rotated image so that all coords are positive.
            # Translation amounts:
            tx = -x_min
            ty = -y_min

            # 5) Build final transform: first rotate about center, then translate
            # M0 is 2x3: [ [a, b, tx0], [c, d, ty0] ]
            M = M0.copy()
            M[0,2] += tx
            M[1,2] += ty

            # 6) Warp into new canvas size
            rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0) if img.shape[2]==4 else (0,0,0))

            # 7) Rotate each bbox's corners using the same transform, then compute new bbox
            new_boxes = []
            for (x, y, bw, bh) in boxes:
                corners = corners_of_box(x, y, bw, bh)   # 4x2
                rotated_corners = rotate_corners(corners, M)  # 4x2

                x_min_b = int(np.floor(rotated_corners[:,0].min()))
                y_min_b = int(np.floor(rotated_corners[:,1].min()))
                x_max_b = int(np.ceil(rotated_corners[:,0].max()))
                y_max_b = int(np.ceil(rotated_corners[:,1].max()))

                new_wb = x_max_b - x_min_b
                new_hb = y_max_b - y_min_b

                # optional: clamp box so it's inside image bounds
                x_min_b_clamped = max(0, x_min_b)
                y_min_b_clamped = max(0, y_min_b)
                x_max_b_clamped = min(new_w - 1, x_max_b)
                y_max_b_clamped = min(new_h - 1, y_max_b)

                # if clamped size becomes zero or negative, skip this box (object fully out)
                if x_max_b_clamped <= x_min_b_clamped or y_max_b_clamped <= y_min_b_clamped:
                    print(f"Warning: box for {img_path} at angle {angle} fell outside canvas; skipping this object.")
                    continue

                new_wb_clamped = x_max_b_clamped - x_min_b_clamped
                new_hb_clamped = y_max_b_clamped - y_min_b_clamped

                new_boxes.append((x_min_b_clamped, y_min_b_clamped, new_wb_clamped, new_hb_clamped))

            if len(new_boxes) == 0:
                print(f"Notice: after rotating {img_path} by {angle}°, no valid boxes remain — skipping image.")
                continue

            # 8) Save rotated image
            base = os.path.basename(img_path)
            name_noext = os.path.splitext(base)[0]
            out_name = f"{name_noext}_rot{angle}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, rotated)

            # 9) Write new annotation line (path num_objs x y w h ...)
            out_line = f"{out_path} {len(new_boxes)}"
            for (nx, ny, nw, nh) in new_boxes:
                out_line += f" {nx} {ny} {nw} {nh}"
            f_out.write(out_line + "\n")
