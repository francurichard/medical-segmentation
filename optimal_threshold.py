import numpy as np
import math

def get_optimal_min_threshold(hu_val, seg_val) -> float:
    interest_pixels = np.array(seg_val, dtype=np.uint8)
    height, width = seg_val.shape
    u_bkg = 0
    u_obj = 0
    no_bkg = 0
    no_obj = 0
    for i in range(height):
        for j in range(width):
            if interest_pixels[i][j] == 0:
                u_bkg += self.hu_val[i][j]
                no_bkg += 1
            elif interest_pixels[i][j] == 1:
                u_obj += self.hu_val[i][j] 
                no_obj += 1
    T = (u_bkg / no_bkg + u_obj / no_obj) / 2

    ok = True

    while ok == True:
        u_bkg = 0
        u_obj = 0
        no_bkg = 0
        no_obj = 0
        for i in range(height):
            for j in range(width):
                if (self.hu_val[i][j] < T): 
                    interest_pixels[i][j] = 0
                    no_bkg += 1
                    u_bkg += self.hu_val[i][j]
                elif self.hu_val[i][j] >= T:
                    interest_pixels[i][j] = 1
                    no_obj += 1
                    u_obj += self.hu_val[i][j]

        new_T = (u_bkg / no_bkg + u_obj / no_obj) / 2

        if T == new_T:
            T = new_T
            ok = False
        elif T != new_T:
            T = new_T

    return T