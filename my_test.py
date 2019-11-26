import os.path as osp
import os
import torch

#rarfile不支持创建rar压缩卷,请用zip/7z
import rarfile
def unrar(rar_file, dir_name):
    # rarfile需要unrar支持,
    # linux下pip install unrar, windows下在winrar文件夹找到unrar,加到path里
    rarobj = rarfile.RarFile(rar_file.decode('utf-8'))
    rarobj.extractall(dir_name.decode('utf-8'))
