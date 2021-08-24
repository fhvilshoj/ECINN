import shutil
import os

def prepare_destination(dest):
    """
        This function will take existing file at destination, say `foo/bar.bas`,
        and move it to `foo/old/[id]_bar.bas
    """
    if not os.path.isfile(dest): return

    *d, f = dest.split('/')
    d = '/'.join(d)

    old_dir = os.path.join(d,'old')
    os.makedirs(old_dir, exist_ok=True)

    olds = len([of for of in os.listdir(old_dir) if f in of])
    f_   = "[%0i]_%s" % (olds, f)
    
    shutil.move(dest, os.path.join(old_dir, f_))

