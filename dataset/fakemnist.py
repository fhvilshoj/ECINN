import torch
import numpy as np 
import os
import os.path
import codecs
import torchvision
import torchvision 
from urllib.error import URLError
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity

"""
    Due to limited flexibility of the FrEIA network, I couldn't update pytorch and torchvision to use the new dataloader,
    which side steps The original MNIST source. Therefore, I had to merge code from the 
    [pytorch 1.4.0 code](https://pytorch.org/docs/1.4.0/_modules/torchvision/datasets/mnist.html#MNIST) 
    with the new MNIST sources from the 
    [torchvision 0.10.0 code](https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html#MNIST).
    This is done in lines [62-184]
"""

class FakeMNIST(torchvision.datasets.MNIST):
    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=False, seed=42): 

        super(FakeMNIST, self).__init__(data_dir, train=train, transform=transform, target_transform=target_transform, download=download)

        if self._check_first():
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
 
            self.reorder_dataset()
            if self.train:
                data_file = self.training_file
            else:
                data_file = self.test_file
            self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def _check_first(self):
        return not os.path.exists(os.path.join(self.processed_folder, "done.txt"))

    def reorder_dataset(self): 
        print("Reordering")
        for data_file in [self.training_file, self.test_file]:
            data, targets = torch.load(os.path.join(self.processed_folder, data_file))

            perm = np.random.permutation(data.shape[0])
            data = data[perm]
            labels = torch.arange(10).long().view(10, 1).repeat(1,data.shape[0]//10)
            labels = labels.view(-1) # [0, 0, 0, ..., 1, 1, 1, ... , 9, 9, 9]
            data[torch.arange(data.shape[0]), labels, 0] = 255

            perm = np.random.permutation(data.shape[0])
            data = data[perm]
            labels = labels[perm]

            with open(os.path.join(self.processed_folder, data_file), 'wb') as f:
                torch.save((data, labels), f)
        with open(os.path.join(self.processed_folder, "done.txt"), 'w') as f:
            f.write("Done")

    mirrors = [
            'https://ossci-datasets.s3.amazonaws.com/mnist/',
            'http://yann.lecun.com/exdb/mnist/',
        ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    def download(self):
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder,        exist_ok=True)
        os.makedirs(self.processed_folder,  exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    print(
                        "Failed to download (trying next)\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))
        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')



def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

