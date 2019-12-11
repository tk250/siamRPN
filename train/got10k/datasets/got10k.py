from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six


class GOT10k(object):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, ArXiv 2018.
    
    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
        list_file (string, optional): If provided, only read sequences
            specified by the file instead of all sequences in the subset.
    """
    def __init__(self, root_dir1, root_dir2=None, subset='train_i', return_meta=False,
                 visible=True, list_file=None):
        super(GOT10k, self).__init__()
        assert subset in ['train', 'val', 'test', 'train_i'], 'Unknown subset.'
        #self.root_dir1 = root_dir1
        self.subset = subset
        self.return_meta = False if subset == 'test' else return_meta
        self.seq_names1 = os.listdir(root_dir1)
        if root_dir2:
            self.seq_names2 = os.listdir(root_dir2)
        else:
            self.seq_names2 = []
        self.seq_names = self.seq_names1 + self.seq_names2
        
        #infrared images
        if subset == 'train_i' and visible == False:   
            self.seq_dirs = [os.path.join(root_dir1, s, 'infrared')
                             for s in self.seq_names1]
            
            self.anno_files = [os.path.join(root_dir1, s, 'infrared.txt')
                             for s in self.seq_names1]
            if root_dir2:
                anno_files2 = [os.path.join(root_dir2, s, 'groundTruth_i.txt')
                                 for s in self.seq_names2]
                seq_dirs2 = [os.path.join(root_dir2, s, 'i')
                                 for s in self.seq_names2]
                self.seq_dirs.extend(seq_dirs2)
                self.anno_files.extend(anno_files2)
        #rgb images
        elif subset == 'train_i' and visible == True:   
            self.seq_dirs = [os.path.join(root_dir1,s, 'visible')
                             for s in self.seq_names1]
            self.anno_files = [os.path.join(root_dir1, s, 'visible.txt')
                             for s in self.seq_names1]
            if root_dir2:
                anno_files2 = [os.path.join(root_dir2, s, 'groundTruth_v.txt')
                                 for s in self.seq_names2]
                seq_dirs2 = [os.path.join(root_dir2, s, 'v')
                                 for s in self.seq_names2]
                self.seq_dirs.extend(seq_dirs2)
                self.anno_files.extend(anno_files2)
        else:
            if list_file is None:
                list_file = os.path.join(root_dir1, subset, 'list.txt')
            self._check_integrity(root_dir1, subset, list_file)
            '''self.seq_names = os.listdir(os.path.join(root_dir, subset))
            self.seq_dirs = [os.path.join(root_dir, subset, s)
                             for s in self.seq_names]
            self.anno_files = [os.path.join(d, 'groundtruth.txt')
                               for d in self.seq_dirs]
            print(sorted(self.anno_files)[:200])'''

            with open(list_file, 'r') as f:
                self.seq_names1 = f.read().strip().split('\n')
                self.seq_dirs = [os.path.join(root_dir1, subset, s)
                                 for s in self.seq_names1]
                self.anno_files = [os.path.join(d, 'groundtruth.txt')
                                   for d in self.seq_dirs]

    
    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))
        if len(img_files) == 0:
            img_files = sorted(glob.glob(os.path.join(
                self.seq_dirs[index], '*.png')))
        if len(img_files) == 0:
            img_files = sorted(glob.glob(os.path.join(
                self.seq_dirs[index], '*.bmp')))
        if self.anno_files[index].endswith('infrared.txt') or self.anno_files[index].endswith('visible.txt') or self.anno_files[index].endswith('groundtruth.txt'):
            anno = np.loadtxt(self.anno_files[index], delimiter=',')
        else:
            anno = np.loadtxt(self.anno_files[index], delimiter=' ')

        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta:
            meta = self._fetch_meta(self.seq_dirs[index])
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_dirs)

    def _check_integrity(self, root_dir, subset, list_file=None):
        assert subset in ['train', 'val', 'test']
        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')
            
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = os.path.join(seq_dir, 'meta_info.ini')
        with open(meta_file) as f:
            meta = f.read().strip().split('\n')[1:]
        meta = [line.split(': ') for line in meta]
        meta = {line[0]: line[1] for line in meta}

        # attributes
        attributes = ['cover', 'absence', 'cut_by_image']
        for att in attributes:
            meta[att] = np.loadtxt(os.path.join(seq_dir, att + '.label'))

        return meta
