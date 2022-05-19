#! /usr/bin/env python3
"""wider_dataset dataset."""

import os
import re
import numpy as np
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.features import *

_PROJECT_URL = 'arma.sourceforge.net/chokepoint/'
# https://github.com/alversafa/chokepoint-bbs

_CITATION = """

"""

_DESCRIPTION = """

"""

class ChokePoint(tfds.core.GeneratorBasedBuilder):
    """ChokePoint Dataset."""

    VERSION = tfds.core.Version('0.1.0')
    HEIGHT = 600
    WIDTH = 800

    def _info(self):
        frame_shape = (self.HEIGHT, self.WIDTH, 3)
        features = {
            'sequences': 
                Sequence({
                    'image': Image(shape=frame_shape),
                    'image/filename': Text(),
                    'face': BBoxFeature()
                })
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=FeaturesDict(features),
            homepage=_PROJECT_URL,
            citation=_CITATION,
            disable_shuffling=False
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        extracted_dirs = {
            'train': tfds.core.ReadOnlyPath(os.getcwd() + "/../datasets/choke_point/chokepoint-groundtruth/G1"),
            'test': tfds.core.ReadOnlyPath(os.getcwd() + "/../datasets/choke_point/chokepoint-groundtruth/G2"),
            'data': tfds.core.ReadOnlyPath(os.getcwd() + "/../datasets/choke_point/chokepoint-data")
        }
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'split': 'test',
                    'extracted_dirs': extracted_dirs
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'split': 'train',
                    'extracted_dirs': extracted_dirs
                })
        ]

    def _generate_examples(self, split, extracted_dirs):
        """Yields examples."""

        pattern_fname = re.compile(r'(.*.jpg)')
        pattern_annot = re.compile(r'.*.jpg,(\d+),(\d+),(\d+),(\d+)$')

        image_dir = extracted_dirs['data']
        annot_path = extracted_dirs[split]

        for i, sequence in enumerate(glob.glob("{}/eval_annotation_seq_*.txt".format(annot_path))):

            records = []
            with tf.io.gfile.GFile(sequence, 'r') as f:
                while True:
                    line = f.readline()
                    match = pattern_fname.match(line)

                    if match is None:
                        break
                    fname = match.group(1)
                    image_fullpath = os.path.join(image_dir, fname)

                    with tf.io.gfile.GFile(image_fullpath, 'rb') as fp:
                        image = tfds.core.lazy_imports.PIL_Image.open(fp)
                        width, height = image.size

                    match = pattern_annot.match(line)

                    if not match:
                        raise ValueError('Cannot parse: %s' % image_fullpath)

                    (ymin, xmin, wbox, hbox) = map(int, match.groups())
                    ymax = np.clip(ymin + hbox, a_min=0, a_max=height)
                    xmax = np.clip(xmin + wbox, a_min=0, a_max=width)
                    ymin = np.clip(ymin, a_min=0, a_max=height)
                    xmin = np.clip(xmin, a_min=0, a_max=width)

                    record = {
                        'image': image_fullpath,
                        'image/filename': fname,
                        'face': tfds.features.BBox(
                            ymin=ymin / height,
                            xmin=xmin / width,
                            ymax=ymax / height,
                            xmax=xmax / width
                        )
                    }
                    records.append(record)
            yield  i, dict(sequences=records)