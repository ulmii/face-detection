"""wider_dataset dataset."""
import tensorflow_datasets as tfds

class WiderDataset(tfds.object_detection.WiderFace):
  """DatasetBuilder for wider_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = tfds.core.ReadOnlyPath("D:/P/mgr/face-detection-ml/datasets/WIDER")
    extracted_dirs = {
      'wider_train': path,
      'wider_val': path,
      'wider_test': path,
      'wider_annot': path,
    }
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'split': 'train',
                'extracted_dirs': extracted_dirs
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'split': 'val',
                'extracted_dirs': extracted_dirs
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'split': 'test',
                'extracted_dirs': extracted_dirs
            })
    ]