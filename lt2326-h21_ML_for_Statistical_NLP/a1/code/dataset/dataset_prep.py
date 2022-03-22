# typing
from typing import Dict, List, Tuple, Union
from numpy.typing import NDArray

# General libraries
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter

#
import numpy as np
from sklearn.model_selection import train_test_split

# image library
import cv2

# Utilities functions
from utils import load_jsonl, write_h5_dataset

bbox = List[List[float]]
roi_dict = Dict[str, Union[str, bool, bbox]]
ignore_dict = Dict[str, bbox]


def images_stats(arr: NDArray) -> Tuple[List[float], List[float]]:

    # Calculate mean and standard dev.
    # images_mu = np.mean(train_images, axis=(0, 1, 2)).tolist()
    # images_std = np.std(train_images, axis=(0, 1, 2)).tolist()
    #  using numpy is tooo slow

    # n * var = ∑(x_i − μ)^2​
    #         = x_1^2        - 2μx_1   + μ^2 +
    #           x_2^2        - 2μx_2   + μ^2 +
    #           ...                          +
    #           x_n^2        - 2μx_n   + μ^2
    #         = ∑(x_i^2      - 2μx_i)  + nμ^2
    #         = ∑x_i^2       - 2μ ∑x_i + nμ^2 & ∑x_i = nμ
    #         = ∑x_i^2       - 2nμ^    + nμ^2
    # n * var = ∑x_i^2       - nμ^2
    #     var = (∑x_i^2 / n) - μ^2

    # arr size (b, h, w, c)
    arr = arr.astype(np.float64)
    count = float(np.prod(arr.shape[:3]))  # pixels count per channel

    # mean: μ
    arr_mu = np.einsum("bhwc->c", arr) / count  # type: NDArray

    # ∑x_i^2 / n
    arr_sq = np.einsum("bhwc,bhwc->bhwc", arr, arr)  # x_i^2
    arr_sq_sum = np.einsum("bhwc->c", arr_sq) / count

    # (∑x_i^2 / n) - μ^2
    arr_var = arr_sq_sum - np.einsum("c,c->c", arr_mu, arr_mu)
    arr_std = np.sqrt(arr_var)  # type: NDArray  # standard deviation

    return arr_mu.tolist(), arr_std.tolist()


# split numpy arrays to train, val, and test based on some target label
def split_data(
    data: NDArray, labels: NDArray, target: NDArray,
    split_sizes: Tuple[float, float]
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:

    test_size = split_sizes[1]
    val_perctg = split_sizes[0] / (1 - test_size)

    # split images and labels based on the the selected cahr label
    trn_val_idx, test_idx, _, _ = train_test_split(range(target.shape[0]),
                                                   target,
                                                   test_size=test_size,
                                                   stratify=target,
                                                   random_state=42)
    trn_val_data = data[trn_val_idx]
    trn_val_labels = labels[trn_val_idx]
    trn_val_target = target[trn_val_idx]

    train_idx, val_idx, _, _ = train_test_split(range(trn_val_target.shape[0]),
                                                trn_val_target,
                                                test_size=val_perctg,
                                                stratify=trn_val_target,
                                                random_state=42)

    train_data = trn_val_data[train_idx]
    val_data = trn_val_data[val_idx]
    test_data = data[test_idx]

    train_labels = trn_val_labels[train_idx]
    val_labels = trn_val_labels[val_idx]
    test_labels = labels[test_idx]

    return (train_data, val_data, test_data, train_labels, val_labels,
            test_labels)


# label_shape = (*img.shape[:2], 2)
def label_pixels(shape: Tuple[int, int, int], roi_bboxes: bbox,
                 ignore_bboxes: bbox, char_bboxes: bbox) -> NDArray[np.uint8]:

    labels = np.zeros(shape, dtype=np.uint8)  # construct label

    # Label pixels.
    # Boundery Boxes (bbox) have x, y, w, h format: lower left
    # corner coord, width, height. Sometimes x, y are in negaive value (i.e.
    # outsie the image). Negative values Cannot be used when slicing array. So
    # we replace them with zero.
    # label rois' pixels
    for bbox in roi_bboxes:
        bbox = np.int0(bbox)
        x, y, w, h = bbox
        x0, y0, _, _ = (bbox > 0) * bbox  # set neg val of bbox coord to 0
        labels[y0:y + h, x0:x + w] = 1

    # label ignore areas' pixels
    for bbox in ignore_bboxes:
        bbox = np.int0(bbox)
        x, y, w, h = bbox
        bbox[bbox < 0] = 0
        x0, y0, _, _ = bbox
        labels[y0:y + h, x0:x + w] = 2

    # label selected char
    for bbox in char_bboxes:
        bbox = np.int0(bbox)
        x, y, w, h = bbox
        x0, y0, _, _ = (bbox > 0) * bbox
        labels[y0:y + h, x0:x + w] = 3

    return labels


def slect_char(
        images_path: Path,
        anntns: Dict[str, Union[roi_dict, ignore_dict]]) -> Tuple[str, int]:
    cs = []
    for path in images_path:
        if path.name in anntns:
            roi_anntns = anntns[path.name]["roi"]
            cs += [roi["text"] for roi in roi_anntns if roi["is_chinese"]]

    char_slctd = Counter(cs).most_common(1)[0]

    return char_slctd


def get_bboxes(roi_anntns: roi_dict,
               ignore_anntns: ignore_dict) -> Tuple[bbox, bbox, bbox]:
    # Get boundery boxes
    # Regions of interests boundery boxes
    roi_bboxes = [
        roi["adjusted_bbox"] for roi in roi_anntns if roi["is_chinese"]
    ]

    # Ignores area boundery boxes
    ignore_bboxes = [ignr["bbox"] for ignr in ignore_anntns]
    ignore_bboxes += [
        roi["adjusted_bbox"] for roi in roi_anntns if not roi["is_chinese"]
    ]

    # selected cahr bboxes
    char_bboxes = [
        roi["adjusted_bbox"] for roi in roi_anntns if roi["text"] == char_slctd
    ]

    return roi_bboxes, ignore_bboxes, char_bboxes


def parse_command(parser):

    parser.add_argument("-d",
                        "--dataset_dir",
                        type=str,
                        default="/scratch/lt2326-h21/a1",
                        help="Dataset directory (Base directory).")
    parser.add_argument("-i",
                        "--image_dir",
                        type=str,
                        default="images",
                        help="Relative directory for images")
    parser.add_argument("-ta",
                        "--train_ann_path",
                        type=str,
                        default="train.jsonl",
                        help="Train annotations file path")
    parser.add_argument("-va",
                        "--val_ann_path",
                        type=str,
                        default="val.jsonl",
                        help="Validation annotations file path")
    parser.add_argument("-o",
                        "--output_dir",
                        type=str,
                        default="/scratch/guszarzmo/lt2326_labs/lab1/data",
                        help="Output directory to save the h5 files")
    parser.add_argument("-c",
                        "--selected_char",
                        type=str,
                        help="Show histogram for characters in dataset")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # paser command
    parser = argparse.ArgumentParser(description="Prepare dataset")
    args = parse_command(parser)

    base_dir = Path(args.dataset_dir)
    # Get images path
    images_dir = base_dir / args.image_dir
    images_path = list(images_dir.glob("*.jpg"))

    # load annotations files
    train_anntns = load_jsonl(base_dir / args.train_ann_path)
    val_anntns = load_jsonl(base_dir / args.val_ann_path)
    anntns = {**train_anntns, **val_anntns}  # merge two dicts
    # assert len(train_anntns) + len(val_anntns) == len(anntns)

    # Select a character to classify, if not selected by the user
    if args.selected_char is None:
        char_slctd, char_count = slect_char(images_path, anntns)
    else:
        raise NotImplementedError

    # Load images as array and label them
    images = []  # type: List[NDArray[np.uint8]] # (h, w, c)
    labels = []  # type: List[NDArray[np.uint8]] # (h, w, c)
    target_char = []  # type: List[int]  # if image has the selected char
    for path in tqdm(images_path, desc="Labeling Images", unit="Image"):
        if path.name in anntns:
            # load image
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

            # get boundery boxes for region of interest (roi), ignore areas,
            # and char to classfy
            roi_anntns = anntns[path.name]["roi"]
            ignore_anntns = anntns[path.name]["ignore"]
            roi_bboxes, ignore_bboxes, char_bboxes = get_bboxes(
                roi_anntns, ignore_anntns)

            # label image pixels
            img_labels = label_pixels(img.shape[:2], roi_bboxes, ignore_bboxes,
                                      char_bboxes)

            target_char.append(int((img_labels == 3).any()))
            images.append(img)
            labels.append(img_labels)

    print("\nSpliting dataset...")

    images = np.stack(images)  # (b, h, w, c)
    labels = np.stack(labels)  # (b, h, w)
    target_char = np.array(target_char)

    # splitting data according to target char
    splits = split_data(images, labels, target_char, [0.15, 0.15])
    train_images, val_images, test_images = splits[:3]
    train_labels, val_labels, test_labels = splits[3:]

    # Calculate mean and standard dev. to use in normaliztion
    images_mu, images_std = images_stats(train_images)

    print(f"Number of samples in the training split:   {train_images.shape[0]}")
    print(f"Number of samples in the Validation split: {val_images.shape[0]}")
    print(f"Number of samples in the test split:       {test_images.shape[0]}")

    print()

    print(f"Images Mean: {images_mu}")
    print(f"Images Std:  {images_std}")

    print()

    print("writing h5 files")

    # forgot that pytorch needs images in bchw order so I do it here
    train_images = np.einsum("bhwc->bchw", train_images)
    val_images = np.einsum("bhwc->bchw", val_images)
    test_images = np.einsum("bhwc->bchw", test_images)

    write_h5_dataset(str(Path(args.output_dir) / "train.h5"), train_images,
                     train_labels, ("train_images", "train_labels"),
                     ("uint8", "uint8"), [images_mu, images_std])

    write_h5_dataset(str(Path(args.output_dir) / "val.h5"), val_images,
                     val_labels, ("val_images", "val_labels"),
                     ("uint8", "uint8"), [images_mu, images_std])

    write_h5_dataset(str(Path(args.output_dir) / "test.h5"), test_images,
                     test_labels, ("test_images", "test_labels"),
                     ("uint8", "uint8"), [images_mu, images_std])
