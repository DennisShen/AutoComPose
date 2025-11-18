import json
import PIL
import PIL.Image
import torchvision.transforms.functional as F
import random

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

base_path = Path(__file__).absolute().parents[1].absolute()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def square_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
        

def validate_sents_mode(dataset, mllm_mode, use_auto_sents, use_human_sents, split):
    """
    Validate that at most one of mllm_mode, use_auto_sents, or use_human_sents is set.
    """
    active_count = sum([
        mllm_mode is not None,
        use_auto_sents != 0,
        use_human_sents != 0
    ])
    
    if active_count > 1:
        raise ValueError("At most one of mllm_mode, use_auto_sents, or use_human_sents may be set.")
    
    if 'train' in split and mllm_mode is None and dataset not in ['fixmypose', 'posefix']:
        raise ValueError(f"{dataset} can only be used with MLLM modes.")
    
    if use_auto_sents != 0 and dataset != 'posefix':
        raise ValueError(f"AUTO mode can only be used with posefix.")


class CPRDataset(Dataset):
    """
    CPR dataset class which manage CPR data
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, rel_caption) when 'train' in split
            - (reference_name, target_name, rel_caption, group_members) when 'test' in split
    """
    def __init__(self, split: str, dataset: str, mode: str, preprocess: callable, 
                 mllm_mode: str = None, use_cycle_loss: bool = False, use_auto_sents: int = 0, use_human_sents: int = 0):
        """
        :param split: dataset split
        :param dataset: dataset name, should be in ['fixmypose', 'aist', 'posefix']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when 'train' in split
                - (reference_name, target_name, rel_caption, group_members) when 'test' in split
        :param preprocess: function which preprocesses the image
        :param mllm_mode: mllm mode, e.g., "mllm-reverse-mirror-filter_3"
        :param use_cycle_loss: Whether to use cycle loss
        :param use_auto_sents: Whether to use the sentences automatically generated by PoseFix
        :param use_human_sents: Whether to use the sentences generated by humans
        """
        self.preprocess = preprocess
        self.dataset_name = dataset
        self.mode = mode
        self.split = split
        self.mllm_mode = None
        self.use_cycle_loss = use_cycle_loss
        self.REVERSE_TABLE = {'mllm':'reverse', 'reverse':'mllm', 'mirror': 'mirror-reverse', 'mirror-reverse': 'mirror'}
        self.use_auto_sents = use_auto_sents
        self.use_human_sents = use_human_sents

        # validate the sentences mode
        validate_sents_mode(dataset, mllm_mode, use_auto_sents, use_human_sents, split)

        # validate dataset mode
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(base_path / f'{dataset}_dataset' / 'captions' / f'{split}.json') as f:
            self.triplets = json.load(f)

        # split test data if containing more than one sentence
        if 'test' in split and dataset in ['aist']:
            new_triplet = []
            for data in self.triplets:
                for sent in data['sents']:
                    new_data = data.copy()
                    new_data['sents'] = [sent]
                    new_triplet.append(new_data)
            print(f"{dataset} {split} dataset in {mode} mode flattened: from {len(self.triplets)} to {len(new_triplet)}.")
            self.triplets = new_triplet

        # get the image names
        with open(base_path / f'{dataset}_dataset' / 'image_splits' / f'{split}.json') as f:
            self.image_names = json.load(f)

        # setup test dataset
        if 'test' in split:
            print(f"{dataset} {split} dataset in {mode} mode initialized.")
        # setup training dataset
        # w/o mllm
        elif mllm_mode is None:
            if use_cycle_loss:
                # use posefix generated sentences w/ cycle loss
                if use_auto_sents != 0:
                    self.auto_mode = ['auto', 'auto-reverse']
                    self.AUTO_REVERSE_TABLE = {'auto':'auto-reverse', 'auto-reverse':'auto'}
                    print(f"{dataset} {split} dataset in {mode} mode initialized. AUTO {use_auto_sents} mode used. Cycle loss activated.")
                else:
                    raise ValueError("Cycle loss must be used with MLLM modes or AUTO mode.")
            else:
                # use posefix generated sentences w/o cycle loss
                if self.use_auto_sents != 0:
                    print(f"{dataset} {split} dataset in {mode} mode initialized. AUTO {use_auto_sents} mode used.")
                # use human generated sentences
                else:
                    print(f"{dataset} {split} dataset in {mode} mode initialized. {use_human_sents} human sentences used.")
        # w/ mllm
        else:
            self.n_mllm_sents = int(mllm_mode.split('_')[1])
            mllm_commands = mllm_mode.split('_')[0].split('-')

            # check the mllm mode
            if 'mllm' not in mllm_commands:
                raise ValueError("Missing 'mllm' in mllm mode")
            if self.n_mllm_sents > 5:
                raise ValueError("Too many sentences per mllm mode")
            if not set(mllm_commands).issubset(['mllm', 'reverse', 'mirror']):
                raise ValueError("Invalid mllm mode")
            if len(set(mllm_commands)) != len(mllm_commands):
                raise ValueError("Duplicated mllm mode")
            if self.use_cycle_loss and ('reverse' not in mllm_commands):
                raise ValueError("Attempt to use cycle loss without reverse descriptions")

            # decode the mllm mode
            if ('reverse' in mllm_commands) and ('mirror' in mllm_commands):
                mllm_commands.append('mirror-reverse')
            self.mllm_mode = mllm_commands

            # make sure every triplet contains all required descriptions
            original_triplets_len = len(self.triplets)
            self.triplets = [d for d in self.triplets if set(self.mllm_mode).issubset(d.keys())]
            print(f"WARNING: MLLM mode now requires each sample to include all four description types: 'mllm', 'mirror', 'reverse', and 'mirror-reverse'. Only {len(self.triplets)} out of {original_triplets_len} samples were retained after filtering.")

            if self.use_cycle_loss:
                print(f"{dataset} {split} dataset in {mode} mode initialized. {mllm_mode} MLLM mode used. Cycle loss activated.")
            else:
                print(f"{dataset} {split} dataset in {mode} mode initialized. {mllm_mode} MLLM mode used.")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']
                reference_name = self.triplets[index]['reference']
                
                if 'train' in self.split:
                    reference_image_path = base_path / f'{self.dataset_name}_dataset' / 'images' / reference_name
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    
                    target_hard_name = self.triplets[index]['target']
                    target_image_path = base_path / f'{self.dataset_name}_dataset' / 'images' / target_hard_name
                    target_image = self.preprocess(PIL.Image.open(target_image_path))

                    # if mllm sentences are used
                    if self.mllm_mode is not None:
                        # [mllm] / [mllm, mirror] / [mllm, reverse] / [mllm, mirror, reverse, mirror-reverse]
                        s_mllm_mode = random.choice(self.mllm_mode)
                        rel_caption = random.choice(self.triplets[index][s_mllm_mode][:self.n_mllm_sents])

                        # mirror and reverse
                        if 'mirror' in s_mllm_mode:
                            reference_image, target_image = F.hflip(reference_image), F.hflip(target_image)
                        if 'reverse' in s_mllm_mode:
                            reference_image, target_image = target_image, reference_image

                        # if cycle loss is used
                        if self.use_cycle_loss:
                            rel_caption_rv = random.choice(self.triplets[index][self.REVERSE_TABLE[s_mllm_mode]][:self.n_mllm_sents])
                            return reference_image, target_image, rel_caption, rel_caption_rv
                    # if posefix sentences are used
                    elif self.use_auto_sents != 0:
                        if self.use_cycle_loss:
                            s_auto_mode = random.choice(self.auto_mode)
                            rel_caption = random.choice(self.triplets[index][s_auto_mode][:self.use_auto_sents])

                            if 'reverse' in s_auto_mode:
                                reference_image, target_image = target_image, reference_image

                            rel_caption_rv = random.choice(self.triplets[index][self.AUTO_REVERSE_TABLE[s_auto_mode]][:self.use_auto_sents])
                            return reference_image, target_image, rel_caption, rel_caption_rv
                        else:
                            rel_caption = random.choice(self.triplets[index]['auto'][:self.use_auto_sents])
                    # if human sentences are used
                    elif self.use_human_sents != 0:
                        rel_caption = random.choice(self.triplets[index]['sents'][:self.use_human_sents])

                    return reference_image, target_image, rel_caption

                elif 'test' in self.split:
                    rel_caption = self.triplets[index]['sents'][0] # test data contains only one sentence after flattening
                    target_hard_name = self.triplets[index]['target']

                    return reference_name, target_hard_name, rel_caption, group_members

            elif self.mode == 'classic':        
                image_name = self.image_names[index]
                image_path = base_path / f'{self.dataset_name}_dataset' / 'images' / image_name
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")